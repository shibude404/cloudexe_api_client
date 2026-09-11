import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from benchmark import (
    TelemetryCollector,
    execute_run,
    normalize_scrape,
    parse_prometheus,
    save_run,
    summarize_metric,
    summarize_telemetry,
)
from mock_server import GIB, SyntheticHandler, metrics_text


def synthetic_config(port=0, scenario="constant"):
    base = f"http://127.0.0.1:{port}"
    return {
        "synthetic_test_data": True,
        "inference_url": f"{base}/v1/completions",
        "api_style": "completions",
        "model": "synthetic-model",
        "prompt": "synthetic prompt",
        "output_tokens": 8,
        "temperature": 0.0,
        "concurrency": 1,
        "warmup_requests": 0,
        "repetitions": 1,
        "request_timeout_seconds": 5,
        "monitoring_url": f"{base}/metrics?scenario={scenario}",
        "monitoring_auth": {"type": "none", "token_env": ""},
        "monitoring_timeout_seconds": 0.1 if scenario == "timeout" else 2,
        "collection_interval_seconds": 0.02,
        "max_acceptable_gap_seconds": 1,
        "gpu_selector": {"gpu": "gpu0", "allocation": "local-test"},
        "metrics": {
            "gpu_power_watts": {"source_name": "synthetic_gpu_power_watts", "source_unit": "watts", "report_unit": "watts", "required": True},
            "gpu_utilization_percent": {"source_name": "synthetic_gpu_activity_percent", "source_unit": "percent", "report_unit": "percent", "required": True},
            "memory_controller_utilization_percent": {"source_name": "synthetic_memory_controller_activity_percent", "source_unit": "percent", "report_unit": "percent", "required": True},
            "gpu_memory_used_gib": {"source_name": "synthetic_gpu_memory_used_bytes", "source_unit": "bytes", "report_unit": "GiB", "conversion_factor": 1 / GIB, "aggregation": "peak", "required": True},
            "gpu_memory_capacity_gib": {"source_name": "synthetic_gpu_memory_capacity_bytes", "source_unit": "bytes", "report_unit": "GiB", "conversion_factor": 1 / GIB, "aggregation": "peak", "required": True},
            "sm_activity_percent": {"source_name": "synthetic_sm_activity_percent", "source_unit": "percent", "report_unit": "percent", "required": False},
        },
    }


def snapshots_for(config, times_and_values, metric="gpu_power_watts"):
    return [{
        "collection_monotonic": stamp,
        "parsed_samples": [{"logical_metric": metric, "collection_monotonic": stamp, "value": value}],
        "errors": [],
    } for stamp, value in times_and_values]


class CalculationTests(unittest.TestCase):
    def test_parse_source_timestamp_and_malformed_line(self):
        parsed, errors = parse_prometheus('power{gpu="0"} 300 123456\nbad line\n')
        self.assertEqual(parsed[0]["source_timestamp"], 123456)
        self.assertEqual(len(errors), 1)

    def test_two_gpu_filter_and_units(self):
        parsed, errors = parse_prometheus(metrics_text("constant", 0))
        self.assertFalse(errors)
        normalized, selection_errors = normalize_scrape(parsed, synthetic_config(), "now", 1.0)
        self.assertFalse(selection_errors)
        values = {item["logical_metric"]: item["value"] for item in normalized}
        self.assertEqual(values["gpu_power_watts"], 300)
        self.assertEqual(values["gpu_memory_used_gib"], 20)
        self.assertTrue(all(item["gpu_identity"]["gpu"] == "gpu0" for item in normalized))

    def test_constant_power(self):
        result = summarize_metric([(0, 300), (1, 300), (2, 300)], 0.5, 1.5, 2)
        self.assertTrue(result["valid"])
        self.assertEqual(result["value"], 300)

    def test_linear_power_and_boundary_interpolation(self):
        result = summarize_metric([(0, 200), (1, 300), (2, 400)], 0, 2, 2)
        self.assertAlmostEqual(result["value"], 300)
        clipped = summarize_metric([(-1, 100), (0, 200), (2, 400), (3, 500)], 0.5, 1.5, 3)
        self.assertAlmostEqual(clipped["value"], 300)

    def test_irregular_intervals_are_time_weighted(self):
        result = summarize_metric([(0, 0), (1, 100), (4, 100)], 0, 4, 4)
        self.assertAlmostEqual(result["value"], 87.5)

    def test_large_gap_invalidates_metric(self):
        result = summarize_metric([(0, 300), (5, 300)], 1, 4, 2)
        self.assertFalse(result["valid"])
        self.assertIn("exceeds", result["reason"])

    def test_missing_utilization_does_not_invalidate_power(self):
        config = synthetic_config()
        snapshots = snapshots_for(config, [(0, 300), (1, 300), (2, 300)])
        summary = summarize_telemetry(snapshots, config, 0.5, 1.5)
        self.assertTrue(summary["metrics"]["gpu_power_watts"]["valid"])
        self.assertFalse(summary["metrics"]["gpu_utilization_percent"]["valid"])

    def test_duplicate_matching_power_is_ambiguous(self):
        config = synthetic_config()
        parsed, _ = parse_prometheus(metrics_text("duplicate_power", 0))
        normalized, errors = normalize_scrape(parsed, config, "now", 0)
        self.assertTrue(any(error["kind"] == "ambiguous" for error in errors))
        self.assertFalse(any(item["logical_metric"] == "gpu_power_watts" for item in normalized))

    def test_memory_peak_and_percentage(self):
        config = synthetic_config()
        parsed, _ = parse_prometheus(metrics_text("constant", 0))
        snapshots = []
        for stamp in (0, 1, 2):
            normalized, errors = normalize_scrape(parsed, config, "now", stamp)
            snapshots.append({"collection_monotonic": stamp, "parsed_samples": normalized, "errors": errors})
        summary = summarize_telemetry(snapshots, config, 0.5, 1.5)
        self.assertEqual(summary["metrics"]["gpu_memory_used_gib"]["value"], 20)
        self.assertEqual(summary["metrics"]["gpu_memory_capacity_gib"]["value"], 80)
        self.assertEqual(summary["gpu_memory_peak_percent"]["value"], 25)


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SyntheticHandler.metric_calls = 0
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), SyntheticHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_manual_metrics_fetch(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/metrics", timeout=2) as response:
            text = response.read().decode()
        self.assertIn('synthetic_gpu_power_watts{gpu="gpu0",allocation="local-test"} 300.0', text)
        self.assertIn('synthetic_gpu_power_watts{gpu="gpu1",allocation="other-test"} 450', text)

    def test_timeout_is_recorded_and_collection_continues(self):
        collector = TelemetryCollector(synthetic_config(self.port, "timeout"))
        first = collector.scrape()
        second = collector.scrape()
        self.assertTrue(first["errors"])
        self.assertTrue(second["errors"])
        self.assertEqual(len(collector.snapshots), 2)

    def test_end_to_end_run_and_artifacts(self):
        config = synthetic_config(self.port)
        run = execute_run(config, 1)
        self.assertEqual(run["total_output_tokens"], 8)
        self.assertGreater(run["aggregate_output_tokens_per_second"], 0)
        self.assertIsNotNone(run["mean_ttft_seconds"])
        metrics = run["telemetry_summary"]["metrics"]
        self.assertEqual(metrics["gpu_power_watts"]["value"], 300)
        self.assertEqual(metrics["gpu_utilization_percent"]["value"], 80)
        self.assertEqual(metrics["memory_controller_utilization_percent"]["value"], 60)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = save_run(config, run, temp_dir)
            expected = {
                "config.sanitized.json", "requests.jsonl", "telemetry_raw.jsonl",
                "telemetry_parsed.csv", "summary.json", "telemetry.svg",
            }
            self.assertEqual({path.name for path in Path(run_dir).iterdir()}, expected)
            self.assertIn("SYNTHETIC TEST DATA", (run_dir / "telemetry.svg").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertTrue(summary["synthetic_test_data"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
