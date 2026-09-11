import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from benchmark import parse_prometheus, select_metrics, summarize_telemetry, run_request


class FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(length))
        chunks = [
            {"choices": [{"text": "hello"}]},
            {"choices": [{"text": " world"}]},
            {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
        ]
        payload = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(payload.encode())


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_prometheus_parse_and_filter(self):
        raw = '# HELP power test\npower{gpu="0"} 300.5\npower{gpu="1"} 99\nutil{gpu="0"} 87\n'
        parsed = parse_prometheus(raw)
        selected = select_metrics(parsed, {"power": "power", "util": "util"}, {"gpu": "0"})
        self.assertEqual(selected, {"power": [300.5], "util": [87.0]})

    def test_telemetry_summary_uses_window(self):
        snapshots = [
            {"monotonic": 1.0, "metrics": {"power": [100]}},
            {"monotonic": 2.0, "metrics": {"power": [200]}},
            {"monotonic": 3.0, "metrics": {"power": [300]}},
        ]
        summary = summarize_telemetry(snapshots, 1.5, 3.0)
        self.assertEqual(summary["snapshot_count"], 2)
        self.assertEqual(summary["power"], 250)

    def test_streaming_request_counts_usage_and_ttft(self):
        config = {
            "inference_url": f"http://127.0.0.1:{self.server.server_port}/v1/completions",
            "model": "fake",
            "prompt": "test",
            "output_tokens": 2,
            "concurrency": 1,
            "request_timeout_seconds": 5,
        }
        result = run_request(config, 0)
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output_tokens, 2)
        self.assertIsNotNone(result.ttft_seconds)
        self.assertGreaterEqual(result.elapsed_seconds, result.ttft_seconds)


if __name__ == "__main__":
    unittest.main()
