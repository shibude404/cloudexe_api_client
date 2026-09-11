"""Dependency-free vLLM benchmark and validity-aware telemetry collector."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import html
import json
import math
import os
import re
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+'
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|Inf|-Inf)'
    r'(?:\s+(-?\d+(?:\.\d+)?))?$'
)
LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def parse_prometheus(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse Prometheus text samples and retain malformed-line errors."""
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            errors.append(f"line {line_number}: malformed Prometheus sample: {line}")
            continue
        name, raw_labels, raw_value, source_timestamp = match.groups()
        value = float(raw_value)
        if not math.isfinite(value):
            errors.append(f"line {line_number}: non-finite value for {name}")
            continue
        labels = {
            key: value.replace(r'\"', '"').replace(r"\\", "\\")
            for key, value in LABEL_RE.findall(raw_labels or "")
        }
        samples.append({
            "source_metric": name,
            "labels": labels,
            "source_value": value,
            "source_timestamp": float(source_timestamp) if source_timestamp else None,
        })
    return samples, errors


def labels_match(labels: dict[str, str], required: dict[str, str]) -> bool:
    return all(labels.get(key) == value for key, value in required.items())


def _metric_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs = config.get("metrics", {})
    if not isinstance(specs, dict):
        raise ValueError("metrics must be an object")
    return specs


def normalize_scrape(
    parsed: list[dict[str, Any]], config: dict[str, Any], collection_utc: str,
    collection_monotonic: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select one series per logical metric and normalize units."""
    normalized: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    selector = config.get("gpu_selector", {})
    for logical_name, spec in _metric_specs(config).items():
        matches = [
            item for item in parsed
            if item["source_metric"] == spec.get("source_name")
            and labels_match(item["labels"], selector)
        ]
        if not matches:
            if spec.get("required", False):
                errors.append({"metric": logical_name, "kind": "missing", "message": "required metric absent"})
            continue
        if len(matches) > 1:
            errors.append({
                "metric": logical_name,
                "kind": "ambiguous",
                "message": f"{len(matches)} series matched selected GPU/allocation",
            })
            continue
        item = matches[0]
        factor = float(spec.get("conversion_factor", 1.0))
        normalized.append({
            "collection_timestamp_utc": collection_utc,
            "collection_monotonic": collection_monotonic,
            "source_timestamp": item["source_timestamp"],
            "gpu_identity": {key: item["labels"].get(key) for key in selector},
            "labels": item["labels"],
            "logical_metric": logical_name,
            "source_metric": item["source_metric"],
            "source_value": item["source_value"],
            "source_unit": spec.get("source_unit", "unknown"),
            "value": item["source_value"] * factor,
            "unit": spec.get("report_unit", spec.get("source_unit", "unknown")),
        })
    return normalized, errors


def _interpolate(left: tuple[float, float], right: tuple[float, float], target: float) -> float:
    if right[0] == left[0]:
        return left[1]
    fraction = (target - left[0]) / (right[0] - left[0])
    return left[1] + fraction * (right[1] - left[1])


def summarize_metric(
    points: list[tuple[float, float]], window_start: float, window_end: float,
    max_gap_seconds: float, aggregation: str = "time_weighted_average",
) -> dict[str, Any]:
    """Summarize one metric with bracketing, interpolation, and gap checks."""
    result: dict[str, Any] = {
        "valid": False, "value": None, "reason": None,
        "window_start_monotonic": window_start, "window_end_monotonic": window_end,
    }
    if window_end <= window_start:
        result["reason"] = "invalid workload window"
        return result
    ordered = sorted(points)
    before = [point for point in ordered if point[0] <= window_start]
    after = [point for point in ordered if point[0] >= window_end]
    if not before or not after:
        result["reason"] = "telemetry does not bracket both workload boundaries"
        return result
    left = before[-1]
    right = after[0]
    relevant_original = [point for point in ordered if left[0] <= point[0] <= right[0]]
    gaps = [b[0] - a[0] for a, b in zip(relevant_original, relevant_original[1:])]
    if gaps and max(gaps) > max_gap_seconds:
        result["reason"] = f"telemetry gap {max(gaps):.6g}s exceeds {max_gap_seconds:.6g}s"
        return result
    start_value = _interpolate(left, next((p for p in ordered if p[0] >= window_start), left), window_start)
    end_left = next((p for p in reversed(ordered) if p[0] <= window_end), right)
    end_value = _interpolate(end_left, right, window_end)
    clipped = [(window_start, start_value)]
    clipped.extend(point for point in ordered if window_start < point[0] < window_end)
    clipped.append((window_end, end_value))
    if aggregation == "peak":
        value = max(point[1] for point in clipped)
    else:
        area = sum(
            (b[0] - a[0]) * (a[1] + b[1]) / 2
            for a, b in zip(clipped, clipped[1:])
        )
        value = area / (window_end - window_start)
    result.update({"valid": True, "value": value, "reason": None, "point_count": len(clipped)})
    return result


def summarize_telemetry(
    snapshots: list[dict[str, Any]], config: dict[str, Any],
    window_start: float, window_end: float,
) -> dict[str, Any]:
    specs = _metric_specs(config)
    max_gap = float(config.get("max_acceptable_gap_seconds", 3.0))
    summary: dict[str, Any] = {"timestamp_basis": "collection_monotonic", "metrics": {}}
    for logical_name, spec in specs.items():
        points = [
            (sample["collection_monotonic"], sample["value"])
            for snapshot in snapshots for sample in snapshot.get("parsed_samples", [])
            if sample["logical_metric"] == logical_name
        ]
        metric_summary = summarize_metric(
            points, window_start, window_end, max_gap,
            aggregation=spec.get("aggregation", "time_weighted_average"),
        )
        metric_summary["unit"] = spec.get("report_unit", spec.get("source_unit", "unknown"))
        snapshot_errors = [
            error for snapshot in snapshots for error in snapshot.get("errors", [])
            if error.get("metric") == logical_name
        ]
        if snapshot_errors:
            metric_summary["errors"] = snapshot_errors
            if any(error.get("kind") == "ambiguous" for error in snapshot_errors):
                metric_summary.update({"valid": False, "value": None, "reason": "ambiguous matching series"})
        summary["metrics"][logical_name] = metric_summary

    used = summary["metrics"].get("gpu_memory_used_gib", {})
    capacity = summary["metrics"].get("gpu_memory_capacity_gib", {})
    summary["gpu_memory_peak_percent"] = {
        "valid": bool(used.get("valid") and capacity.get("valid") and capacity.get("value")),
        "value": (100 * used["value"] / capacity["value"])
        if used.get("valid") and capacity.get("valid") and capacity.get("value") else None,
        "unit": "percent",
        "reason": None if used.get("valid") and capacity.get("valid") and capacity.get("value")
        else "valid memory-used and capacity values required",
    }
    return summary


@dataclass
class RequestResult:
    request_id: int
    success: bool
    start_utc: str
    end_utc: str
    elapsed_seconds: float
    ttft_seconds: float | None
    output_tokens: int | None
    error: str | None
    events: list[dict[str, Any]]


def _event_has_output(event: dict[str, Any]) -> bool:
    return any(choice.get("text") or (choice.get("delta") or {}).get("content") for choice in event.get("choices", []))


def run_request(config: dict[str, Any], request_id: int) -> RequestResult:
    api_key = os.environ.get(config.get("api_key_env", "CLOUDEXE_API_KEY"), "")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body: dict[str, Any] = {
        "model": config["model"], "max_tokens": int(config["output_tokens"]),
        "stream": True, "stream_options": {"include_usage": True},
        "temperature": config.get("temperature", 0.0),
    }
    if config.get("api_style", "completions") == "chat":
        body["messages"] = [{"role": "user", "content": config["prompt"]}]
    else:
        body["prompt"] = config["prompt"]
    request = urllib.request.Request(
        config["inference_url"], data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    start_utc, start, first_token = utc_now(), time.perf_counter(), None
    events: list[dict[str, Any]] = []
    output_tokens, error = None, None
    try:
        with urllib.request.urlopen(request, timeout=float(config.get("request_timeout_seconds", 60))) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                events.append(event)
                if first_token is None and _event_has_output(event):
                    first_token = time.perf_counter()
                usage = event.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    output_tokens = int(usage["completion_tokens"])
        if output_tokens is None:
            raise ValueError("stream ended without usage.completion_tokens")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    return RequestResult(
        request_id, error is None, start_utc, utc_now(), end - start,
        first_token - start if first_token is not None else None,
        output_tokens, error, events,
    )


class TelemetryCollector:
    """Periodic raw-endpoint collector with synchronous boundary samples."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.snapshots: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def scrape(self) -> dict[str, Any]:
        stamp = {"collection_timestamp_utc": utc_now(), "collection_monotonic": time.perf_counter()}
        try:
            headers: dict[str, str] = {}
            auth = self.config.get("monitoring_auth", {})
            token = os.environ.get(auth.get("token_env", ""), "") if auth.get("token_env") else ""
            if auth.get("type") == "bearer" and token:
                headers["Authorization"] = f"Bearer {token}"
            request = urllib.request.Request(self.config["monitoring_url"], headers=headers)
            with urllib.request.urlopen(request, timeout=float(self.config.get("monitoring_timeout_seconds", 5))) as response:
                raw = response.read().decode()
            parsed, parse_errors = parse_prometheus(raw)
            normalized, selection_errors = normalize_scrape(
                parsed, self.config, stamp["collection_timestamp_utc"], stamp["collection_monotonic"]
            )
            stamp.update({
                "raw_response": raw, "parsed_samples": normalized,
                "errors": [{"kind": "parse", "message": error} for error in parse_errors] + selection_errors,
            })
        except Exception as exc:
            stamp.update({"raw_response": None, "parsed_samples": [], "errors": [{
                "kind": "collection", "message": f"{type(exc).__name__}: {exc}"
            }]})
        with self._lock:
            self.snapshots.append(stamp)
        return stamp

    def start(self) -> None:
        self.scrape()
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self, final_scrape: bool = True) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        if final_scrape:
            self.scrape()

    def _collect(self) -> None:
        interval = float(self.config.get("collection_interval_seconds", 1.0))
        while not self._stop.wait(interval):
            self.scrape()


def execute_run(config: dict[str, Any], repetition: int) -> dict[str, Any]:
    collector = TelemetryCollector(config)
    collector.start()
    start_utc, start = utc_now(), time.perf_counter()
    concurrency = int(config["concurrency"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        requests = list(executor.map(lambda i: run_request(config, i), range(concurrency)))
    end, end_utc = time.perf_counter(), utc_now()
    collector.stop(final_scrape=True)
    successful = [item for item in requests if item.success]
    total_tokens = sum(item.output_tokens or 0 for item in successful)
    ttfts = [item.ttft_seconds for item in successful if item.ttft_seconds is not None]
    return {
        "synthetic_test_data": bool(config.get("synthetic_test_data", False)),
        "repetition": repetition, "workload_start_utc": start_utc, "workload_end_utc": end_utc,
        "workload_start_monotonic": start, "workload_end_monotonic": end,
        "workload_elapsed_seconds": end - start,
        "successful_requests": len(successful), "failed_requests": len(requests) - len(successful),
        "total_output_tokens": total_tokens,
        "aggregate_output_tokens_per_second": total_tokens / (end - start) if successful else None,
        "mean_ttft_seconds": sum(ttfts) / len(ttfts) if ttfts else None,
        "request_results": [asdict(item) for item in requests],
        "telemetry_summary": summarize_telemetry(collector.snapshots, config, start, end),
        "raw_telemetry": collector.snapshots,
    }


def sanitized_config(config: dict[str, Any]) -> dict[str, Any]:
    secrets = {"api_key", "prometheus_token", "authorization", "token"}
    return {key: "<redacted>" if key.lower() in secrets else value for key, value in config.items()}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_svg(path: Path, run: dict[str, Any]) -> None:
    """Write a simple dependency-free multi-panel telemetry plot."""
    metrics = ["gpu_power_watts", "gpu_utilization_percent", "memory_controller_utilization_percent", "gpu_memory_used_gib"]
    samples = [sample for snap in run["raw_telemetry"] for sample in snap.get("parsed_samples", [])]
    width, panel_h, margin = 900, 150, 55
    height = 55 + panel_h * len(metrics)
    start, end = run["workload_start_monotonic"], run["workload_end_monotonic"]
    all_times = [sample["collection_monotonic"] for sample in samples] or [start, end]
    min_t, max_t = min(all_times), max(all_times)
    span = max(max_t - min_t, 1e-9)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="25" font-family="sans-serif" font-weight="bold">SYNTHETIC TEST DATA — Telemetry</text>']
    for index, metric in enumerate(metrics):
        top = 45 + index * panel_h
        points = [(s["collection_monotonic"], s["value"]) for s in samples if s["logical_metric"] == metric]
        max_v = max((value for _, value in points), default=1) or 1
        parts.append(f'<rect x="{margin}" y="{top}" width="{width-2*margin}" height="110" fill="none" stroke="#bbb"/>')
        parts.append(f'<text x="5" y="{top+15}" font-family="sans-serif" font-size="11">{html.escape(metric)}</text>')
        if points:
            coords = " ".join(
                f"{margin+(t-min_t)/span*(width-2*margin):.1f},{top+105-v/max_v*95:.1f}" for t, v in points
            )
            parts.append(f'<polyline points="{coords}" fill="none" stroke="#1261a0" stroke-width="2"/>')
        for boundary, color, label in ((start, "#198754", "start"), (end, "#dc3545", "end")):
            x = margin + (boundary - min_t) / span * (width - 2 * margin)
            parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+110}" stroke="{color}" stroke-dasharray="4 3"/>')
            parts.append(f'<text x="{x+2:.1f}" y="{top+108}" font-family="sans-serif" font-size="9">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_run(config: dict[str, Any], run: dict[str, Any], output_root: str | Path) -> Path:
    run_dir = Path(output_root) / f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.sanitized.json").write_text(json.dumps(sanitized_config(config), indent=2), encoding="utf-8")
    _write_jsonl(run_dir / "requests.jsonl", run["request_results"])
    _write_jsonl(run_dir / "telemetry_raw.jsonl", run["raw_telemetry"])
    parsed = [sample for snapshot in run["raw_telemetry"] for sample in snapshot.get("parsed_samples", [])]
    columns = [
        "collection_timestamp_utc", "collection_monotonic", "source_timestamp", "logical_metric",
        "source_metric", "source_value", "source_unit", "value", "unit", "gpu_identity", "labels",
    ]
    with (run_dir / "telemetry_parsed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for sample in parsed:
            writer.writerow({key: json.dumps(sample[key]) if isinstance(sample.get(key), dict) else sample.get(key) for key in columns})
    summary = {key: value for key, value in run.items() if key not in {"request_results", "raw_telemetry"}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_svg(run_dir / "telemetry.svg", run)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    for warmup_id in range(int(config.get("warmup_requests", 1))):
        warmup = run_request(config, -(warmup_id + 1))
        if not warmup.success:
            raise SystemExit(f"warm-up failed: {warmup.error}")
    for repetition in range(1, int(config.get("repetitions", 1)) + 1):
        run = execute_run(config, repetition)
        print(save_run(config, run, config.get("output_directory", "results")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
