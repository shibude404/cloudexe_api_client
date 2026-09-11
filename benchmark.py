"""Cloudexe vLLM API benchmark client (standard library only).

The client uses OpenAI-compatible streaming completions, measures client-side
TTFT and aggregate output throughput, scrapes Prometheus text telemetry, and
keeps raw evidence. Provider metric names remain configuration values.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+'
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|Inf|-Inf)'
    r'(?:\s+\d+)?$'
)
LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def parse_prometheus(text: str) -> list[dict[str, Any]]:
    """Parse Prometheus text exposition samples without external packages."""
    parsed = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        labels = {
            key: bytes(value, "utf-8").decode("unicode_escape")
            for key, value in LABEL_RE.findall(raw_labels or "")
        }
        value = float(raw_value)
        if math.isfinite(value):
            parsed.append({"name": name, "labels": labels, "value": value})
    return parsed


def labels_match(labels: dict[str, str], required: dict[str, str]) -> bool:
    return all(labels.get(key) == value for key, value in required.items())


def select_metrics(
    samples: list[dict[str, Any]], metric_names: dict[str, str], labels: dict[str, str]
) -> dict[str, list[float]]:
    selected = {logical_name: [] for logical_name in metric_names}
    reverse = {metric_name: logical_name for logical_name, metric_name in metric_names.items() if metric_name}
    for sample in samples:
        logical_name = reverse.get(sample["name"])
        if logical_name and labels_match(sample["labels"], labels):
            selected[logical_name].append(sample["value"])
    return selected


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_telemetry(
    snapshots: list[dict[str, Any]], start_monotonic: float, end_monotonic: float
) -> dict[str, Any]:
    in_window = [
        item for item in snapshots
        if start_monotonic <= item["monotonic"] <= end_monotonic and not item.get("error")
    ]
    names = sorted({name for item in in_window for name in item.get("metrics", {})})
    summary: dict[str, Any] = {"snapshot_count": len(in_window)}
    for name in names:
        # Sum duplicate per-GPU series at a timestamp. With one selected B300,
        # this normally contains a single value.
        points = [
            (item["monotonic"], sum(item["metrics"].get(name, [])))
            for item in in_window if item["metrics"].get(name)
        ]
        if len(points) == 1:
            summary[name] = points[0][1]
        elif len(points) >= 2:
            area = sum(
                (right_t - left_t) * (left_value + right_value) / 2
                for (left_t, left_value), (right_t, right_value) in zip(points, points[1:])
            )
            covered = points[-1][0] - points[0][0]
            summary[name] = area / covered if covered > 0 else None
        else:
            summary[name] = None
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
    for choice in event.get("choices", []):
        if choice.get("text"):
            return True
        delta = choice.get("delta") or {}
        if delta.get("content"):
            return True
    return False


def run_request(config: dict[str, Any], request_id: int) -> RequestResult:
    api_key = os.environ.get(config.get("api_key_env", "CLOUDEXE_API_KEY"), "")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body: dict[str, Any] = {
        "model": config["model"],
        "max_tokens": int(config["output_tokens"]),
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": config.get("temperature", 0.0),
    }
    if config.get("api_style", "completions") == "chat":
        body["messages"] = [{"role": "user", "content": config["prompt"]}]
    else:
        body["prompt"] = config["prompt"]

    request = urllib.request.Request(
        config["inference_url"], data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    start_utc = utc_now()
    start = time.perf_counter()
    first_token = None
    events: list[dict[str, Any]] = []
    output_tokens = None
    error = None
    try:
        with urllib.request.urlopen(request, timeout=float(config.get("request_timeout_seconds", 600))) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
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
    except Exception as exc:  # Preserve failures as data for benchmark auditing.
        error = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    return RequestResult(
        request_id=request_id,
        success=error is None,
        start_utc=start_utc,
        end_utc=utc_now(),
        elapsed_seconds=end - start,
        ttft_seconds=(first_token - start) if first_token is not None else None,
        output_tokens=output_tokens,
        error=error,
        events=events,
    )


class TelemetryCollector:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.snapshots: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.config.get("prometheus_url"):
            return
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _collect(self) -> None:
        interval = float(self.config.get("telemetry_interval_seconds", 1.0))
        token_env = self.config.get("prometheus_token_env", "")
        while not self._stop.is_set():
            stamp = {"timestamp_utc": utc_now(), "monotonic": time.perf_counter()}
            try:
                headers = {}
                token = os.environ.get(token_env, "") if token_env else ""
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                req = urllib.request.Request(self.config["prometheus_url"], headers=headers)
                with urllib.request.urlopen(req, timeout=10) as response:
                    raw = response.read().decode("utf-8")
                parsed = parse_prometheus(raw)
                stamp["raw"] = raw
                stamp["metrics"] = select_metrics(
                    parsed,
                    self.config.get("metric_names", {}),
                    self.config.get("required_metric_labels", {}),
                )
            except Exception as exc:
                stamp["error"] = f"{type(exc).__name__}: {exc}"
            self.snapshots.append(stamp)
            self._stop.wait(interval)


def execute_run(config: dict[str, Any], repetition: int) -> dict[str, Any]:
    collector = TelemetryCollector(config)
    collector.start()
    start_utc = utc_now()
    start = time.perf_counter()
    concurrency = int(config["concurrency"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(lambda i: run_request(config, i), range(concurrency)))
    end = time.perf_counter()
    end_utc = utc_now()
    collector.stop()

    successful = [result for result in results if result.success]
    total_tokens = sum(result.output_tokens or 0 for result in successful)
    elapsed = end - start
    ttfts = [result.ttft_seconds for result in successful if result.ttft_seconds is not None]
    return {
        "repetition": repetition,
        "workload_start_utc": start_utc,
        "workload_end_utc": end_utc,
        "workload_elapsed_seconds": elapsed,
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "total_output_tokens": total_tokens,
        "aggregate_output_tokens_per_second": total_tokens / elapsed if successful and elapsed > 0 else None,
        "mean_ttft_seconds": average(ttfts),
        "request_results": [asdict(result) for result in results],
        "telemetry_summary": summarize_telemetry(collector.snapshots, start, end),
        "raw_telemetry": collector.snapshots,
    }


def sanitized_config(config: dict[str, Any]) -> dict[str, Any]:
    secret_fields = {"api_key", "prometheus_token", "authorization"}
    return {key: value for key, value in config.items() if key.lower() not in secret_fields}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = load_config(args.config)

    for warmup_id in range(int(config.get("warmup_requests", 1))):
        result = run_request(config, -(warmup_id + 1))
        if not result.success:
            raise SystemExit(f"warm-up failed: {result.error}")

    runs = [execute_run(config, rep + 1) for rep in range(int(config.get("repetitions", 3)))]
    output_dir = Path(config.get("output_directory", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    artifact = {"created_utc": utc_now(), "config": sanitized_config(config), "runs": runs}
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
