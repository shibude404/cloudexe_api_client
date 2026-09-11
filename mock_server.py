"""Local synthetic vLLM and Prometheus-style HTTP service."""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

GIB = 1024 ** 3


def metrics_text(scenario: str, call: int) -> str:
    power = 300.0 if scenario != "linear" else 200.0 + min(call, 2) * 100.0
    lines = [
        '# HELP synthetic_gpu_power_watts Invented GPU board power fixture.',
        f'synthetic_gpu_power_watts{{gpu="gpu0",allocation="local-test"}} {power}',
        'synthetic_gpu_activity_percent{gpu="gpu0",allocation="local-test"} 80',
        'synthetic_memory_controller_activity_percent{gpu="gpu0",allocation="local-test"} 60',
        f'synthetic_gpu_memory_used_bytes{{gpu="gpu0",allocation="local-test"}} {20 * GIB}',
        f'synthetic_gpu_memory_capacity_bytes{{gpu="gpu0",allocation="local-test"}} {80 * GIB}',
        'synthetic_sm_activity_percent{gpu="gpu0",allocation="local-test"} 75',
        'synthetic_gpu_power_watts{gpu="gpu1",allocation="other-test"} 450',
        'synthetic_gpu_activity_percent{gpu="gpu1",allocation="other-test"} 25',
        'synthetic_memory_controller_activity_percent{gpu="gpu1",allocation="other-test"} 35',
        f'synthetic_gpu_memory_used_bytes{{gpu="gpu1",allocation="other-test"}} {10 * GIB}',
        f'synthetic_gpu_memory_capacity_bytes{{gpu="gpu1",allocation="other-test"}} {40 * GIB}',
        'synthetic_sm_activity_percent{gpu="gpu1",allocation="other-test"} 20',
    ]
    if scenario == "missing_utilization":
        lines = [line for line in lines if "synthetic_gpu_activity_percent" not in line]
    elif scenario == "duplicate_power":
        lines.append('synthetic_gpu_power_watts{gpu="gpu0",allocation="local-test",source="duplicate"} 301')
    elif scenario == "malformed":
        lines.append("this is not valid Prometheus text")
    return "\n".join(lines) + "\n"


class SyntheticHandler(BaseHTTPRequestHandler):
    metric_calls = 0

    def log_message(self, *_args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/metrics":
            self.send_error(404)
            return
        scenario = parse_qs(parsed.query).get("scenario", ["constant"])[0]
        if scenario == "timeout":
            time.sleep(2)
        type(self).metric_calls += 1
        payload = metrics_text(scenario, type(self).metric_calls).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if urlparse(self.path).path not in {"/v1/completions", "/v1/chat/completions"}:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        output_tokens = int(request.get("max_tokens", 4))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for index in range(output_tokens):
            event = {"choices": [{"text": f" token{index}"}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.01)
        usage = {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": output_tokens}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), SyntheticHandler)
    print(f"Synthetic service: http://{args.host}:{args.port}")
    print(f"Metrics: http://{args.host}:{args.port}/metrics")
    print("All values are invented test fixtures, not Cloudexe/B300 specifications.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
