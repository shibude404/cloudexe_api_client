# Cloudexe API benchmark client

This dependency-free Python client can be validated entirely on a laptop. The
included HTTP service returns invented inference responses and Prometheus-style
GPU telemetry. No GPU or Cloudexe access is used by local tests.

## Current collector map

- `load_config()` reads endpoint, authentication, metric, unit, label, interval,
  and coverage settings.
- `TelemetryCollector.scrape()` fetches one response, preserves it, parses and
  normalizes selected metrics, and records errors.
- `TelemetryCollector.start()/stop()` obtain boundary samples and poll in a
  background thread.
- `parse_prometheus()` parses metrics, labels, values, and optional source
  timestamps.
- `normalize_scrape()` selects exactly one series per logical metric for the
  configured GPU/allocation and applies unit conversions.
- `summarize_metric()` performs boundary interpolation, time-weighted averaging
  or peak selection, and maximum-gap validation.
- `summarize_telemetry()` keeps each metric independently valid or invalid.
- `execute_run()` coordinates inference and telemetry and computes throughput
  and client-observed TTFT.
- `save_run()` writes inspectable JSONL/CSV/JSON artifacts and an SVG plot.

## Local fixture values

The `/metrics` endpoint exposes two fake GPUs. The configured `gpu0` values are:

| Metric | Value |
|---|---:|
| GPU board power | 300 W |
| GPU activity | 80% |
| Memory-controller activity | 60% |
| GPU memory used | 20 GiB |
| GPU memory capacity | 80 GiB |
| SM activity | 75% |

`gpu1` has different values and labels so filtering can be verified. These are
test fixtures, not B300 specifications or Cloudexe metric names.

## Setup and tests

Python 3.12 is required; no third-party packages are required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m unittest -v test_benchmark.py
```

## Manual mock endpoint

Terminal 1:

```powershell
.\.venv\Scripts\python.exe mock_server.py --port 8765
```

Terminal 2:

```powershell
Invoke-WebRequest http://127.0.0.1:8765/metrics | Select-Object -Expand Content
.\.venv\Scripts\python.exe benchmark.py --config config.synthetic.json
```

Each synthetic run writes:

```text
results/synthetic_<timestamp>/
  config.sanitized.json
  requests.jsonl
  telemetry_raw.jsonl
  telemetry_parsed.csv
  summary.json
  telemetry.svg
```

The plot and summary are explicitly labelled synthetic. Warm-up completes
before collection. The collector obtains one sample before workload start and
one after workload completion so summaries can interpolate both boundaries.

## Failure scenarios

Append a scenario to the monitoring URL in a copied configuration:

```text
?scenario=constant
?scenario=linear
?scenario=missing_utilization
?scenario=duplicate_power
?scenario=malformed
?scenario=timeout
```

Collection errors are saved and polling continues. Missing or insufficient data
invalidates only the affected metric; it is never silently replaced by zero.

## Cloudexe adaptation checklist

Before live use, replace placeholders in `config.example.json` after confirming:

- Raw scrape endpoint versus historical query API.
- Authentication requirements.
- Exact metric names, definitions, units, and GPU/allocation labels.
- Source timestamps and underlying telemetry update interval.
- Supported inference route, streaming usage response, and model name.

Polling frequency does not imply measurement freshness. If Cloudexe supplies a
historical query API, add a retrieval adapter while reusing normalization,
selection, validity, and calculation logic.
