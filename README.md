# Cloudexe API benchmark client

This starter sends OpenAI-compatible streaming requests to a Cloudexe-hosted
vLLM endpoint and scrapes a Prometheus text endpoint during the same workload.
It records aggregate output tokens/s, client-observed TTFT, GPU power, GPU
utilization, memory-controller utilization, GPU memory-capacity usage, and SM
activity when those metrics are available.

## Before using a paid endpoint

Ask Cloudexe for the inference and Prometheus URLs, authentication formats,
served model name, exact metric names/units/labels, and a sample response. Copy
`config.example.json` to `config.json` and replace every `REPLACE_...` value.
Do not put secrets in the JSON file.

In PowerShell, set secrets only in environment variables:

```powershell
$env:CLOUDEXE_API_KEY = "..."
$env:CLOUDEXE_PROMETHEUS_TOKEN = "..."
```

## Local tests

From this directory:

```powershell
python -m unittest -v test_benchmark.py
```

The tests use a local fake streaming server and synthetic Prometheus data. They
do not contact Cloudexe or run a GPU experiment.

## Run

```powershell
Copy-Item config.example.json config.json
# Edit config.json after Cloudexe supplies endpoint details.
python benchmark.py --config config.json
```

Each invocation writes a timestamped JSON artifact under `results/`. The API
client intentionally rejects a stream that does not contain an actual
`usage.completion_tokens` value instead of treating requested `max_tokens` as
the generated-token count.

## Important limitations before live validation

- Prometheus metric names and GPU labels are placeholders.
- The telemetry summary uses trapezoidal time weighting across timestamped
  samples. Confirm the provider sampling semantics and coverage requirements
  before treating it as the final power methodology.
- The configured prompt text must be generated/tokenized to the intended input
  length; character count is not token count.
- API request concurrency is not vLLM's internal batch size.
