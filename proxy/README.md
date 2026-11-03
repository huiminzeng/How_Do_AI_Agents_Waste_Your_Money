# Tracing proxy

A LiteLLM proxy that records every model call an agent makes. Benchmark-agnostic:
anything that speaks the OpenAI chat-completions API can be pointed at it.

```
benchmark harness (patched to import llm_cost_tracker.tracer)
  → the tracer stamps each request with an X-Trace-Context header (benchmark, domain, model, run, task)
  → LiteLLM proxy (run_proxy.sh, :4000) forwards to the provider named by the alias
  → trace_logger.py appends one row per call to that trajectory's calls.jsonl
```

## What is recorded

```
artifacts/trajectories/
  <owner>/<credential>/<benchmark>/<domain>/<model>/trace_<trace_id>/calls.jsonl
```

`<owner>` is the `user_id` trace field (`shared` when unset), `<credential>` the `api_key`
field (`no_key`), `<domain>` the `domain` field (`no_domain` for benchmarks without one).
Every row carries the four token counts (input, cache read, cache write, output),
`reasoning_tokens`, the visible text and the reasoning text, tool calls as objects,
`cost_usd`, `requested_model` (the alias the harness asked for), `request_params` (the
effort / thinking / sampling settings the call actually ran with), `response_id` (the
provider's id), `tools` (tool names + schema hash) and, on a turn the provider ended
abnormally, `stop_details`. Next to `calls.jsonl` in each trace directory:

| file | contents |
|---|---|
| `tools_<hash>.json` | the full tool schema, written once per distinct hash |
| `raw.jsonl` | the provider's response body as litellm received it, one line per non-streamed call |
| `errors.jsonl` | one row per failed call (provider 4xx/5xx, timeouts, unknown alias) with the request messages and params |
| `calls.jsonl.attemptN` | a superseded attempt's files, moved aside together when a task is re-run |

Dollar costs are token costs only, per call:
`(input × input_price + cache_read × cache_read_price + cache_write × cache_write_price + output × output_price) / 1e6`,
with prices from the live LiteLLM price map, overridable in `pricing.py`.

Calls without trace headers are logged under `artifacts/trajectories/untraced/`. The
benchmarks' own helper models (τ²'s user simulator, MCP-Atlas's scorer) call their provider
directly and are not recorded; LLMaJ's judges go through their own proxy (`LLMaJ/proxy/`)
and are recorded under `artifacts/llmaj/judge_spend/`.

## Running it

```bash
proxy/setup.sh      # once: proxy/.venv with litellm 1.100.1 and this package
./run_proxy.sh      # from the repository root; reads .env, listens on :4000 (PROXY_PORT)
```

The benchmark run scripts point their clients at it through `OPENAI_BASE_URL` and send
`PROXY_API_KEY` as their API key. Provider keys are read by the proxy process only.

| Variable | Read by | Meaning |
|---|---|---|
| `PROXY_API_KEY` | proxy and clients | the proxy's master key; what clients send as their OpenAI key |
| `OPENAI_BASE_URL`, `TRACE_PROXY_BASE_URL` | clients | where to send calls (`http://localhost:4000/v1`) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `KIMI_API_KEY` / `KIMI_BASE_URL`, `SGLANG_BASE_URL` | proxy | the providers behind the aliases in `config.yaml` |
| `LLM_MONITOR_LOG` | proxy | trace root, as a path to `calls.jsonl` (default `<repo>/artifacts/trajectories/calls.jsonl`) |
| `PROXY_PORT`, `PROXY_HOST` | proxy | listen address (default `127.0.0.1:4000`) |

## Package structure

```
llm_cost_tracker/
  proxy/          config.yaml (one alias per model × reasoning effort), run_proxy.sh,
                  trace_logger.py (the callback), litellm_patches.py
  tracer/         HTTP-layer trace context injection, imported by the patched harnesses
  pricing.py      price table and per-call cost
tests/            pytest
```

Models are aliases in `llm_cost_tracker/proxy/config.yaml`; the comments there record
why each alias carries the parameters it does (effort placement, cache markers, the
Responses route for tool use with reasoning):

```yaml
- model_name: my-model
  litellm_params:
    model: openai/my-model
    api_base: https://api.openai.com/v1   # pin it: never inherit $OPENAI_BASE_URL (self-loop)
    api_key: os.environ/OPENAI_API_KEY
```

## Tracer integration

What the benchmark patches do. Install once at process startup; trace scopes nest via
decorator or context manager, and `reset=True` on each task attempt makes the proxy start
that trajectory's `calls.jsonl` afresh (the previous attempt is moved aside):

```python
from llm_cost_tracker import tracer

tracer.install(os.environ.get("TRACE_PROXY_BASE_URL") or os.environ.get("OPENAI_BASE_URL"))

with tracer.trace(benchmark="my_benchmark", domain="retail", model="kimi-k3-high",
                  run_id=tracer.run_id_from_dir(output_dir, env_var="MY_BENCH_RUN_ID"),
                  unit_id="42", trial_id="t1", reset=True):
    client.chat.completions.create(...)
```

Standard trace keys: `benchmark`, `domain`, `model`, `run_id`, `unit_id`, `trial_id`.
`run_id_from_dir` resolves the run id as: env override → basename of the harness's own
output directory → `"default"`.
