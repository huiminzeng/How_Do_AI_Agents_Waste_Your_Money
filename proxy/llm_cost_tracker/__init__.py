"""llm_cost_tracker — Standalone LLM cost tracking via litellm proxy.

Usage:
    # Start the proxy (logs all LLM calls to JSONL):
    litellm --config llm_cost_tracker/proxy/config.yaml --port 4000

    # Point any OpenAI-compatible client at it:
    export OPENAI_BASE_URL=http://localhost:4000/v1
    export OPENAI_API_KEY=sk-tau-local-1234

    # Calls are logged under artifacts/trajectories/<owner>/<credential>/<benchmark>/<domain>/<model>/trace_*/calls.jsonl

No application-specific imports. Works with any benchmark or application
that uses OpenAI-compatible LLM calls.
"""

__version__ = "0.1.0"

# Re-export tracer module for convenient usage:
#   from llm_cost_tracker import tracer
from . import tracer  # noqa: E402,F401
