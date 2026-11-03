#!/usr/bin/env bash
# Start the LiteLLM proxy for LLM cost monitoring.
#
# Usage:
#   ./run_proxy.sh                 # start proxy on port 4000
#   PROXY_PORT=8000 ./run_proxy.sh # custom port
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRACKER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$TRACKER_ROOT/.." && pwd)"

# A log root given on the command line must survive `source .env`, which also sets
# LLM_MONITOR_LOG. Without this guard a test proxy started with
# `LLM_MONITOR_LOG=/tmp/x/calls.jsonl ./run_proxy.sh` would silently write into the real
# artifacts/trajectories/. Same pattern as the OPENAI_BASE_URL guard in the benchmark run scripts.
_PRESERVE_LLM_MONITOR_LOG="${LLM_MONITOR_LOG-__unset__}"
[ -f "$PROJECT_ROOT/.env" ] && set -a && source "$PROJECT_ROOT/.env" && set +a
if [ "${_PRESERVE_LLM_MONITOR_LOG}" != "__unset__" ]; then export LLM_MONITOR_LOG="$_PRESERVE_LLM_MONITOR_LOG"; fi

# The proxy runs from its OWN venv (proxy/.venv, litellm 1.100.1, made by proxy/setup.sh), kept
# apart from the harness clients' venvs so upgrading one never silently changes the other.
# Hard failure, no fallback: the last fallback (tau2-bench/.venv, litellm 1.81.11) merged
# parallel tool calls on streamed Responses routes and nothing in the logs said so.
#
# Create it with:
#   cd proxy && uv venv --python 3.12 .venv \
#     && VIRTUAL_ENV=$PWD/.venv uv pip install 'litellm[proxy]==1.100.1'
if [ ! -x "$TRACKER_ROOT/.venv/bin/litellm" ]; then
  echo "Error: no proxy venv at $TRACKER_ROOT/.venv (see the comment above this line)." >&2
  exit 1
fi
export PATH="$TRACKER_ROOT/.venv/bin:$PATH"

PROXY_PORT="${PROXY_PORT:-4000}"
CONFIG="$SCRIPT_DIR/config.yaml"
export LLM_MONITOR_LOG="${LLM_MONITOR_LOG:-$PROJECT_ROOT/artifacts/trajectories/calls.jsonl}"
: "${PROXY_API_KEY:?set PROXY_API_KEY in $PROJECT_ROOT/.env (any string; the key clients send)}"
# Aliases whose provider key is unset fail at call time with a 401, not at startup:
# say so here so a missing key is a known gap rather than a surprise mid-run.
for k in OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY KIMI_API_KEY; do
  [ -n "${!k:-}" ] || echo "    note: $k unset - aliases that use it will fail (see config.yaml)"
done

# .env sets OPENAI_BASE_URL for *clients* (this proxy). The proxy itself must never
# inherit it: LiteLLM's OpenAI provider would use it as the upstream api_base and the
# proxy would call itself in a loop. config.yaml pins api_base per model; unset here too.
unset OPENAI_BASE_URL OPENAI_API_BASE

echo "==> Starting LiteLLM proxy on http://localhost:${PROXY_PORT}"
echo "    Config: $CONFIG"
echo "    Log root: $LLM_MONITOR_LOG"
echo "    litellm:  $(python -c 'import importlib.metadata as m; print(m.version("litellm"))' 2>/dev/null || echo unknown)  ($(command -v litellm))"
echo "    Writes:   artifacts/trajectories/<owner>/<credential>/<benchmark>/<domain>/<model>/trace_<trace_id>/calls.jsonl"
echo "              (<owner> = user_id trace field, 'shared' when unset; <credential> = api_key field, 'no_key';"
echo "               <domain> = domain field, 'no_domain' for benchmarks without one)"
echo ""
echo "    Point your client at:"
echo "      export OPENAI_BASE_URL=http://localhost:${PROXY_PORT}/v1"
echo "      export OPENAI_API_KEY=\"\$PROXY_API_KEY\"   # proxy master_key, not your OpenAI key"
echo ""

export PYTHONPATH="${TRACKER_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"

python - <<'PY'
import importlib
module = importlib.import_module("trace_logger")
if not hasattr(module, "proxy_logger"):
    raise RuntimeError("trace_logger.proxy_logger not found")
print("    Callback import check: OK")
PY

# Loopback by default. The proxy authenticates with a single shared master key and
# takes the caller's identity from a header it cannot verify, so anything that can
# reach it can spend under any name. Override only if you genuinely want it exposed.
exec litellm --config "$CONFIG" --port "$PROXY_PORT" --host "${PROXY_HOST:-127.0.0.1}"
