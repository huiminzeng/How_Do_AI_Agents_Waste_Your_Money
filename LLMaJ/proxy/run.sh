#!/usr/bin/env bash
# Start the LiteLLM proxy that serves the LLMaJ judges.
#
#   ./proxy/run.sh
#
# Port 4500. The experiment proxy on 4000 is untouched: different
# config file, different port, different trace root.
#
# Judge calls are recorded under artifacts/llmaj/judge_spend/, never beside the agents' traces,
# so what the judging cost never mixes with what the experiments cost.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
LLMAJ="$(cd "$DIR/.." && pwd)"
ROOT="$(cd "$LLMAJ/.." && pwd)"

[ -f "$ROOT/.env" ] && set -a && source "$ROOT/.env" && set +a

# Same venv and the same llm_cost_tracker.proxy.trace_logger as the cost proxy (proxy/).
TRACKER="${COST_TRACKER_ROOT:-$ROOT/proxy}"
if [ ! -x "$TRACKER/.venv/bin/litellm" ]; then
  echo "Error: no proxy venv at $TRACKER/.venv - run proxy/setup.sh first" >&2
  exit 1
fi
export PATH="$TRACKER/.venv/bin:$PATH"

PORT="${LLMAJ_PROXY_PORT:-4500}"

# Read out of paths.py, not written here, so the shell and the package cannot
# disagree about where cost records go. Hard-coding it once already cost a
# session: the tree was renamed, this line was not, and a running proxy kept
# recreating the old directory and writing there with nothing failing.
#
# Loaded as a plain file rather than by importing the package: `import llmaj`
# pulls in packages the proxy's venv does not have.
# That import failed silently inside $( ), leaving LLM_MONITOR_LOG EMPTY - and an
# empty value means the tracker writes nowhere at all. Verified non-empty below.
LLM_MONITOR_LOG="$(python - <<PY
import runpy
ns = runpy.run_path("$DIR/../llmaj/paths.py")
print(ns["JUDGE_SPEND"] / "calls.jsonl")
PY
)"
if [ -z "$LLM_MONITOR_LOG" ]; then
  # An empty value is NOT "record nothing": trace_logger falls back to
  # <cwd>/artifacts/trajectories, where nobody looks for judge records. That
  # happened, silently, for 61 calls. Refuse rather than let the fallback run.
  echo "Could not resolve the trace path from llmaj/paths.py."
  echo "An empty value makes the tracker fall back to \$PWD/artifacts/trajectories — records"
  echo "would be written somewhere nobody reads. Refusing to start."
  exit 1
fi
export LLM_MONITOR_LOG
mkdir -p "$(dirname "$LLM_MONITOR_LOG")"

# .env sets OPENAI_BASE_URL for clients. If the proxy inherits it, LiteLLM's
# OpenAI provider treats it as the upstream and the proxy calls itself.
unset OPENAI_BASE_URL OPENAI_API_BASE

# --- checks before spending anything ---------------------------------------
: "${PROXY_API_KEY:?set PROXY_API_KEY (the proxy master key; .env has it)}"
for k in OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY KIMI_API_KEY; do
  [ -n "${!k:-}" ] || echo "    warning: $k unset — judges using it will 401"
done
if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  echo "Port ${PORT} is already in use. Stop that proxy first, or set LLMAJ_PROXY_PORT."
  exit 1
fi

echo "==> LLMaJ judge proxy on http://127.0.0.1:${PORT}"
echo "    config   $DIR/config.yaml"
echo "    traces   $LLM_MONITOR_LOG"
echo ""
echo "    judges:"
grep -E '^\s*-\s*model_name:' "$DIR/config.yaml" | sed 's/.*model_name: /      /'
echo ""
echo "    point LLMaJ at it:"
echo "      export LLMAJ_BASE_URL=http://127.0.0.1:${PORT}/v1"
echo ""

export PYTHONPATH="$TRACKER:$DIR:${PYTHONPATH:-}"
python -c "import importlib; m=importlib.import_module('llm_cost_tracker.proxy.trace_logger'); \
  assert hasattr(m,'proxy_logger'); print('    callback import: OK')"

exec litellm --config "$DIR/config.yaml" --port "$PORT" --host 127.0.0.1
