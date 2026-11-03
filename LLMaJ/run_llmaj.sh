#!/usr/bin/env bash
# Evaluate one agent model with LLMaJ, across every benchmark we have traces for.
#
# One of these per screen session:
#
#   ./run_llmaj.sh kimi-k3
#   ./run_llmaj.sh gemini-3.1-pro-preview
#   ./run_llmaj.sh gpt-5.6-terra
#   ./run_llmaj.sh claude-sonnet-5
#
# Everything after the agent name is passed through to evaluate.py:
#
#   ./run_llmaj.sh kimi-k3 --dry-run          count the work, spend nothing
#   ./run_llmaj.sh kimi-k3 --limit 10         fewer trajectories per cell
#   ./run_llmaj.sh kimi-k3 --judge gemini-pro-high
#   ./run_llmaj.sh kimi-k3 --only efficiency
#
# Sessions do not contend: each agent writes to its own directories, and a
# trajectory whose shard exists is skipped, so an interrupted session resumes by
# being re-run with the same command.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"

[ $# -ge 1 ] || { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
AGENT="$1"; shift

export PROXY_API_KEY="$(awk -F= '/^PROXY_API_KEY=/{print $2}' "$ROOT/.env")"
export LLMAJ_BASE_URL="${LLMAJ_BASE_URL:-http://127.0.0.1:4500/v1}"
# The interpreter LLMaJ was installed into (`pip install -e .`): LLMaJ/.venv if
# there is one, else whatever python3 is on PATH. PYTHON overrides.
PYTHON="${PYTHON:-$([ -x "$DIR/.venv/bin/python" ] && echo "$DIR/.venv/bin/python" || command -v python3)}"

# The judge proxy is LLMaJ's own on 4500. 4000 serves the experiments and
# their config is frozen; a judge pointed at them would mix judge spend into the
# experiment records.
if ! curl -sf -o /dev/null -H "Authorization: Bearer $PROXY_API_KEY" \
     "${LLMAJ_BASE_URL%/v1}/v1/models"; then
  echo "No judge proxy at $LLMAJ_BASE_URL — start it first:"
  echo "    cd $DIR && ./proxy/run.sh"
  exit 1
fi

exec "$PYTHON" "$DIR/evaluate.py" "$AGENT" "$@"
