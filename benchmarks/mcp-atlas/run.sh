#!/usr/bin/env bash
# Run MCP-Atlas with the agent routed through the proxy (traced). The sandbox
# (36 MCP servers) and the TypeScript harness are started by upstream's run_all.sh;
# this only points LLM_BASE_URL / LLM_API_KEY at the proxy and tags the run.
#
#   MCP_MODEL=<alias> MCP_RUN_ID=my_run ./run.sh [--num-tasks 5] [other run_all.sh args]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/../_common.sh"; load_env
cd "$here/src"
export MCP_MODEL="${MCP_MODEL:-gpt-5.2-chat}" MCP_RUN_ID="${MCP_RUN_ID:-mcp_$(date +%Y%m%d_%H%M)}"
proxy_serves "$MCP_MODEL"
# upstream reads these from src/.env; override for this run without editing the file.
# The harness appends /v1/chat/completions itself, so it gets the origin without /v1.
export LLM_BASE_URL="${OPENAI_BASE_URL%/v1}" LLM_API_KEY="$PROXY_API_KEY" MCP_ATLAS_RUN_ID="$MCP_RUN_ID"
# upstream's run_all.sh also insists the two lines are non-empty in src/.env itself
for kv in "LLM_BASE_URL=$LLM_BASE_URL" "LLM_API_KEY=$LLM_API_KEY"; do
  k="${kv%%=*}"; grep -qE "^$k=." .env || { sed -i "/^$k=/d" .env; echo "$kv" >> .env; }
done
export LLM_MONITOR_LOG="${LLM_MONITOR_LOG:-$repo_root/artifacts/trajectories/calls.jsonl}"
export PYTHON="$here/src/.venv/bin/python"     # run_all.sh runs run_eval.py with this
./run_all.sh --model "$MCP_MODEL" --output "outputs_${MCP_RUN_ID}.csv" "$@"; rc=$?
# upstream stops the harness's shell but not the node process under it, and its next
# run would silently reuse that orphan (with this run's settings). Stop ours.
port="$(grep -E '^PORT=' .env | tail -1 | cut -d= -f2)"; port="${port:-3001}"
for p in $(ss -tlnp 2>/dev/null | grep ":${port} " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do
  case "$(readlink "/proc/$p/cwd" 2>/dev/null)" in "$here"/src/*) kill "$p" 2>/dev/null && echo "==> stopped harness process $p" ;; esac
done
exit $rc
