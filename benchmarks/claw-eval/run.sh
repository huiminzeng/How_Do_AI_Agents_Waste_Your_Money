#!/usr/bin/env bash
# Run a claw-eval batch with the agent routed through the proxy (traced); the
# benchmark's own judge and user simulator call their provider directly.
#
#   CLAW_EVAL_MODEL=<alias> CLAW_EVAL_TRIALS=1 CLAW_EVAL_PARALLEL=3 ./run.sh
#
# Extra arguments go to `claw_eval.cli batch`.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/../_common.sh"; load_env
cd "$here/src"
export CLAW_EVAL_MODEL="${CLAW_EVAL_MODEL:-gpt-5.2-chat}"
export CLAW_EVAL_CONFIG="${CLAW_EVAL_CONFIG:-$here/src/config_proxy_smoke.yaml}"
export CLAW_EVAL_TASKS_DIR="${CLAW_EVAL_TASKS_DIR:-$here/src/tasks}"
export CLAW_EVAL_TRIALS="${CLAW_EVAL_TRIALS:-1}" CLAW_EVAL_PARALLEL="${CLAW_EVAL_PARALLEL:-1}"
export LLM_MONITOR_LOG="${LLM_MONITOR_LOG:-$repo_root/artifacts/trajectories/calls.jsonl}"
proxy_serves "$CLAW_EVAL_MODEL"
exec .venv/bin/python -m claw_eval.cli batch \
  --config "$CLAW_EVAL_CONFIG" --tasks-dir "$CLAW_EVAL_TASKS_DIR" \
  --model "$CLAW_EVAL_MODEL" --base-url "$OPENAI_BASE_URL" --api-key "$PROXY_API_KEY" \
  --trials "$CLAW_EVAL_TRIALS" --parallel "$CLAW_EVAL_PARALLEL" "$@"
