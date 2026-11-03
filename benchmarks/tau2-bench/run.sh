#!/usr/bin/env bash
# Run τ²-bench with the agent routed through the proxy (traced) and the user
# simulator calling its provider directly (not traced, not part of the measurement).
#
#   DOMAIN=retail NUM_TASKS=5 NUM_TRIALS=2 AGENT_MODEL=openai/<alias> ./run.sh
#
# Extra arguments go to `tau2 run`.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/../_common.sh"; load_env
cd "$here/src"
export DOMAIN="${DOMAIN:-retail}"
export AGENT_MODEL="${AGENT_MODEL:-openai/gpt-5.2-chat}"      # alias served by the proxy
export USER_MODEL="${USER_MODEL:-gpt-5.2-chat-latest}"        # direct to the provider
export NUM_TASKS="${NUM_TASKS:-5}" NUM_TRIALS="${NUM_TRIALS:-2}" MAX_RETRIES="${MAX_RETRIES:-0}" MAX_CONCURRENCY="${MAX_CONCURRENCY:-5}" TEMP="${TEMP:-1.0}"
export OPENAI_DIRECT_API_KEY="${OPENAI_DIRECT_API_KEY:-${OPENAI_API_KEY:-}}"
export OPENAI_DIRECT_BASE_URL="${OPENAI_DIRECT_BASE_URL:-https://api.openai.com/v1}"
case "$USER_MODEL" in
  claude-*) [ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "USER_MODEL=$USER_MODEL: set ANTHROPIC_API_KEY in .env (the user simulator calls Anthropic directly)" >&2; exit 1; } ;;
  *) [ -n "$OPENAI_DIRECT_API_KEY" ] || { echo "USER_MODEL=$USER_MODEL: set OPENAI_API_KEY in .env (the user simulator calls OpenAI directly), or pick USER_MODEL=claude-sonnet-5" >&2; exit 1; } ;;
esac
proxy_serves "$AGENT_MODEL"
export OPENAI_API_KEY="$PROXY_API_KEY"
uv run tau2 run \
  --domain "$DOMAIN" \
  --agent-llm "$AGENT_MODEL" \
  --agent-llm-args "{\"api_base\":\"$OPENAI_BASE_URL\",\"api_key\":\"$OPENAI_API_KEY\",\"temperature\":$TEMP}" \
  --user-llm "$USER_MODEL" \
  --user-llm-args "$(case "$USER_MODEL" in \
      claude-*) echo "{\"api_key\":\"${ANTHROPIC_API_KEY:-}\",\"temperature\":$TEMP}" ;; \
      *) echo "{\"api_base\":\"$OPENAI_DIRECT_BASE_URL\",\"api_key\":\"$OPENAI_DIRECT_API_KEY\",\"temperature\":$TEMP}" ;; \
    esac)" \
  --num-tasks "$NUM_TASKS" --num-trials "$NUM_TRIALS" --max-retries "$MAX_RETRIES" --max-concurrency "$MAX_CONCURRENCY" \
  "$@"
