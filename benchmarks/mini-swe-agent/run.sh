#!/usr/bin/env bash
# Run mini-swe-agent on SWE-bench through the proxy. Every model call of the agent
# is traced; grading is a separate step (grade.sh).
#
#   MINI_SWE_MODEL=openai/<alias> MINI_SWE_SUBSET=verified MINI_SWE_SPLIT=test \
#   MINI_SWE_WORKERS=4 MINI_SWE_TRIAL_ID=t1 ./run.sh [--slice 0:10]
#
# The alias MUST be prefixed openai/ or litellm routes by name and never reaches the
# proxy. The model registry prices proxy aliases for litellm; without it every task
# fails with "This model isn't mapped yet".
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/../_common.sh"; load_env
cd "$here/src"; export PATH="$here/src/.venv/bin:$PATH"
export MINI_SWE_MODEL="${MINI_SWE_MODEL:-openai/gpt-5.2-chat}"
case "$MINI_SWE_MODEL" in openai/*) ;; *) echo "MINI_SWE_MODEL must start with openai/ (got '$MINI_SWE_MODEL')" >&2; exit 2;; esac
export MINI_SWE_SUBSET="${MINI_SWE_SUBSET:-lite}" MINI_SWE_SPLIT="${MINI_SWE_SPLIT:-dev}" MINI_SWE_TRIAL_ID="${MINI_SWE_TRIAL_ID:-t1}" MINI_SWE_WORKERS="${MINI_SWE_WORKERS:-1}"
export MINI_SWE_OUTPUT="${MINI_SWE_OUTPUT:-$here/outputs/$MINI_SWE_TRIAL_ID}"
export MINI_SWE_RUN_ID="$(basename "$MINI_SWE_OUTPUT")"
export LITELLM_MODEL_REGISTRY_PATH="${LITELLM_MODEL_REGISTRY_PATH:-$here/model_registry.json}"
proxy_serves "$MINI_SWE_MODEL"
export OPENAI_API_KEY="$PROXY_API_KEY"
mini-extra swebench -c swebench.yaml -m "$MINI_SWE_MODEL" \
  -c "model.model_kwargs.api_base=$OPENAI_BASE_URL" -c "model.model_kwargs.api_key=$OPENAI_API_KEY" -c "model.model_kwargs.drop_params=true" \
  --subset "$MINI_SWE_SUBSET" --split "$MINI_SWE_SPLIT" -o "$MINI_SWE_OUTPUT" -w "$MINI_SWE_WORKERS" "$@"
