#!/usr/bin/env bash
# Score one run's answers with MCP-Atlas's own LLM-as-judge claim-coverage scorer.
# Writes src/scoring_<run id>/scored_<alias>.csv, which LLMaJ reads as the grade.
#
#   MCP_MODEL=<alias> MCP_RUN_ID=my_run ./score.sh          # the run ./run.sh made
#
# The judge is EVAL_LLM_MODEL / EVAL_LLM_BASE_URL / EVAL_LLM_API_KEY. If src/.env does not
# set them, gemini-3.1-pro-preview is called directly on Gemini's OpenAI-compatible
# endpoint with GEMINI_API_KEY from the root .env - not through the proxy, so the
# scorer's calls never mix with the agents' traces.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/../_common.sh"; load_env
cd "$here/src"
: "${MCP_MODEL:?set MCP_MODEL=<alias>}" "${MCP_RUN_ID:?set MCP_RUN_ID=<the run id ./run.sh used>}"
in="outputs_${MCP_RUN_ID}.csv"; [ -f "$in" ] || { echo "no $in in $PWD (run ./run.sh first)" >&2; exit 1; }
out="scoring_${MCP_RUN_ID}"; mkdir -p "$out"
# ground truth = the dataset the run answered, exported verbatim to CSV once
gt="$out/groundtruth.csv"
[ -f "$gt" ] || .venv/bin/python - "$gt" <<'PY'
import sys
from datasets import load_dataset
load_dataset("ScaleAI/MCP-Atlas", split="train").to_csv(sys.argv[1], index=False)
PY
if ! grep -qE '^EVAL_LLM_API_KEY=.' .env; then
  : "${GEMINI_API_KEY:?set GEMINI_API_KEY in the root .env (the judge the scorer uses), or EVAL_LLM_* in src/.env}"
  export EVAL_LLM_MODEL="${EVAL_LLM_MODEL:-gemini-3.1-pro-preview}" \
    EVAL_LLM_BASE_URL="${EVAL_LLM_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai}" \
    EVAL_LLM_API_KEY="${EVAL_LLM_API_KEY:-$GEMINI_API_KEY}"
fi
.venv/bin/python services/scoring/score_claims.py \
  --groundtruth-file "$gt" --model-file "$in" --model-name "$MCP_MODEL" --output-dir "$out" "$@"
echo "==> $(ls "$out"/scored_*.csv)"
