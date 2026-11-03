#!/usr/bin/env bash
# Grade a run's preds.json with the SWE-bench harness (Docker). Writes the report
# next to src/ as <model>.<run_id>.json, like SWE-bench itself does.
#
#   PREDICTIONS=outputs/t1/preds.json DATASET=SWE-bench/SWE-bench_Verified SPLIT=test RUN_ID=t1 ./grade.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/../_common.sh"
cd "$here/src"
: "${PREDICTIONS:?set PREDICTIONS=<path to preds.json>}"
.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name "${DATASET:-SWE-bench/SWE-bench_Lite}" --split "${SPLIT:-dev}" \
  --predictions_path "$PREDICTIONS" --max_workers "${MAX_WORKERS:-4}" --run_id "${RUN_ID:-$(basename "$(dirname "$PREDICTIONS")")}" "$@"
