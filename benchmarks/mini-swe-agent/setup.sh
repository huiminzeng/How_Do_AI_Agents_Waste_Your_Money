#!/usr/bin/env bash
# mini-swe-agent v2.4.2 + our cost-tracking patch, and the SWE-bench evaluator
# (unmodified, pinned) in the same venv. Running tasks and grading patches both
# need Docker: SWE-bench ships one image per task instance.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/../_common.sh"
fetch_upstream "$here"
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
# The evaluator as a source checkout, installed editable: a wheel built from the git
# URL leaves out harness/constants/fixtures/*.Cargo.lock and `import swebench` fails.
if [ ! -d "$here/SWE-bench/.git" ]; then
  git clone -q https://github.com/SWE-bench/SWE-bench.git "$here/SWE-bench"
  git -C "$here/SWE-bench" checkout -q f7bbbb2
fi
cd "$here/src"
rm -rf .venv; uv venv -q --python 3.12 .venv
VIRTUAL_ENV="$PWD/.venv" uv pip install -q -e . -e "$repo_root/proxy" -e "$here/SWE-bench"
command -v docker >/dev/null || echo "warning: docker not found; SWE-bench runs and grading need it" >&2
echo "==> mini-swe-agent ready: $here/src  (evaluator: $here/SWE-bench @ f7bbbb2)"
