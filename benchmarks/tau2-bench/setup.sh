#!/usr/bin/env bash
# tau2-bench @ the pinned upstream commit + our cost-tracking patch, in its own uv env.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/../_common.sh"
fetch_upstream "$here"
cd "$here/src"
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
uv sync --python python3.12
uv pip install --python .venv/bin/python -e "$repo_root/proxy"     # the tracer the patch imports
echo "==> tau2-bench ready: $here/src"
