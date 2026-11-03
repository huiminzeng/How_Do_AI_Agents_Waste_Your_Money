#!/usr/bin/env bash
# One-time: the proxy's own virtualenv (litellm 1.100.1, pinned - see README) plus
# this package, editable, so trace_logger / pricing / litellm_patches import from here.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
[ -x "$here/.venv/bin/python" ] || uv venv --python 3.12 "$here/.venv"
VIRTUAL_ENV="$here/.venv" uv pip install -q 'litellm[proxy]==1.100.1' -e "$here"
echo "==> proxy venv ready: $here/.venv  (litellm $("$here/.venv/bin/python" -c 'import importlib.metadata as m;print(m.version("litellm"))'))"
