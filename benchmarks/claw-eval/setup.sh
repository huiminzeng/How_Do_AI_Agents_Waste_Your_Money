#!/usr/bin/env bash
# claw-eval @ the pinned upstream commit + our cost-tracking patch, in its own venv.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/../_common.sh"
fetch_upstream "$here"
cd "$here/src"
python3 -m venv .venv
.venv/bin/pip install -q -e . -e "$repo_root/proxy" 'litellm[proxy]'
echo "==> claw-eval ready: $here/src"
