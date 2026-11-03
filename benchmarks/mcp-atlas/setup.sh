#!/usr/bin/env bash
# MCP-Atlas @ the pinned upstream commit + our patch (tracer header on every model
# call; the uvx MCP-SDK pin and two runtime mounts without which 9 of the 36 tool
# servers never come up). Needs Docker, Node and Python.
set -euo pipefail
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/../_common.sh"
fetch_upstream "$here"
cd "$here/src"
docker image inspect agent-environment:latest >/dev/null 2>&1 || {
  docker pull ghcr.io/scaleapi/mcp-atlas:1.2.7 && docker tag ghcr.io/scaleapi/mcp-atlas:1.2.7 agent-environment:latest; }
make install-harness      # services/agent-harness (npm)
# the runner's and the scorer's python deps, in their own venv (upstream's `make
# install-python` installs into whatever python is on PATH)
[ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv
VIRTUAL_ENV="$PWD/.venv" uv pip install -q -r requirements.txt
[ -f .env ] || { cp env.template .env; echo "==> wrote $here/src/.env from env.template: fill in the MCP servers' third-party keys"; }
echo "==> mcp-atlas ready: $here/src"
