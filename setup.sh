#!/usr/bin/env bash
# One-time installs. Each block has its own environment; nothing is installed globally.
#
#   ./setup.sh              everything below
#   ./setup.sh proxy        the tracing proxy (litellm 1.100.1 in proxy/.venv)
#   ./setup.sh llmaj        the judge package (LLMaJ/.venv)
#   ./setup.sh tau2-bench | claw-eval | mini-swe-agent | mcp-atlas
#                           clone the pinned upstream, apply our patch, install
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
what=("$@"); [ ${#what[@]} -gt 0 ] || what=(proxy llmaj tau2-bench claw-eval mini-swe-agent mcp-atlas)
for w in "${what[@]}"; do
  echo; echo "########## $w"
  case "$w" in
    proxy) "$ROOT/proxy/setup.sh" ;;
    llmaj) cd "$ROOT/LLMaJ"; [ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv
           VIRTUAL_ENV="$PWD/.venv" uv pip install -q -e .; echo "==> LLMaJ ready: $PWD/.venv" ;;
    tau2-bench|claw-eval|mini-swe-agent|mcp-atlas) "$ROOT/benchmarks/$w/setup.sh" ;;
    *) echo "unknown block: $w" >&2; exit 2 ;;
  esac
done
