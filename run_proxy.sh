#!/usr/bin/env bash
# Start the tracing proxy on :4000 (PROXY_PORT to change). Reads .env from this directory;
# writes traces under ./artifacts/trajectories/. Everything a benchmark sends goes through here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/proxy/llm_cost_tracker/proxy/run_proxy.sh" "$@"
