#!/usr/bin/env bash
# Shared by every benchmarks/<name>/setup.sh and run.sh. Source it; do not run it.
#
#   bench_root   this benchmark's directory (where UPSTREAM and the patch live)
#   repo_root    the repository root (where .env and proxy/ live)
#   src          the upstream checkout, created by setup.sh at $bench_root/src
#
# load_env      sources $repo_root/.env but lets OPENAI_BASE_URL / TRACE_PROXY_BASE_URL
#               given on the command line win over the file (so one shell can point a
#               single run at a test proxy without editing .env).
# fetch_upstream clones the pinned upstream commit into src/ and applies the patch.
#               Idempotent: a src/ that already has the patch applied is left alone.

_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$_common_dir/.." && pwd)"

load_env() {
  local keep_base="${OPENAI_BASE_URL-__unset__}" keep_trace="${TRACE_PROXY_BASE_URL-__unset__}"
  if [ -f "$repo_root/.env" ]; then set -a; . "$repo_root/.env"; set +a; fi
  [ "$keep_base" != "__unset__" ] && export OPENAI_BASE_URL="$keep_base"
  [ "$keep_trace" != "__unset__" ] && export TRACE_PROXY_BASE_URL="$keep_trace"
  # The proxy is the only thing a benchmark talks to. Its client key is the proxy's
  # master key, never a provider key; provider keys live in the proxy's environment.
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:4000/v1}"
  export TRACE_PROXY_BASE_URL="${TRACE_PROXY_BASE_URL:-$OPENAI_BASE_URL}"
  : "${PROXY_API_KEY:?set PROXY_API_KEY in $repo_root/.env (the proxy master key)}"
}

PATCH_COMMIT_MSG="cost tracking patch (this repository)"

fetch_upstream() {
  local bench_root="$1" src="$1/src"
  # shellcheck disable=SC1091
  . "$bench_root/UPSTREAM"       # repo= commit= date= note=
  if [ -d "$src/.git" ]; then
    # Our patch is committed on top of the pinned commit; a checkout whose HEAD is that
    # commit is ready, whatever run artefacts have accumulated in the tree since.
    if [ "$(git -C "$src" log -1 --format=%s)" = "$PATCH_COMMIT_MSG" ] \
       && git -C "$src" merge-base --is-ancestor "$commit" HEAD 2>/dev/null; then
      echo "==> $src already at $commit with the patch applied"; return 0
    fi
    echo "==> $src exists but is not upstream@$commit + patch; remove it to re-fetch" >&2; return 1
  fi
  echo "==> cloning $repo @ $commit"
  git clone -q "$repo" "$src"
  git -C "$src" checkout -q "$commit"
  git -C "$src" apply "$bench_root/cost_tracking.patch"
  git -C "$src" add -A && git -C "$src" -c user.name=setup -c user.email=setup@local commit -q -m "$PATCH_COMMIT_MSG"
  echo "==> patched: $(git -C "$src" diff --stat HEAD~1 | tail -1)"
}

# The proxy must serve the alias a run asks for; checking first turns a silent
# "This model isn't mapped" failure on every task into one line here.
proxy_serves() {
  local alias="${1#openai/}" models
  models="$(curl -sf -m 10 -H "Authorization: Bearer $PROXY_API_KEY" "${OPENAI_BASE_URL%/v1}/v1/models" || true)"
  case "$models" in
    *"\"$alias\""*) return 0 ;;
    *) echo "proxy at $OPENAI_BASE_URL is not running or does not serve $alias (see proxy/config.yaml)" >&2; return 1 ;;
  esac
}
