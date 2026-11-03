"""LiteLLM proxy custom logger for LLM cost monitoring.

Writes one JSONL row per LLM call. Works with any application that routes
calls through the litellm proxy — completely decoupled from specific benchmarks.

Trace grouping uses incoming trace headers (`x-trace-context` preferred, legacy
`x-trace-meta` supported). Calls without valid trace metadata are logged under
a default "untraced" benchmark group.

Every row also carries ``request_params``: the effort / thinking / sampling settings
this call ACTUALLY ran with, taken from the body litellm sent to the provider, from
litellm's merged params, and from what the provider echoed back, so a row says which
proxy config produced it - see ``_effort_record``. A row whose tool-call arguments
are not valid JSON gets ``tool_args_invalid`` (count) and a printed warning, so a broken
stream assembly shows up on the first call rather than after a whole cell. A turn whose end the
normalised ``stop_reason`` cannot explain - an Anthropic refusal (seen here as
``content_filter``) or a Gemini ``MALFORMED_FUNCTION_CALL`` (seen as ``end_turn``) - carries
``stop_details`` with the provider's own reason; see ``_stop_details``.

Each trace directory holds:
  calls.jsonl        one row per successful call (the analysis contract), now also with
                     ``requested_model`` (the alias the client called), ``prices`` (the per-million
                     rates behind cost_usd), ``response_id`` (the provider's id from the raw body;
                     litellm's when streamed) and ``tools`` (names + schema hash)
  tools_<hash>.json  the full tool schema the harness sent, written once per distinct hash
  raw.jsonl          the provider's response body as litellm received it, one line per
                     non-streamed call, keyed by (trace_id, round). Streams have no single
                     body and are not written here.
  errors.jsonl       one row per FAILED call (HTTP 4xx/5xx, timeouts, connection errors):
                     litellm reports these through ``log_failure_event``, which the logger
                     never implemented, so retries and provider rejections were invisible.

Registered in config.yaml:
    litellm_settings:
      callbacks: llm_cost_tracker.proxy.trace_logger.proxy_logger
"""
from __future__ import annotations

import re
import base64
import json
import os
import threading
try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix platforms
    fcntl = None  # type: ignore[assignment]
from binascii import Error as BinasciiError
from hashlib import sha256
import collections
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from litellm.integrations.custom_logger import CustomLogger

# Runtime fixes for litellm bugs the proxy must not pass on (see the module docstring).
# The proxy loads this module as its callback, so importing here applies them at start.
try:
    import litellm_patches as _litellm_patches
except ImportError:  # imported as llm_cost_tracker.proxy.trace_logger (tests, tooling)
    from . import litellm_patches as _litellm_patches
_litellm_patches.apply()
_litellm_patches.apply_tool_message_marker()


def _sanitize_path_segment(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "untraced"
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    sanitized = "".join(out).strip("._")
    return sanitized or "untraced"


def _log_paths(
    trace_meta: Optional[Dict[str, Any]] = None,
    model: str = "",
    trace_id: str = "",
) -> List[str]:
    """Resolve output paths for call logs.

    Finest-grained layout (single output):
      artifacts/trajectories/<owner>/<credential>/<benchmark>/<domain>/<model>/trace_<id>/calls.jsonl

    Benchmark, then domain and model, then one directory per trace, so a single
    task/trial can be read without scanning the whole cell.

    ``<owner>`` comes from the ``user_id`` trace field and defaults to
    ``shared``, which is what every runner that does not set it gets. It exists
    so one user's subtree can be handed out on its own, certain to contain nobody
    else's calls — isolation by path rather than by filtering after the fact.

    ``<credential>`` comes from the ``api_key`` trace field and defaults to
    ``no_key``. Nesting it *under* the owner rather than folding it into the
    same segment is deliberate: credentials get rotated, and a rotation should
    start a new subtree while leaving everything that account ever ran readable
    from one place.

    ``<domain>`` comes from the ``domain`` trace field. Benchmarks without a
    notion of domain (SWE, claw-eval) fall back to ``no_domain`` so the depth
    stays predictable.

    If LLM_MONITOR_LOG is set to a file path, its directory is used as the
    trace root and the subfolders above are applied beneath it:
      LLM_MONITOR_LOG=/abs/path/calls.jsonl
      -> /abs/path/<owner>/<credential>/<benchmark>/<domain>/<model>/trace_<id>/calls.jsonl
    """
    meta = trace_meta or {}
    owner = _sanitize_path_segment(meta.get("user_id", "shared"))
    credential = _sanitize_path_segment(meta.get("api_key", "no_key"))
    benchmark = _sanitize_path_segment(meta.get("benchmark", "untraced"))
    domain = _sanitize_path_segment(meta.get("domain", "no_domain"))
    model_segment = _sanitize_path_segment(model or "unknown_model")
    configured = os.environ.get("LLM_MONITOR_LOG")
    if configured:
        root = os.path.dirname(configured) or os.getcwd()
    else:
        root = os.path.join(os.getcwd(), "artifacts", "trajectories")
    trace_segment = _sanitize_path_segment(trace_id or "untraced")
    return [
        os.path.join(
            root,
            owner,
            credential,
            benchmark,
            domain,
            model_segment,
            f"trace_{trace_segment}",
            "calls.jsonl",
        )
    ]


#: Superseded attempts are kept beside the canonical log as ``calls.jsonl.attempt1``,
#: ``…attempt2``, and so on.
ATTEMPT_SUFFIX = ".attempt"


def _archive_superseded_log(log_path: str) -> str:
    """Move a previous attempt's log aside instead of deleting it.

    A retry re-enters the same trace scope with a fresh scope id but the *same*
    trace id — ``attempt`` and ``simulation_id`` are deliberately excluded from
    the path hash — so it resolves to this same file. Deleting here discarded the
    earlier attempt's calls outright, which silently under-reported spend: a
    throttled run that burned 32 requests could end up reporting the two calls of
    its final attempt.

    Renaming keeps ``calls.jsonl`` meaning "the current attempt" while leaving
    every earlier attempt on disk for cost accounting.
    """
    index = 1
    while os.path.exists(f"{log_path}{ATTEMPT_SUFFIX}{index}"):
        index += 1
    archived = f"{log_path}{ATTEMPT_SUFFIX}{index}"
    try:
        os.replace(log_path, archived)
    except OSError as exc:  # pragma: no cover - fall back rather than lose the reset
        print(f"[trace_logger] could not archive {log_path}: {exc}; removing instead")
        try:
            os.remove(log_path)
        except OSError:
            pass
        return ""
    return archived


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _infer_caller(messages: List[Dict[str, Any]], tools: Optional[list]) -> str:
    """Infer caller role from message context. Override via x-caller header."""
    if tools:
        return "agent"
    system = ""
    for m in messages or []:
        if m.get("role") == "system":
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            system = content
            break
    if "user interacting with an agent" in system:
        return "user_sim"
    return "agent"


def _dig_headers(kwargs: Dict[str, Any]) -> Dict[str, str]:
    """Incoming request headers, lower-cased.

    litellm hands the proxy's request to every callback as
    ``litellm_params.proxy_server_request.headers`` (verified on 1.100.1 for the
    Anthropic, Gemini, Responses and OpenAI-compatible routes, streamed and not).
    """
    lp = kwargs.get("litellm_params") or {}
    req = lp.get("proxy_server_request") if isinstance(lp, dict) else None
    headers = req.get("headers") if isinstance(req, dict) else None
    if not isinstance(headers, dict):
        return {}
    return {str(k).lower(): v for k, v in headers.items()}


def _uget(obj, key: str, default=0):
    """Read `key` from either dict-like or attribute-like objects.

    LiteLLM/OpenAI SDK usage payloads can be plain dicts or typed objects.
    This helper keeps extraction code provider-agnostic.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _usage_zero() -> Dict[str, int]:
    """Return a normalized zero-usage payload.

    Used when response usage is missing or unparsable.
    """
    return {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}


def _usage_prompt_completion(usage: Any) -> tuple[int, int]:
    """Extract base prompt/output token counts with naming fallbacks.

    Field mapping:
      - prompt tokens: `usage.prompt_tokens` or `usage.input_tokens`
      - output tokens: `usage.completion_tokens` or `usage.output_tokens`
    """
    prompt = int((_uget(usage, "prompt_tokens", None) or _uget(usage, "input_tokens", 0)) or 0)
    completion = int((_uget(usage, "completion_tokens", None) or _uget(usage, "output_tokens", 0)) or 0)
    return prompt, completion


def _extract_usage_anthropic_bedrock(usage: Any) -> Dict[str, int]:
    """Extract usage for Anthropic/Bedrock-style explicit cache fields.

    Expected fields:
      - `cache_read_input_tokens`
      - `cache_creation_input_tokens`

    Normalization rule:
      input_tokens (uncached) = prompt - cache_read - cache_write
      cache_read_tokens       = cache_read_input_tokens
      cache_write_tokens      = cache_creation_input_tokens
    """
    prompt, completion = _usage_prompt_completion(usage)
    cache_read = int((_uget(usage, "cache_read_input_tokens", 0)) or 0)
    cache_write = int((_uget(usage, "cache_creation_input_tokens", 0)) or 0)
    input_tokens = prompt - cache_read - cache_write
    return {
        "input_tokens": max(input_tokens, 0),
        "output_tokens": completion,
        "cache_read_tokens": max(cache_read, 0),
        "cache_write_tokens": max(cache_write, 0),
    }


def _extract_usage_openrouter_openai(usage: Any) -> Dict[str, int]:
    """
    Extract usage for OpenRouter/OpenAI-style `prompt_tokens_details`.

    Expected fields:
      - `prompt_tokens_details.cached_tokens` (cache read)
      - optional `prompt_tokens_details.cache_write_tokens` (cache write)

    Normalization rules:
      - if write is present:
          input_tokens = prompt - cached_tokens - cache_write_tokens
      - if write is absent:
          input_tokens = prompt - cached_tokens
          cache_write_tokens = 0

    This keeps `input_tokens` as "input.uncached" while avoiding inferred writes.
    """
    prompt, completion = _usage_prompt_completion(usage)
    details = _uget(usage, "prompt_tokens_details", None)
    cache_read = int((_uget(details, "cached_tokens", 0)) or 0)
    cache_write = int((_uget(details, "cache_write_tokens", 0)) or 0)
    if cache_write > 0:
        input_tokens = prompt - cache_read - cache_write
    else:
        input_tokens = prompt - cache_read
    return {
        "input_tokens": max(input_tokens, 0),
        "output_tokens": completion,
        "cache_read_tokens": max(cache_read, 0),
        "cache_write_tokens": max(cache_write, 0),
    }


def _extract_usage_sglang(usage: Any) -> Dict[str, int]:
    """
    Extract usage for SGLang OpenAI-compatible chat responses.

    SGLang's canonical OpenAI-style usage fields are:
      - `usage.prompt_tokens`
      - `usage.completion_tokens`
      - `usage.total_tokens`

    Optional cache reporting (when SGLang cache reporting is enabled and the
    request asks for cached-token details) is:
      - `usage.prompt_tokens_details.cached_tokens`

    SGLang OpenAI-compatible usage does not natively expose a write-token field.
    For faithful reporting, we therefore do NOT infer cache writes here.

    Normalization rule:
      input_tokens (uncached) = prompt - cached_tokens
      cache_read_tokens       = cached_tokens
      cache_write_tokens      = 0
    """
    prompt, completion = _usage_prompt_completion(usage)
    details = _uget(usage, "prompt_tokens_details", None)
    cache_read = int((_uget(details, "cached_tokens", 0)) or 0)
    cache_write = 0
    input_tokens = prompt - cache_read
    return {
        "input_tokens": max(input_tokens, 0),
        "output_tokens": completion,
        "cache_read_tokens": max(cache_read, 0),
        "cache_write_tokens": 0,
    }


def _extract_usage_no_cache_fields(usage: Any) -> Dict[str, int]:
    """Fallback when provider exposes no cache usage fields.

    Normalization rule:
      input_tokens = prompt
      cache_read_tokens = 0
      cache_write_tokens = 0
    """
    prompt, completion = _usage_prompt_completion(usage)
    return {
        "input_tokens": max(prompt, 0),
        "output_tokens": max(completion, 0),
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }


def _looks_sglang_model(model: str) -> bool:
    m = (model or "").lower()
    return ("qwen" in m) or ("sglang" in m) or ("sentinel-qwen" in m)


def _looks_bedrock_anthropic_model(model: str) -> bool:
    m = (model or "").lower()
    return ("anthropic" in m) or ("claude" in m) or ("bedrock/" in m)


def _extract_usage(response_obj, model: str = "") -> Dict[str, int]:
    """Dispatch usage extraction to provider/schema-specific handlers.

    Detection strategy:
      1) Model hint for SGLang-like names.
      2) Presence of Anthropic/Bedrock explicit fields.
      3) Presence of OpenRouter/OpenAI `prompt_tokens_details` fields.
      4) Fallback to no-cache schema.

    Output schema is always:
      {
        "input_tokens": <uncached_input_for_this_call>,
        "output_tokens": <completion_tokens>,
        "cache_read_tokens": <cache_read_tokens>,
        "cache_write_tokens": <cache_write_tokens>
      }
    """
    usage = getattr(response_obj, "usage", None)
    if usage is None and isinstance(response_obj, dict):
        usage = response_obj.get("usage")
    if usage is None:
        return _usage_zero()

    prompt_details = _uget(usage, "prompt_tokens_details", None)
    has_openrouter_fields = (
        _uget(prompt_details, "cached_tokens", None) is not None
        or _uget(prompt_details, "cache_write_tokens", None) is not None
    )
    has_anthropic_fields = (
        _uget(usage, "cache_read_input_tokens", None) is not None
        or _uget(usage, "cache_creation_input_tokens", None) is not None
    )

    if _looks_sglang_model(model):
        return _extract_usage_sglang(usage)
    if has_anthropic_fields or _looks_bedrock_anthropic_model(model):
        return _extract_usage_anthropic_bedrock(usage)
    if has_openrouter_fields:
        return _extract_usage_openrouter_openai(usage)
    return _extract_usage_no_cache_fields(usage)


def _parse_json_dict(raw: Any) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _parse_trace_context_b64(raw: Any) -> Optional[Dict[str, Any]]:
    """Parse tracer-style x-trace-context header (base64(json(dict)))."""
    if not raw:
        return None
    try:
        payload = base64.b64decode(str(raw), validate=True)
        parsed = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError, BinasciiError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _parse_trace_meta(headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Read trace context from either legacy or tracer-native header formats.

    Supported formats:
      - x-trace-meta: plain JSON object string
      - x-trace-context: base64(json(object))
    """
    # Backward-compatible path used by existing tau2/swe patches.
    parsed = _parse_json_dict(headers.get("x-trace-meta"))
    if parsed is not None:
        return parsed
    # Tracer-native path (stackable context header).
    return _parse_trace_context_b64(headers.get("x-trace-context"))


def _canonical_meta_json(trace_meta: Dict[str, Any]) -> str:
    return json.dumps(trace_meta, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


_VOLATILE_TRACE_KEYS = frozenset({
    # Per-attempt UUID — excluded from trace_id hash (tau2 retries append by default).
    "simulation_id",
    # Scope token: new id per tracer.trace(reset=True) scope; drives log truncation.
    "_scope_id",
    "scope_id",
    # Reset flags — control truncate-on-first-call, not file identity.
    "reset",
    "replace",
    "trace_reset",
    # Common volatile keys from external runners.
    "attempt",
    "attempt_id",
    "retry",
    "retry_count",
    "timestamp",
    "ts",
})


def _is_reset_requested(trace_meta: Optional[Dict[str, Any]]) -> bool:
    if not trace_meta:
        return False
    for key in ("reset", "replace", "trace_reset"):
        val = trace_meta.get(key)
        if val is None:
            continue
        if isinstance(val, bool):
            if val:
                return True
            continue
        if isinstance(val, (int, float)):
            if val:
                return True
            continue
        if str(val).lower() in {"true", "1", "yes", "on"}:
            return True
    return False


def _scope_id_from_meta(trace_meta: Dict[str, Any]) -> str:
    """Resolve scope identity for reset-once semantics.

    Prefer explicit tracer ``_scope_id``; fall back to other volatile ids
    (e.g. tau2 ``simulation_id``) for legacy callers without ``reset=True``.
    """
    for key in ("_scope_id", "scope_id", "simulation_id", "attempt_id", "attempt"):
        val = trace_meta.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _scope_sidecar_path(log_path: str) -> str:
    return log_path + ".scope"


def _flock_excl(file_obj) -> None:
    if fcntl is not None:
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)


def _funlock(file_obj) -> None:
    if fcntl is not None:
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)


def _resolve_model(trace_meta: Optional[Dict[str, Any]], request_model: str) -> str:
    """Prefer explicit request model; fall back to model declared in trace context."""
    req = (request_model or "").strip()
    if req:
        return req
    if trace_meta:
        ctx_model = trace_meta.get("model")
        if ctx_model is not None and str(ctx_model).strip():
            return str(ctx_model).strip()
    return ""


def _stable_trace_meta(trace_meta: Dict[str, Any], model: str = "") -> Dict[str, Any]:
    """Return stable trace identity fields used for trace_id hashing.

    Keep benchmark/task-level identity and drop per-attempt fields so retries
    append into the same trace file instead of exploding folder count.

    `model` must be the alias the client CALLED, not the deployment it resolved
    to (see `_identity_model`): kimi-k3-low and kimi-k3-high both resolve to
    `kimi-k3`, the mcp-atlas harness stamps a constant run_id, and with the resolved
    name in the hash the two effort arms of one task would share a trace_id - and the
    second run's reset=true would truncate the first run's trajectory.
    45 kimi-k3-low mcp_atlas trajectories were destroyed that way before the run
    was stopped.
    """
    stable = {k: v for k, v in trace_meta.items() if k not in _VOLATILE_TRACE_KEYS}
    if model:
        stable["model"] = model
    return stable


def _identity_model(trace_meta: Optional[Dict[str, Any]], kwargs: Dict[str, Any]) -> str:
    """The model name that goes into the trace id: the requested alias when the
    router recorded one, else the resolved model. The directory layout keeps using
    the resolved model (`_log_paths`), so effort arms stay in their base model's
    folder and are told apart by `requested_model` on every row, as before."""
    return _requested_model(kwargs) or _resolve_model(trace_meta, kwargs.get("model") or "")


def _trace_id_from_meta(trace_meta: Dict[str, Any], model: str = "") -> str:
    payload = _canonical_meta_json(_stable_trace_meta(trace_meta, model=model)).encode("utf-8")
    return f"meta-{sha256(payload).hexdigest()[:20]}"


def _trace_name_from_meta(trace_meta: Dict[str, Any]) -> str:
    parts = []
    for key in sorted(trace_meta):
        value = trace_meta[key]
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            value_text = json.dumps(value, sort_keys=True, ensure_ascii=False)
        else:
            value_text = str(value)
        parts.append(f"{key}={value_text}")
    return "|".join(parts)


def _compute_cost(usage: Dict[str, int], model: str) -> Optional[float]:
    """Compute dollar cost from token usage. Returns None for unpriced models."""
    try:
        from llm_cost_tracker.pricing import (
            get_token_price, get_cache_pricing, has_real_pricing,
        )
        if not has_real_pricing(model):
            return None
        in_price, out_price = get_token_price(model)
        cache_read_price, cache_write_price = get_cache_pricing(model)
        cost = (
            usage["input_tokens"] * in_price / 1_000_000
            + usage["cache_read_tokens"] * cache_read_price / 1_000_000
            + usage["cache_write_tokens"] * cache_write_price / 1_000_000
            + usage["output_tokens"] * out_price / 1_000_000
        )
        return round(cost, 6)
    except Exception:
        return None


#: Request-side keys worth keeping per row. Effort/thinking first (the ablation axis),
#: then the sampling/limit params every harness sets differently.
_EFFORT_KEYS = (
    "reasoning_effort", "reasoning", "thinking", "output_config", "extra_body",
    "temperature", "max_tokens", "max_output_tokens", "max_completion_tokens",
)


def _effort_record(kwargs: Dict[str, Any], response_obj: Any) -> Dict[str, Any]:
    """The effort / thinking / sampling settings this call actually carried.

    Three sources, kept apart so a reader can see where a value came from and
    which one won (measured: an alias-level ``output_config`` beats a
    client ``reasoning_effort`` on Anthropic, while on Gemini, OpenAI Responses and
    the kimi-k3 gateway the client value wins):

      outbound   effort-related keys of the body litellm sent to the provider
                 (``additional_args.complete_input_dict``). Present for Anthropic,
                 Gemini and OpenAI-compatible chat routes; absent on the Responses
                 route, where litellm does not expose the translated body.
      requested  litellm's merged ``optional_params`` (client param + alias config).
      echoed     what the provider says it applied. The Responses API returns
                 ``reasoning`` (effort, summary mode) on every response.

    Never raises: a logging failure must not take down a run.
    """
    out: Dict[str, Any] = {}
    try:
        add = kwargs.get("additional_args") or {}
        cid = add.get("complete_input_dict") if isinstance(add, dict) else None
        if isinstance(cid, str):
            try:
                cid = json.loads(cid)
            except (TypeError, ValueError):
                cid = None
        if isinstance(cid, dict):
            outbound = {k: cid[k] for k in _EFFORT_KEYS if k in cid}
            gen_cfg = cid.get("generationConfig")
            if isinstance(gen_cfg, dict):  # Gemini nests thinking + sampling here
                outbound["generationConfig"] = {
                    k: v for k, v in gen_cfg.items()
                    if k in ("thinkingConfig", "temperature", "max_output_tokens")
                }
            if outbound:
                out["outbound"] = outbound
        opt = kwargs.get("optional_params") or {}
        if isinstance(opt, dict):
            requested = {k: opt[k] for k in _EFFORT_KEYS if k in opt}
            if requested:
                out["requested"] = requested
        echo = getattr(response_obj, "reasoning", None)
        if echo is not None:
            out["echoed"] = echo.model_dump() if hasattr(echo, "model_dump") else echo
    except Exception as exc:  # pragma: no cover - defensive
        out["error"] = str(exc)[:160]
    return out


def _invalid_tool_args(tool_calls: Optional[List[Dict[str, Any]]]) -> int:
    """Count tool calls whose ``arguments`` string is not valid JSON.

    A non-zero count means the record is wrong, not that the model misbehaved:
    the one way this has happened so far is litellm's chunk builder concatenating
    the arguments of parallel calls (``{"a":1}{"b":2}``) when a streamed
    Responses-route reply arrived with every call at index 0 (litellm < 1.98).
    """
    bad = 0
    for tc in tool_calls or []:
        args = ((tc or {}).get("function") or {}).get("arguments")
        if args is None or args == "":
            continue
        if not isinstance(args, str):
            continue
        try:
            json.loads(args)
        except (TypeError, ValueError):
            bad += 1
    return bad


def _stop_details(kwargs: Dict[str, Any], stop_reason: Optional[str]) -> Optional[Dict[str, Any]]:
    """Why a turn ended when the normalised ``stop_reason`` cannot say. None otherwise.

    Two providers end turns in ways litellm's chat vocabulary flattens:

    * Anthropic ``stop_reason: "refusal"`` (Claude 4.7+ / 5.x): HTTP 200, empty content,
      zero output tokens, and the reason in ``stop_details`` (``category`` such as
      ``reasoning_extraction``, plus ``explanation``). litellm reports ``content_filter``
      and drops the details.
    * Gemini ``finishReason: "MALFORMED_FUNCTION_CALL"`` (and any other non-STOP reason
      litellm maps to "stop"): thinking tokens are billed, no part but the thought comes
      back, and litellm reports a plain stop. Measured: 1 of 3 identical
      four-call prompts to gemini-3.1-pro ended this way; the row looked like a model
      that chose to say nothing (output_tokens == reasoning_tokens, no content).

    Both survive only in the raw upstream body litellm hands the callback as
    ``original_response`` (a JSON string on non-streamed calls; unavailable on streams),
    so this is best effort and never raises.
    """
    raw = kwargs.get("original_response")
    if not isinstance(raw, str):
        return None
    if stop_reason in ("content_filter", "refusal") and "stop_details" in raw:
        try:
            body = json.loads(raw)
        except (TypeError, ValueError):
            return None
        details = body.get("stop_details") if isinstance(body, dict) else None
        return details if isinstance(details, dict) else None
    if stop_reason == "end_turn" and "finishReason" in raw:
        try:
            body = json.loads(raw)
        except (TypeError, ValueError):
            return None
        cands = body.get("candidates") if isinstance(body, dict) else None
        fr = (cands[0] or {}).get("finishReason") if isinstance(cands, list) and cands else None
        if isinstance(fr, str) and fr != "STOP":
            return {"type": "gemini_finish_reason", "finish_reason": fr}
    return None


def _sibling(log_path: str, name: str) -> str:
    return os.path.join(os.path.dirname(log_path) or ".", name)


#: Files kept next to calls.jsonl that belong to the same attempt and move with it on reset.
_SIBLING_LOGS = ("raw.jsonl", "errors.jsonl")


def _tools_record(tools: Any, log_path: str) -> Optional[Dict[str, Any]]:
    """Names + schema hash for the row; the full schema once per hash next to the log.

    The schema is re-sent on every call and can run to tens of KB (mcp_atlas: 36 servers),
    so it is not repeated per row. The hash lets a reader prove the tool surface did not
    change mid-trace - an mcp_atlas run can silently lose 9 servers' tools, and
    nothing else in the rows would show it.
    """
    if not isinstance(tools, list) or not tools:
        return None
    try:
        canon = json.dumps(tools, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    except Exception:
        return None
    digest = sha256(canon.encode("utf-8")).hexdigest()[:16]
    names: List[str] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else None
        name = (fn or {}).get("name") or t.get("name")
        if name:
            names.append(str(name))
    path = _sibling(log_path, f"tools_{digest}.json")
    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "x", encoding="utf-8") as f:
                f.write(canon)
        except FileExistsError:
            pass
        except OSError as exc:  # pragma: no cover
            print(f"[trace_logger] could not write {path}: {exc}")
    return {"names": names, "hash": digest, "count": len(tools)}


def _provider_response_id(raw: Any, response_obj: Any) -> Optional[str]:
    """The provider's own id for this response, falling back to litellm's.

    litellm's normalised object carries a synthetic ``chatcmpl-…`` / base64 id on the
    anthropic/ and Responses routes; the provider's id (``msg_…``,
    ``resp_…``, Gemini ``responseId``) is in the raw body, which is what a support
    ticket or a billing export needs.
    """
    if isinstance(raw, dict):
        for key in ("id", "responseId"):
            val = raw.get(key)
            if isinstance(val, str) and val:
                return val
    val = getattr(response_obj, "id", None)
    return val if isinstance(val, str) and val else None


def _raw_body(kwargs: Dict[str, Any]) -> Any:
    """The provider's response body as litellm received it, or None (streams)."""
    raw = kwargs.get("original_response")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _requested_model(kwargs: Dict[str, Any]) -> Optional[str]:
    """The alias the client asked for (e.g. ``claude-fable-5-1-max``).

    The row's ``model`` is the resolved deployment name shared by every effort alias,
    so without this the effort level could only be inferred back from request_params.
    litellm's router records the alias as ``litellm_params.metadata.model_group``.
    """
    lp = kwargs.get("litellm_params") or {}
    md = lp.get("metadata") if isinstance(lp, dict) else None
    if isinstance(md, dict):
        for key in ("model_group", "original_model_group"):
            val = md.get(key)
            if isinstance(val, str) and val:
                return val
    val = kwargs.get("model")
    return val if isinstance(val, str) and val else None


def _unit_prices(model: str) -> Optional[Dict[str, float]]:
    """USD per million tokens used for this row's cost_usd, so a later price-table
    change cannot make recorded costs irreproducible."""
    try:
        from llm_cost_tracker.pricing import get_token_price, get_cache_pricing, has_real_pricing
        if not has_real_pricing(model):
            return None
        in_price, out_price = get_token_price(model)
        cache_read, cache_write = get_cache_pricing(model)
        return {"input": in_price, "cache_read": cache_read, "cache_write": cache_write, "output": out_price, "unit": "usd_per_million_tokens"}
    except Exception:
        return None


class TraceLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._rounds: Dict[str, int] = defaultdict(int)
        # log_path -> active scope_id (fast path within this process)
        self._active_scopes: Dict[str, str] = {}
        # Failures already written to errors.jsonl, keyed by litellm's per-request
        # ``litellm_call_id``. litellm invokes BOTH failure hooks for proxy-level failures
        # (unknown alias), once each for provider errors; without this the former appeared
        # twice. Not keyed by id(exc): a freed exception's address is
        # reused by the next one, and a burst of five back-to-back failures lost one row.
        self._seen_errors: "collections.deque[Any]" = collections.deque(maxlen=4096)

    #: Model aliases served by the kimi-k3 gateway, which rejects any
    #: field it does not know with "unsupported parameter(s)". tau2 puts a
    #: non-standard `name` beside `function` on every tool_call entry
    #: (tau2/utils/llm_utils.py to_litellm_messages); OpenAI and Anthropic
    #: ignore it, the kimi gateway returns 400 (every tau2 simulation would fail
    #: at the first tool result). Stripped here, at
    #: the proxy, so no harness is touched and every kimi alias behaves the same.
    STRIP_TOOL_CALL_NAME_PREFIXES = ("kimi-k3",)
    #: The kimi-k3 gateway caps a completion at 128 tokens when the
    #: request carries no max_tokens; a reasoning model spends those on its
    #: chain of thought and returns an EMPTY reply (finish_reason=length,
    #: 128 output tokens all reasoning; tau2 then fails the task). claw-eval and mcp-atlas send
    #: max_tokens=131072 themselves; tau2 sends none. The same default is set
    #: here for every kimi alias when the client sent none, so the three
    #: efforts and the baseline all run under one ceiling.
    KIMI_DEFAULT_MAX_TOKENS = 131072
    #: Message fields the kimi gateway accepts, per role (None = any other role).
    KIMI_MESSAGE_KEYS = {
        "system": {"role", "content", "name"},
        "user": {"role", "content", "name"},
        "assistant": {"role", "content", "tool_calls", "reasoning_content", "name"},
        "tool": {"role", "content", "tool_call_id", "name"},
        None: {"role", "content", "name", "tool_calls", "tool_call_id"},
    }

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            model = str(data.get("model") or "")
            if model.startswith(self.STRIP_TOOL_CALL_NAME_PREFIXES) and call_type in ("completion", "acompletion"):
                if data.get("max_tokens") is None and data.get("max_completion_tokens") is None:
                    data["max_tokens"] = self.KIMI_DEFAULT_MAX_TOKENS
                n = 0
                dropped = 0
                for m in data.get("messages") or []:
                    if not isinstance(m, dict):
                        continue
                    # Only the fields the chat schema defines. mini-swe-agent echoes
                    # litellm's own Message objects back verbatim, which carry
                    # `provider_specific_fields`, `function_call: null` and the
                    # like; the kimi gateway answers 400 "unsupported parameter(s)" to
                    # every one of them. `reasoning_content` is kept:
                    # the endpoint accepts it back on assistant turns.
                    for k in [k for k in m if k not in self.KIMI_MESSAGE_KEYS.get(m.get("role"), self.KIMI_MESSAGE_KEYS[None])]:
                        m.pop(k, None)
                        dropped += 1
                    if m.get("role") == "assistant":
                        for tc in m.get("tool_calls") or []:
                            if isinstance(tc, dict) and "name" in tc and isinstance(tc.get("function"), dict):
                                tc.pop("name", None)
                                n += 1
                if n or dropped:
                    print(f"[trace_logger] {model}: stripped {n} tool_call.name and {dropped} non-standard message field(s)")
        except Exception as exc:  # never let a sanitiser fail a request
            print(f"[trace_logger] pre-call sanitiser skipped: {exc!r}")
        return data

    def _maybe_reset_log(
        self,
        trace_id: str,
        trace_meta: Dict[str, Any],
        paths: List[str],
    ) -> None:
        """Truncate canonical calls.jsonl once per scope when reset=true."""
        if not _is_reset_requested(trace_meta):
            return
        scope_id = _scope_id_from_meta(trace_meta)
        if not scope_id:
            print(
                "[trace_logger] reset=true but no scope id "
                "(_scope_id, simulation_id, etc.); skipping truncate"
            )
            return
        for path in paths:
            self._reset_log_for_scope(path, trace_id, scope_id)

    def _reset_log_for_scope(self, log_path: str, trace_id: str, scope_id: str) -> None:
        """Reset log once per scope; durable across proxy restarts via sidecar file."""
        sidecar = _scope_sidecar_path(log_path)
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

        with self._lock:
            if self._active_scopes.get(log_path) == scope_id:
                return

            try:
                with open(sidecar, "a+", encoding="utf-8") as sf:
                    _flock_excl(sf)
                    try:
                        sf.seek(0)
                        stored = sf.read().strip()
                        if stored == scope_id:
                            self._active_scopes[log_path] = scope_id
                            return
                        if os.path.isfile(log_path):
                            _archive_superseded_log(log_path)
                        for name in _SIBLING_LOGS:
                            sib = _sibling(log_path, name)
                            if os.path.isfile(sib):
                                _archive_superseded_log(sib)
                        sf.seek(0)
                        sf.truncate(0)
                        sf.write(scope_id)
                        sf.flush()
                        os.fsync(sf.fileno())
                    finally:
                        _funlock(sf)
            except OSError as exc:
                print(f"[trace_logger] reset failed for {log_path}: {exc}")
                return

            self._active_scopes[log_path] = scope_id
            self._rounds[trace_id] = 0

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record_error(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record_error(kwargs, response_obj, start_time, end_time)

    def _record_error(self, kwargs, response_obj, start_time, end_time) -> None:
        try:
            self._write_error(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            print(f"[trace_logger] failed to log error: {e}")

    def _write_error(self, kwargs, response_obj, start_time, end_time) -> None:
        """One row in errors.jsonl per failed call. litellm hands the exception as
        ``response_obj`` (and again as ``kwargs['exception']``)."""
        exc = response_obj if isinstance(response_obj, BaseException) else kwargs.get("exception")
        call_id = kwargs.get("litellm_call_id")
        dedupe_key = call_id if call_id else (id(exc), str(start_time)) if exc is not None else None
        with self._lock:
            if dedupe_key is not None:
                if dedupe_key in self._seen_errors:
                    return
                self._seen_errors.append(dedupe_key)
        messages = kwargs.get("messages") or []
        opt = kwargs.get("optional_params") or {}
        headers = _dig_headers(kwargs)
        trace_meta = _parse_trace_meta(headers)
        model = _resolve_model(trace_meta, kwargs.get("model") or "")
        if trace_meta:
            trace_id = _trace_id_from_meta(trace_meta, model=_identity_model(trace_meta, kwargs))
            trace_name = headers.get("x-trace-name") or _trace_name_from_meta(trace_meta)
        else:
            trace_id, trace_name = "untraced", "untraced"
        paths = _log_paths(trace_meta, model=model, trace_id=trace_id)
        if trace_meta is not None:
            self._maybe_reset_log(trace_id, trace_meta, paths)
        try:
            latency = (end_time - start_time).total_seconds()
        except Exception:
            latency = 0.0
        with self._lock:
            after_round = self._rounds[trace_id]
        row = {
            "kind": "error",
            "ts": _now(),
            "trace_id": trace_id,
            "trace_name": trace_name,
            "trace_meta": trace_meta,
            "after_round": after_round,
            "caller": headers.get("x-caller") or _infer_caller(messages, opt.get("tools")),
            "model": model,
            "requested_model": _requested_model(kwargs),
            "latency_s": round(latency, 3),
            "error_type": type(exc).__name__ if exc is not None else None,
            "status_code": getattr(exc, "status_code", None),
            "llm_provider": getattr(exc, "llm_provider", None),
            "message": (str(exc) if exc is not None else "")[:2000],
            "request_messages": messages,
            "request_params": _effort_record(kwargs, None),
            "tools": _tools_record(opt.get("tools"), paths[0]) if paths else None,
        }
        line = json.dumps(row, ensure_ascii=False, default=str)
        with self._lock:
            for path in paths:
                err_path = _sibling(path, "errors.jsonl")
                os.makedirs(os.path.dirname(err_path), exist_ok=True)
                with open(err_path, "a", encoding="utf-8") as f:
                    _flock_excl(f)
                    try:
                        f.write(line + "\n")
                        f.flush()
                    finally:
                        _funlock(f)

    def _record(self, kwargs, response_obj, start_time, end_time) -> None:
        try:
            self._write_row(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            print(f"[trace_logger] failed to log call: {e}")

    def _write_row(self, kwargs, response_obj, start_time, end_time) -> None:
        messages = kwargs.get("messages") or []
        opt = kwargs.get("optional_params") or {}
        tools = opt.get("tools")
        headers = _dig_headers(kwargs)
        trace_meta_parsed = _parse_trace_meta(headers)
        model = _resolve_model(trace_meta_parsed, kwargs.get("model") or "")

        if trace_meta_parsed:
            trace_meta = trace_meta_parsed
            trace_id = _trace_id_from_meta(trace_meta, model=_identity_model(trace_meta, kwargs))
            trace_name = headers.get("x-trace-name") or _trace_name_from_meta(trace_meta)
        else:
            trace_meta = None
            trace_id = "untraced"
            trace_name = "untraced"
        caller = headers.get("x-caller") or _infer_caller(messages, tools)

        paths = _log_paths(trace_meta, model=model, trace_id=trace_id)
        if trace_meta is not None:
            self._maybe_reset_log(trace_id, trace_meta, paths)

        usage = _extract_usage(response_obj, model=model)
        cost = _compute_cost(usage, model)

        # Reasoning-token detail. OpenAI-standard placement is
        # usage.completion_tokens_details.reasoning_tokens (OpenAI, Gemini, the kimi gateway, and
        # Anthropic via litellm >= 1.98, which fills it from output_tokens_details.thinking_tokens);
        # SGLang puts it at usage.reasoning_tokens top-level. Kept separate from
        # output_tokens, which already includes reasoning. Three states are meaningful:
        # N thought for N tokens; 0 the model can think and chose not to; None the
        # provider never reports it (Bedrock).
        reasoning_tokens = None
        try:
            _u = getattr(response_obj, "usage", None)
            details = _uget(_u, "completion_tokens_details", None)
            if details is not None:
                reasoning_tokens = _uget(details, "reasoning_tokens", None)
            if reasoning_tokens is None:
                reasoning_tokens = _uget(_u, "reasoning_tokens", None)
        except Exception:
            pass

        # Response message. Two shapes reach the callback:
        #   chat routes (anthropic/, gemini/, openai-compatible)  -> ModelResponse with .choices
        #   openai/responses/ route                               -> ResponsesAPIResponse with .output
        # litellm hands the callback the RAW Responses object on that route (streamed or not),
        # so it is read directly rather than through a chat translation.
        resp_msg: Dict[str, Any] = {"role": "assistant", "content": None}
        stop_reason = None
        tool_called = None
        output_items = getattr(response_obj, "output", None)
        if output_items is not None and not getattr(response_obj, "choices", None):
            texts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            think: List[str] = []
            for it in output_items:
                d = it if isinstance(it, dict) else (it.model_dump() if hasattr(it, "model_dump") else {})
                kind = d.get("type")
                if kind == "message":
                    for part in (d.get("content") or []):
                        pd = part if isinstance(part, dict) else {}
                        if pd.get("text"):
                            texts.append(pd["text"])
                elif kind == "function_call":
                    tool_calls.append({
                        "id": d.get("call_id") or d.get("id"),
                        "type": "function",
                        "function": {"name": d.get("name"), "arguments": d.get("arguments")},
                    })
                elif kind == "reasoning":
                    for sm in (d.get("summary") or []):
                        sd = sm if isinstance(sm, dict) else {}
                        if sd.get("text"):
                            think.append(sd["text"])
            if texts:
                resp_msg["content"] = "\n".join(texts)
            if think:
                resp_msg["reasoning_content"] = "\n".join(think)
            if tool_calls:
                resp_msg["tool_calls"] = tool_calls
                tool_called = tool_calls[0]["function"]["name"]
            # `status` is "completed"/"incomplete"; map to the chat vocabulary below.
            st = getattr(response_obj, "status", None)
            if st == "completed":
                stop_reason = "tool_calls" if tool_calls else "stop"
            elif st:
                stop_reason = str(st)
        else:
            try:
                choice = response_obj.choices[0]
                stop_reason = getattr(choice, "finish_reason", None)
                m = choice.message
                resp_msg["content"] = getattr(m, "content", None)
                # Reasoning text arrives as message.reasoning_content (kimi-k3, gemini summaries,
                # Anthropic summarized thinking) or under provider_specific_fields.
                _reasoning = getattr(m, "reasoning_content", None)
                if _reasoning is None and hasattr(m, "provider_specific_fields"):
                    _psf = getattr(m, "provider_specific_fields") or {}
                    _reasoning = _psf.get("reasoning_content")
                if _reasoning:
                    resp_msg["reasoning_content"] = _reasoning
                tcs = getattr(m, "tool_calls", None)
                if tcs:
                    resp_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tcs
                    ]
                    tool_called = tcs[0].function.name
            except Exception:
                pass

        try:
            latency = (end_time - start_time).total_seconds()
        except Exception:
            latency = 0.0

        # Normalize finish_reason to one vocabulary. A turn that carries tool calls is a
        # tool_use turn whatever the provider's streamed final chunk said: litellm's
        # gemini/ stream (1.100.1) closes with finish_reason "stop" even when the turn
        # holds four function calls, so the raw value would split identical turns into
        # two classes by provider.
        if stop_reason == "tool_calls":
            stop_reason = "tool_use"
        elif stop_reason == "stop":
            stop_reason = "end_turn"
        if resp_msg.get("tool_calls") and stop_reason in ("end_turn", None):
            stop_reason = "tool_use"

        with self._lock:
            self._rounds[trace_id] += 1
            round_idx = self._rounds[trace_id]

        row = {
            "kind": "call",
            "ts": _now(),
            "trace_id": trace_id,
            "trace_name": trace_name,
            "trace_meta": trace_meta,
            "round": round_idx,
            "caller": caller,
            "model": model,
            "requested_model": _requested_model(kwargs),
            "latency_s": round(latency, 3),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "cache_read_tokens": usage["cache_read_tokens"],
            "cache_write_tokens": usage["cache_write_tokens"],
            "reasoning_tokens": reasoning_tokens,
            "cost_usd": cost,
            "prices": _unit_prices(model),
            "stop_reason": stop_reason,
            "tool_called": tool_called,
            "request_messages": messages,
            "response_message": resp_msg,
            "request_params": _effort_record(kwargs, response_obj),
            "response_id": None,  # filled below once the raw body is known
            "tools": _tools_record(tools, paths[0]) if paths else None,
        }
        details = _stop_details(kwargs, stop_reason)
        if details:
            row["stop_details"] = details
        n_bad_args = _invalid_tool_args(resp_msg.get("tool_calls"))
        if n_bad_args:
            row["tool_args_invalid"] = n_bad_args
            print(
                f"[trace_logger] WARNING {n_bad_args} tool call(s) with non-JSON arguments "
                f"recorded for trace {trace_id} round {round_idx} (model={model}); "
                "the stream assembly merged parallel calls - see _invalid_tool_args"
            )

        raw = _raw_body(kwargs)
        row["response_id"] = _provider_response_id(raw, response_obj)
        line = json.dumps(row, ensure_ascii=False, default=str)
        raw_line = None
        if raw is not None:
            raw_line = json.dumps({"trace_id": trace_id, "round": round_idx, "ts": row["ts"],
                                   "response_id": row["response_id"], "raw": raw},
                                  ensure_ascii=False, default=str)
        with self._lock:
            for path in paths:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    _flock_excl(f)
                    try:
                        f.write(line + "\n")
                        f.flush()
                    finally:
                        _funlock(f)
                if raw_line is not None:
                    with open(_sibling(path, "raw.jsonl"), "a", encoding="utf-8") as f:
                        _flock_excl(f)
                        try:
                            f.write(raw_line + "\n")
                            f.flush()
                        finally:
                            _funlock(f)


proxy_logger = TraceLogger()
