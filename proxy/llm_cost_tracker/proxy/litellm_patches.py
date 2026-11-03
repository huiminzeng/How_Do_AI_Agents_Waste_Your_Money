"""Runtime fixes for litellm bugs the proxy would otherwise pass on to every harness.

Imported by ``trace_logger`` so that loading the callback applies them. Each patch states
the litellm version it was written against and carries a probe (`still_needed`) that says
whether upstream still has the bug; when a probe reports False the patch is a no-op and
should be deleted.

RESPONSES_TWO_CHOICES (litellm 1.100.1)
    ``LiteLLMResponsesTransformationHandler._convert_response_output_to_choices`` turns a
    Responses API reply that holds a message AND function calls into TWO chat choices:
    choices[0] = the text with finish_reason "stop", choices[1] = the tool calls. Every
    chat client reads choices[0], so the tool calls vanish on non-streamed calls while the
    proxy's own record (built from the raw Responses object) still shows them. Upstream
    issue #17246 fixed the streamed variant (PR #17652); the non-streamed split remains in
    1.100.1 (reproduced with gpt-6-astra and gpt-5.6-terra: client content
    present, tool_calls == [], finish_reason "stop"). tau2 and mcp-atlas call these models
    non-streamed, so without this every text+tool turn would look to them like the model
    refused to act. With the fix at the proxy, no harness has to stream to dodge it.

    The fix merges the trailing tool_calls choice into the first text choice: content from
    the text choices, tool_calls from the last, reasoning from whichever carried it,
    finish_reason "tool_calls", index 0.

ANTHROPIC_TOOL_MESSAGE_MARKER (litellm 1.100.1)
    ``AnthropicCacheControlHook._safe_insert_cache_control_in_message`` puts a cache marker
    on a list-content message's LAST CONTENT BLOCK. For a ``role: tool`` message that block
    becomes ``tool_result.content[-1]``, and Anthropic rejects the request: "cache_control
    may not be specified within tool_result.content. Instead, place it directly on
    tool_result." (every mini-swe-agent SWE-bench trajectory would die at round 2:
    mini-swe sends tool results as content lists, tau2 and claw-eval as strings). litellm's
    own tool_result conversion already honours a MESSAGE-level ``cache_control`` on a tool
    message by placing it on the tool_result block (factory.convert_to_anthropic_tool_result),
    so the fix is to set it there for tool messages and leave every other role alone.

    Why it matters: it lets config.yaml mark the LAST message (``index: -1``) again. The
    ``role: user`` alternative marks EVERY user message on
    1.100.1 (``_resolve_target_indices`` returns all matches), hits Anthropic's cap of 4
    breakpoints by the fourth user turn, and leaves no marker after the last user message
    during tool-call streaks - measured on a fable smoke run: fresh (uncached)
    input rose from 0% to 34% of the prompt by round 12, cache writes stopped, and the
    proxy log filled with "Reached the provider limit of 4 cache breakpoints".
"""
from __future__ import annotations

from typing import Any, Sequence

__all__ = ["apply", "still_needed", "apply_tool_message_marker", "tool_message_marker_still_needed"]

_applied = False
_applied_tool_marker = False


def _merge_choices(choices: list[Any]) -> list[Any]:
    if len(choices) < 2:
        return choices
    last = choices[-1]
    last_tools = getattr(getattr(last, "message", None), "tool_calls", None)
    if not last_tools:
        return choices
    texts = [c for c in choices[:-1] if getattr(getattr(c, "message", None), "content", None)]
    if len(texts) != len(choices) - 1:
        return choices  # something other than text choices precedes the tool calls; leave it
    first = texts[0]
    first.message.content = "\n".join(c.message.content for c in texts)
    first.message.tool_calls = last_tools
    for attr in ("reasoning_content", "reasoning_items"):
        if getattr(first.message, attr, None) is None and getattr(last.message, attr, None) is not None:
            setattr(first.message, attr, getattr(last.message, attr))
    first.finish_reason = "tool_calls"
    first.index = 0
    return [first]


def still_needed() -> bool:
    """True while litellm's converter still splits text + tool calls into two choices."""
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler as H,
    )
    original = getattr(H._convert_response_output_to_choices, "__wrapped_original__", H._convert_response_output_to_choices)
    items = [
        ResponseOutputMessage(id="msg_probe", role="assistant", status="completed", type="message",
                              content=[ResponseOutputText(type="output_text", text="probe", annotations=[])]),
        ResponseFunctionToolCall(id="fc_probe", call_id="call_probe", name="f", arguments="{}", type="function_call", status="completed"),
    ]
    return len(original(items)) != 1


def apply() -> None:
    global _applied
    if _applied:
        return
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler as H,
    )
    original = H._convert_response_output_to_choices
    if getattr(original, "__wrapped_original__", None) is not None:
        _applied = True
        return
    if not still_needed():
        print("[litellm_patches] RESPONSES_TWO_CHOICES: upstream no longer splits choices; delete this patch")
        _applied = True
        return

    def patched(output_items: Sequence[object], handle_raw_dict_callback=None):
        return _merge_choices(original(output_items, handle_raw_dict_callback))

    patched.__wrapped_original__ = original  # type: ignore[attr-defined]
    H._convert_response_output_to_choices = staticmethod(patched)  # type: ignore[assignment]
    _applied = True
    print("[litellm_patches] RESPONSES_TWO_CHOICES applied (text + tool calls merged into one choice)")


def tool_message_marker_still_needed() -> bool:
    """True while litellm puts a tool message's marker inside its content list."""
    from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook as H
    original = getattr(H._safe_insert_cache_control_in_message, "__wrapped_original__",
                       H._safe_insert_cache_control_in_message)
    msg = {"role": "tool", "tool_call_id": "call_probe", "content": [{"type": "text", "text": "probe"}]}
    out = original(msg, {"type": "ephemeral"}, False)
    return "cache_control" not in out and any("cache_control" in b for b in out.get("content", []))


def apply_tool_message_marker() -> None:
    global _applied_tool_marker
    if _applied_tool_marker:
        return
    from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook as H
    original = H._safe_insert_cache_control_in_message
    if getattr(original, "__wrapped_original__", None) is not None:
        _applied_tool_marker = True
        return
    if not tool_message_marker_still_needed():
        print("[litellm_patches] ANTHROPIC_TOOL_MESSAGE_MARKER: upstream places tool markers correctly; delete this patch")
        _applied_tool_marker = True
        return

    def patched(message, control, openai_dialect=False):
        if not openai_dialect and isinstance(message, dict) and message.get("role") == "tool":
            message["cache_control"] = control
            return message
        return original(message, control, openai_dialect)

    patched.__wrapped_original__ = original  # type: ignore[attr-defined]
    H._safe_insert_cache_control_in_message = staticmethod(patched)  # type: ignore[assignment]
    _applied_tool_marker = True
    print("[litellm_patches] ANTHROPIC_TOOL_MESSAGE_MARKER applied (tool messages carry the cache marker at message level)")

