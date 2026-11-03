"""Tests for the per-row request_params record and the tool-argument check."""

from __future__ import annotations

import json

from llm_cost_tracker.proxy import trace_logger as tl


class _Echo:
    def __init__(self, d):
        self._d = d

    def model_dump(self):
        return dict(self._d)


class _Resp:
    def __init__(self, reasoning=None):
        if reasoning is not None:
            self.reasoning = reasoning


def test_effort_record_anthropic_outbound_and_requested():
    kwargs = {
        "additional_args": {"complete_input_dict": {
            "model": "claude-fable-5-1", "max_tokens": 4000, "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "high"},
        }},
        "optional_params": {"reasoning_effort": "low", "output_config": {"effort": "high"}, "max_tokens": 4000},
    }
    rec = tl._effort_record(kwargs, _Resp())
    assert rec["outbound"] == {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"},
        "max_tokens": 4000,
    }
    assert rec["requested"]["output_config"] == {"effort": "high"}
    assert rec["requested"]["reasoning_effort"] == "low"
    assert "messages" not in str(rec)
    assert "echoed" not in rec


def test_effort_record_gemini_generation_config():
    kwargs = {"additional_args": {"complete_input_dict": {
        "contents": [], "system_instruction": {},
        "generationConfig": {"max_output_tokens": 4000, "temperature": 1.0,
                             "thinkingConfig": {"thinkingLevel": "low", "includeThoughts": True},
                             "candidateCount": 1},
    }}}
    rec = tl._effort_record(kwargs, _Resp())
    assert rec["outbound"]["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "low"
    assert "candidateCount" not in rec["outbound"]["generationConfig"]
    assert "contents" not in rec["outbound"]


def test_effort_record_responses_echo_when_body_missing():
    kwargs = {"optional_params": {"reasoning": {"effort": "low"}, "max_output_tokens": 4000}}
    rec = tl._effort_record(kwargs, _Resp(reasoning=_Echo({"effort": "low", "summary": "detailed"})))
    assert "outbound" not in rec
    assert rec["requested"]["reasoning"] == {"effort": "low"}
    assert rec["echoed"] == {"effort": "low", "summary": "detailed"}


def test_effort_record_accepts_json_string_body_and_never_raises():
    kwargs = {"additional_args": {"complete_input_dict": '{"model": "x", "reasoning_effort": "max"}'}}
    assert tl._effort_record(kwargs, _Resp())["outbound"] == {"reasoning_effort": "max"}
    assert tl._effort_record({"additional_args": "garbage"}, None) == {}
    assert tl._effort_record({}, object()) == {}


def test_invalid_tool_args_counts_only_unparseable_strings():
    calls = [
        {"id": "a", "type": "function", "function": {"name": "f", "arguments": '{"q": 1}'}},
        {"id": "b", "type": "function", "function": {"name": "f", "arguments": '{"q":1}{"q":2}'}},
        {"id": "c", "type": "function", "function": {"name": "f", "arguments": ""}},
        {"id": "d", "type": "function", "function": {"name": "f", "arguments": None}},
    ]
    assert tl._invalid_tool_args(calls) == 1
    assert tl._invalid_tool_args(None) == 0
    assert tl._invalid_tool_args([]) == 0


def test_stop_details_only_on_refusal_and_from_raw_body():
    raw = '{"id":"msg_1","stop_reason":"refusal","stop_details":{"type":"refusal","category":"reasoning_extraction","explanation":"blocked"},"content":[]}'
    assert tl._stop_details({"original_response": raw}, "content_filter") == {
        "type": "refusal", "category": "reasoning_extraction", "explanation": "blocked"}
    assert tl._stop_details({"original_response": raw}, "tool_use") is None
    assert tl._stop_details({"original_response": "not json {"}, "content_filter") is None
    assert tl._stop_details({}, "content_filter") is None


def test_stop_reason_is_tool_use_whenever_tool_calls_are_present(tmp_path, monkeypatch):
    """A streamed gemini turn arrives with finish_reason 'stop' although it holds tool calls."""
    import types
    monkeypatch.setenv("LLM_MONITOR_LOG", str(tmp_path / "calls.jsonl"))
    fn = types.SimpleNamespace(name="get_weather", arguments='{"city": "Lima"}')
    tc = types.SimpleNamespace(id="call_1", function=fn)
    msg = types.SimpleNamespace(content=None, tool_calls=[tc], reasoning_content=None)
    choice = types.SimpleNamespace(finish_reason="stop", message=msg)
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    resp = types.SimpleNamespace(choices=[choice], usage=usage)
    kwargs = {"messages": [{"role": "user", "content": "hi"}], "model": "gemini-3.1-pro-preview",
              "litellm_params": {"proxy_server_request": {"headers": {"x-trace-meta": '{"benchmark": "t", "unit_id": "u"}'}}}}
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    tl.TraceLogger()._write_row(kwargs, resp, now, now)
    rows = [json.loads(l) for f in tmp_path.rglob("calls.jsonl") for l in open(f)]
    assert len(rows) == 1 and rows[0]["stop_reason"] == "tool_use" and rows[0]["tool_called"] == "get_weather"


def test_stop_details_records_gemini_malformed_function_call():
    raw = '{"candidates":[{"content":{"parts":[{"thought":true,"text":"..."}],"role":"model"},"finishReason":"MALFORMED_FUNCTION_CALL"}],"usageMetadata":{"thoughtsTokenCount":238}}'
    assert tl._stop_details({"original_response": raw}, "end_turn") == {"type": "gemini_finish_reason", "finish_reason": "MALFORMED_FUNCTION_CALL"}
    ok = raw.replace("MALFORMED_FUNCTION_CALL", "STOP")
    assert tl._stop_details({"original_response": ok}, "end_turn") is None
    assert tl._stop_details({"original_response": raw}, "tool_use") is None


def test_litellm_patch_merges_text_and_tool_calls_into_one_choice():
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler as H,
    )
    from llm_cost_tracker.proxy import litellm_patches
    litellm_patches.apply()
    items = [
        ResponseOutputMessage(id="m1", role="assistant", status="completed", type="message",
                              content=[ResponseOutputText(type="output_text", text="I will check Oslo.", annotations=[])]),
        ResponseFunctionToolCall(id="fc1", call_id="call_1", name="get_weather", arguments='{"city":"Oslo"}', type="function_call", status="completed"),
        ResponseFunctionToolCall(id="fc2", call_id="call_2", name="get_weather", arguments='{"city":"Lima"}', type="function_call", status="completed"),
    ]
    choices = H._convert_response_output_to_choices(items)
    assert len(choices) == 1
    assert choices[0].finish_reason == "tool_calls" and choices[0].index == 0
    assert choices[0].message.content == "I will check Oslo."
    assert [tc["function"]["name"] if isinstance(tc, dict) else tc.function.name for tc in choices[0].message.tool_calls] == ["get_weather", "get_weather"]
    # tool-only and text-only replies are untouched
    assert len(H._convert_response_output_to_choices(items[1:])) == 1
    assert H._convert_response_output_to_choices(items[:1])[0].finish_reason == "stop"


def _fake_kwargs(tmp_path, monkeypatch, **extra):
    import datetime as dt
    monkeypatch.setenv("LLM_MONITOR_LOG", str(tmp_path / "calls.jsonl"))
    kwargs = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-6-astra",
              "optional_params": {"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
                                            {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]},
              "litellm_params": {"proxy_server_request": {"headers": {"x-trace-meta": '{"benchmark": "t", "unit_id": "u", "reset": "true", "_scope_id": "s1"}'}}}}
    kwargs.update(extra)
    now = dt.datetime.now(dt.timezone.utc)
    return kwargs, now


def _rows(tmp_path, name):
    return [json.loads(l) for f in tmp_path.rglob(name) for l in open(f)]


def test_failed_call_is_written_to_errors_jsonl_not_calls(tmp_path, monkeypatch):
    kwargs, now = _fake_kwargs(tmp_path, monkeypatch)
    class BadRequest(Exception):
        status_code = 400
        llm_provider = "anthropic"
    exc = BadRequest('tool_choice: type "any" is not supported for this model.')
    tl.TraceLogger().log_failure_event(kwargs, exc, now, now)
    errs = _rows(tmp_path, "errors.jsonl")
    assert len(errs) == 1 and errs[0]["kind"] == "error" and errs[0]["status_code"] == 400
    assert errs[0]["error_type"] == "BadRequest" and "tool_choice" in errs[0]["message"]
    assert errs[0]["trace_meta"]["unit_id"] == "u" and errs[0]["tools"]["names"] == ["get_weather", "bash"]
    assert _rows(tmp_path, "calls.jsonl") == []


def test_success_row_has_response_id_tools_hash_and_raw_sidecar(tmp_path, monkeypatch):
    import types
    raw = '{"id": "resp_abc", "output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}], "status": "completed", "usage": {"input_tokens": 3, "output_tokens": 2}}'
    kwargs, now = _fake_kwargs(tmp_path, monkeypatch, original_response=raw)
    msg = types.SimpleNamespace(content="hello", tool_calls=None, reasoning_content=None)
    resp = types.SimpleNamespace(id="chatcmpl-synthetic", choices=[types.SimpleNamespace(finish_reason="stop", message=msg)],
                                 usage={"prompt_tokens": 3, "completion_tokens": 2})
    tl.TraceLogger()._write_row(kwargs, resp, now, now)
    rows = _rows(tmp_path, "calls.jsonl"); assert len(rows) == 1
    row = rows[0]
    assert row["response_id"] == "resp_abc"  # the provider's id from the raw body, not litellm's synthetic one
    assert row["tools"]["names"] == ["get_weather", "bash"] and row["tools"]["count"] == 2 and len(row["tools"]["hash"]) == 16
    tools_files = list(tmp_path.rglob("tools_*.json")); assert len(tools_files) == 1
    assert json.loads(tools_files[0].read_text())[0]["function"]["name"] == "get_weather"
    raws = _rows(tmp_path, "raw.jsonl"); assert len(raws) == 1
    assert raws[0]["trace_id"] == row["trace_id"] and raws[0]["round"] == row["round"] and raws[0]["raw"]["id"] == "resp_abc"


def test_streamed_call_writes_no_raw_sidecar(tmp_path, monkeypatch):
    import types
    kwargs, now = _fake_kwargs(tmp_path, monkeypatch)  # no original_response string (streams hand an iterator)
    kwargs["original_response"] = object()
    msg = types.SimpleNamespace(content="hi", tool_calls=None, reasoning_content=None)
    resp = types.SimpleNamespace(id="chatcmpl-1", choices=[types.SimpleNamespace(finish_reason="stop", message=msg)], usage={"prompt_tokens": 1, "completion_tokens": 1})
    tl.TraceLogger()._write_row(kwargs, resp, now, now)
    rows = _rows(tmp_path, "calls.jsonl")
    assert len(rows) == 1 and list(tmp_path.rglob("raw.jsonl")) == []
    assert rows[0]["response_id"] == "chatcmpl-1"  # streamed: only litellm's id exists


def test_reset_archives_raw_and_errors_with_calls(tmp_path, monkeypatch):
    kwargs, now = _fake_kwargs(tmp_path, monkeypatch, original_response='{"id": "r1"}')
    import types
    msg = types.SimpleNamespace(content="a", tool_calls=None, reasoning_content=None)
    resp = types.SimpleNamespace(id="r1", choices=[types.SimpleNamespace(finish_reason="stop", message=msg)], usage={"prompt_tokens": 1, "completion_tokens": 1})
    logger = tl.TraceLogger()
    logger._write_row(kwargs, resp, now, now)
    logger.log_failure_event(kwargs, RuntimeError("boom"), now, now)
    # a new scope for the same trace -> the previous attempt's three files move aside together
    kwargs["litellm_params"]["proxy_server_request"]["headers"]["x-trace-meta"] = '{"benchmark": "t", "unit_id": "u", "reset": "true", "_scope_id": "s2"}'
    logger._write_row(kwargs, resp, now, now)
    d = next(tmp_path.rglob("calls.jsonl")).parent
    names = sorted(p.name for p in d.iterdir())
    assert "calls.jsonl.attempt1" in names and "raw.jsonl.attempt1" in names and "errors.jsonl.attempt1" in names
    assert len(_rows(tmp_path, "calls.jsonl")) == 1 and "errors.jsonl" not in names


def test_requested_model_alias_and_prices_and_error_dedupe(tmp_path, monkeypatch):
    import types
    kwargs, now = _fake_kwargs(tmp_path, monkeypatch)
    kwargs["model"] = "claude-fable-5-1"
    kwargs["litellm_params"]["metadata"] = {"model_group": "claude-fable-5-1-max"}
    msg = types.SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None)
    resp = types.SimpleNamespace(id="msg_1", choices=[types.SimpleNamespace(finish_reason="stop", message=msg)], usage={"prompt_tokens": 1, "completion_tokens": 1})
    logger = tl.TraceLogger()
    logger._write_row(kwargs, resp, now, now)
    row = _rows(tmp_path, "calls.jsonl")[0]
    assert row["model"] == "claude-fable-5-1" and row["requested_model"] == "claude-fable-5-1-max"
    assert "prices" in row  # None when pricing is unavailable offline, a dict with per-million rates otherwise
    import asyncio
    # the same failed request reaches both hooks -> one row
    kwargs["litellm_call_id"] = "call-A"
    exc = RuntimeError("same failure delivered twice")
    logger.log_failure_event(kwargs, exc, now, now)
    asyncio.run(logger.async_log_failure_event(kwargs, exc, now, now))
    assert len(_rows(tmp_path, "errors.jsonl")) == 1
    # five different requests failing back to back -> five rows, even if the exception objects
    # are freed and their addresses reused between calls
    for i in range(5):
        k = dict(kwargs); k["litellm_call_id"] = f"call-{i}"
        logger.log_failure_event(k, RuntimeError(f"burst {i}"), now, now)
    errs = _rows(tmp_path, "errors.jsonl")
    assert len(errs) == 6 and errs[-1]["requested_model"] == "claude-fable-5-1-max"
