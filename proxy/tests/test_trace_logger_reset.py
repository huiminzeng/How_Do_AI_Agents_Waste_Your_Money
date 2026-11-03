"""Tests for trace log reset semantics."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from llm_cost_tracker.proxy import trace_logger as tl
from llm_cost_tracker.tracer.context import trace


def test_is_reset_requested_accepts_bool_and_int():
    assert tl._is_reset_requested({"reset": True})
    assert tl._is_reset_requested({"reset": 1})
    assert not tl._is_reset_requested({"reset": False})
    assert not tl._is_reset_requested({"reset": 0})
    assert tl._is_reset_requested({"replace": "yes"})


def test_scope_id_falls_back_to_simulation_id():
    meta = {"simulation_id": "sim-abc", "benchmark": "tau2"}
    assert tl._scope_id_from_meta(meta) == "sim-abc"


def test_normalize_reset_adds_scope_id():
    with trace(benchmark="mini_swe", run_id="batch1", unit_id="inst-1", reset=True) as scope:
        ctx = scope._merged()
        assert ctx["reset"] == "true"
        assert "_scope_id" in ctx
        assert len(ctx["_scope_id"]) >= 8


def test_reset_not_in_stable_hash():
    meta = {
        "benchmark": "mini_swe",
        "run_id": "batch1",
        "unit_id": "inst-1",
        "reset": "true",
        "_scope_id": "abc123",
        "simulation_id": "volatile-uuid",
    }
    stable = tl._stable_trace_meta(meta, model="gpt-5.2-chat")
    assert "reset" not in stable
    assert "_scope_id" not in stable
    assert "simulation_id" not in stable
    assert stable["model"] == "gpt-5.2-chat"


def test_resolve_model_prefers_request():
    meta = {"model": "from-context"}
    assert tl._resolve_model(meta, "from-request") == "from-request"
    assert tl._resolve_model(meta, "") == "from-context"


def test_maybe_reset_truncates_once_per_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MONITOR_LOG", str(tmp_path / "calls.jsonl"))
    logger = tl.TraceLogger()
    meta = {
        "benchmark": "mini_swe",
        "run_id": "batch1",
        "unit_id": "inst-1",
        "reset": "true",
        "_scope_id": "scope-a",
    }
    trace_id = tl._trace_id_from_meta(meta, model="gpt-5.2-chat")
    paths = tl._log_paths(meta, model="gpt-5.2-chat", trace_id=trace_id)
    log_path = Path(paths[0])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text('{"kind":"call","round":1}\n{"kind":"call","round":2}\n')

    logger._maybe_reset_log(trace_id, meta, paths)
    assert not log_path.exists()

    log_path.write_text('{"kind":"call","round":1}\n')
    logger._maybe_reset_log(trace_id, meta, paths)
    assert log_path.read_text().count("round") == 1


def test_run_id_separates_experiments():
    meta_a = {"benchmark": "tau2", "run_id": "exp_a", "task_id": "1", "seed": "42"}
    meta_b = {"benchmark": "tau2", "run_id": "exp_b", "task_id": "1", "seed": "42"}
    id_a = tl._trace_id_from_meta(meta_a, model="gpt-5-chat")
    id_b = tl._trace_id_from_meta(meta_b, model="gpt-5-chat")
    assert id_a != id_b


def test_maybe_reset_new_scope_truncates_again(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MONITOR_LOG", str(tmp_path / "calls.jsonl"))
    logger = tl.TraceLogger()
    base = {
        "benchmark": "mini_swe",
        "run_id": "batch1",
        "unit_id": "inst-1",
        "reset": "true",
    }
    meta_a = {**base, "_scope_id": "scope-a"}
    meta_b = {**base, "_scope_id": "scope-b"}
    trace_id = tl._trace_id_from_meta(meta_a, model="gpt-5.2-chat")
    paths = tl._log_paths(meta_a, model="gpt-5.2-chat", trace_id=trace_id)
    log_path = Path(paths[0])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("stale\n")

    logger._maybe_reset_log(trace_id, meta_a, paths)
    log_path.write_text("attempt1\n")

    logger._maybe_reset_log(trace_id, meta_b, paths)
    assert not log_path.exists()


def test_reset_survives_new_logger_instance(tmp_path, monkeypatch):
    """Sidecar file makes reset-once durable across proxy restarts."""
    monkeypatch.setenv("LLM_MONITOR_LOG", str(tmp_path / "calls.jsonl"))
    meta = {
        "benchmark": "mini_swe",
        "run_id": "batch1",
        "unit_id": "inst-1",
        "reset": "true",
        "_scope_id": "scope-a",
    }
    trace_id = tl._trace_id_from_meta(meta, model="gpt-5.2-chat")
    paths = tl._log_paths(meta, model="gpt-5.2-chat", trace_id=trace_id)
    log_path = Path(paths[0])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    tl.TraceLogger()._maybe_reset_log(trace_id, meta, paths)
    log_path.write_text("call1\n")

    # Simulate proxy restart: new logger, same scope — must not truncate.
    tl.TraceLogger()._maybe_reset_log(trace_id, meta, paths)
    assert log_path.read_text() == "call1\n"
