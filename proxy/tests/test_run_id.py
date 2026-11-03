"""Tests for tracer.run_id_from_dir."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_cost_tracker.tracer.run_id import run_id_from_dir


def test_run_id_prefers_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAU2_RUN_ID", "custom-exp")
    assert run_id_from_dir(tmp_path / "native_name", env_var="TAU2_RUN_ID") == "custom-exp"


def test_run_id_falls_back_to_dir_basename(tmp_path: Path) -> None:
    exp_dir = tmp_path / "my_experiment_2026"
    exp_dir.mkdir()
    assert run_id_from_dir(exp_dir, env_var="TAU2_RUN_ID") == "my_experiment_2026"


def test_run_id_default_when_missing() -> None:
    assert run_id_from_dir(None, env_var="TAU2_RUN_ID", default="fallback") == "fallback"


def test_run_id_ignores_empty_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINI_SWE_RUN_ID", "   ")
    exp_dir = tmp_path / "batch_a"
    exp_dir.mkdir()
    assert run_id_from_dir(exp_dir, env_var="MINI_SWE_RUN_ID") == "batch_a"
