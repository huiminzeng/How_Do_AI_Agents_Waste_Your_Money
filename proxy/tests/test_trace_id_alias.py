"""Two effort aliases of one model, same task, same run: two trace ids.

Regression: kimi-k3-low and kimi-k3-high resolve to the same
deployment (`kimi-k3`); hashing the resolved name made their traces for one
mcp_atlas task share a directory, and the later run's reset truncated the earlier.
"""
from llm_cost_tracker.proxy import trace_logger as tl

META = {"benchmark": "mcp_atlas", "run_id": "mcp_untitled", "unit_id": "689bd255", "model": "kimi-k3-high",
        "_scope_id": "uuid-1"}


def _kwargs(alias: str) -> dict:
    return {"model": "kimi-k3", "litellm_params": {"metadata": {"model_group": alias}}}


def test_aliases_get_distinct_ids():
    low = tl._trace_id_from_meta(META, model=tl._identity_model(META, _kwargs("kimi-k3-low")))
    high = tl._trace_id_from_meta(META, model=tl._identity_model(META, _kwargs("kimi-k3-high")))
    assert low != high


def test_same_alias_same_id_across_attempts():
    a = dict(META, _scope_id="uuid-1")
    b = dict(META, _scope_id="uuid-2")
    k = _kwargs("kimi-k3-low")
    assert tl._trace_id_from_meta(a, model=tl._identity_model(a, k)) == tl._trace_id_from_meta(b, model=tl._identity_model(b, k))


def test_without_router_alias_falls_back_to_resolved_model():
    k = {"model": "kimi-k3"}
    assert tl._identity_model(META, k) == "kimi-k3"
