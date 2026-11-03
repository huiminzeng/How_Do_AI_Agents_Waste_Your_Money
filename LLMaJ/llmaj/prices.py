"""Per-model token prices, in USD per million tokens.

Copied from `proxy/llm_cost_tracker/pricing.py` rather than imported, so this package stays runnable on its own. Hardcoding prices is only safe
because it is checked: `tokens.py` reprices every call from these numbers and compares the
total against the `cost_usd` the proxy recorded at the time. A drift shows up as a
mismatch, not as a wrong pie chart.

Note kimi-k3's input tier is 10x the others' and its cache-read tier is 1.5x, which is why
its cost profile looks unlike anything else even at similar token volumes.
"""

from __future__ import annotations

PER_MILLION: dict[str, dict[str, float]] = {
    "kimi-k3": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.0},
    "gemini-3.1-pro-preview": {
        "input": 2.0, "output": 12.0, "cache_read": 0.2, "cache_write": 2.0,
    },
    "gpt-5.6-terra": {
        "input": 2.0, "output": 12.0, "cache_read": 0.2, "cache_write": 2.5,
    },
    "claude-sonnet-5": {
        "input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.5,
    },
    # Same source as the others: LiteLLM's live price JSON as fetched by the proxy's
    # pricing.py (fable cache reads are $0.25/M, a
    # quarter of the usual 10% rule; astra lists a cache_write price that OpenAI never
    # bills because it reports no cache-write tokens - kept so the two tables agree).
    "claude-fable-5-1": {
        "input": 10.0, "output": 50.0, "cache_read": 0.25, "cache_write": 12.5,
    },
    "gpt-6-astra": {
        "input": 10.0, "output": 50.0, "cache_read": 1.0, "cache_write": 12.5,
    },
    # Verified against the proxy: opus-5's 500-instance SWE run logged $179.42, and
    # these four rates over its recorded token counts reproduce it exactly. A model missing here is not an error anywhere - `rates()`
    # returns None and `tokens.py` reports zero dollars - so the reprice check
    # printed at judge start is what catches it.
    "claude-opus-5": {
        "input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25,
    },
    # sentinel-qwen3-4b-thinking-rl: a Qwen3-4B-Thinking fine-tune served
    # self-hosted; rates are the ones the proxy recorded on every row of its tau2 run.
    "sentinel-qwen3-4b-thinking-rl": {"input": 0.08, "output": 0.24, "cache_read": 0.08, "cache_write": 0.08},
    # Qwen3-30B-A3B-Thinking-2507-tau2-consecutive-tool-penalty: the 30B
    # MoE fine-tune, self-served; rates are the ones the proxy recorded on all 5,251
    # rows of its tau2 run.
    "qwen3-30b-a3b-thinking-2507-tau2-consecutive-tool-penalty":
        {"input": 0.12, "output": 0.5, "cache_read": 0.12, "cache_write": 0.12},
    # Reasoning-effort ABLATION arms of kimi-k3: same model, same
    # prices, a different effort. Listed as models of their own so every stage
    # keeps them apart from the max-effort baseline; `EFFORT_ARMS` says what
    # they are arms of.
    "kimi-k3-low": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.0},
    "kimi-k3-high": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.0},
    # The no-thinking arm: a self-hosted deployment, because the gateway rejects
    # `none`; billed the same, so the rates are the baseline's.
    "kimi-k3-none": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.0},
}

#: arm -> (baseline model, effort).
EFFORT_ARMS: dict[str, tuple[str, str]] = {
    "kimi-k3-low": ("kimi-k3", "low"),
    "kimi-k3-high": ("kimi-k3", "high"),
    "kimi-k3-none": ("kimi-k3", "none"),
}


def is_arm(model: str) -> bool:
    return model in EFFORT_ARMS


#: Aliases seen in the `model` field of trace records.
ALIASES: dict[str, str] = {
    "k3-gw": "kimi-k3",
    "openai/kimi-k3": "kimi-k3",
    "gemini-3.1-pro-low": "gemini-3.1-pro-preview",
    "gemini-3.1-pro-high": "gemini-3.1-pro-preview",
    "gpt-5.6-terra-low": "gpt-5.6-terra",
    "claude-sonnet-5-low": "claude-sonnet-5",
    "claude-sonnet-5-medium": "claude-sonnet-5",
    "claude-opus-5-low": "claude-opus-5",
    "claude-opus-5-medium": "claude-opus-5",
    "claude-fable-5-1-low": "claude-fable-5-1",
    "claude-fable-5-1-medium": "claude-fable-5-1",
    "claude-fable-5-1-xhigh": "claude-fable-5-1",
    "claude-fable-5-1-max": "claude-fable-5-1",
    "gpt-6-astra-low": "gpt-6-astra",
    "gpt-6-astra-medium": "gpt-6-astra",
    "gpt-6-astra-xhigh": "gpt-6-astra",
    "openai/kimi-k3-low": "kimi-k3-low",
    "openai/kimi-k3-high": "kimi-k3-high",
    # The proxy files this model's traces under the lower-cased name but stamps the
    # served (mixed-case) name on every row; tau2's results directory carries the
    # served name too.
    "Qwen3-30B-A3B-Thinking-2507-tau2-consecutive-tool-penalty":
        "qwen3-30b-a3b-thinking-2507-tau2-consecutive-tool-penalty",
}


def canonical(model: str) -> str:
    """The priced model behind a name, whatever the name carries around it.

    Three things get stripped, and they are different kinds of decoration:
      `openai/` and friends   the provider prefix the harness asked with
      `__<harness>`           the CELL's name in the harness ablation. A cell
                              called `kimi-k3__opencode` is kimi-k3 run under
                              opencode: same weights, same bill, same price
                              list. Leaving it on made `rates()` return None,
                              every dollar figure for those cells would come
                              out 0.00 and the composition file would record a
                              -100% reprice error without anything failing.
      an alias                effort levels and the like, via ALIASES

    The suffix is stripped HERE and not at each call site because this is the
    one function every price lookup goes through; `exact_tokens` had the same
    bug separately and needed its own fix.
    """
    name = (model or "").split("/")[-1].split("__")[0]
    return ALIASES.get(name, ALIASES.get(model, name))


def rates(model: str) -> dict[str, float] | None:
    return PER_MILLION.get(canonical(model))


def names(model: str) -> list[str]:
    """Every spelling a run may have used for `model`: the canonical id first, then
    each alias that resolves to it (effort-level aliases such as claude-fable-5-1-max).

    Harness outputs are named after the alias the run was launched with (tau2's
    simulation directory, claw-eval's trace directory, mcp_atlas's scored CSV), while
    the proxy files traces under the resolved model. Anything that looks a run up by
    model name has to try all of these.
    """
    return [model] + sorted(a for a, c in ALIASES.items() if c == model and a != model)

