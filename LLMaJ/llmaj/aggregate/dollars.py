"""What the waste cost, in dollars. Output side only, and every figure counted.

PRICES COME FROM A TABLE THAT IS CHECKED AGAINST THE BILL. `prices.py` holds the
rates and `tokens.py` reprices every cell with them, then compares the total
against what the proxy actually recorded; the disagreement is below 1e-5 dollars
on every cell. This module divides the repriced dollars by the token counts to
get the per-token rate, which stays correct if a cell was ever priced
differently.

EVERY TOKEN FIGURE HERE WAS COUNTED, NOT ESTIMATED. `exact_tokens` runs the
model's own tokenizer over each stream three ways - full, with all marks
stripped, with one kind's marks stripped - and this module multiplies those
counts by the output rate: no marked-
share applied to a billed count, and no input-side
estimate (the old one converted repeated tool responses to
tokens by a JSON ratio, which is exactly the conversion this project forbids).
A cell without a `waste_dollars_by_kind` in its exact_tokens.json is an error
here, not a fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths

#: Input categories, in the order they appear in a composition file.
INPUT_KEYS = ("input", "cache_read", "cache_write")
OUTPUT_KEYS = ("visible", "reasoning")


@dataclass
class Prices:
    """Effective per-token prices for one cell, divided out of what it paid."""

    cell_key: str
    fresh_input: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    blended_input: float = 0.0
    output: float = 0.0
    def to_json(self) -> dict:
        return {"cell_key": self.cell_key,
                "per_million": {
                    "fresh_input": round(self.fresh_input * 1e6, 4),
                    "cache_read": round(self.cache_read * 1e6, 4),
                    "cache_write": round(self.cache_write * 1e6, 4),
                    "blended_input": round(self.blended_input * 1e6, 4),
                    "output": round(self.output * 1e6, 4)}}


def _div(num: float, den: float) -> float:
    return num / den if den else 0.0


def prices_for(agent_model: str, benchmark: str, domain: str) -> Prices | None:
    """Read one cell's composition file and divide out its prices."""
    path = (paths.AGENT_SPEND / agent_model
            / paths.cell_seg(benchmark, domain) / "composition.json")
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    c, dol = d["counts"], d["dollars"]
    in_tok = sum(c.get(k, 0) for k in INPUT_KEYS)
    in_dol = sum(dol.get(k, 0.0) for k in INPUT_KEYS)
    out_tok = sum(c.get(k, 0) for k in OUTPUT_KEYS)
    out_dol = sum(dol.get(k, 0.0) for k in OUTPUT_KEYS)
    return Prices(
        cell_key=d.get("cell_key", ""),
        fresh_input=_div(dol.get("input", 0.0), c.get("input", 0)),
        cache_read=_div(dol.get("cache_read", 0.0), c.get("cache_read", 0)),
        cache_write=_div(dol.get("cache_write", 0.0), c.get("cache_write", 0)),
        blended_input=_div(in_dol, in_tok),
        output=_div(out_dol, out_tok),
    )


@dataclass
class Waste:
    """One cell's waste in dollars, input and output kept separate."""

    cell_key: str = ""
    prices: Prices | None = None
    #: bucket -> dollars. Exact tokens, one price, so a number not a range.
    output: dict[str, float] = field(default_factory=dict)
    #: bucket -> kind -> dollars, output side. Same palette as the input split.
    output_by_kind: dict = field(default_factory=dict)
    #: The token figures the dollars came from, so a reader can re-price them.
    output_tokens: dict[str, float] = field(default_factory=dict)
    #: Reasoning tokens billed with no text to audit, and what they cost. Not
    #: waste - we never looked at them - but the size of what the audit missed.
    unaudited_tokens: float = 0.0
    unaudited_dollars: float = 0.0
    #: What the audited trajectories actually cost, from the proxy's own
    #: `cost_usd`. Without it the waste figures are absolute dollars with no
    #: sense of scale - $0.21 means nothing until you know whether the run cost
    #: $2 or $200.
    spent_output: float = 0.0
    spent_input: float = 0.0
    spent_total: float = 0.0
    n_calls: int = 0
    #: Tokens behind `spent_output` / `spent_input`. `spent_input_tokens` counts
    #: fresh + cache_read + cache_write, because all three are input the model
    #: had to be given - the price differs 10x between them but the count does
    #: not, and the count is what a token-side share divides by.
    spent_output_tokens: int = 0
    spent_input_tokens: int = 0
    #: Buckets billed but refused a dollar figure, because the text we have is
    #: not the text that was billed (gemini's reasoning summary).
    unpriced: dict = field(default_factory=dict)

    @property
    def output_total(self) -> float:
        return sum(self.output.values())

    @property
    def share_of_output(self) -> float:
        """Wasted output dollars over output dollars spent.

        THE headline ratio. The judge audits what the agent PRODUCED, and this
        is that waste against that spend - numerator and denominator covering
        the same thing.
        """
        return (self.output_total / self.spent_output) if self.spent_output else 0.0

    def to_json(self) -> dict:
        return {
            "cell_key": self.cell_key,
            "prices": self.prices.to_json() if self.prices else None,
            "output_waste_dollars": {k: round(v, 6)
                                     for k, v in self.output.items()},
            "output_waste_dollars_total": round(self.output_total, 6),
            "output_waste_tokens": {k: round(v, 1)
                                    for k, v in self.output_tokens.items()},
            "output_waste_by_kind": {b: {k: round(v, 6) for k, v in d.items()}
                                     for b, d in self.output_by_kind.items()},
            "unaudited_reasoning_tokens": round(self.unaudited_tokens, 1),
            "unaudited_reasoning_dollars": round(self.unaudited_dollars, 6),
            "unpriced_buckets": {k: round(v, 2)
                                 for k, v in self.unpriced.items()},
            "spent": {"output": round(self.spent_output, 6),
                      "input": round(self.spent_input, 6),
                      "total": round(self.spent_total, 6),
                      "n_calls": self.n_calls,
                      "output_tokens": self.spent_output_tokens,
                      "input_tokens": self.spent_input_tokens},
            "waste_share_of_output_dollars": self.share_of_output,
        }


def spent_on(agent_model: str, benchmark: str, domain: str,
             trace_ids: set) -> tuple[float, float, float, int, int, int]:
    """What those exact trajectories cost, from the proxy's own `cost_usd`.

    Read per trace rather than taken from the cell's `composition.json`: the
    audit covers 36 of 200 trajectories, and the cell total would be a
    denominator five times too large.

    `cost_usd` is the proxy's own figure, priced by `prices.py` and checked
    against what was recorded - see `reprice_error`, under 1e-5 dollars on every
    cell. Output and input are split by repricing the token counts, because the
    record carries one total.
    """
    from ..preprocess.sources import cells_for, _read_calls
    p = prices_for(agent_model, benchmark, domain)
    if p is None:
        return 0.0, 0.0, 0.0, 0
    out_tok = fresh = cached = written = 0
    total = 0.0
    calls = 0
    for cell in cells_for(agent_model):
        if cell.benchmark != benchmark or cell.domain != domain:
            continue
        files = []
        if cell.layout == "monitor":
            for root in paths.MONITOR_ROOTS:
                if root.exists():
                    files += sorted(root.glob(cell.pattern))
        else:
            f = paths.RESULTS / cell.pattern
            if f.exists():
                files = [f]
        for f in files:
            for rec in _read_calls(f):
                if rec.get("trace_id") not in trace_ids:
                    continue
                calls += 1
                total += rec.get("cost_usd") or 0.0
                out_tok += rec.get("output_tokens") or 0
                fresh += rec.get("input_tokens") or 0
                cached += rec.get("cache_read_tokens") or 0
                written += rec.get("cache_write_tokens") or 0
    out_cost = out_tok * p.output
    in_cost = (fresh * p.fresh_input + cached * p.cache_read
               + written * p.cache_write)
    # The token totals accumulated to price the cell are returned too: they are
    # the DENOMINATORS a share needs: an input-side waste
    # share has nothing to divide by without `fresh + cached + written`, and the
    # figure that compares input against reasoning against visible+action cannot
    # be drawn from dollars alone.
    return out_cost, in_cost, total, calls, out_tok, fresh + cached + written


def build(agent_model: str, benchmark: str, domain: str,
          trace_ids: set | None = None, *, exact: dict) -> Waste:
    """Price one cell's waste from its exact_tokens.json. Nothing else feeds it.

    `exact` is REQUIRED and must carry `waste_dollars_by_kind` on every priced
    bucket. There is no fallback of any kind: when one existed
    it produced plausible numbers that were wrong by 4-5 points per kind, and a
    wrong number that looks right is worse than an error.
    """
    p = prices_for(agent_model, benchmark, domain)
    w = Waste(cell_key=(p.cell_key if p else
                        f"{agent_model}/{benchmark}/{domain}"), prices=p)
    if p is None:
        return w

    if not (exact.get("buckets") if exact else None):
        raise ValueError(
            f"{agent_model}/{benchmark}/{domain}: exact_tokens.json is missing "
            f"or has no buckets. Run the token-count stage first.")
    for name, b in (exact.get("buckets") or {}).items():
        if not b.get("billed_tokens"):
            continue
        if not b.get("priceable"):
            # Refused for what the text IS, not for a statistic: a
            # provider's summary of a chain of thought is not the object
            # that was billed. Recorded as a refusal so the page can say so.
            w.unpriced[name] = 0.0
            continue
        w.output_tokens[name] = b["waste_tokens"]
        w.output[name] = b["waste_dollars"]
        # Per kind is COUNTED with the tokenizer, one kind stripped at a time
        # (exact_tokens.Bucket.waste_by_kind).
        # Absent means the count stage never ran; EMPTY means the bucket had no
        # waste (one trajectory whose reply was all substance), which is a result.
        if "waste_dollars_by_kind" not in b:
            raise ValueError(
                f"{agent_model}/{benchmark}/{domain}: exact_tokens.json has no "
                f"waste_dollars_by_kind for bucket {name!r}. Re-run the "
                f"token-count stage (scripts_exact.py).")
        w.output_by_kind[name] = dict(b["waste_dollars_by_kind"])
    # Reasoning that was billed with no text to read. Held out of both
    # sides on the output tables, and named rather than silently dropped.
    blind = (exact.get("buckets") or {}).get("reasoning") or {}
    if blind.get("billed_tokens") and not blind.get("text_tokens"):
        w.unaudited_tokens = blind["billed_tokens"]
        w.unaudited_dollars = blind["billed_tokens"] * p.output
    if trace_ids:
        (w.spent_output, w.spent_input, w.spent_total, w.n_calls,
         w.spent_output_tokens,
         w.spent_input_tokens) = spent_on(agent_model, benchmark, domain,
                                          trace_ids)
    return w


def table(w: Waste) -> str:
    if w.prices is None:
        return f"  {w.cell_key}: no composition.json, cannot price"
    p = w.prices
    lines = [
        f"  {w.cell_key}   output ${p.output * 1e6:.2f}/M  "
        f"input ${p.fresh_input * 1e6:.2f}/M fresh, "
        f"${p.cache_read * 1e6:.2f}/M cached",
        "",
        f"  {'side':<24}{'tokens':>12}{'dollars':>13}",
        "  " + "-" * 49,
    ]
    for name, dollars in w.output.items():
        lines.append(f"  output · {name:<15}{w.output_tokens[name]:>12,.0f}"
                     f"{dollars:>13,.4f}")
    for name in (w.unpriced or {}):
        lines.append(
            f"  {name:<24}{'not priced':>12}{'':>13}   the text is a summary, "
            f"not the billed thing")
    if w.unaudited_tokens:
        lines.append(
            f"  not audited (no text) {w.unaudited_tokens:>14,.0f}"
            f"{w.unaudited_dollars:>13,.4f}   <- neither wasted nor kept")
    if w.spent_total:
        lines += [
            "",
            f"  these {w.n_calls:,} calls cost ${w.spent_total:.4f}"
            f"  (output ${w.spent_output:.4f}, input ${w.spent_input:.4f})",
            f"    wasted, as a share of OUTPUT spend   "
            f"{w.share_of_output * 100:>5.1f}%   <- what the judge audits",
        ]
    return "\n".join(lines)


def save(w: Waste, judge_id: str, run_id: str, agent_model: str,
         benchmark: str, domain: str) -> Path:
    """Alongside the cell's other results, so a reader finds all three units.

    Tokens and dollars. Filed under the
    judge and run because the output half depends on which judge marked what;
    the input half does not, and carries the same value under every judge.
    """
    path = paths.ensure_parent(
        paths.cell_dir(paths.EFFICIENCY_TASK, judge_id, run_id, agent_model,
                       benchmark, domain) / "dollars.json")
    path.write_text(json.dumps(w.to_json(), indent=2) + "\n")
    return path
