"""Waste in tokens, counted rather than converted.

Every count here is produced by the model's own tokenizer over the stored
text: the full stream, the stream with all marks stripped, and the stream with
one kind's marks stripped at a time.

This module removes all three by asking the question the bill answers:

    waste_tokens = tokens(the stream) - tokens(the stream with the marked
                                               spans deleted)

Nothing is estimated. Token boundaries merging when text is removed, JSON
tokenising differently from prose, code tokenising differently again - all of
it is handled because the tokeniser sees the real before and after.

WHICH COUNTER, AND WHY IT CAN BE TRUSTED

Text streams (reasoning, visible) are counted as the stored text. Tool calls are
counted two ways, by model:

  gemini, claude-sonnet-5   NATIVE. The calls are read back as objects from
      calls.jsonl (preprocess/structured_calls.py) and handed to the provider's
      count_tokens API in its own form - a function_call part, a tool_use block -
      after proving the objects re-render to the stored string byte for byte.
      Against the bill on real action-only turns: gemini exact to a
      constant 2k-1 on 40/40 turns; sonnet within +-1 on 66/66 turns after its
      6 + 10(k-1) placeholder constant. The rendered `name({args})` text was
      off by -129..+51 tokens per turn - low on short calls, high on long ones -
      and is no longer counted for these models. Marked ranges are cut out of
      the OBJECTS: a call covered end to end is dropped whole, a range inside
      an argument is removed from the decoded value by position.
  kimi-k3, gpt-5.6-terra    Rendered text with a local tokeniser (HF file;
      tiktoken o200k_base). Validated: 61-64% of turns exact,
      residual 1.4-1.5% of the output bill, uncorrelated with content type.

Prose and code count exact to a per-turn constant for every model (sonnet -2,
gemini 0). What remains in the envelope for the native models is text cut at
the 8,000-character storage cap: on sonnet/claw_eval, six truncated replies
carry all 14,814 tokens of it (COUNT_TOKENS_PROBE.md).

TWO THINGS THE COUNT DELIBERATELY EXCLUDES
The envelope is billed and belongs to no stream, so it is reported as its own
line rather than smeared across the streams - on tau2 it is 18-24% of the
output bill, which is spend no amount of editing the text could remove.

And gemini's reasoning is refused a dollar figure, not because of any spread
statistic but because the text is not the billed object: it returns a
provider-written summary of its chain of thought. Billed reasoning over summary
tokens runs 0.66x to 34.86x, and the summary has a ~204-token floor regardless
of how little the model actually thought, so a large part of it is the
summariser's own writing. The summary's own waste share is still reported - it
is a real measurement of a real text - it just cannot be priced.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

from .. import paths
from ..evaluate.efficiency import merged_spans
from ..schema import EffVerdict, Span, Turn

#: Streams to a billing bucket, the same split `waste_tokens` uses.
REASONING, OUTPUT = "reasoning", "visible+action"
BUCKET_OF = {"reasoning": REASONING, "visible": OUTPUT, "action": OUTPUT}

#: What each model's reasoning text IS. Not a statistic - a fact about the API,
#: and the three cases need different words in a report, not one "unpriceable":
#:
#:   raw      the chain of thought itself. Auditable and priceable.
#:   summary  a provider-written precis of it. Real text, real waste share, but
#:            a different object from the billed tokens - 0.66x to 34.86x them,
#:            with a ~204-token floor of the summariser's own framing.
#:   absent   no text at all. claude-sonnet-5 returns a thinking block whose
#:            text is empty and whose only content is a 564-character encrypted
#:            signature; gpt-5.6-terra returns nothing. Billed, unknowable - and
#:            that is "no evidence", never "no waste".
RAW_CHAIN = ("kimi-k3", "kimi-k3-low", "kimi-k3-high", "kimi-k3-none",
             "sentinel-qwen3-4b-thinking-rl",
             "qwen3-30b-a3b-thinking-2507-tau2-consecutive-tool-penalty")
#: claude-fable-5-1 and gpt-6-astra return a provider-written summary of their
#: reasoning, like gemini's - counted, never priced. A sonnet/terra cell run
#: without the summary setting carries no text and is "absent".
SUMMARY_ONLY = ("gemini-3.1-pro-preview", "claude-fable-5-1", "gpt-6-astra")


def reasoning_kind(agent_model: str, has_text: bool | None = None) -> str:
    """What the stored reasoning text IS for a model - or, given `has_text`, for
    one cell of it.

    Per cell matters: the same model can be run with the provider's summary
    switched on or off, so one cell carries a summary while another carries
    nothing, and a model-level answer would misfile one of them.
    """
    # Match on the model, not the cell name. A harness-ablation cell is called
    # `<model>__<harness>`: same model, same raw chain of thought, same
    # tokenizer (`_counter_for` already matches by prefix) - only the harness
    # differs. Matching the full string sent all seven of those cells down the
    # `summary` branch, where reasoning is counted but never priced and is left
    # out of `audited`, so their headline waste share covered visible+action
    # alone while every native kimi cell's covered reasoning too - two different
    # measures under one name.
    if agent_model.split("__")[0] in RAW_CHAIN:
        return "raw"
    if has_text is not None:
        return "summary" if has_text else "absent"
    if agent_model in SUMMARY_ONLY:
        return "summary"
    return "absent"

#: JSON escapes the PROXY introduced, not the model. Gemini and Anthropic return
#: tool arguments as a structured object; normalising them to the OpenAI string
#: form runs them through json.dumps, and one newline becomes two characters that
#: tokenise apart. Measured on gemini: counting the escaped form overshoots the
#: bill by 58 tokens on a 3,691-token turn, and undoing the escapes lands at +7 -
#: the same constant every other tool call shows. Applied only where the proxy
#: created them; kimi and terra come from OpenAI-format APIs where the model
#: itself generated the string.
#: Models whose tool calls are counted as the provider's own objects rather
#: than as our rendered `name({args})` text. Both have a token-count API that
#: accepts the call in its native form; measured on real turns, the native
#: count matches the bill to a constant (below) while the rendered text was off
#: by -129..+51 tokens per turn and always low on short calls, which is where
#: most detours and duplicate fetches live (COUNT_TOKENS_PROBE.md).
#: Anthropic models share one tokenizer and one message envelope. claude-fable-5-1 was
#: calibrated against 30 real tau2 turns (max effort): text-only
#: turns match count_tokens minus 6 on 20/20, k-call turns match the per-call
#: tool_use count minus (6 + 10(k-1)) within +-1 on 10/10, k = 1..5 - the same two
#: constants sonnet measured.
ANTHROPIC = ("claude-sonnet-5", "claude-fable-5-1")

#: OpenAI models whose billed text is reproduced by tiktoken o200k. gpt-6-astra
#: checked on 33 real tau2 turns (xhigh): text-only turns
#: bill exactly o200k + 6 on 9/9, one-call turns bill the rendered call + 11
#: (19/24) or + 13 (5/24) - the same constant-envelope behaviour terra showed,
#: so a different tokenizer is ruled out (it would scatter the differences).
OPENAI_O200K = ("gpt-5.6-terra", "gpt-6-astra")

#: Reasoning inside `content` (no parser at serve time); see preprocess/inline_think.py.
INLINE_THINK = ("qwen3-30b-a3b-thinking-2507-tau2-consecutive-tool-penalty",)

NATIVE_ACTION = ("gemini-3.1-pro-preview",) + ANTHROPIC


def _native_envelope(model: str, k: int) -> int:
    """billed tokens of a k-call turn = raw native count + this.

    Measured on real action-only turns against the bill:
      gemini   native = billed - (2k - 1)   exact on 40/40 turns, k = 1..4
      sonnet   sum of per-call counts, each with a placeholder tool_result,
               = billed + 6 + 10(k - 1)     within +-1 on 66/66 turns, k = 1..4
    """
    if k <= 0:
        return 0
    if model == "gemini-3.1-pro-preview":
        return 2 * k - 1
    if model in ANTHROPIC:
        return -(6 + 10 * (k - 1))
    raise KeyError(model)


# --------------------------------------------------------------- the counters

class Counter:
    """One model's token counter, plus a disk cache keyed by the text.

    Cached because two of the four are network calls and the point of writing
    this to disk is that a rebuild costs nothing. The key is the text itself,
    so a cache entry can never belong to a different string.
    """

    def __init__(self, model: str):
        self.model = model
        self.path = paths.RESULTS_OUT / "token_cache" / f"{paths.agent_seg(model)}.json"
        self.cache: dict[str, int] = {}
        if self.path.exists():
            try:
                self.cache = json.load(self.path.open())
            except (OSError, ValueError):
                self.cache = {}
        self.lock = threading.Lock()
        self.calls = 0
        self._fn = _counter_for(model)

    def __call__(self, text: str) -> int:
        # Whitespace-only counts as nothing, and is not sent. It arises when the
        # judge marked a whole stream: strip the spans and only the joins remain.
        # Anthropic rejects such a request outright ("text content blocks must
        # contain non-whitespace text"), which surfaced as a counter failure on
        # exactly the streams where the finding was strongest.
        if not text or not text.strip():
            return 0
        # sha256, not hash(): Python randomises string hashing per process,
        # so a cache keyed on hash() never survives a restart - the first
        # rebuild re-made every network call while looking like a cache hit.
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
        with self.lock:
            hit = self.cache.get(key)
        if hit is not None:
            return hit
        n = self._fn(text)
        with self.lock:
            self.cache[key] = n
            self.calls += 1
        return n

    def count_calls(self, calls: list[tuple[str, dict]]) -> int:
        """Billed-equivalent tokens of a turn's tool calls, as objects.

        `calls` are (name, argument dict). Cached under a key that includes
        the model and the canonical JSON of the calls, so a re-run costs no
        network calls and a different model can never hit another's entry.
        """
        if not calls:
            return 0
        canon = json.dumps([self.model, calls], sort_keys=True, ensure_ascii=False)
        key = "calls:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()[:24]
        with self.lock:
            hit = self.cache.get(key)
        if hit is not None:
            return hit
        n = _native_calls_for(self.model)(calls) + _native_envelope(self.model, len(calls))
        with self.lock:
            self.cache[key] = n
            self.calls += 1
        return n

    def save(self) -> None:
        paths.ensure_parent(self.path)
        json.dump(self.cache, self.path.open("w"))


def _native_calls_for(model: str):
    """Raw native count of a list of (name, args) calls; envelope added by caller."""
    if model == "gemini-3.1-pro-preview":
        from google import genai
        from google.genai import types
        cl = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        def _g(calls):
            parts = [types.Part(function_call=types.FunctionCall(name=n, args=a))
                     for n, a in calls]
            for _ in range(6):
                try:
                    return cl.models.count_tokens(
                        model="gemini-3.1-pro-preview",
                        contents=[types.Content(role="model", parts=parts)]).total_tokens
                except Exception:
                    time.sleep(0.5)
            raise RuntimeError("gemini count_tokens (function_call) failed after 6 tries")
        return _g
    if model in ANTHROPIC:
        import anthropic
        cl = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        def _one(name, args):
            # The API refuses a tool_use without a tool_result after it, so
            # each call is counted alone with a fixed placeholder result and
            # a fixed tool definition; both are constant and are part of the
            # envelope measured in _native_envelope.
            tools = [{"name": name, "description": "x",
                      "input_schema": {"type": "object", "properties": {}}}]
            msgs = [{"role": "user", "content": "x"},
                    {"role": "assistant", "content": [
                        {"type": "tool_use", "id": "toolu_000000000000000000000001",
                         "name": name, "input": args}]},
                    {"role": "user", "content": [
                        {"type": "tool_result",
                         "tool_use_id": "toolu_000000000000000000000001", "content": "ok"}]}]
            for _ in range(6):
                try:
                    full = cl.messages.count_tokens(model=model, tools=tools,
                                                    messages=msgs).input_tokens
                    base = cl.messages.count_tokens(model=model, tools=tools,
                                                    messages=msgs[:1]).input_tokens
                    return full - base
                except Exception:
                    time.sleep(0.5)
            raise RuntimeError("anthropic count_tokens (tool_use) failed after 6 tries")
        return lambda calls: sum(_one(n, a) for n, a in calls)
    raise KeyError(f"no native tool-call counter for {model}")


_APPROX_NOTED: set = set()


def _counter_for(model: str):
    if model.startswith("kimi-k3"):
        import tempfile
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        p = hf_hub_download("baseten/kimi-k3-tokenizer", "tokenizer.json")
        d = json.load(open(p))
        # The decoder uses a variant newer than the installed tokenizers can
        # parse, and it is only needed to turn ids back into text. Dropped so
        # the encoder - the only half this module uses - loads.
        d.pop("decoder", None)
        f = os.path.join(tempfile.mkdtemp(), "tokenizer.json")
        json.dump(d, open(f, "w"))
        tok = Tokenizer.from_file(f)
        return lambda t: len(tok.encode(t, add_special_tokens=False).ids)
    if model == "sentinel-qwen3-4b-thinking-rl":
        # A Qwen3-4B-Thinking fine-tune: the base model's tokenizer, which the
        # fine-tune does not change. Verified against the proxy's billed
        # completion tokens on real rows.
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3-4B-Thinking-2507",
                                                  "tokenizer.json"))
        return lambda t: len(tok.encode(t, add_special_tokens=False).ids)
    if model in INLINE_THINK:
        # The 30B Qwen3 fine-tune: same tokenizer family, taken from its own base
        # repo. The reasoning/reply split and the reasoning_tokens reconstruction
        # for this model live in preprocess/inline_think.py, which owns the
        # tokenizer so both count with one object.
        from ..preprocess.inline_think import counter
        return counter(model)
    if model in OPENAI_O200K:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        # o200k_harmony gives an identical count on all 1,878 turns, so the
        # choice between them is not a source of error.
        return lambda t: len(enc.encode(t))
    if model == "gemini-3.1-pro-preview":
        from google import genai
        cl = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        def _g(t: str) -> int:
            for _ in range(5):
                try:
                    # An empty string counts 1: a turn marker the caller is not
                    # asking about. Subtracted so the number is the text alone.
                    return cl.models.count_tokens(
                        model="gemini-3.1-pro-preview", contents=t).total_tokens - 1
                except Exception:
                    time.sleep(0.4)
            raise RuntimeError("gemini count_tokens failed after 5 tries")
        return _g
    if model in ANTHROPIC:
        import anthropic
        cl = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        def _a(t: str) -> int:
            for _ in range(5):
                try:
                    # The messages envelope is 6 tokens, measured against a
                    # single-character message.
                    return cl.messages.count_tokens(
                        model=model,
                        messages=[{"role": "user", "content": t}]).input_tokens - 6
                except Exception:
                    time.sleep(0.4)
            raise RuntimeError("anthropic count_tokens failed after 5 tries")
        return _a
    # Not in the table: count with o200k and say so. The counts are then an
    # approximation of the provider's bill, not a reconstruction of it; add the
    # model's own tokenizer above for exact numbers.
    import tiktoken
    if model not in _APPROX_NOTED:
        _APPROX_NOTED.add(model)
        print(f"     note: no tokenizer entry for {model}; counting with o200k_base "
              f"(approximate). Add one in llmaj/aggregate/exact_tokens.py for exact counts.", flush=True)
    enc = tiktoken.get_encoding("o200k_base")
    return lambda t: len(enc.encode(t))


# ------------------------------------------------------------------- the roll

@dataclass
class Bucket:
    #: Billed tokens for this bucket, straight from the proxy's record.
    billed: int = 0
    #: Tokens of the text we hold, counted. Their difference from `billed` is
    #: the message envelope, which belongs to no stream.
    text: int = 0
    #: Tokens that would not have been billed had the marked spans not been
    #: written. Counted, never converted.
    waste: int = 0
    #: Text a usable verdict covered, and text it did not. Both in tokens, so
    #: coverage is on the same scale as everything else here.
    judged: int = 0
    dollars: float = 0.0
    waste_dollars: float = 0.0
    #: kind -> text tokens removed when ONLY that kind's marks are stripped.
    #: Counted, not apportioned: the earlier route split `waste` by marked
    #: tokens per kind, and a token of a tool call is not a token
    #: of prose - detour (calls, code) came out 4-5 points too low and padding
    #: (prose) as much too high on gpt-5.6-terra and sonnet-5.
    waste_by_kind: dict = field(default_factory=dict)

    def dollars_by_kind(self) -> dict:
        """`waste_dollars` split by the per-kind token counts.

        The per-kind strips do not sum exactly to `waste` (tokens merge across
        a cut), so they are scaled to it rather than priced one by one.
        """
        t = sum(self.waste_by_kind.values())
        if not t:
            return {}
        return {k: self.waste_dollars * n / t for k, n in self.waste_by_kind.items()}

    @property
    def envelope(self) -> int:
        """Billed tokens that belong to no stream: role markers, the tool_use
        block, the call id, the tool name.

        Signed on purpose. It goes NEGATIVE where our rendering carries text the
        provider did not bill - kimi's mcp_atlas actions are written
        `name({args})` while the provider billed its own serialisation, so we
        count 3,183 tokens more than the bill. Clamping that to zero would hide
        a real disagreement between our text and the billed text behind a
        plausible-looking number."""
        return self.billed - self.text

    @property
    def share(self) -> float:
        """Waste over the BILL. Only meaningful when the text we counted is the
        text that was billed - see `share_of_text` for when it is not."""
        return self.waste / self.billed if self.billed else 0.0

    @property
    def share_of_text(self) -> float:
        """Waste over the text we actually read.

        The right denominator for a bucket whose text is not the billed object.
        gemini's claw_eval summary is 21,294 tokens against 190,668 billed, so
        dividing its waste by the bill gave 6.6% where the summary's own waste
        is 59.0% - a number about neither thing. Published alongside `share`
        rather than instead of it, and `priceable` says which one to read."""
        return self.waste / self.text if self.text else 0.0

    @property
    def coverage(self) -> float:
        return self.judged / self.text if self.text else 0.0

    def to_json(self, priceable: bool) -> dict:
        d = {"billed_tokens": self.billed, "text_tokens": self.text,
             "envelope_tokens": self.envelope, "waste_tokens": self.waste,
             # `waste_share` is over the bill where the text is what was
             # billed, and over the text where it is not (a provider's
             # reasoning summary: marked summary tokens over summary tokens).
             # There is no third figure: marked SUMMARY tokens over the
             # BILLED hidden reasoning would put two different objects in one
             # ratio and measure neither.
             "waste_share": self.share if priceable else self.share_of_text,
             "coverage": self.coverage, "priceable": priceable,
             "waste_tokens_by_kind": dict(self.waste_by_kind)}
        if priceable:
            d["billed_dollars"] = round(self.dollars, 6)
            d["waste_dollars"] = round(self.waste_dollars, 6)
            d["waste_dollars_by_kind"] = {k: round(v, 6)
                                          for k, v in self.dollars_by_kind().items()}
        return d


@dataclass
class Cell:
    agent_model: str
    benchmark: str
    domain: str
    buckets: dict[str, Bucket] = field(default_factory=dict)
    n_traces: int = 0
    calls: int = 0
    #: One record per counted turn: {trace_id, round, context_tokens,
    #: buckets: {name: {billed, waste, waste_by_kind}}}. The cell totals are
    #: the sums of these, so anything drawn from them (waste by position in
    #: the trajectory, by context length) reconciles with the cell by
    #: construction rather than by a second count.
    rounds: list = field(default_factory=list)
    #: Marked ranges inside tool calls that lay on JSON syntax only and so
    #: removed no token. Reported, not hidden.
    structure_only_ranges: int = 0

    def bucket(self, name: str) -> Bucket:
        return self.buckets.setdefault(name, Bucket())

    def priceable(self, name: str) -> bool:
        """Whether this bucket's tokens may carry a dollar figure.

        One rule, and it is about what the text IS rather than how it behaves:
        a bucket can be priced when the text we counted is the text that was
        billed. That holds for every visible+action bucket and for a raw chain
        of thought. It fails for a provider-written summary, which is a
        different object from the reasoning it describes.
        """
        if name != REASONING:
            return True
        # Same split as `reasoning_kind`: the harness suffix names the cell,
        # not the model whose chain of thought this is.
        return self.agent_model.split("__")[0] in RAW_CHAIN

    def audited_names(self) -> list[str]:
        """The buckets this report treats as the agent's audited output.

        Replies and tool calls always. Reasoning ONLY when its text is the
        billed object (kimi-k3's raw chain of thought): the judge read what
        was paid for, so its marks are marks on the bill. A provider-written
        summary (gemini, gpt-5.6-terra, gpt-6-astra, claude-fable-5-1) is a
        different object from the reasoning that was billed - on one
        gpt-6-astra cell 145 of 702 summaries hold more
        tokens than the reasoning billed beside them, which a billed text
        cannot do - so that reasoning is in no cost figure at all. It stays in
        the trace viewer, where the summary is shown as what it is.
        """
        names = [OUTPUT] if (self.buckets.get(OUTPUT) or Bucket()).billed else []
        r = self.buckets.get(REASONING)
        if r and r.billed and self.priceable(REASONING):
            names.append(REASONING)
        return names

    def audited_json(self) -> dict:
        """The audited buckets pooled into one: the denominator every cost
        figure divides by, written once here so nothing pools it a second time
        differently."""
        names = self.audited_names()
        bs = [self.buckets[n] for n in names]
        billed = sum(b.billed for b in bs)
        text = sum(b.text for b in bs)
        waste = sum(b.waste for b in bs)
        judged = sum(b.judged for b in bs)
        by_kind: dict[str, int] = {}
        usd_kind: dict[str, float] = {}
        for b in bs:
            for k, v in b.waste_by_kind.items():
                by_kind[k] = by_kind.get(k, 0) + v
            for k, v in b.dollars_by_kind().items():
                usd_kind[k] = usd_kind.get(k, 0.0) + v
        return {"buckets": names,
                "billed_tokens": billed, "text_tokens": text,
                "waste_tokens": waste,
                "waste_share": waste / billed if billed else 0.0,
                "coverage": judged / text if text else 0.0,
                "billed_dollars": round(sum(b.dollars for b in bs), 6),
                "waste_dollars": round(sum(b.waste_dollars for b in bs), 6),
                "waste_tokens_by_kind": by_kind,
                "waste_dollars_by_kind": {k: round(v, 6)
                                          for k, v in usd_kind.items()}}

    def to_json(self) -> dict:
        return {
            "agent_model": self.agent_model, "benchmark": self.benchmark,
            "domain": self.domain, "n_traces": self.n_traces,
            # What the cost figures are over. See `audited_names`.
            "audited": self.audited_json(),
            "notes": {"structure_only_ranges": self.structure_only_ranges},
            # `calls` is deliberately NOT published. It counts what THIS run
            # sent, which is zero on a cached rebuild, so writing it makes the
            # same inputs produce a different file - the one property this
            # artefact exists to have.
            "reasoning_kind": reasoning_kind(
                self.agent_model,
                has_text=bool((self.buckets.get(REASONING) or Bucket()).text)),
            "buckets": {k: v.to_json(self.priceable(k))
                        for k, v in self.buckets.items()},
            "rounds": [
                {"trace_id": r["trace_id"], "round": r["round"],
                 "context_tokens": r["context_tokens"],
                 "buckets": {k: {"billed_tokens": b["billed"],
                                 "waste_tokens": b["waste"],
                                 "waste_tokens_by_kind": dict(b["waste_by_kind"])}
                             for k, b in sorted(r["buckets"].items())}}
                for r in sorted(self.rounds, key=lambda r: (r["trace_id"], r["round"]))],
        }


def _parse_calls(calls: list[tuple[str, str]]) -> list[tuple[str, dict]]:
    return [(n, json.loads(a) if a else {}) for n, a in calls]


def _string_values(raw: str) -> list[tuple[int, int, tuple]]:
    """Every string VALUE in a JSON document, at any depth: (inner start,
    inner end, path). `path` is the sequence of keys / list indices from the
    root, so the cut can be applied to the decoded object at that place.

    A small scanner over the raw text, because the cut has to happen at raw
    positions - the judge's ranges are positions in the rendered string - and
    then be applied to the decoded value so the result is still valid JSON.
    Nested strings (an `edits[0].old`, a `params.query`) are listed too: a
    fifth to a third of claw_eval / mcp_atlas tool calls carry nested arguments.
    """
    out: list[tuple[int, int, tuple]] = []
    stack: list[list] = []          # per container: [kind, key_or_index, expect_value]
    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]
        if ch == '"':
            j = i + 1
            while j < n and raw[j] != '"':
                j += 2 if raw[j] == "\\" else 1
            k = j + 1
            while k < n and raw[k] in " \t\r\n":
                k += 1
            top = stack[-1] if stack else None
            if top is not None and top[0] == "{" and k < n and raw[k] == ":" and not top[2]:
                top[1] = json.loads(raw[i:j + 1]); top[2] = True
            elif top is not None and (top[0] == "[" or top[2]):
                out.append((i + 1, j, tuple(fr[1] for fr in stack)))
                if top[0] == "{":
                    top[2] = False
            i = j + 1
            continue
        if ch == "{":
            stack.append(["{", None, False])
        elif ch == "[":
            stack.append(["[", 0, True])
        elif ch in "}]":
            if stack:
                stack.pop()
            if stack and stack[-1][0] == "{":
                stack[-1][2] = False
        elif ch == ",":
            if stack and stack[-1][0] == "[":
                stack[-1][1] += 1
            elif stack and stack[-1][0] == "{":
                stack[-1][2] = False
        elif ch not in " \t\r\n" and stack and stack[-1][0] == "{" and stack[-1][2]:
            # a non-string value (number, true, null, or a container handled above)
            pass
        i += 1
    return out


def _set_path(obj, path: tuple, value):
    cur = obj
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value


def _get_path(obj, path: tuple):
    cur = obj
    for k in path:
        cur = cur[k]
    return cur


def _decodes(raw_fragment: str) -> bool:
    try:
        json.loads('"' + raw_fragment + '"')
        return True
    except ValueError:
        return False


def _decoded_offsets(lit: str, start: int, end: int) -> tuple[int, int]:
    """Decoded [start, end) inside a string literal's inner text.

    A judge's range can begin or end in the middle of an escape sequence
    (right after the backslash of a \\" for instance). Each boundary is moved
    outward until the prefix it closes decodes: one character for a two-
    character escape, up to five for a \\uXXXX one.
    """
    while start > 0 and not _decodes(lit[:start]):
        start -= 1
    while end < len(lit) and not _decodes(lit[:end]):
        end += 1
    if not (_decodes(lit[:start]) and _decodes(lit[:end])):
        raise ValueError(f"undecodable literal boundary in {lit[:60]!r}")
    return len(json.loads('"' + lit[:start] + '"')), len(json.loads('"' + lit[:end] + '"'))


#: Marked ranges that fell only on JSON syntax (braces, keys, commas) and so
#: removed nothing. Per process; `build` reads and resets it per cell and writes
#: the cell's count into exact_tokens.json (`notes.structure_only_ranges`), so
#: it is never silently large.
STRUCTURE_ONLY_RANGES = 0


def _cut_calls(calls: list[tuple[str, str]], ranges, who: str) -> list[tuple[str, dict]]:
    """The turn's calls with the marked ranges removed, as objects.

    `ranges` are (start, end, kind) in the rendered text. A call whose text is
    covered end to end is dropped: a wasted call is wasted whole, envelope
    included. A call marked in part has the marked text removed from inside
    the string argument it falls in, by decoded position, so the result is
    always valid JSON. Marked text that lies on JSON syntax rather than in a
    value removes nothing and is counted in STRUCTURE_ONLY_RANGES.
    """
    global STRUCTURE_ONLY_RANGES
    from ..preprocess.structured_calls import segments
    out = []
    for (a, b), (name, args) in zip(segments(calls), calls):
        cov = sorted((max(a, s), min(b, e)) for s, e, _ in ranges if s < b and e > a)
        covered = sum(e - s for s, e in cov)
        if not covered:
            out.append((name, json.loads(args) if args else {}))
            continue
        if covered >= (b - a) - 2:      # whole call marked; tolerate the trailing paren
            continue
        obj = json.loads(args) if args else {}
        astart = a + len(name) + 1       # rendered position of args[0]
        lits = _string_values(args)
        # Apply cuts per string value, from the end so earlier offsets hold.
        cuts_by_path: dict[tuple, list[tuple[int, int]]] = {}
        for s_, e_ in cov:
            rs, re_ = s_ - astart, e_ - astart          # raw args coordinates
            touched = False
            for ls, le, path in lits:
                os_, oe = max(rs, ls), min(re_, le)
                if oe <= os_:
                    continue
                touched = True
                d0, d1 = _decoded_offsets(args[ls:le], os_ - ls, oe - ls)
                cuts_by_path.setdefault(path, []).append((d0, d1))
            if not touched:
                STRUCTURE_ONLY_RANGES += 1
        for path, cuts in cuts_by_path.items():
            v = _get_path(obj, path)
            if not isinstance(v, str):
                raise ValueError(f"{who}: cut inside non-string argument {path!r}")
            for d0, d1 in sorted(cuts, reverse=True):
                v = v[:d0] + v[d1:]
            _set_path(obj, path, v)
        out.append((name, obj))
    return out


def _strip(text: str, ranges) -> str:
    """The stream with every marked range removed, in one pass."""
    if not ranges:
        return text
    out = [text[:ranges[0][0]]]
    for i in range(len(ranges) - 1):
        out.append(text[ranges[i][1]:ranges[i + 1][0]])
    out.append(text[ranges[-1][1]:])
    return "".join(out)


def _stream_text(turn: Turn, stream: str) -> str:
    return {"reasoning": turn.reasoning_content, "visible": turn.visible_text,
            "action": turn.action}[stream]


def build(agent_model: str, benchmark: str, domain: str, counter: Counter,
          groups, price_per_token: float, workers: int = 8) -> Cell:
    """One cell, counted. `groups` yields (turns, verdicts) per trajectory."""
    from concurrent.futures import ThreadPoolExecutor

    cell = Cell(agent_model, benchmark, domain)
    jobs = []
    calls_before = counter.calls
    try:
        return _build(cell, agent_model, benchmark, domain, counter, groups,
                      price_per_token, workers, jobs, calls_before)
    finally:
        # Every count bought so far is kept even when the cell fails half way
        # (a stored action text that no longer matches its calls raises); the
        # re-run then costs only what is still missing.
        counter.save()


def _build(cell, agent_model, benchmark, domain, counter, groups, price_per_token,
           workers, jobs, calls_before):
    from concurrent.futures import ThreadPoolExecutor
    global STRUCTURE_ONLY_RANGES
    STRUCTURE_ONLY_RANGES = 0
    for turns, verdicts in groups:
        usable = {(v.round, v.stream): v for v in verdicts
                  if not v.error and v.aligned and v.citation_ok and v.cls}
        if not usable:
            # A trajectory with no usable verdict is a run failure, not a
            # finding: counting its tokens would assert it wasted nothing.
            continue
        cell.n_traces += 1
        from ..preprocess.structured_calls import context_for
        ctx = context_for(agent_model, benchmark, domain, turns[0].trace_id) if turns else {}
        for turn in turns:
            billed = turn.output_tokens or 0
            if not billed:
                continue
            reason = turn.reasoning_tokens or 0
            cell.bucket(REASONING).billed += reason
            cell.bucket(OUTPUT).billed += billed - reason
            rec = {"trace_id": turn.trace_id, "round": turn.round,
                   "context_tokens": ctx.get(turn.round), "buckets": {}}
            def rb(name, _rec=rec):
                return _rec["buckets"].setdefault(
                    name, {"billed": 0, "waste": 0, "waste_by_kind": {}})
            if reason:
                rb(REASONING)["billed"] += reason
            rb(OUTPUT)["billed"] += billed - reason
            ri = len(cell.rounds)
            cell.rounds.append(rec)
            for stream in ("reasoning", "visible", "action"):
                text = _stream_text(turn, stream)
                if not text:
                    continue
                v = usable.get((turn.round, stream))
                if stream == "action" and agent_model in NATIVE_ACTION:
                    # Count the calls as objects. First prove that the objects
                    # are the ones the judge saw: rendering them must give the
                    # stored string byte for byte.
                    from ..preprocess.structured_calls import calls_for, render
                    calls = calls_for(agent_model, benchmark, domain, turn.trace_id).get(turn.round)
                    who = f"{agent_model}/{benchmark}/{domain} {turn.trace_id} round {turn.round}"
                    if calls is None or render(calls) != text:
                        raise ValueError(f"{who}: stored action text does not match "
                                         f"the calls in calls.jsonl; refusing to count")
                    # Objects to count, prepared here (pure work, and the one
                    # place STRUCTURE_ONLY_RANGES is touched); the counting -
                    # two HTTP calls per object on Anthropic - runs in the pool
                    # like the text jobs.
                    full_calls = _parse_calls(calls)
                    if v is None:
                        jobs.append((ri, "native", BUCKET_OF[stream], full_calls, None, {}))
                        continue
                    ranges = merged_spans(v.spans, text)
                    cut_calls = _cut_calls(calls, ranges, who)
                    per_calls = {k: _cut_calls(calls, [r for r in ranges if r[2] == k], who)
                                 for k in sorted({k for _, _, k in ranges})}
                    jobs.append((ri, "native", BUCKET_OF[stream], full_calls, cut_calls, per_calls))
                    continue
                if v is None:
                    jobs.append((ri, BUCKET_OF[stream], text, None, {}))
                    continue
                raw = _stream_text(turn, stream)
                ranges = merged_spans(v.spans, raw)
                cut = _strip(raw, ranges)
                # One more strip per kind present, so each kind's tokens are
                # counted with the tokenizer rather than apportioned by
                # characters.
                cuts = {k: _strip(raw, [r for r in ranges if r[2] == k])
                        for k in sorted({k for _, _, k in ranges})}
                jobs.append((ri, BUCKET_OF[stream], text, cut, cuts))

    def work(job):
        ri, job = job[0], job[1:]
        if job[0] == "native":              # tool calls counted as objects
            _, name, full_calls, cut_calls, per_calls = job
            full = counter.count_calls(full_calls)
            if cut_calls is None:
                return ri, name, full, None, {}
            cut = counter.count_calls(cut_calls)
            per = {k: full - counter.count_calls(c) for k, c in per_calls.items()}
            return ri, name, full, cut, per
        name, full, cut, cuts = job
        if cut is None:
            return ri, name, counter(full), None, {}
        nf = counter(full)
        return ri, name, nf, counter(cut), {k: nf - counter(c) for k, c in cuts.items()}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ri, name, full, cut, per in ex.map(work, jobs):
            b = cell.bucket(name)
            b.text += full
            if cut is not None:
                b.judged += full
                b.waste += full - cut
                r = cell.rounds[ri]["buckets"].setdefault(
                    name, {"billed": 0, "waste": 0, "waste_by_kind": {}})
                r["waste"] += full - cut
                for k, n in per.items():
                    b.waste_by_kind[k] = b.waste_by_kind.get(k, 0) + n
                    r["waste_by_kind"][k] = r["waste_by_kind"].get(k, 0) + n
    for name, b in cell.buckets.items():
        # PER TOKEN, matching `dollars.Prices.output`. A per-million figure
        # here would be a factor of a million off with nothing to catch the
        # mistake (every dollar would read as zero). One unit, named in the
        # parameter - a conversion at the call site is a place to
        # get it wrong again.
        b.dollars = b.billed * price_per_token
        b.waste_dollars = b.waste * price_per_token
    cell.calls = counter.calls - calls_before       # THIS cell's new counts
    cell.structure_only_ranges = STRUCTURE_ONLY_RANGES
    return cell


def path_for(judge_id, run_id, agent_model, benchmark, domain):
    return paths.summary_file(paths.EFFICIENCY_TASK, judge_id, run_id,
                              agent_model, benchmark,
                              domain).with_name("exact_tokens.json")


def save(cell: Cell, judge_id: str, run_id: str):
    p = paths.ensure_parent(path_for(judge_id, run_id, cell.agent_model,
                                     cell.benchmark, cell.domain))
    p.write_text(json.dumps(cell.to_json(), indent=2))
    return p


def load(judge_id, run_id, agent_model, benchmark, domain) -> dict | None:
    p = path_for(judge_id, run_id, agent_model, benchmark, domain)
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def table(cell: Cell) -> str:
    """One cell's counted waste, for the run log.

    Prints what was counted and what was refused, because a refusal is the
    part a reader would otherwise mistake for a zero.
    """
    kind = reasoning_kind(cell.agent_model,
                          has_text=bool((cell.buckets.get(REASONING) or Bucket()).text))
    out = [f"    counted {cell.n_traces} trajectories"
           f"  ({cell.calls} new token counts)"]
    for name in (OUTPUT, REASONING):
        b = cell.buckets.get(name)
        if not b or not b.billed:
            continue
        if cell.priceable(name):
            out.append(f"      {name:<15}{b.waste:>9,} / {b.billed:>9,} tok"
                       f" = {b.share * 100:>5.1f}%   ${b.waste_dollars:.4f}")
        elif b.text:
            out.append(f"      {name:<15}{b.waste:>9,} / {b.text:>9,} tok"
                       f" = {b.share_of_text * 100:>5.1f}% of the {kind} text,"
                       f" not priced")
        else:
            out.append(f"      {name:<15}{'':>9} {b.billed:>9,} tok billed,"
                       f" no text returned, not priced")
    return "\n".join(out)
