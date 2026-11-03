"""Drive one judge over whole trajectories, asking what the output was worth.

The unit is a trajectory, not a turn, and that is forced by the definitions:
"was the result used", "the repeats", "already in the context" are all claims
about turns OTHER than the one being judged. It is also cheaper - the system
prompt goes out once per trajectory instead of once per turn, measured 2.8x on
kimi-k3/swe_bench.

Resumable the same way the utility pass is, and for the same reason: a shard
that exists is done, so an interrupted run needs no progress file to keep in
sync.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .. import paths
from ..prompts.efficiency_classes import (DEPENDS_FIELD, KEPT_FIELD, KIND_SEVERITY,
                                          MATCH_CHARS, MATCH_WINDOW, MAX_SPANS,
                                          OPENING_CHARS, SPAN_KINDS, STREAMS,
                                          WASTE_SHARE_BUCKETS)
from ..prompts.efficiency_prompt import EFF_PROMPT_VERSION, build, snapshot
from ..preprocess.duplicates import find as find_repeats
from ..preprocess import policies
from ..preprocess.sample import load_turns
from ..schema import EffVerdict, Span, Turn
from .judge import Judge, trace_meta_for

#: What one sentence of a deliverable is, per cell. It sets the reference point
#: for every must-have verdict, so it is data the caller supplies rather than
#: something guessed here. The benchmark's NAME is deliberately absent: the judge
#: should reason from what was being produced, not from a prior about swe_bench.
DELIVERABLE: dict[str, str] = {
    "swe_bench/all": "A patch to the repository that fixes the reported bug.",
    "tau2/airline": (
        "A reply to the customer, plus any reservation change the airline's "
        "policy allows, made through the booking tools."
    ),
    "tau2/retail": (
        "A reply to the customer, plus any order change the store's policy "
        "allows - return, exchange, cancellation, address or payment edit - "
        "made through the tools."
    ),
    "mcp_atlas/all": "A correct factual answer to the user's question, in text.",
    # claw_eval is the one cell whose deliverable is not one sentence. Keyed by
    # shape rather than by task, and `None` covers 149 of the 161 tasks - see
    # preprocess/claw_tasks for why `category` cannot be mapped wholesale.
    "claw_eval/all": {
        None: (
            "Whatever the user's opening request asks for, carried out through "
            "the tools available - an answer, an action taken, text drafted, a "
            "report produced, or NO_REPLY when the standing instructions say no "
            "reply is warranted. The request is quoted verbatim as the first "
            "thing the agent received; read it rather than assume a shape."
        ),
        "execute": (
            "The operation carried out on the files in the container, and the "
            "output or artefact the user asked for. The work is a side effect on "
            "those files, not the reply about it."
        ),
        "review": (
            "The defect, or the execution order, that the user asked about in the "
            "code they supplied - with the reasoning that establishes it. Here "
            "the reasoning is part of what was asked for, not overhead on it."
        ),
        "guarded": (
            "What the user legitimately asked for, and NOTHING that instructions "
            "embedded in the content asked for. Refusing part of the request can "
            "be the correct deliverable: these tasks include over-asks and "
            "injected instructions that the standing instructions forbid acting "
            "on."
        ),
    },
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "round": {"type": "integer"},
                    "streams": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "stream": {"type": "string", "enum": list(STREAMS)},
                                "opening": {"type": "string"},
                                DEPENDS_FIELD: {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                                KEPT_FIELD: {"type": "string"},
                                "spans": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "quote": {"type": "string"},
                                            "kind": {
                                                "type": "string",
                                                "enum": list(SPAN_KINDS),
                                            },
                                            "why": {"type": "string"},
                                        },
                                        "required": ["quote", "kind", "why"],
                                    },
                                },
                            },
                            # `spans` is required even though an empty list is a
                            # legal answer. Making it optional lets a judge omit
                            # the field and have it read as "nothing removable"
                            # without ever having been asked - which is exactly
                            # the free default the inversion removes.
                            "required": ["stream", "opening", "spans",
                                         KEPT_FIELD, DEPENDS_FIELD],
                        },
                    },
                },
                "required": ["round", "streams"],
            },
        }
    },
    "required": ["turns"],
}

#: A trajectory's worth of verdicts is long, and on kimi-k3 the judge's own
#: reasoning is most of it (88% of output tokens in our runs), scaling with
#: the trajectory's rounds rather than with anything the prompt controls: a
#: 52-round mcp_atlas trajectory needs ~136 verdicts, ~21,500 tokens of answer
#: and ~95,000 of reasoning, and the long Qwen tau2 trajectories pushed the
#: judge past 131,072 with no content at all. The cap only bounds output, it
#: reserves nothing, so raising it costs nothing on calls that fit; a call that
#: still stops with `length` here is limited by the model, not by us. A judge
#: whose provider caps lower learns that cap from its 400 (judge.Judge.ask).
MAX_OUTPUT_TOKENS = 262144


@dataclass
class EffStats:
    traces: int = 0
    skipped: int = 0
    verdicts: int = 0
    errors: int = 0
    spans: int = 0
    spans_unverified: int = 0
    misaligned: int = 0
    #: Verdicts naming a round that does not exist later in the trajectory.
    bad_citations: int = 0
    #: Verdicts claiming must-have while citing nobody - the failure the
    #: citation requirement exists to make visible.
    unmarked_uncited: int = 0
    spans_capped: int = 0
    #: Repeated calls the judge missed and we added. High here is a
    #: statement about the judge, not about the agent.
    repeats_injected: int = 0
    #: Same call, different result - flagged in the prompt for the judge to rule
    #: on, because no mechanical test separates verification from a failed trial.
    rechecks_flagged: int = 0
    #: Shards that existed but held no usable verdict, so were re-sent.
    retried: int = 0


def _stream_text(turn: Turn, stream: str) -> str:
    return {"reasoning": turn.reasoning_content,
            "visible": turn.visible_text,
            "action": turn.action}[stream]


def find_folded(text: str, quote: str) -> tuple[int, int] | None:
    """Locate `quote` in `text`, tolerating the characters a judge rewrites.

    A plain `quote in text` would reject correct spans: the judge writes
    "I\'m sorry" where the source has "I\u2019m sorry", and a correct finding
    would be recorded as a fabrication - the one number that is supposed to say
    how far the judge can be trusted. So quotes are folded through `_FOLD`, the
    same way `opening` alignment is.

    Folding is length-preserving for every entry in `_FOLD` except U+2026, which
    is why the ellipsis is handled by falling back to a squashed search that
    returns no offsets rather than by mapping it: offsets into the folded string
    must stay valid in the raw one, or the highlight would drift.
    """
    if quote in text:
        i = text.find(quote)
        return i, i + len(quote)
    if "\u2026" in text or "\u2026" in quote:
        return None
    a, b = text.translate(_FOLD), quote.translate(_FOLD)
    if len(a) != len(text) or len(b) != len(quote):
        return None
    i = a.find(b)
    return (i, i + len(b)) if i >= 0 else None


#: How closely a quote must resemble a window of the source to count as
#: located, and the shortest run of identical characters that may anchor the
#: window. Both measured on the 386 quotes that exact matching rejected across
#: all 20 cells:
#:
#:     >=0.95   266   68.9%      located window / quote length: median 1.02,
#:     >=0.90    59   15.3%      range 0.85-1.17 - the window is the quote's
#:     >=0.80    11    2.8%      size, not a runaway match
#:     < 0.80    50   13.0%
#:
#: 0.90 is where the two controls separate. Verified quotes matched against
#: their own stream: 100% at >=0.95, median 1.000 (n=2,437). The same quotes
#: matched against a DIFFERENT turn of the same stream: 1.1% at >=0.90, median
#: 0.180 (n=2,365) - and those are text that genuinely repeats across turns, so
#: the quote is still in the trajectory. Sensitivity 100%, false accept 1.1%.
#:
#: Why this matters more than it looks: at exact matching the fabrication rate
#: read 2.12% of all spans and rose with trajectory length, 0.7% -> 5.6%, which
#: looks exactly like a judge that reconstructs from memory on long inputs. 84%
#: of it was punctuation and whitespace the judge tidied while copying. The real
#: rate is 0.18%.
FUZZY_MIN = 0.90
FUZZY_BLOCK = 8


def find_fuzzy(text: str, quote: str) -> tuple[int, int] | None:
    """Locate a quote the judge copied imperfectly, keeping raw-text offsets.

    Runs only when `find_folded` has already failed, so it costs nothing on the
    18,207-span common path - 386 calls across every cell.

    Offsets must index the RAW text, because they are what the page highlights
    and what `merged_spans` unions. That rules out comparing squashed strings:
    squashing collapses whitespace runs and the offsets stop being valid. So the
    comparison runs on the `_FOLD`-translated text, which is length-preserving
    for every entry except U+2026, and the ellipsis case is refused the same way
    `find_folded` refuses it.

    The window is bounded by the outermost matching blocks of at least
    FUZZY_BLOCK characters rather than by a sliding scan: a quote the judge
    retyped differs in scattered places, and the first and last verbatim runs
    are where it starts and ends.
    """
    if "\u2026" in text or "\u2026" in quote:
        return None
    a, b = text.translate(_FOLD), quote.translate(_FOLD)
    if len(a) != len(text) or len(b) != len(quote):
        return None
    if not a or not b:
        return None
    blocks = [x for x in SequenceMatcher(None, a, b, autojunk=False)
              .get_matching_blocks() if x.size >= FUZZY_BLOCK]
    if not blocks:
        return None
    start = min(x.a for x in blocks)
    end = max(x.a + x.size for x in blocks)
    if end <= start:
        return None
    if SequenceMatcher(None, a[start:end], b,
                       autojunk=False).ratio() < FUZZY_MIN:
        return None
    return start, end


def find_span(text: str, quote: str) -> tuple[int, int] | None:
    """Where a quote sits in its stream: exact first, then imperfect copy.

    THE definition of "the judge quoted something real". Locating a quote and
    verifying it are the same act, so there is one function for both and no
    stored boolean to disagree with it - `Span.verified` is re-derived from this
    wherever the source text is at hand, which is what lets the locator improve
    without re-running the judge.
    """
    if not quote:
        return None
    found = find_folded(text, quote)
    return found if found is not None else find_fuzzy(text, quote)


def merged_spans(spans: list[Span], text: str) -> list[tuple[int, int, str]]:
    """The character ranges the spans cover, merged, each carrying one kind.

    THE single place a span is located. Every number and every highlight is
    derived from this list, because the alternative was tried and failed twice:
    `verified` folded curly quotes while `_waste_chars` did not, so round 6 of
    meta-17b470d4 rendered a highlighted `restated` span above the line "0%
    removable, 0 of 358 chars"; and the per-kind totals summed quote lengths
    while the waste total summed a union, so overlapping quotes would have made
    the pie's wedges add up to more than the slice they sit in.

    Overlaps merge and the FIRST kind seen wins, which is the same rule the page
    paints by - a nested quote does not recolour its container. Ordering is by
    start then by length descending, so the longest quote at a position is the
    one whose kind survives.

    A quote that cannot be located contributes nothing: it is a fabrication,
    and counting it would let one inflate the score. Located-ness is decided
    here by `find_span`, never read from `Span.verified`.
    """
    hits: list[tuple[int, int, str]] = []
    for sp in spans:
        if not sp.quote:
            continue
        # `sp.verified` is deliberately NOT consulted. It was written at judge
        # time by whatever locator that run shipped with, and gating on it would
        # freeze the measurement to the weakest locator the pipeline ever had.
        found = find_span(text, sp.quote)
        if found is None:
            continue
        kind = sp.kind or "padding"
        hits.append((found[0], found[1], kind))
        start = found[1]
        while True:
            i = text.find(sp.quote, start)
            if i < 0:
                break
            hits.append((i, i + len(sp.quote), kind))
            start = i + 1
    if not hits:
        return []
    hits.sort(key=lambda h: (h[0], -h[1]))
    merged: list[list] = []
    for a, b, kind in hits:
        if merged and a < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b, kind])
    return [(a, b, k) for a, b, k in merged]


def _waste_chars(spans: list[Span], text: str) -> int:
    """Characters covered by at least one verified span. A union, not a sum."""
    return sum(b - a for a, b, _ in merged_spans(spans, text))


def bucket(share: float) -> str:
    """The stream's class, read off how much of it was marked.

    A summary of the marking, never an input to it. The judge states no class;
    the thresholds only give the summary table three columns to print, and the
    continuous `waste_share` is what any claim should be made from.
    """
    label = WASTE_SHARE_BUCKETS[0][1]
    for lo, name in WASTE_SHARE_BUCKETS:
        if share >= lo:
            label = name
    return label


def inject_repeats(verdicts: list[EffVerdict], turns: list[Turn]) -> int:
    """Add the mechanically found repeated calls as spans on the action stream.

    Not left to the judge, on evidence: shown both 706-character responses of a
    repeated call, it still called the second call must-have, and the
    comparison is one a language model is badly placed to make: two long
    JSON blobs several turns apart, byte-identical, with the received block
    truncated at 1,200 characters so a longer response shows two matching
    PREFIXES and an invisible tail.

    Injected as its own span, with `source="mechanical"`, so a judge that also
    found it keeps its own mark and the two can be compared. Returns how many
    were added that the judge had missed.

    Every repeat is marked, user-asked ones included: the same bytes re-entered
    the context whether or not anyone asked, and `Repeat.kind` grades the two
    differently (`duplicate-fetch` vs `detour`) rather than excusing one.
    """
    repeats = find_repeats(turns).by_round()
    if not repeats:
        return 0
    added = 0
    for v in verdicts:
        if v.stream != "action" or v.error:
            continue
        r = repeats.get(v.round)
        if r is None:
            continue
        # The judge already found this one. Leave its span alone and record the
        # agreement by not injecting a second, identical mark.
        if any(sp.kind in ("duplicate-fetch", "detour") and sp.verified
               for sp in v.spans):
            continue
        text = _stream_text(by_round_of(turns).get(v.round), "action")
        quote = r.action.strip()
        v.spans.append(Span(quote=quote, why=r.why(), kind=r.kind,
                            verified=bool(quote)
                            and find_folded(text or "", quote) is not None,
                            source="mechanical"))
        v.waste_chars = _waste_chars(v.spans, text or "")
        v.waste_share = v.waste_chars / v.n_chars if v.n_chars else 0.0
        v.cls = bucket(v.waste_share)
        added += 1
    return added


def by_round_of(turns: list[Turn]) -> dict[int, Turn]:
    return {t.round: t for t in turns}


def verdicts_from(reply: dict, turns: list[Turn], judge_id: str, run_id: str,
                  trace_id: str) -> tuple[list[EffVerdict], list[str]]:
    """Turn the judge's reply into records, checking it against the input.

    Three things are checked here rather than trusted, each because getting it
    wrong would be invisible in the totals:

      * the round number must exist in the trajectory. A misaligned array - the
        17th verdict landing on the 18th turn - is the failure mode of asking
        for many labels in one call, so the round is echoed and verified.
      * the stream must actually be present in that turn. A verdict on a stream
        the turn does not have is dropped, never counted as waste.
      * `opening` must be a prefix of that stream. This is the one check that
        catches a SHIFTED array - a verdict meant for round 4 arriving labelled
        round 3 passes everything else, because round 3 exists and its stream is
        non-empty. Whitespace is normalised before comparing, and only the first
        OPENING_CHARS are compared, because a judge that tidies indentation is
        aligned and merely untidy.
      * every quote must be a substring of the stream it came from.
    """
    by_round = {t.round: t for t in turns}
    out: list[EffVerdict] = []
    problems: list[str] = []
    # One verdict per (round, stream). The judge sometimes lists the same one
    # twice in a single reply and gives two different answers - measured on the
    # stored runs, 815 such records across 22 cells, including a 221-character
    # action called `not-necessary` with 220 wasted characters in the first
    # entry and `must-have` with 0 in the second. Both were kept and both were
    # summed, so that span entered the denominator twice and the numerator
    # once. Its effect on the shipped numbers is small (<=0.06pp on one cell's
    # waste share) but it is an input the parser should have refused.
    #
    # The first entry wins, and it is recorded rather than quietly dropped.
    # Two alternatives were rejected: dropping both loses the stream's
    # character count, which is the denominator every waste share is built on
    # and the one thing a failed verdict must still contribute; and keeping
    # whichever is more conservative would silently pick a side in a
    # disagreement the judge itself could not resolve.
    seen: set[tuple] = set()

    for entry in reply.get("turns") or []:
        rd = entry.get("round")
        turn = by_round.get(rd)
        if turn is None:
            problems.append(f"round {rd} is not in this trajectory")
            continue
        for st in entry.get("streams") or []:
            name = st.get("stream")
            if name not in STREAMS:
                problems.append(f"round {rd}: unknown stream {name!r}")
                continue
            text = _stream_text(turn, name)
            if not text:
                problems.append(f"round {rd}: {name} was not present")
                continue
            if (rd, name) in seen:
                prev = next(v for v in out if v.round == rd and v.stream == name)
                problems.append(
                    f"round {rd} {name}: listed more than once in one reply; "
                    f"keeping the first ({prev.cls}, {prev.waste_chars} of "
                    f"{prev.n_chars} chars wasted) and dropping the repeat")
                continue
            seen.add((rd, name))
            opening = str(st.get("opening") or "")
            aligned = _prefix_matches(opening, text)
            if not aligned:
                problems.append(
                    f"round {rd} {name}: opening does not match the stream "
                    f"(got {opening[:40]!r})")
            # The citation is checked like `opening`: a round that does not
            # exist cannot have used anything.
            #
            # "Strictly later" applies to ACTION ONLY. A turn's reasoning feeds
            # that turn's own action, so reasoning may cite its own round; and a
            # visible message is consumed by the USER at the round it is sent,
            # so for the final reply - the one message most likely to be the
            # deliverable - "cite a later round" would be unsatisfiable.
            #
            # Only a tool call genuinely reaches forward: its result arrives in
            # the next round and is used after that. So only action is bound.
            cited = [int(x) for x in (st.get(DEPENDS_FIELD) or [])
                     if isinstance(x, (int, float))]
            floor = rd + 1 if name == "action" else rd
            bad_cites = [c for c in cited if c not in by_round or c < floor]
            if bad_cites:
                problems.append(
                    f"round {rd} {name}: cites round(s) {bad_cites} as using "
                    f"what it kept, but no such round is at or after {floor}")
            raw_spans = st.get("spans") or []
            spans = []
            for sp in raw_spans[:MAX_SPANS]:
                quote = str(sp.get("quote") or "")
                kind = str(sp.get("kind") or "")
                if kind not in SPAN_KINDS:
                    problems.append(f"round {rd} {name}: unknown span kind "
                                    f"{kind!r}")
                spans.append(Span(quote=quote, why=str(sp.get("why") or ""),
                                  kind=kind,
                                  verified=bool(quote)
                                  and find_folded(text, quote) is not None))
            ranges = merged_spans(spans, text)
            waste = sum(b - a for a, b, _ in ranges)
            share = waste / len(text) if text else 0.0
            out.append(EffVerdict(
                trace_id=trace_id, round=rd, stream=name,
                judge_id=judge_id, judge_run_id=run_id,
                prompt_version=EFF_PROMPT_VERSION,
                cls=bucket(share), why="",
                n_chars=len(text), opening=opening, aligned=aligned,
                depended_on_by=cited, kept=str(st.get(KEPT_FIELD) or ""),
                citation_ok=not bad_cites, spans=spans,
                waste_chars=waste, waste_share=share,
                spans_capped=len(raw_spans) >= MAX_SPANS,
            ))
    return out, problems


#: Characters a judge routinely rewrites while copying, which say nothing about
#: whether it copied the RIGHT text. Found by three verdicts being rejected as
#: misaligned when the only difference was U+2019 -> U+0027: the judge had
#: normalised a curly apostrophe. Rejecting those discarded correct verdicts and
#: made the judge look unreliable.
_FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00a0": " ", "\u2026": "...",
})


def _squash(text: str) -> str:
    """Normalise what a copy may legitimately change, and nothing else.

    Whitespace runs and the quote/dash characters above. NOT case, NOT
    punctuation in general: those could differ between two turns, and the whole
    point of the comparison is to tell two turns apart.
    """
    return " ".join(text.translate(_FOLD).split())


def _prefix_matches(opening: str, text: str) -> bool:
    """Whether `opening` names the start of `text`, allowing for how copying works.

    Compares MATCH_CHARS, not OPENING_CHARS: the last few characters the judge
    was asked for land inside a token, where a model generating token by token
    has to guess. Every observed mismatch diverged at character 37-39 of 40 and
    was a correct verdict thrown away.

    A stream shorter than MATCH_CHARS is compared over its whole length.
    """
    if not opening:
        return False
    a = _squash(opening)
    b = _squash(text)
    n = min(len(a), len(b), MATCH_CHARS)
    if n == 0:
        return False
    # Anywhere in the opening window, not only at position 0: a judge shown a
    # wrapped reply copies the reply. Two different turns do not share their
    # first 20 characters, so the window costs no discrimination.
    return a[:n] in b[:MATCH_WINDOW + n]


def _has_usable(path) -> bool:
    """Whether a shard holds a usable verdict FROM THE CURRENT PROMPT VERSION.

    The version check is the load-bearing half. `run_id` is date-based and stable
    within a day, and the shard path does not carry the prompt version, so
    without this a prompt change followed by a re-run on the same day skips every
    trajectory and reports "3 done before" - the new prompt never goes out and
    nothing says so. That is worse than a crash: the run looks finished and the
    results are from the old question.

    A shard from another version is not deleted here. `run` counts it as a retry
    and overwrites it, so the old answers survive exactly as long as they are the
    ones on disk.
    """
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("prompt_version") != EFF_PROMPT_VERSION:
                return False
            # Same three conditions the statistics use. `citation_ok` was
            # missing: a trajectory whose only verdicts all failed the citation
            # check would count as done here and contribute nothing there, so it
            # would sit in every denominator with an empty numerator and never be
            # re-judged. No trajectory is in that state today - checked across
            # all 654 shards - which is exactly when to close it.
            if (not d.get("error") and d.get("aligned")
                    and d.get("citation_ok") and d.get("cls")):
                return True
    return False


def _shard_version(path) -> str:
    """The prompt version a shard was written under, or "" if unreadable."""
    try:
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    return str(json.loads(line).get("prompt_version") or "")
    except (OSError, ValueError):
        pass
    return ""


def _shard(row: dict, judge_id: str, run_id: str):
    return paths.verdict_shard(paths.EFFICIENCY_TASK, judge_id, run_id, row["agent_model"],
                               row["benchmark"], row["domain"],
                               row["trace_id"])


def _user_deliverables() -> dict:
    """Deliverable sentences for benchmarks this file does not list, from a JSON
    file keyed `<benchmark>/<domain>` or `<benchmark>`: LLMAJ_DELIVERABLES, or
    artifacts/llmaj/deliverables.json."""
    p = _user_deliverables.path()
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {k: v for k, v in d.items() if isinstance(v, str) and not k.startswith("_")}


_user_deliverables.path = lambda: Path(os.environ.get("LLMAJ_DELIVERABLES") or (paths.WORK / "deliverables.json"))


def deliverable_for(row: dict) -> str:
    """The sentence that tells the judge what this trajectory was producing.

    One per cell, except claw_eval, where the entry is a dict keyed by task
    shape. The shape is looked up from the task definition rather than carried
    in the index, so nothing has to be re-extracted to change the mapping.
    """
    key = f"{row['benchmark']}/{row['domain']}"
    if key not in DELIVERABLE:
        extra = _user_deliverables()
        if key in extra or row["benchmark"] in extra:
            return extra.get(key) or extra[row["benchmark"]]
        raise KeyError(
            f"no deliverable sentence for {key}. Every must-have verdict is "
            f"relative to it, so add one to {_user_deliverables.path()} "
            f"(a JSON object: {{\"{key}\": \"<what the agent was producing>\"}}) "
            f"rather than let the judge infer what was being produced."
        )
    entry = DELIVERABLE[key]
    if isinstance(entry, str):
        return entry
    from ..preprocess.claw_tasks import shape
    return entry[shape(row.get("task_id"))]


def _judge_one(judge: Judge, row: dict, run_id: str, policy: str,
               overwrite: bool) -> tuple[str, list, list, int, int]:
    """One trajectory: one call, one shard. Returns what the caller tallies.

    Pulled out of `run` so trajectories can go concurrently. Everything it
    touches is per-trajectory - its own shard path, its own turns, its own call -
    so nothing needs a lock. The two shared things are handled outside: the
    prompt snapshot is written once before any worker starts, and the policy is
    loaded once and passed in.

    Returns ("skipped"|"retried"|"done"|"empty", verdicts, problems, injected,
    rechecks) rather than mutating a shared stats object, because a counter
    incremented from four threads is the kind of bug that shows up as a total
    that is slightly wrong and never as a crash.
    """
    path = _shard(row, judge.judge_id, run_id)
    state = "done"
    if path.exists() and not overwrite:
        if _has_usable(path):
            return "skipped", [], [], 0, 0
        state = "retried"
        old = _shard_version(path)
        why = (f"was judged under prompt {old}, now {EFF_PROMPT_VERSION}"
               if old and old != EFF_PROMPT_VERSION
               else "previous attempt produced no usable verdict")
        print(f"  {row['trace_id'][:16]}: {why}, re-judging", flush=True)

    turns = load_turns(row)
    live = [t for t in turns if t.reasoning_content or t.visible_text or t.action]
    if not live:
        return "empty", [], [], 0, 0

    meta = trace_meta_for(judge.judge_id, run_id, EFF_PROMPT_VERSION,
                          row["agent_model"], row["benchmark"], row["domain"],
                          task="efficiency")
    report = find_repeats(turns)
    reply, err = judge.ask(
        build(turns, deliverable_for(row), policy, report.rechecks_by_round()),
        RESPONSE_SCHEMA, "efficiency_audit", meta, MAX_OUTPUT_TOKENS)
    injected = 0
    if reply is None:
        verdicts, problems = [], [err or "no reply"]
    else:
        verdicts, problems = verdicts_from(reply, turns, judge.judge_id,
                                           run_id, row["trace_id"])
        injected = inject_repeats(verdicts, turns)

    paths.ensure_parent(path)
    with path.open("w") as fh:
        for v in verdicts:
            fh.write(json.dumps(v.to_json()) + "\n")
        # Problems are written into the shard, not printed and forgotten: a
        # trajectory whose reply was rejected must not look like one with
        # nothing to report.
        for pr in problems:
            fh.write(json.dumps(EffVerdict(
                trace_id=row["trace_id"], round=-1, stream="",
                judge_id=judge.judge_id, judge_run_id=run_id,
                prompt_version=EFF_PROMPT_VERSION, error=pr).to_json()) + "\n")
    return state, verdicts, problems, injected, len(report.rechecks)


def run(judge: Judge, rows: list[dict], run_id: str, overwrite: bool = False,
        progress: bool = True, workers: int = 1) -> EffStats:
    """Judge every trajectory, `workers` at a time.

    Serial was 6 minutes per trajectory measured - 535s, 321s and 241s for
    three - because 88% of the judge's output is its own reasoning and that does
    not compress. 360 trajectories serially is 36 hours, so concurrency is not an
    optimisation here.

    The cap that matters is upstream, not local: every judge session shares
    one endpoint, and several sessions each running `workers` in parallel
    multiply against it. One 503 there takes out every
    in-flight call. Concurrency also changes the blast radius of one
    outage from one trajectory to `workers` of them.
    """
    snapshot()          # before the first worker, so a run that dies leaves its question
    stats = EffStats()
    # One document per cell for three of the four benchmarks - identical across
    # agents, so loaded once rather than per trajectory. Empty for mcp_atlas,
    # which has no system prompt, and an empty string renders no block at all
    # rather than an empty one that would invite the judge to reason about its
    # absence.
    #
    # claw_eval is per trajectory: two thirds of its system message is the tool
    # environment of the specific task, so one stored document would show every
    # trajectory but one another task's tools. `policies.for_trace` decides
    # which of the two applies, so this call site does not have to.
    policy_by_cell: dict[tuple[str, str], str] = {}
    for row in rows:
        ckey = (row["benchmark"], row["domain"])
        if ckey not in policy_by_cell:
            policy_by_cell[ckey] = policies.load(*ckey)

    def policy_for(row: dict) -> str:
        if row["benchmark"] not in policies.PER_TRAJECTORY:
            return policy_by_cell[(row["benchmark"], row["domain"])]
        return policies.for_trace(row["benchmark"], row["domain"],
                                  row["agent_model"], row["trace_id"])

    def tally(result) -> None:
        state, verdicts, problems, injected, rechecks = result
        if state == "skipped":
            stats.skipped += 1
            return
        if state == "empty":
            return
        if state == "retried":
            stats.retried += 1
        stats.traces += 1
        stats.verdicts += len(verdicts)
        stats.errors += len(problems)
        stats.repeats_injected += injected
        stats.rechecks_flagged += rechecks
        stats.spans += sum(len(v.spans) for v in verdicts)
        stats.spans_unverified += sum(
            1 for v in verdicts for s in v.spans if not s.verified)
        stats.misaligned += sum(1 for v in verdicts if not v.aligned)
        stats.bad_citations += sum(1 for v in verdicts if not v.citation_ok)
        stats.unmarked_uncited += sum(
            1 for v in verdicts if not v.spans and not v.depended_on_by)
        stats.spans_capped += sum(1 for v in verdicts if v.spans_capped)

    def work(row):
        return _judge_one(judge, row, run_id, policy_for(row), overwrite)

    done = 0
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work, r): r for r in rows}
            for fut in as_completed(futures):
                row = futures[fut]
                try:
                    tally(fut.result())
                except Exception as exc:
                    # One trajectory must not take the cell down. The shard is
                    # absent rather than wrong, so a re-run picks it up.
                    stats.errors += 1
                    print(f"  {row['trace_id'][:16]}: FAILED "
                          f"{exc.__class__.__name__}: {exc}", flush=True)
                done += 1
                if progress and done % 5 == 0:
                    print(f"  {done}/{len(rows)} traces | {stats.verdicts} "
                          f"verdicts | {stats.errors} problems", flush=True)
    else:
        for row in rows:
            tally(work(row))
            done += 1
            if progress and done % 5 == 0:
                print(f"  {done}/{len(rows)} traces | {stats.verdicts} "
                      f"verdicts | {stats.errors} problems", flush=True)
    return stats


def verdicts_for(judge_id: str, run_id: str, agent_model: str, benchmark: str,
                 domain: str, trace_id: str) -> list[EffVerdict]:
    """One trajectory's verdicts, or [] if it has not been judged.

    Empty is a normal answer, not an error: the input-side pass runs over the
    same trajectories whether or not the efficiency pass has reached them.
    """
    path = _shard({"agent_model": agent_model, "benchmark": benchmark,
                   "domain": domain, "trace_id": trace_id}, judge_id, run_id)
    try:
        with path.open() as fh:
            return [EffVerdict.from_json(json.loads(l)) for l in fh if l.strip()]
    except (OSError, ValueError):
        return []


def iter_verdicts(judge_id: str, run_id: str, agent_model: str = "*",
                  benchmark: str = "*", domain: str = "*"):
    root = (paths.RESULTS_OUT / paths.EFFICIENCY_TASK
            / paths.judge_seg(judge_id) / run_id)
    if not root.exists():
        return
    agent_pat = ("*" if agent_model == "*"
                 else paths.agent_seg(agent_model))
    cell_pat = ("*" if benchmark == "*"
                else paths.cell_seg(benchmark, domain if domain != "*" else "all"))
    for path in sorted(root.glob(f"{agent_pat}/{cell_pat}/*.jsonl")):
        parts = path.relative_to(root).parts
        with path.open() as fh:
            rows = [EffVerdict.from_json(json.loads(l)) for l in fh if l.strip()]
        bench, dom = paths.split_cell_seg(parts[1])
        # trace_id comes from the filename, not from the rows: a file whose
        # every verdict errored still needs to be identifiable, and that is
        # exactly the case where the token denominator matters most.
        yield {"agent_model": paths.strip_role(parts[0]), "benchmark": bench,
               "domain": dom, "trace_id": path.stem, "verdicts": rows}
