"""The three efficiency classes and the rules that separate them.

Parallel to `states.py`, and deliberately independent of it. A turn carries both
a functional label ("what activity is this") and an efficiency label ("was this
worth producing"); neither constrains the other, and they are produced by
separate passes with opposite information requirements:

    functional   one turn at a time, action WITHHELD - the label must not be a
                 restatement of what the turn did next (TAXONOMY.md 3.1)
    efficiency   the whole trajectory at once, action SHOWN - "necessary",
                 "rolled back" and "repeats" are all statements about turns
                 other than this one, and cannot be answered locally

Measured on kimi-k3/swe_bench: literal duplicate commands are 0.1% of
turns, and the three of them that exist are re-running a test after a fix, which
is necessary. So there is no cheap mechanical proxy for "necessary" - hence a
judge, and hence the verbatim-span requirement below as its check.
"""

from __future__ import annotations

MUST = "must-have"
OK = "ok-to-have"
WASTE = "not-necessary"

CLASSES: tuple[str, ...] = (MUST, OK, WASTE)

#: The three streams a turn can contain. A stream the turn does not have is
#: absent from the verdict, never labelled `not-necessary` - gpt-5.6 and
#: claude-sonnet-5 emit no reasoning text at all, and scoring that as waste
#: would report them as infinitely efficient.
STREAMS: tuple[str, ...] = ("reasoning", "visible", "action")

STREAM_DESC: dict[str, str] = {
    "reasoning": "the model's private thinking for this turn",
    "visible": "the text the user would see",
    "action": "the tool calls the turn sent, with their arguments",
}

#: The class is DERIVED; the judge never states it. It marks removable stretches
#: and nothing else, so what it does NOT mark is its claim of necessity - saying
#: nothing asserts that every character was load-bearing, which is visible and
#: checkable against the source. A per-stream label would have nowhere to put
#: "60% of this is the user's own words read back"; the class is a summary we
#: compute from coverage.
DEFINITIONS: dict[str, str] = {
    MUST: (
        "Later work depends on it, and you can name which turn. Remove it and "
        "that turn breaks."
    ),
    OK: (
        "Nothing depends on it, but it could have mattered when it was written - "
        "a trial, an exploration, a route that turned out not to lead anywhere."
    ),
    WASTE: (
        "Nothing depends on it and nothing could have. The information was "
        "already present, or nothing reads it."
    ),
}

#: Answered BEFORE the class, and the class follows from them. Naming a turn is a
#: claim, not a category, and a claim can be checked.
DEPENDS_FIELD = "depended_on_by"
KEPT_FIELD = "kept"
#: Older spelling of the same field, still read.
BREAKS_FIELD = "what_breaks"

#: Rules for MARKING, not for choosing a class. Rule 1 carries the inversion
#: (nothing marked = everything necessary). Rule 8 is why `prev_observation` is
#: in the prompt: a re-fetch the customer asked for is compliance, not waste,
#: however identical the bytes.
BOUNDARY_RULES: tuple[str, ...] = (
    "If nothing at all uses a stream, mark the whole thing and leave "
    "`depended_on_by` empty. That is a finding, not a failure to answer.",

    "Reading the request back is `restated`, even when compressed and even when "
    "it is the opening move. So is quoting the standing instructions back, when "
    "they are shown to you. Summarising in order to reach a conclusion is not: "
    "the conclusion is new, so mark the summary and keep the conclusion.",

    "Applying a rule is not repeating it. \"Basic economy is not refundable\" "
    "copied out of the standing instructions is `restated`; \"this booking is "
    "basic economy, so it is not refundable\" joins the rule to a fact and the "
    "result was in neither. Mark the part that was already written down and keep "
    "the part that was not.",

    "A call repeated with a different result is flagged for you in a note, and "
    "it is yours to rule on. If the agent changed something and is confirming "
    "the change, the re-check is necessary and you mark nothing. If it was "
    "trying a variation that did not help, it is `detour`. The note tells you "
    "the result differed; it does not tell you which of those happened.",

    "Getting the same information twice: the first time stands, the repeats are "
    "`duplicate-fetch`. Re-running a test after an edit is not a repeat - the "
    "earlier result no longer holds. Re-reading a file nobody touched is.",

    "Proximity is not necessity. A comment inside a necessary patch, a preamble "
    "before a necessary edit, a sign-off after a correct answer: the work around "
    "them stays and they go.",

    f"A stream cannot outrank what it serves. Reasoning that sets up a step you "
    f"marked `detour` is itself at most `detour`.",

    "Judge against the task, not against whether the run succeeded - you are not "
    "told whether it did.",

    "A repeat the user asked for is still a repeat, but a milder one. Only a "
    "repeat with nothing but tool output in between is `duplicate-fetch`. When "
    "the customer asked for a double-check in between, the call is `detour`: it "
    "could have come out differently and it complied with a request - but the "
    "same bytes re-entered the context and the agent already had them.",

    "Quote word for word. Quotes are located in the source text, and one that is "
    "not found there is recorded as a fabrication.",
)

#: Spans are the measurement and the whole answer: waste is the summed length
#: of quoted text, counted rather than estimated. 40 because the task is "mark
#: every removable stretch" and one long reasoning block can hold a dozen. A cap
#: that binds is reported, never silently applied: `spans_capped` in the summary.
MAX_SPANS = 40

#: What kind of removable content a span is. Writing too much (`padding`) and
#: doing too much (`detour`) have different fixes, and three further kinds were
#: each found by hand in pilot output:
#:
#:   restated         reasoning that reads the user's own request back ("User
#:                    asks escalate/have someone double-check original booking
#:                    details" against the customer's own words), or quotes the
#:                    standing instructions back.
#:
#:   duplicate-fetch  a call whose response is byte-identical to an earlier one
#:                    (the same get_reservation_details twice, the same 706
#:                    characters back). Also detected mechanically
#:                    (`preprocess/duplicates.py`), so the judge's marking can be
#:                    scored against ground truth.
#:
#:   unread           produced, and then nothing ever reads it.
SPAN_KINDS = ("restated", "duplicate-fetch", "padding", "detour", "unread")
SPAN_KIND_DESC = {
    "restated": "says again what is already in the context - the user's own "
                "words, an earlier tool result, or the agent's own earlier turn",
    "duplicate-fetch": "a call whose result is the same as one already in the "
                       "context, with nothing but tool output in between - "
                       "nobody asked for it and nothing could have changed",
    "padding": "words with no content: filler openers, commentary on style, "
               "sign-offs, restated plans",
    "detour": "a step that need not have been taken at all - a trial, an "
              "exploration, a route later abandoned",
    "unread": "produced, and then nothing ever reads it",
}

#: Severity is DERIVED from kind, not asked for. The user's three-way taxonomy
#: applies to characters now rather than to streams, and the mapping is fixed:
#: `detour` is the one kind that could have paid off when it was written, which
#: is exactly the definition of ok-to-have. The rest could not have.
#:
#: Deriving it removes a second thing for the judge to get wrong, and makes the
#: ok/waste boundary a property of the taxonomy rather than of a judgement call.
KIND_SEVERITY: dict[str, str] = {
    "restated": WASTE,
    "duplicate-fetch": WASTE,
    "padding": WASTE,
    "detour": OK,
    "unread": WASTE,
}

#: Unmarked text is must-have by construction - that is the whole point of the
#: inversion. These thresholds only bucket a stream for the summary table; the
#: continuous `waste_share` is the number that gets reported.
WASTE_SHARE_BUCKETS = ((0.0, MUST), (0.25, OK), (0.75, WASTE))

#: How many characters of each stream the judge must copy back. The ONE check
#: that proves a verdict sits on the turn it was meant for: round numbers are
#: copied by the judge, so a hallucinated round is caught but a SHIFTED array is
#: not - a verdict meant for round 4 arriving labelled round 3 passes every other
#: check, because round 3 exists and its stream is non-empty.
#:
#: Spans could serve the same purpose but are optional: a stream with nothing
#: removable has none, and the first real run produced zero spans across 12
#: verdicts - so nothing at all was verified. An opening is required on every
#: verdict.
#:
#: How many characters the judge is ASKED to copy, and how many are actually
#: COMPARED. They differ on purpose.
#:
#: Asking for exactly N characters puts the cut-off inside a token, and a model
#: generating token by token cannot stop mid-word - it guesses a plausible
#: continuation. Measured across three agents: every failed match
#: diverged at character 37-39 of 40, never earlier.
#:
#:     asked for 40:  'search_direct_flight({"origin":"EWR","da'
#:     actual:        'search_direct_flight({"origin":"EWR","de'
#:                                                          ^ char 39
#:
#: So ask for a few more than are compared, and compare only the part that is
#: safely inside what was asked for. 20 characters still separates any two turns
#: in practice - `search_direct_flight({` and `{"message":"Your` are not close -
#: while leaving five characters of slack at the ragged end.
OPENING_CHARS = 25
MATCH_CHARS = 20

#: How far into the stream the copied opening may start.
#:
#: A judge shown `{"message":"Yes. Please provide your user ID..."}` copies the
#: sentence, not the wrapper - reasonably, since the wrapper is transport, not
#: something a user reads. Requiring a match at position 0 discarded 50 correct
#: verdicts, 16% of gpt-5.6-terra's visible stream, because that model wraps
#: 94-98% of its tau2 replies in {"message": ...}. (Its own doing, not the
#: harness's: the system prompt says "generate valid JSON only", meant for tool
#: calls, and it applied that to prose. On mcp_atlas the same model wraps 0%.)
#:
#: A window rather than a special case for that wrapper: any prefix a judge might
#: skip is covered, and 80 characters is far too small to reach a different turn.
#: Surveyed across all 42 agent/cell/stream combinations - terra on
#: tau2 is the only one that is not plain text.
MATCH_WINDOW = 80

#: Verbatim so the quote can be checked against the source. A quote that is not
#: a substring is a fabrication, and the rate of those is a direct read on how
#: far the rest of the verdict can be trusted.
SPAN_MUST_BE_VERBATIM = True
