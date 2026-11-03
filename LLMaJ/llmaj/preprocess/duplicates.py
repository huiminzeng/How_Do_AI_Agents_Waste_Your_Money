"""Duplicate tool calls, found mechanically rather than by a judge.

The one waste kind that does not need judgement: if the same call goes out twice
and the same bytes come back, the second response added nothing to the context.
No model opinion is involved, so this is ground truth - and it is deliberately
NOT shown to the judge. `duplicate-fetch` is a span kind the judge is asked to
mark on its own, and keeping this hidden makes it the only kind whose marking can
be scored for recall and precision. Every other kind we can only read.

One exception is built in, and it is the same lesson that put `prev_observation`
into the prompt: a re-fetch the customer asked for is compliance, not waste.
meta-17b470d4 sent get_reservation_details twice, 706 identical characters both
times - and in between the customer said "I'm pretty sure I did purchase the
insurance - can you please double check on your end?". So `strict` requires that
nothing but tool output arrived in between; `all_repeats` keeps the wider set,
because "the agent complied twice" is still worth counting separately.

Measured on kimi-k3/tau2 (656 trajectories, 4,563 calls that got a
response): 32 repeats in 29 trajectories, 27,611 characters of identical response
re-entering the context. 0.7% of calls. Small, and reported anyway - the point is
a list of what was wasted, not a ranking of what wasted most.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schema import Turn

#: Substitutions applied before comparing two responses, for the parts that
#: change without carrying information.
#:
#: Needed because 17 repeats on mcp_atlas were the SAME connection failure twice
#: and differed only in which upstream IP was tried:
#:
#:     Cannot connect to host overpass-api.de:443 ... ('65.109.112.52', 443)
#:     Cannot connect to host overpass-api.de:443 ... ('162.55.144.139', 443)
#:
#: Retrying a service that is down carries no new information, so those are
#: waste, and a byte comparison misses all of them.
#:
#: Deliberately narrow. A looser version that also collapsed any 6+ digit number
#: was tested and matched ZERO additional cases, so it bought nothing and could
#: have erased a real change - an order total, a reservation id, a count. The
#: rule is that a substitution must be provably information-free, not merely
#: convenient.
_NOISE: tuple[tuple[str, str], ...] = (
    (r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<ip>"),
    (r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
     r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?", "<ts>"),
    (r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
     "<uuid>"),
)


def normalise(text: str) -> str:
    """A response with its information-free parts collapsed."""
    for pat, rep in _NOISE:
        text = re.sub(pat, rep, text)
    return " ".join(text.split())


@dataclass
class Repeat:
    """One call that had already been made, with the same result."""

    round: int
    #: The earlier round that made the same call.
    first_round: int
    action: str
    #: Length of the response that came back a second time.
    response_chars: int
    #: True when nothing but tool output arrived between the two calls, so the
    #: repeat cannot be explained as complying with a request.
    strict: bool
    #: Roles seen between the two calls, in order. Kept so a reviewer can see
    #: WHY a repeat was not counted strict without re-deriving it.
    roles_between: tuple[str, ...] = ()
    #: How many turns actually read the repeated response.
    #:
    #: A call sent in round N gets its result back as round N+1's
    #: `prev_observation`, so the response is in the request of rounds N+1
    #: through the last - NOT of round N, where the call was still being
    #: written. Counting from N inclusive overstated a 3-read repeat as 4, a 33%
    #: overstatement of the whole input side on a short trajectory.
    turns_remaining: int = 0

    @property
    def kind(self) -> str:
        """The span kind this repeat should be marked as.

        `duplicate-fetch` when nothing but tool output arrived in between, so
        nobody asked and nothing could have changed. `detour` when the user
        asked: the call complied with a request and could in principle have come
        back different, but the same bytes still re-entered the context and the
        agent already had them. Either way it is marked - the user's point is
        that compliance does not make the waste disappear.
        """
        return "duplicate-fetch" if self.strict else "detour"

    @property
    def context_char_reads(self) -> int:
        """Characters the repeat costs on the INPUT side.

        The response re-enters the context and is read again on every remaining
        turn. Measured on kimi-k3/tau2: 169,903 character-reads against 1,931
        characters of repeated calls, so the input side is 88x the output side.
        Kept strictly separate from the output buckets in `waste_tokens` - the
        denominators are different and a combined figure would mean nothing.
        """
        return self.response_chars * self.turns_remaining

    def reads_note(self) -> str:
        n = self.turns_remaining
        if n <= 0:
            return ("the trajectory ended before the response was read again, "
                    "so the context cost is the one read")
        return (f"re-read on {n} further "
                f"{'turn' if n == 1 else 'turns'} "
                f"({self.context_char_reads:,} character-reads)")

    def why(self) -> str:
        """The sentence shown next to the injected span."""
        asked = ("nothing but tool output arrived in between" if self.strict
                 else "the user asked for it, which does not un-spend the "
                      "tokens")
        return (f"Already called in round {self.first_round}, returning the same "
                f"{self.response_chars:,} characters; {asked}. That response is "
                f"{self.reads_note()}.")


@dataclass
class Recheck:
    """The same call again, with a genuinely different result.

    NOT waste, and not decidable here. Two things look identical to this pass:

        the agent changed something and is verifying it    necessary
        the agent tried a variation and it did not help    ok-to-have, a trial

    Telling those apart needs to read the reasoning around them, so this pass
    only locates them and the efficiency judge decides. A similarity threshold
    was considered and rejected: at 0.92, `get_user_details` called again after a
    cancellation - the reservation list shortens, most of the record is
    unchanged - scores as a near-duplicate, and calling that waste would punish
    exactly the verification we want agents to do.
    """

    round: int
    first_round: int
    action: str
    #: Length of each response, so a reviewer can see how much changed.
    first_chars: int = 0
    chars: int = 0
    #: Same meaning as `Repeat.turns_remaining` - rounds strictly after this one,
    #: which are the rounds whose request carries this response. Needed because
    #: a re-check the judge rules removable costs input exactly as an identical
    #: repeat does, and `context_waste.add_judged_rechecks` charges it.
    turns_remaining: int = 0

    @property
    def context_char_reads(self) -> int:
        """Characters this re-check costs on the INPUT side.

        Same arithmetic as `Repeat.context_char_reads` and deliberately not
        shared with it: the two are counted into separate fields because one is
        a byte comparison and the other is a judgement, and a single property on
        a base class would invite them being summed.
        """
        return self.chars * self.turns_remaining

    def note(self) -> str:
        return (f"Round {self.round} repeats the call from round "
                f"{self.first_round}, and the result came back different "
                f"({self.first_chars:,} then {self.chars:,} characters). Decide "
                f"whether the re-check was needed.")


@dataclass
class Report:
    all_repeats: list[Repeat] = field(default_factory=list)
    #: Same call, different result. For the judge, not for us.
    rechecks: list[Recheck] = field(default_factory=list)


    @property
    def strict(self) -> list[Repeat]:
        return [r for r in self.all_repeats if r.strict]

    @property
    def rounds(self) -> set[int]:
        """Rounds a judge should have marked `duplicate-fetch`."""
        return {r.round for r in self.strict}

    @property
    def repeated_response_chars(self) -> int:
        """Response bytes that came back a second time, counted once each.

        Named away from `waste_chars`, which is the efficiency judge's marked
        OUTPUT characters. These are INPUT characters and the two are never
        summed - different denominators, different prices.
        """
        return sum(r.response_chars for r in self.strict)

    @property
    def context_char_reads(self) -> int:
        """Input-side cost of EVERY repeat, strict or user-asked.

        Both are counted: the user's instruction was that a repeat the customer
        asked for still appended the same bytes to the context window.
        """
        return sum(r.context_char_reads for r in self.all_repeats)

    def by_round(self) -> dict[int, "Repeat"]:
        return {r.round: r for r in self.all_repeats}

    def rechecks_by_round(self) -> dict[int, "Recheck"]:
        return {r.round: r for r in self.rechecks}

    def to_json(self) -> list[dict]:
        return {
            "repeats": [{"round": r.round, "first_round": r.first_round,
                         "action": r.action, "response_chars": r.response_chars,
                         "turns_remaining": r.turns_remaining,
                         "kind": r.kind, "strict": r.strict,
                         "roles_between": list(r.roles_between)}
                        for r in self.all_repeats],
            "rechecks": [{"round": r.round, "first_round": r.first_round,
                          "action": r.action, "first_chars": r.first_chars,
                          "chars": r.chars} for r in self.rechecks],
        }


def find(turns: list[Turn]) -> Report:
    """Repeated calls in one trajectory.

    A call's result is the NEXT turn's `prev_observation`, and only when that
    turn's `prev_role` is "tool" - otherwise the call's result never came back
    inside this trajectory and there is nothing to compare.
    """
    ordered = sorted(turns, key=lambda t: t.round)
    by_round = {t.round: t for t in ordered}

    # (action, normalised response) -> the first round that produced that pair,
    # and action -> the rounds that called it, for the recheck pass.
    seen: dict[tuple[str, str], int] = {}
    by_action: dict[str, list[tuple[int, int]]] = {}
    report = Report()

    for i, turn in enumerate(ordered):
        if not turn.action:
            continue
        nxt = ordered[i + 1] if i + 1 < len(ordered) else None
        if nxt is None or nxt.prev_role != "tool":
            continue
        raw = (nxt.prev_observation or "").strip()
        if not raw:
            continue
        act = turn.action.strip()
        key = (act, normalise(raw))
        prior = by_action.setdefault(act, [])
        first = seen.get(key)
        if first is None:
            seen[key] = turn.round
            # Called before, but the result differs: the judge's case, not ours.
            if prior:
                report.rechecks.append(Recheck(
                    round=turn.round, first_round=prior[-1][0], action=act,
                    first_chars=prior[-1][1], chars=len(raw),
                    turns_remaining=sum(1 for t in ordered
                                        if t.round > turn.round)))
            prior.append((turn.round, len(raw)))
            continue
        prior.append((turn.round, len(raw)))
        roles = tuple(
            by_round[r].prev_role
            for r in range(first + 1, turn.round + 1)
            if r in by_round and by_round[r].prev_role
        )
        report.all_repeats.append(Repeat(
            round=turn.round, first_round=first, action=turn.action,
            response_chars=len(key[1]),
            strict=all(role == "tool" for role in roles),
            roles_between=roles,
            turns_remaining=sum(1 for t in ordered if t.round > turn.round),
        ))
    return report
