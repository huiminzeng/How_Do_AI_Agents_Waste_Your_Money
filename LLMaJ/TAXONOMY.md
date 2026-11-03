# Reasoning-State Taxonomy — specification

What this is for: Kimi K3 emitted reasoning text on 25,385 agent turns across four
benchmarks. We know how many tokens it spent thinking; we do not know **what it was
thinking about**. This document defines the label set that answers that, and the graph
built on top of it.

Nothing here is implemented yet. This is the contract that the labelling code and the
analysis code both have to satisfy.

---

## 1. Unit of analysis

**One label per agent turn**, where a turn is one LLM call that produced reasoning text
(`response_message.reasoning_content`). Not per sentence — per turn. Rationale: the
transition graph is over turns, and a per-sentence taxonomy would need a second,
different aggregation step to become comparable to it.

Turns with no reasoning text are **not** labelled and **not** treated as a state. They
are a real phenomenon — 59% of K3's tau2 turns emit only 1–5 structural tokens and no
text — but they carry no content to classify. See §6 for how they affect transitions.

## 2. The eight states

Mutually exclusive. Every labelled turn gets exactly one. Each definition below is
followed by a verbatim excerpt from the actual K3 traces, so the boundaries are drawn
against real text rather than imagined text.

| # | State | Definition |
|---|---|---|
| 1 | `FRAME` | Establishing what the task is and what constrains it: parsing the request, restating it, identifying policy/permission/format requirements that bound the solution. |
| 2 | `PLAN` | Choosing what to do next: strategy, ordering, meta-reasoning about approach. Distinct from FRAME in that FRAME asks *what is being asked*, PLAN asks *what do I do*. |
| 3 | `RETRIEVE` | Deciding to read from the environment, or issuing a read. Read-only intent. |
| 4 | `INTERPRET` | Making sense of what a tool returned. Occurs *after* a result arrives, *before* the next decision. |
| 5 | `DERIVE` | Reasoning to a conclusion from facts already in context — arithmetic, aggregation, inference. **No tool involved.** |
| 6 | `MUTATE` | Changing the environment or producing the deliverable: writes, edits, patches, sending a reply that commits to something. |
| 7 | `VERIFY` | Checking a result against an expectation: running tests, re-reading a value to confirm, sanity-checking a computation. |
| 8 | `RECOVER` | Reacting to something that went wrong: diagnosing a failure, retracting, abandoning an approach for another. |

### Anchored examples (verbatim from traces)

- `FRAME` — tau2 round 1: *"We need respond requiring auth at beginning. User asks
  general/policy + likely wants cancel part pending order. Need ask identity email or
  name+zip."*
- `PLAN` — tau2 round 5: *"We need inspect pending order(s)… Need retrieve #W2166301."*
- `INTERPRET` — mcp_atlas round 6: *"The Purchase History shows various Product
  Categories… The Inventory collection has no records with Inventory Date of
  2015-04-28."*
- `DERIVE` — mcp_atlas round 3: *"Average = sum / 7. Let me compute: 34684.1286761754 +
  35738.4635420804 = …"*
- `RECOVER` — swe_bench round 50: *"It downloaded a cp38 wheel but the system python may
  be a different version… cp38 wheel won't import on a different python. Let me just
  check the source instead."*

### Why these eight and not four

A four-state set (plan / read / write / check) is defensible statistically but was
rejected on the grounds that every benchmark's graph would look alike. The three states
that carry the discriminating power are `INTERPRET`, `DERIVE` and `RECOVER`:

- `INTERPRET` separates *reading* from *understanding what was read*. Collapsing it hides
  the "kept querying without absorbing the answer" pattern.
- `DERIVE` is where tool-free reasoning lives. It is near-absent in tau2 and common in
  mcp_atlas; a merged taxonomy would report that difference as noise.
- `RECOVER` is the direct measure of self-correction, and the state most likely to
  separate models. Folded into `PLAN` it becomes invisible.

### Benchmark-specific behaviour needs no extra states

An earlier draft added one state per benchmark (`POLICY` for tau2, `COMPUTE` for
mcp_atlas, `ENVIRONMENT` for swe_bench). That was dropped: each of those behaviours is
already a clean instance of one of the eight, and per-benchmark state sets would make the
transition matrices non-comparable — which defeats the main purpose.

| Behaviour | Where it lands | Benchmark it is common in |
|---|---|---|
| Reasoning about policy / auth requirements | `FRAME` | tau2 |
| Numeric computation and aggregation | `DERIVE` | mcp_atlas |
| Toolchain / environment troubleshooting | `RECOVER` | swe_bench |

The prediction this makes is falsifiable, which is the point: tau2 should be `FRAME`-heavy,
mcp_atlas `RETRIEVE`/`INTERPRET`-heavy, swe_bench `RECOVER`/`VERIFY`-heavy. If all four
benchmarks come out with the same distribution, the taxonomy is not measuring what it
claims to.

## 3. Boundary rules

Ambiguity is the main threat to inter-judge agreement, so the contested pairs get explicit
tie-breakers. These are decisions, not derivations — they need to be applied uniformly
rather than argued case by case.

1. **`RETRIEVE` vs `MUTATE`** — decided by *intent to change state*, not by which tool
   exists. A read issued in order to prepare a write is still `RETRIEVE`.
2. **`RETRIEVE` vs `INTERPRET`** — if the turn's reasoning is dominated by *what came
   back*, it is `INTERPRET`; if by *what to fetch next*, `RETRIEVE`. When a turn does both
   (common), label by which one the turn's decision hinges on.
3. **`DERIVE` vs `INTERPRET`** — `INTERPRET` reads a tool result at face value; `DERIVE`
   combines or transforms facts to produce something not present in any single result.
4. **`VERIFY` vs `RECOVER`** — `VERIFY` is checking *without* having found a problem;
   once a problem is identified and the turn is about responding to it, it is `RECOVER`.
5. **`FRAME` vs `PLAN`** — `FRAME` is about the problem's constraints, `PLAN` about the
   solver's next move. A turn that restates the task *and* picks a first action is `PLAN`
   (the action is the operative content).
6. **Mixed turns** — label the state that the reasoning is *driving toward*, as expressed
   in the reasoning text itself. Earlier drafts said "label by the outgoing decision";
   that was unusable, because the judge is deliberately not shown the outgoing tool call
   (§3.1). Where the text commits to a next action ("Need retrieve #W2166301"), that
   action's state wins; where it does not, the dominant activity in the text wins.

### 3.1 What the judge is shown

The rule: **the judge sees what the model saw before it reasoned, and nothing that came
after.** Reasoning text alone is not enough — several boundaries in §3 are undecidable
without the preceding observation — but showing the consequences of the turn corrupts the
labels in a different way.

| Shown | Withheld |
|---|---|
| the reasoning text | **the outgoing tool call** |
| the immediately preceding observation (last entry of `request_messages`: a tool result or a user message) | the assistant's visible reply |
| a one-line statement of the task | whether the turn succeeded |
| | **the round number** |
| | **the benchmark's identity** |

Availability is verified, not assumed: `request_messages` is present on **100%** of K3's
reasoning-bearing calls in all four sources, and its last entry is the preceding
observation — median length 192 characters (tau2) to 600 (mcp_atlas), small enough to
include verbatim. Truncate at 1,500 characters head+tail; swe_bench observations have a
long tail of file dumps.

Why each exclusion:

- **Outgoing tool call.** Two reasons. It is reserved as the independent cross-check
  below, which is destroyed if the judge sees it. And a judge that sees the turn end in a
  write will label `MUTATE` regardless of what the reasoning says — that is labelling by
  outcome, not by reasoning, and it would manufacture agreement with the cross-check.
- **Round number.** §4 makes the state distribution's round-dependence a headline result.
  A judge told "this is round 1" has a prior for `FRAME`, and the result becomes partly an
  artefact of the prompt.
- **Benchmark identity.** §2 predicts tau2 is `FRAME`-heavy, mcp_atlas
  `RETRIEVE`/`INTERPRET`-heavy, swe_bench `RECOVER`/`VERIFY`-heavy, and calls that
  prediction falsifiable. Naming the benchmark to the judge makes it self-fulfilling.
  Leakage through the observation text itself (a git diff is recognisably swe_bench)
  cannot be prevented and is accepted; what is avoided is the explicit prime.

Cost of the added context: roughly +150 tokens per turn against a 307-token median
reasoning text, i.e. about +50% on the input side — $28–100 per judge for the full set
rather than $19–67.

### A cross-check the judges do not see

`tool_called` is recorded on every turn independently of any judge. Tool names carry read
/ write semantics (`get_*`, `find_*`, `search_*` are reads; `modify_*`, `cancel_*`,
`update_*`, file edits are writes). This gives an **independent validation signal** for
the `RETRIEVE`/`MUTATE` boundary: a turn labelled `MUTATE` whose tool is `get_user_details`
is a disagreement worth inspecting.

It is deliberately **not** used as a hard constraint. Overriding the judge with the tool
name would inflate apparent agreement while hiding real judge disagreement, and the
mapping is imperfect anyway (a turn can reason about a write and then call a read).

## 4. Multiple judges

Labels are produced by **several different LLMs independently**, not one. Every stored
label therefore carries the judge that produced it, and **no analysis may pool labels
across judges without saying so**.

### Output schema (one row per turn per judge)

```
judge_id           model name + version + effort setting, e.g. "gemini-3-flash-preview"
judge_run_id       one id per (judge, prompt version, batch) — lets a re-label be
                   distinguished from the original
benchmark          tau2 | mcp_atlas | swe_bench | claw_eval
domain             retail | airline | all | general
trace_id           joins back to the trace in artifacts/trajectories/
task_id / unit_id  the benchmark's own task identifier
round              turn index within the trajectory — required, see §5
state              one of the eight
confidence         judge's own confidence, if the judge reports one
rationale          one sentence, for spot-checking and for adjudicating disagreements
prompt_version     hash or tag of the labelling prompt
```

`round` is mandatory because the strongest structural finding available in this data is
round-dependent: K3's share of turns that think at all falls monotonically from 100% at
round 1 to 8% by round 16. Any state distribution reported without conditioning on round
is confounded by that decay.

### Reporting rules

- Per-judge results are the primary output. A cross-judge consensus figure may be
  reported **in addition**, never instead.
- Every graph, table and percentage states its `judge_id` in the caption.
- Agreement between judges is itself a result: report pairwise agreement and a
  chance-corrected statistic (Gwet's AC1 or Krippendorff's α — AC1 is preferred here
  because the state distribution will be skewed, which deflates κ).
- Where judges disagree systematically on a particular boundary, that boundary is a
  finding about the taxonomy, not noise to average away.
- **A judge must not label its own traces.** The traces are Kimi K3's; a K3 judge is
  excluded to avoid self-preference.

## 5. The transition graph

**It is a weighted directed graph, not a DAG.** Agent behaviour contains cycles —
`VERIFY → RECOVER → MUTATE → VERIFY` is the debugging loop — and the cycles are the
informative part. Calling it a DAG would require deleting exactly what we want to see.
The precise object is the transition matrix of a first-order Markov chain, estimated by
counting.

- **Nodes** — the eight states, plus `START` and `END` sentinels.
- **Edges** — `P(next state | current state)`, counted over consecutive rounds within a
  single trajectory.
- **One graph per (judge × benchmark).** Never pooled across judges. There is no model
  axis: kimi-k3 is the only subject (§7), so the comparison this design is built to
  support is **one model across four benchmarks**, not one benchmark across models.

### Sample-size discipline

Available transitions (pairs of consecutive rounds where **both** turns have reasoning
text), K3 only:

Measured by the extraction stage, not estimated:

| benchmark | trajectories | turns | labellable turns | transitions |
|---|---|---|---|---|
| swe_bench | 500 | 16,660 | 16,490 | 15,826 |
| mcp_atlas | 500 | 3,272 | 3,250 | 2,730 |
| claw_eval | 483 | 2,420 | 2,401 | 1,903 |
| tau2/retail | 456 | 5,955 | 2,420 | 1,150 |
| tau2/airline | 200 | 2,378 | 978 | 498 |
| **total** | **2,139** | **30,685** | **25,539** | **22,107** |

The `turns` column is the reason `labellable` is not simply "all turns": tau2/retail emits
5,955 turns but only 2,420 carry reasoning text. claw_eval's counts are still rising — the
18-task rerun is in flight, and this snapshot already includes 154 turns it has produced.

Eight states means 64 possible edges. On tau2/airline that is 7.8 transitions per edge if
they were spread evenly — they will not be; real transition matrices are sparse, so the
~20 edges that carry the main structure should get ~25 each while the tail gets 0–2. The
tail is not estimable and must not be drawn as though it were:

```
n >= 30    solid edge, probability labelled
n = 10-29  dashed edge, probability labelled with n
n < 10     not drawn; counted in an "other" residual so the rows still sum to 1
```

tau2/airline is the binding constraint at 498 transitions, and it gets worse when split
per model. Default to **pooling tau2 retail + airline** (1,648) for graph purposes, and
split them only when the question is specifically about domain differences.

### How the matrix is estimated

Four things that were left implicit and are now fixed, because each one changes the
numbers:

1. **`START` / `END` are the first and last turn *with reasoning text* in a trajectory**,
   not the first and last turn. A trajectory whose opening turns are textless starts at
   its first labelled turn. This follows from §6 — a textless turn breaks the chain, and
   the same rule has to apply at the ends or the sentinels would bridge what the interior
   is forbidden to bridge.
2. **Rows are normalised over observed transitions only**, before any edge is dropped for
   sample size. Dropping first and normalising after would inflate the surviving edges.
3. **The `n < 10` residual is a real row entry, not a rounding remainder.** Each row
   carries an explicit `other` mass equal to the summed probability of its undrawn edges,
   and the row sums to 1 including it. A row whose `other` mass exceeds 0.20 is flagged:
   its structure is mostly in the tail and the drawn edges misrepresent it.
4. **The first-order assumption is checked, not asserted.** Fit a second-order chain on
   the benchmarks that can support it (swe_bench, 15,826 transitions) and compare by BIC.
   If second-order wins decisively, the first-order graph is still reported — it is the
   comparable object — but the finding is recorded, because "the next state depends on
   more than the current one" is itself informative about the reasoning structure.

### Comparing two matrices

The main result in the section above is a comparison of distances, so the distance needs a
definition and a null. Neither existed in earlier drafts, which made "between-benchmark ≫
between-judge" an eyeball claim.

**Distance.** Treat each row of a transition matrix as a probability distribution over the
eight states plus `other`. Take the Jensen-Shannon divergence row by row, then average the
rows weighted by each state's occupancy in the reference matrix. JS is chosen over KL
because it is symmetric and finite when a state is unobserved in one matrix but not the
other — which will happen (`DERIVE` is near-absent in tau2). Occupancy weighting stops a
rarely-visited state's noisy row from dominating.

**Null.** A distance is meaningless without knowing how large it gets from sampling alone.
Bootstrap **over trajectories, not over turns** — turns within a trajectory are dependent,
so resampling turns would understate the variance badly. For each comparison, resample
trajectories with replacement within one (judge, benchmark) cell, recompute the matrix, and
build the distribution of distances between two such resamples. That is the noise floor. A
between-benchmark distance counts only if it sits outside the 95th percentile of that
floor.

**Reporting.** The claim is a comparison of two distributions, not two point estimates:
report between-benchmark and between-judge distances each with bootstrap intervals. If the
intervals overlap, the honest conclusion is that this data cannot separate reasoning
structure from judge disagreement — see the section above, that outcome must be reportable.

### What the graph answers that a phase-proportion table cannot

The proportion view says "33% of turns are MUTATE". The transition view says:

- `P(VERIFY → RECOVER)` — how often checking actually catches something, i.e. whether
  self-correction happens or verification is ceremonial.
- `P(RETRIEVE → RETRIEVE)` — querying without absorbing; a proxy for floundering.
- `P(PLAN → MUTATE)` vs `P(PLAN → RETRIEVE)` — acting before looking.
- Expected path length from `START` to `END`, and how much of it is spent in cycles.

### The question this is actually built to answer

Earlier work here established that **behaviour belongs to the benchmark, not the model**:
persistence does not travel across benchmarks (ρ = 0.02), and cost predictability is ~80%
a property of the benchmark. That was measured from outcome-level statistics — steps,
tokens, cost — which are coarse.

The transition graph tests the same claim on a completely independent signal: the
*internal structure of reasoning*. One model, four environments. If the finding holds,
K3's transition matrix should differ more between benchmarks than the judges differ from
each other on any single benchmark. Concretely:

- **Between-benchmark distance** — the distance between K3's transition matrix on
  swe_bench and on tau2, under one fixed judge.
- **Between-judge distance** — the distance between two judges' matrices on the same
  benchmark.

If between-benchmark ≫ between-judge, reasoning structure is environment-determined and
the earlier finding is confirmed from a second, unrelated direction. If they are
comparable, the graph is measuring judge disagreement rather than model behaviour, and
nothing about the model can be claimed from it. **That second outcome is a real
possibility and the design must be able to report it** — which is why per-judge results
are primary (§4) and why the judges are drawn from three different vendors.

Note this makes the between-judge figure do double duty: it is both a reliability check
on the taxonomy and the denominator for the main result.

Cross-model comparison is deliberately absent. gpt-5.6-terra and gemini-3.1-pro return a
*summary* of their reasoning rather than the chain, and claude-sonnet-5 returns an empty
thinking block with a signature and nothing else (verified against the raw
Anthropic API). Labelling a summary would measure how a model describes its reasoning, not
how it reasons — a different question, and one this taxonomy is not built for.

## 6. Turns without reasoning text

59% of K3's tau2 turns emit 1–5 tokens and no text; on swe_bench it is 1%. These are not
missing data — `reasoning_tokens` is present on 100% of calls, and the 1–5 tokens are the
structural overhead of an empty thinking block. The model chose not to reason.

Consequences:

- They are excluded from labelling (nothing to label).
- A transition is counted **only** when both endpoints have text. A textless turn between
  two labelled turns **breaks** the sequence rather than being bridged, because bridging
  would fabricate an edge that was never taken. The cost is lost sample; the alternative
  is invented structure.
- Their frequency is reported as a first-class statistic per (benchmark, round). "K3 does
  not think at all on 59% of tau2 turns" is a finding, and it is round-dependent.

## 7. Scope and provenance

### Which models can be labelled at all

Measured on the recorded runs, not assumed:

| Model | Route | Reasoning text | Fit for labelling |
|---|---|---|---|
| kimi-k3 | OpenAI-compatible endpoint / SGLang, open weights | **raw chain of thought** | **primary subject** |
| gemini-3.1-pro | Google AI Studio | a *rewritten summary*, not the chain | secondary, with a stated caveat |
| gpt-5.6-terra | OpenAI responses | none (0% of calls) | cannot be labelled |
| gpt-5.6-sol | OpenAI responses | none | cannot be labelled |
| claude-sonnet-5 | Anthropic direct | none (0% of calls) | cannot be labelled |

So the taxonomy is applied to K3. Any gemini figure must be reported separately and
labelled as summary-derived; K3 and gemini rows must never be pooled.

### Evidence that gemini's text is a summary, not the chain

Established using Gemini's own tokenizer (`count_tokens`) against the
provider's own billed counts, on stored `calls.jsonl` rows:

- **Visible text reconciles exactly.** `output_tokens - reasoning_tokens` equals the
  tokenized length of `content` in 12 of 12 sampled calls, difference 0. This
  establishes that the tokenizer and our stored counts are both trustworthy, and is the
  baseline the next line is measured against.
- **Thought text does not reconcile.** On the same calls, the tokenized length of
  `reasoning_content` versus billed `reasoning_tokens` ranges 0.33x–2.36x. It errs in
  *both* directions: long thinking comes back compressed (1966 -> 1025), short thinking
  comes back expanded (86 -> 203).
- **Spread, not median, is the test.** chars-per-billed-token has P95/P5 = 4.1 for
  gemini thought text against 1.5 for its own visible text. A median alone is
  misleading here — the median of that ratio happens to land near 4.
- **The same measure clears K3.** K3 thought text has P95/P5 = 1.8, tighter than its own
  visible-text baseline of 2.5. K3 text length tracks the billed count; gemini's does not.
- **Reproduced directly.** Three identical `include_thoughts=True` calls to
  gemini-3.1-pro returned thought parts of 2080, 2960 and **2** characters while billing
  1059, 1065 and 1196 thinking tokens. The 2-character case is the limit of the same
  behaviour, not a separate bug.
- **The opaque handle.** Gemini attaches a binary `thought_signature` to the *answer*
  part (never to the thought part). It is the multi-turn reasoning-state handle, and it
  is not decodable by us.

Rate in our own data: 0.5% of 5,700 gemini calls bill thinking tokens and return no
thought text at all. That is provider behaviour, not a logging gap — see the direct
reproduction above.

### What this does not affect

`reasoning_tokens`, `output_tokens`, `input_tokens` and cost are unaffected: they are
the provider's own counts, they reconcile, and `content` contains only visible text
(0 of 400 sampled gemini calls had answer text stored into `reasoning_content`; 7 had
overlap because the model quoted its own upcoming reply inside its thoughts). Every
cost- and token-volume figure stands. Only the labelling of gemini
*text* is affected.

### Source of text

- `response_message.reasoning_content` in the proxy's `calls.jsonl`, which also carries
  `round`, `trace_id`, `tool_called` and `trace_meta`.
- Same model and effort across benchmarks; which run each sample came from is recorded,
  because the gateway may change underneath.
