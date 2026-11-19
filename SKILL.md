---
name: llmaj-token-waste
description: Prepare an AI agent's trajectories for LLMaJ, the token-waste judge of the paper "How Do AI Agents Waste Your Money?": find them, convert them to the judge's format, check them, and get them ready for the user to submit. Use when a user wants to know how much of their agent's output is waste.
---

# Send your agent's trajectories to LLMaJ

You are reading this on behalf of a user who ran an AI agent and wants to know how much of
its output was waste. Your job: find their trajectories, convert them to the judge's format,
check them, and leave the user with exactly what they need to submit on the project's page.
Nothing in this document runs the judge; we run it.

**You do not submit for the user.** Submitting means agreeing to terms about their data and
answering a captcha, on https://agenteconomics.org/posts/how-agents-waste-money/play. That is theirs to do. You prepare; they submit.

## What this is

The paper *How Do AI Agents Waste Your Money?* (arXiv:2611.07238, https://arxiv.org/abs/2611.07238; project page https://agenteconomics.org/posts/how-agents-waste-money) measures **token waste**: the
spans of an agent's output that could be deleted without changing the outcome. An
LLM-as-a-Judge (LLMaJ) reads a whole trajectory and marks four kinds of removable span:
*restated* (repeats what is already in the context), *duplicate-fetch* (a tool call whose
result is already there), *padding* (words with no content) and *detour* (a step that need
not have been taken). The marked spans are re-tokenised against the billed tokens, so the
result is a waste ratio in tokens and dollars, per trajectory and per model.

Code: https://github.com/huiminzeng/How_Do_AI_Agents_Waste_Your_Money. The paper's own labeled trajectories: https://huggingface.co/datasets/huiminz3/How_Do_AI_Agents_Waste_Your_Money.

## The workflow

1. **You** (the agent) find the user's trajectories, convert them to the judge's JSONL and check them.
2. **You** count them. Up to 5 trajectories: the user uploads the file on the page. More than 5: you
   upload them to a public Hugging Face dataset under the user's account, and the user gives the
   page its name.
3. **The user** submits on https://agenteconomics.org/posts/how-agents-waste-money/play: they read the terms and agree to them.
4. **We** run LLMaJ. The page gives the user a private link the moment they submit; it shows
   where the request stands, and then their results page: every trajectory
   replayed turn by turn with the spans the judge marked, the numbers (waste ratio, strict waste
   ratio, success rate), and a download of the labeled trajectories.

## Limits

- **32 trajectories per person**, in total, and 3 million characters in total. Tell the user if they
  have more, and ask which to send.
- **One trajectory is one run of one task**: at most 100 turns and 500,000 characters (the system
  prompt plus, per turn, what it received, its reasoning, its reply and its tool calls). A longer run
  is refused; do not split it, cut it, or join several runs into one.
- **5 trajectories** in one upload from the page; more go through a Hugging Face dataset.
- **One model per request.** Trajectories of two models are two requests.
- Results are kept for **30 days**, then deleted.

## Supported benchmarks (v1)

The first version judges trajectories from the four benchmarks of the paper:

| benchmark | `benchmark` value | `domain` |
|---|---|---|
| τ²-bench | `tau2` | `retail` or `airline` |
| Claw-Eval | `claw_eval` | `all` |
| SWE-bench Verified | `swe_verified` | `all` |
| MCP-Atlas | `mcp_atlas` | `all` |

Other benchmarks and task sets: more to come. For now a request with another `benchmark` value is
refused when it is submitted, so tell the user rather than preparing one.

## Step 1: find the trajectories

Ask the user where the agent's run left its logs; if you cannot ask, look in the usual places
(the harness's output directory, `data/simulations/`, `*.traj.json`, `artifacts/trajectories/`) and
say which file you took. Recognised sources:

- **τ²-bench**: `data/simulations/<run>/results.json` (one file, many simulations). Carries the
  task id, the reward, the agent policy and the provider's raw responses; the converter takes the
  success rate, the system prompt, the reasoning text and the reasoning tokens from them, keeps
  one trial per task, and drops the scripted opening greeting, which is not a model call.
- **mini-swe-agent / SWE-bench**: one `*.traj.json` per task, a list of `{role, content}` messages.
- **The paper's tracing proxy**: `artifacts/trajectories/.../trace_<id>/calls.jsonl`, one
  directory per trajectory.
- **Anything else that logs OpenAI-style messages**: a JSON list of `{role, content,
  tool_calls?}`, or an object with a `messages` list. One file per trajectory.

The judge needs, per turn, what the agent *received* (the user's message or the previous tool
result) and what it *produced* (reasoning if the model exposes it, the visible reply, the tool
calls). Logs that only keep the final answer cannot be judged.

## Step 2: convert and check

Clone the repository and run the converter. It recognises the source by its shape, writes one
JSONL file in the judge's format, and reports anything that would stop the judge. Nothing is
sent anywhere.

```bash
git clone https://github.com/huiminzeng/How_Do_AI_Agents_Waste_Your_Money
cd How_Do_AI_Agents_Waste_Your_Money/LLMaJ
python3 scripts_convert_trajectories.py <file-or-directory> --model <model-name> --benchmark <benchmark>
python3 scripts_convert_trajectories.py <the written file> --check
```

The converter writes `<benchmark>.jsonl`, or `<benchmark>-<domain>.jsonl` for τ²-bench
(`tau2-retail.jsonl`, `tau2-airline.jsonl`); `-o` names the file yourself. It refuses to write when
the source does not name the model or the benchmark and you did not pass them, when a check
fails, or when the output still contains an API key.

`--model` is the model as it should be shown: the model name without the provider route
(`gpt-6-astra-xhigh`, not `openai/gpt-6-astra-xhigh`), with the effort setting kept as a suffix
when the run used one (`-low`, `-high`, `-max`). `--benchmark` is one of the values in the table
above, or the user's own name for another task set; τ²-bench logs carry theirs. Read the output:
`ERROR` lines must be fixed; `note` lines tell the user what the judge will and will not see
(no reasoning text in the log, no billed token counts, a turn without received text).

If the converter does not recognise the user's format, write the JSONL yourself from the spec
below, then run `--check` on it. If the converter leaves out something the log does carry (a
system prompt, reasoning text, token counts), add it back from the log and run `--check` again:
adding fields the converter missed is fine, trimming or rewording the agent's own text is not.

### The format

One JSON object per line, one trajectory per object:

```json
{"trace_id": "t-0001", "task_id": "task-17", "model": "my-model", "benchmark": "tau2", "domain": "retail",
 "run_id": "run1", "system_prompt": "<the agent's standing instructions, if any>", "solved": 1,
 "turns": [
   {"round": 1, "received": {"role": "user", "text": "Hi, I need to change the delivery address for an order."},
    "reasoning": "<chain of thought, or empty if the model exposes none>",
    "visible": "<the reply the user sees, or empty>",
    "tool_calls": [{"name": "find_user_id_by_email", "arguments": {"email": "a@b.c"}}],
    "output_tokens": 61, "reasoning_tokens": 27},
   {"round": 2, "received": {"role": "tool", "text": "<what the tool returned>"},
    "reasoning": "...", "visible": "", "tool_calls": [{"name": "modify_pending_order_address", "arguments": {}}]}
 ]}
```

| field | required | meaning |
|---|---|---|
| `trace_id` | yes | unique within the file |
| `task_id` | yes | the benchmark's task id |
| `model` | yes | the model as it should be shown |
| `benchmark`, `domain` | yes / optional | see the table above; `domain` defaults to `all` |
| `run_id` | optional | names the run; one dataset should hold one run per task |
| `system_prompt` | optional | lets the judge check "already in the context" against the instructions |
| `solved` | optional | the benchmark's own grade, 0-1, passed through as it is (fractional grades included); needed for the success rate |
| `turns[].round` | yes | 1, 2, 3, … in order |
| `turns[].received` | optional | `{role: user|tool, text}`, what the agent had just been given |
| `turns[].reasoning` | optional | the model's chain of thought, verbatim, if exposed |
| `turns[].visible` | optional | the reply the user sees |
| `turns[].tool_calls` | optional | `[{name, arguments}]` |
| `turns[].output_tokens`, `reasoning_tokens` | optional | the provider's billed counts; estimated with o200k when absent |

Each turn needs at least one of `reasoning`, `visible` or `tool_calls`. Do not paste the whole
conversation history into every turn: `received` is only what arrived since the previous turn.
`benchmark` and `domain` never contain a dash; the dash is only in the file name.

## Step 3: more than 5 trajectories? Upload to Hugging Face

Skip this step when there are 5 or fewer: the user uploads the JSONL file on the page.

The dataset is the user's: public, under their account, one dataset per model, one file per
benchmark named `<benchmark>.jsonl` (for τ², `tau2-retail.jsonl` / `tau2-airline.jsonl`).

```bash
pip install -U huggingface_hub
hf auth login                      # the user's token, write access; `hf auth whoami` shows who is logged in
hf repos create <user>/<dataset-name> --repo-type dataset     # older versions: hf repo create
hf upload <user>/<dataset-name> <benchmark>.jsonl <benchmark>.jsonl --repo-type dataset
hf upload <user>/<dataset-name> README.md README.md --repo-type dataset
```

`README.md` says: the model, the benchmark and its version, the harness, how many tasks and
which trial, the date, where the grades come from, and that the trajectories are in the LLMaJ
format of the paper. Then check the result: open `https://huggingface.co/datasets/<user>/<dataset-name>`,
confirm it is public and lists the files, and download one file back and run `--check` on it.

Upload only converted JSONL, never the raw log. τ²-bench's `results.json` carries the user
simulator's provider key and the agent's proxy key and address; other harness logs may carry
theirs. The converter refuses to write an output that still contains a key, and the converted
files are what goes up.

## Step 4: hand over to the user

Tell the user, in these words or close to them:

- **5 trajectories or fewer:** "Your trajectories are ready in `<path to the .jsonl>`. Open
  https://agenteconomics.org/posts/how-agents-waste-money/play, choose *upload them here*, and pick that file."
- **More than 5:** "Your trajectories are in the dataset
  `https://huggingface.co/datasets/<user>/<dataset-name>`. Open https://agenteconomics.org/posts/how-agents-waste-money/play, choose *a Hugging Face
  dataset*, and paste that address."

Then tell them what the page will ask, so nothing surprises them: an email address (it counts
their trajectories against the per-person limit; nothing is emailed to it); terms to read and agree to (their trajectories' text is sent to the judge model, a third
party; the results page is unlisted but anyone with its link can open it; everything is deleted
after 30 days); and one optional permission (use of the trajectories for research).

Also tell them how many trajectories there are, which model and benchmark, and anything the
converter's `note` lines said the judge will not be able to see.

## What comes back

A private link, shown on the page the moment the user submits. Nothing is emailed, so tell them
to save it. It first shows where the request stands, then the results: each trajectory replayed turn
by turn: the transcript with every removable span highlighted and the judge's reason for it,
the task as a floor plan, the type of each action and each step of reasoning, and the token
curves. Its download button gives the labeled trajectories: the same JSONL, each turn with the
judge's spans, their kinds and reasons, and what the judge kept, plus the per-cell numbers.

Judging takes a few minutes per trajectory and requests are judged in the order they
arrive, so expect results within hours, not minutes. Results submitted this way are a
sample and are not listed in the community table on the project page. That table is for complete
benchmark runs, which are requested with a pull request instead: see
https://github.com/huiminzeng/How_Do_AI_Agents_Waste_Your_Money/blob/main/community/README.md.

## Step 5 (optional): replay the result on the user's own machine

The results page already shows the walkthrough. If the user wants it locally, or after the 30
days are over, build it from the labeled file they downloaded:

```bash
cd How_Do_AI_Agents_Waste_Your_Money/LLMaJ
python scripts_walkthrough.py /path/to/<benchmark>.labeled.jsonl
```

It prints the path of an `index.html` (under `artifacts/walkthrough/<name>/`); open it in a
browser. It needs no API key and nothing but Python. One trajectory shows at a time, chosen
from the menu at the top; `--limit N` keeps the page small for a large file. The floor plan
knows the four supported benchmarks; another benchmark gets a generic plan.

## Rules

- Do not submit on the user's behalf, and do not agree to the terms for them.
- Upload only converted JSONL, never raw logs (they may contain keys and other secrets). A file
  that still contains an API key is refused.
- No personal information about other people in what is submitted. If the trajectories contain
  some (real names, addresses, account details of real customers), tell the user before they submit.
- One run per task: if the agent ran a task several times, keep one trial per task (the τ²-bench
  converter keeps the first and says so; for other sources, pick one and say which).
- Keep the agent's text complete and unedited: a trimmed trajectory judges as less wasteful than
  it was. Adding back fields the converter missed is not editing.
- Do not run the judge yourself on the user's behalf for a submission; the listed numbers come from our run.
