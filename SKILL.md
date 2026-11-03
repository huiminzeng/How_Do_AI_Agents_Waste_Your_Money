---
name: llmaj-token-waste
description: Prepare an AI agent's trajectories for LLMaJ, the token-waste judge of the paper "How Do AI Agents Waste Your Money?", upload them to Hugging Face, and submit them for judging. Use when a user wants to know how much of their agent's output is waste.
---

# Send your agent's trajectories to LLMaJ

You are reading this on behalf of a user who ran an AI agent and wants to know how much of
its output was waste. Your job: find their trajectories, convert them to the judge's format,
check them, upload them to a Hugging Face dataset under the user's account, and give the user
the message that submits them. Nothing in this document runs the judge; we run it.

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

1. **You** (the agent) convert the user's trajectories to the judge's JSONL, check them, and
   upload them to a public Hugging Face dataset under the user's account.
2. **The user** sends us the dataset name (by email, or in a Discussion on our dataset page).
3. **We** run LLMaJ on it and send back the labeled trajectories (every span the judge marked,
   with its kind and reason) and the numbers: waste ratio, strict waste ratio, and the share by
   kind. If the trajectories are from one of the supported benchmarks, the result is also
   listed on the project page's community table, as reported, under the user's name.

## Supported benchmarks (v1)

The first version judges trajectories from the four benchmarks of the paper:

| benchmark | `benchmark` value | `domain` |
|---|---|---|
| τ²-bench | `tau2` | `retail` or `airline` |
| Claw-Eval | `claw_eval` | `all` |
| SWE-bench Verified | `swe_verified` | `all` |
| MCP-Atlas | `mcp_atlas` | `all` |

Other benchmarks and task sets: more to come. You may still convert and upload them (use your
own `benchmark` name), and say so in the message; they will be judged when their deliverable
sentence is in place.

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

## Step 3: upload to Hugging Face

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

## Step 4: tell us

Give the user this message to send, filled in:

```
Subject: LLMaJ submission: <model> on <benchmark>

Dataset: https://huggingface.co/datasets/<user>/<dataset-name>
Model: <model name>          Harness: <the harness, or "native">
Benchmark: <benchmark>       Tasks: <how many>, <task sample, e.g. the paper's 100 or all>
Grades: <where the success rate comes from: "solved" in the file / attached report / none>
Name to list: <how the user wants to be credited; their Hugging Face username if you cannot ask>
```

Send it to huiminz3@illinois.edu and lu.yuxuan@northeastern.edu, or post it as a Discussion on
https://huggingface.co/datasets/huiminz3/How_Do_AI_Agents_Waste_Your_Money.

## What comes back

The labeled trajectories (the same JSONL, each turn with the judge's spans, their kinds and
reasons, and what the judge kept), the per-trajectory and per-cell numbers, and a line in the
community table for supported benchmarks. Judging takes a few minutes per trajectory on our
side; expect an answer within days, not minutes.

## Rules

- Upload only converted JSONL, never raw logs (they may contain keys and other secrets).
- One run per task: if the agent ran a task several times, keep one trial per task (the τ²-bench
  converter keeps the first and says so; for other sources, pick one and say which).
- Keep the agent's text complete and unedited: a trimmed trajectory judges as less wasteful than
  it was. Adding back fields the converter missed is not editing.
- Do not run the judge yourself on the user's behalf for a submission; the listed numbers come from our run.
