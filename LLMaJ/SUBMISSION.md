# Judging your own agent with LLMaJ

The step-by-step for a person. `skill.md` on the project page says the same thing to a coding
agent; this file is for doing it by hand, and for running the judge yourself if you prefer not
to send us anything.

Two ways to get the labeled trajectories:

- **Send them to us.** Convert your logs to the judge's format (steps 1-3), upload them to a
  public Hugging Face dataset under your account (step 4), send us the dataset name (step 5).
  We run LLMaJ and send back the labeled trajectories and the numbers; results on the paper's
  four benchmarks are listed on the project page's community table.
- **Run the judge locally** (step 6). Same converted file, your own judge key, nothing leaves
  your machine.

Supported in this version: τ²-bench, Claw-Eval, SWE-bench Verified, MCP-Atlas. More to come.
Other task sets can be converted and judged locally the same way; they need one extra sentence
(step 6).

## 0. Prerequisites

- Python 3.11 or newer, `git`, and for step 4 `pip install -U huggingface_hub`.
- Trajectories from a run of your agent. The judge needs, for every turn, what the agent
  *received* (the user's message or the previous tool result) and what it *produced*
  (reasoning if the model exposes it, the visible reply, the tool calls). A log that keeps only
  the final answer cannot be judged.

```bash
git clone https://github.com/huiminzeng/How_Do_AI_Agents_Waste_Your_Money
cd How_Do_AI_Agents_Waste_Your_Money/LLMaJ
```

The converter and checker (`scripts_convert_trajectories.py`) are plain Python with no
dependencies; `./setup.sh llmaj` from the repository root is only needed for step 6.

## 1. Find the trajectories

| you ran | the file(s) |
|---|---|
| τ²-bench | `data/simulations/<run>/results.json` (one file, every simulation of the run) |
| mini-swe-agent / SWE-bench | one `*.traj.json` per task |
| this repository's proxy | `artifacts/trajectories/.../trace_<id>/calls.jsonl`, one directory per trajectory |
| anything that logs OpenAI-style messages | a JSON list of `{role, content, tool_calls?}`, or an object with a `messages` list; one file per trajectory |

If your harness logs something else, see "The format" below and write the JSONL yourself.

## 2. Convert

```bash
python3 scripts_convert_trajectories.py <file-or-directory> --model <model-name> --benchmark <benchmark>
```

- `--model`: the model as it should be shown, without the provider route (`gpt-6-astra-xhigh`,
  not `openai/gpt-6-astra-xhigh`), with the effort setting as a suffix if the run used one.
  τ²-bench logs name the model themselves; for them `--model` is optional.
- `--benchmark`: `tau2`, `claw_eval`, `swe_verified`, `mcp_atlas`, or your own name for another
  task set. τ²-bench logs carry theirs.

The converter recognises the source by its shape and writes `<benchmark>.jsonl`, or for τ²-bench
`tau2-retail.jsonl` / `tau2-airline.jsonl` (`-o` names the file yourself). For τ²-bench it also
takes the success rate (`solved`), the agent policy (`system_prompt`), the reasoning text and the
reasoning tokens from the provider's raw responses, keeps one trial per task, and drops the
scripted opening greeting, which is not a model call.

It refuses to write when the source does not name the model or the benchmark and you did not
pass them, when a check fails, or when the output still contains an API key. Nothing is sent
anywhere.

What you will see:

```
15 trajectories, 166 turns, from 15 tau2
  note   trajectory 3 (…): no reasoning text in any turn; only the replies and tool calls will be judged
wrote tau2-retail.jsonl (0.3 MB, 15 trajectories): ready for the judge
```

`ERROR` lines must be fixed. `note` lines tell you what the judge will not see: no reasoning
text in the log, no billed token counts (then they are estimated with o200k), a turn without
received text.

## 3. Check

```bash
python3 scripts_convert_trajectories.py tau2-retail.jsonl --check
```

Run this on any file before uploading, including one you wrote yourself. The same checks run in
the browser on the project page (Play with LLMaJ, "Check or convert"), where you can also paste a
raw τ²-bench or messages-list file and download the conversion, if you would rather not clone.

If the converter left out something your log does carry (a system prompt, reasoning text, token
counts), add it back from the log and check again. Adding fields the converter missed is fine;
trimming or rewording the agent's own text is not: a trimmed trajectory judges as less wasteful
than it was.

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
| `benchmark`, `domain` | yes / optional | `tau2` + `retail`/`airline`, `claw_eval`, `swe_verified`, `mcp_atlas`, or your own; `domain` defaults to `all`; no dashes in either |
| `run_id` | optional | names the run; one file holds one trial per task |
| `system_prompt` | optional | lets the judge check "already in the context" against the instructions |
| `solved` | optional | the benchmark's own grade, 0-1, as it is; needed for the success rate |
| `turns[].round` | yes | 1, 2, 3, … in order |
| `turns[].received` | optional | `{role: user|tool, text}`: what the agent had just been given, only what arrived since the previous turn |
| `turns[].reasoning` | optional | the model's chain of thought, verbatim, if exposed |
| `turns[].visible` | optional | the reply the user sees |
| `turns[].tool_calls` | optional | `[{name, arguments}]` |
| `turns[].output_tokens`, `reasoning_tokens` | optional | the provider's billed counts; estimated when absent |

Each turn needs at least one of `reasoning`, `visible` or `tool_calls`.

## 4. Upload to Hugging Face

One dataset per model, public, under your account; one file per benchmark, named as in step 2.

```bash
hf auth login                                                  # your token, write access
hf repos create <user>/<dataset-name> --repo-type dataset      # older versions: hf repo create
hf upload <user>/<dataset-name> tau2-retail.jsonl tau2-retail.jsonl --repo-type dataset
hf upload <user>/<dataset-name> README.md README.md --repo-type dataset
```

The `README.md` says: the model, the benchmark and its version, the harness, how many tasks and
which trial, the date, where the grades come from, and that the trajectories are in the LLMaJ
format. Then open `https://huggingface.co/datasets/<user>/<dataset-name>`, confirm it is public
and lists the files.

Upload only the converted JSONL, never the raw log: τ²-bench's `results.json` carries the user
simulator's provider key and the agent's proxy key, and other harness logs may carry theirs.

## 5. Tell us

Email huiminz3@illinois.edu and lu.yuxuan@northeastern.edu, or post a Discussion on
https://huggingface.co/datasets/huiminz3/How_Do_AI_Agents_Waste_Your_Money:

```
Subject: LLMaJ submission: <model> on <benchmark>

Dataset: https://huggingface.co/datasets/<user>/<dataset-name>
Model: <model name>          Harness: <the harness, or "native">
Benchmark: <benchmark>       Tasks: <how many>, <task sample, e.g. the paper's 100 or all>
Grades: <where the success rate comes from: "solved" in the file / attached report / none>
Name to list: <how you want to be credited>
```

You get back the labeled trajectories (the same JSONL, each turn with the judge's spans, their
kinds and reasons, and what the judge kept) and the numbers: waste ratio, strict waste ratio,
share by kind, per trajectory and for the set. Expect days, not minutes.

## 6. Or run the judge yourself

Needs a judge key. The paper's judge is Kimi-K3 (`KIMI_API_KEY` + `KIMI_BASE_URL`, an
OpenAI-compatible endpoint serving `kimi-k3`); `LLMaJ/proxy/config.yaml` also lists Claude,
GPT and Gemini judges, each needing its provider key. Costs a few dollars per 100 trajectories.

```bash
cd How_Do_AI_Agents_Waste_Your_Money
cp .env.example .env            # PROXY_API_KEY (any string) + the judge's key
./setup.sh llmaj
cd LLMaJ && ./proxy/run.sh      # the judge proxy on :4500, in its own terminal
```

Then, in `LLMaJ/`:

```bash
python3 scripts_import_trajectories.py tau2-retail.jsonl          # writes the trajectories where the judge looks
./run_llmaj.sh <model-name> --judge kimi-k3 --only efficiency --run run1 --re-extract --limit 100
python3 scripts_report_cell.py <model-name> --benchmark tau2 --domain retail --run run1
```

`<model-name>` is the `model` field of your file. The run is resumable: a trajectory whose
verdict exists is skipped. Results land under
`artifacts/llmaj/results/efficiency/kimi-k3_llmaj/run1/<model-name>_agent/<cell>/`:
`<trace_id>.jsonl` are the verdicts (per turn and per stream: the spans with `quote`, `kind`,
`why`, what the judge kept, which later round depends on it), `summary.json` the per-class
shares, `exact_tokens.json` the waste in tokens by kind and by stream, per trajectory and for the
cell (`audited.waste_share` is the waste ratio). `scripts_report_cell.py` prints the cell's
numbers in one object.

For a task set the paper did not run, the judge needs one sentence saying what the agent was
supposed to produce. Put it in `artifacts/llmaj/deliverables.json` at the repository root before
running, keyed by your `benchmark` value:

```json
{"my_tasks": "A patch to the repository that fixes the reported bug."}
```

Two things to know about models the paper did not run: the marked spans are re-tokenised with
each model's own tokenizer, and for a model not in `llmaj/aggregate/exact_tokens.py` the
pipeline counts with o200k and prints a note; and if your model exposes its full chain of
thought (not a provider-written summary), add its name to `RAW_CHAIN` in the same file so the
reasoning is audited and priced like Kimi-K3's.
