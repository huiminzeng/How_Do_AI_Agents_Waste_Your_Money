# Community results: asking for a complete benchmark run to be evaluated

The [project page](https://agenteconomics.org/posts/how-agents-waste-money/play) has a table of
community results: complete benchmark runs of other people's agents, judged with the LLMaJ of
this repository and listed as reported. This folder is how a run gets there.

You open a pull request. **The pull request is a request, not the data**: it adds one small
file that says what you ran and where the trajectories are. The trajectories themselves stay in
a public Hugging Face dataset under your account. We run the judge.

This is separate from *Play with LLMaJ* on the project page, where anyone can have a sample of
up to 32 trajectories judged and get a private results page. Those samples are not listed.

## What qualifies

- **A complete run of one benchmark**: every task of the benchmark's task set, or the task
  sample the paper used. Say which. A handful of tasks is a sample; use *Play with LLMaJ* for that.
- **One of the paper's benchmarks**: τ²-bench (`tau2`, domain `retail` or `airline`),
  Claw-Eval (`claw_eval`), SWE-bench Verified (`swe_verified`), MCP-Atlas (`mcp_atlas`).
- **One model and one harness per request.** Two models, or the same model under two harnesses,
  are two requests.
- **One trial per task.** If the agent ran a task several times, keep one and say which.
- **The benchmark's own grade for every task** (`solved` in the file), so that the success
  rate can be shown beside the waste ratio.
- **Unedited.** The agent's text is complete and as it was produced: a trimmed trajectory
  judges as less wasteful than it was.

## Steps

1. **Convert and check your trajectories**, as [SKILL.md](../SKILL.md) (for your coding agent) or
   [LLMaJ/SUBMISSION.md](../LLMaJ/SUBMISSION.md) (by hand) describes:

   ```bash
   cd LLMaJ
   python3 scripts_convert_trajectories.py <file-or-directory> --model <model-name> --benchmark <benchmark>
   python3 scripts_convert_trajectories.py <the written file> --check
   ```

2. **Upload them to a public Hugging Face dataset** under your own account: one file per
   benchmark cell (`tau2-retail.jsonl`, `swe_verified.jsonl`, ...), plus a `README.md` saying what
   the run was. Converted JSONL only, never raw logs: they can contain API keys. The dataset must
   be public and not gated.

3. **Add one file to this folder**: copy [`submissions/TEMPLATE.json`](submissions/TEMPLATE.json) to
   `submissions/<model>_<organization>_<date>.json` (lower case, no spaces; for example
   `my-agent-v2_example-lab_2026-11-19.json`) and fill it in. Nothing else goes in the pull request:
   no trajectories, no results.

4. **Open the pull request.** In its description say, in a sentence or two, what the agent is,
   and anything a reader of the number should know: changes to the benchmark or the harness, tasks
   left out and why.

## The file

| field | |
|---|---|
| `model` | The model as it should be shown, without the provider route, with the effort setting as a suffix when the run used one (`-low`, `-high`, `-max`). The same string as `model` in the trajectories. |
| `model_organization` | Who made the model. |
| `submitting_organization` | Who ran it: a lab, a company, or your name. Shown in the table's "by" column. |
| `harness` | What drove the model: `native` for the benchmark's own agent loop, otherwise the harness's name. |
| `benchmark`, `domain` | As in the trajectories. `domain` is `retail` or `airline` for `tau2`, `all` otherwise. |
| `task_set` | Which tasks, in words: `"all 114 tasks"`, `"the paper's sample of 100"`. |
| `trajectories` | How many trajectories the dataset holds for this cell. |
| `dataset` | The dataset's address: `https://huggingface.co/datasets/<user>/<name>`. |
| `dataset_revision` | The commit of the dataset to judge (the 40-character hash on its "Files" tab). The dataset can change afterwards; the number is tied to this revision. |
| `submission_date` | The day you open the pull request, `YYYY-MM-DD`. |
| `references` | Optional: links to the model, a paper, the code of the harness. |
| `notes` | Optional: anything that changes what the number means. |

There is no contact field. The pull request is where we reach you.

## What happens next

We check the file and the dataset: the format, that the run is complete, one trial per task,
no credentials in the text. If something has to be fixed we say so in the pull request. Then
we run LLMaJ on that revision of the dataset, with the judge and prompts of the paper, and post
the numbers in the pull request: waste ratio, strict waste ratio, share by kind, success rate.
When you agree they are what should be listed, the pull request is merged and the line is
added to the table.

Results are listed as reported and are not merged into the paper's figures: a different task
sample or harness changes what the number means.

This is a request to a small research team. We take runs on as we can, and may decline one
that does not qualify or that we cannot afford to judge.
