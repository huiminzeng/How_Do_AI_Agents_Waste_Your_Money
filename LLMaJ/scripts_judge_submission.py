#!/usr/bin/env python3
"""Judge a submitted Hugging Face dataset: download it, run LLMaJ, write the labeled trajectories.

    python3 scripts_judge_submission.py <user>/<dataset> [--judge kimi-k3] [--run ID] [--limit N] [--workers 3]

The dataset holds trajectories in the judge's JSONL (one file per benchmark, as skill.md
describes). Everything about one submission lives under one directory, apart from this
repository's own artifacts:

    artifacts/submissions/<user>/<dataset>/
      input/      the dataset as downloaded (README.md, *.jsonl)
      work/       LLMaJ's data root for this submission: artifacts/trajectories + artifacts/llmaj
      output/     <file>.labeled.jsonl per input file, summary.json, community_entry.json

The labeled trajectories are in the format of the paper's dataset: each turn carries
`labels.efficiency.<stream>` with the judge's spans (kind, quote, why, verified), what it
kept, which rounds depend on it, and the per-stream waste share; plus `tokens` per turn
(billed and waste tokens by bucket). summary.json has the cell numbers (waste ratio, strict
waste ratio, share by kind, success rate from the file's `solved`), and community_entry.json
is the line to add to the project page's community table.

Needs the judge proxy (LLMaJ/proxy/run.sh) and PROXY_API_KEY, like run_llmaj.sh. The
supported benchmarks are the paper's four; any other `benchmark` value needs a sentence in
--deliverable (what the agent was producing), or the run stops before spending.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from scripts_convert_trajectories import check  # noqa: E402

#: `benchmark` values in a submission -> the trace tag the judge's discovery maps to the paper's cells.
TAGS = {"tau2": "tau2", "claw_eval": "claw_eval", "mcp_atlas": "mcp_atlas",
        "swe_verified": "mini_swe", "swe_bench": "mini_swe"}
#: The name shown on the community table.
SHOWN = {"tau2": "tau2", "claw_eval": "claw_eval", "swe_bench": "swe_verified", "mcp_atlas": "mcp_atlas"}
PYTHON = str(HERE / ".venv/bin/python") if (HERE / ".venv/bin/python").exists() else sys.executable


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def run(cmd: list[str], env: dict, log) -> None:
    log.write("$ " + " ".join(cmd) + "\n"); log.flush()
    p = subprocess.run(cmd, cwd=str(HERE), env=env, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        sys.exit(f"{Path(cmd[1]).name} exited with {p.returncode}; see {log.name}")


def download(repo_id: str, into: Path) -> list[Path]:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id, repo_type="dataset", local_dir=str(into),
                      allow_patterns=["*.jsonl", "*.md", "*.json", "*.txt"])
    files = sorted(p for p in into.rglob("*.jsonl") if ".cache" not in p.parts)
    if not files:
        sys.exit(f"{repo_id}: no .jsonl file in the dataset")
    return files


def prepare(files: list[Path], work: Path, deliverable: str | None, log, do_import: bool = True) -> tuple[str, dict, dict]:
    """Check every file, note the grades, rewrite the benchmark names for the judge, import."""
    from llmaj.evaluate.efficiency import DELIVERABLE
    models, solved, benches = set(), {}, {}
    upload = work / "upload"
    upload.mkdir(parents=True, exist_ok=True)
    for f in files:
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        problems = [m for k, m in sum((check(t, f"{f.name} trajectory {i}") for i, t in enumerate(rows, 1)), []) if k == "bad"]
        if problems:
            sys.exit(f"{f.name}: not in the judge's format:\n  " + "\n  ".join(problems[:10]))
        out = []
        for t in rows:
            models.add(t["model"])
            bench = t["benchmark"]
            tag = TAGS.get(bench, bench)
            dom = t.get("domain") or "all"
            if tag not in TAGS.values() and f"{bench}/{dom}" not in DELIVERABLE and bench not in DELIVERABLE and not deliverable:
                sys.exit(f"{f.name}: benchmark {bench!r} is not one of the supported four; pass --deliverable "
                         f"'<what the agent was producing>' to judge it anyway")
            if t.get("solved") is not None:
                solved[(tag, dom, t["trace_id"])] = float(t["solved"])
            benches[(tag, dom)] = benches.get((tag, dom), 0) + 1
            out.append({**t, "benchmark": tag})
        (upload / f.name).write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in out), encoding="utf-8")
    if len(models) != 1:
        sys.exit(f"one model per dataset: found {sorted(models)}")
    model = models.pop()
    work_llmaj = work / "artifacts" / "llmaj"
    work_llmaj.mkdir(parents=True, exist_ok=True)
    if deliverable:
        (work_llmaj / "deliverables.json").write_text(json.dumps(
            {f"{b}/{d}": deliverable for b, d in benches} | {b: deliverable for b, _ in benches}, indent=2))
    if do_import:
        env = dict(os.environ, LLMAJ_DATA_ROOT=str(work))
        for f in sorted(upload.glob("*.jsonl")):
            run([PYTHON, str(HERE / "scripts_import_trajectories.py"), str(f)], env, log)
    return model, solved, benches


def judge(model: str, args, work: Path, log) -> None:
    env = dict(os.environ, LLMAJ_DATA_ROOT=str(work))
    env.setdefault("LLMAJ_BASE_URL", "http://127.0.0.1:4500/v1")
    cmd = [PYTHON, str(HERE / "evaluate.py"), model, "--only", "efficiency", "--judge", args.judge,
           "--run", args.run, "--limit", str(args.limit), "--no-top-up", "--workers", str(args.workers), "--re-extract"]
    run(cmd, env, log)


def collect(work: Path, model: str, judge_id: str, run_id: str, solved: dict, out_dir: Path,
            repo_id: str, harness: str) -> dict:
    """Join the extracted turns, the verdicts and the token counts into labeled trajectories."""
    traces = work / "artifacts/llmaj/input/traces" / model
    index = work / "artifacts/llmaj/input/traces/index.jsonl"
    task_of = {}
    for line in (index.open(encoding="utf-8") if index.exists() else []):
        r = json.loads(line)
        task_of[r.get("trace_id")] = r.get("task_id")
    results = work / "artifacts/llmaj/results/efficiency" / f"{judge_id}_llmaj" / run_id / f"{model}_agent"
    summary = {"dataset": repo_id, "model": model, "harness": harness, "judge": judge_id, "run": run_id,
               "prompt_version": None, "cells": []}
    entries = []
    for cell_dir in sorted(p for p in traces.iterdir() if p.is_dir()):
        cell = cell_dir.name
        bench, _, dom = cell.partition("-")
        dom = dom or "all"
        rdir = results / cell
        labels: dict[str, dict[int, dict]] = {}
        problems: dict[str, list[str]] = defaultdict(list)
        for shard in rdir.glob("*.jsonl") if rdir.exists() else []:
            for line in shard.open(encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                summary["prompt_version"] = summary["prompt_version"] or r.get("prompt_version")
                if r.get("error"):
                    problems[shard.stem].append(r["error"]); continue
                if not r.get("aligned"):
                    problems[shard.stem].append(f"round {r.get('round')} {r.get('stream')}: verdict not aligned"); continue
                labels.setdefault(r["trace_id"], {}).setdefault(int(r["round"]), {})[r["stream"]] = {
                    "spans": [{"kind": s.get("kind"), "quote": s.get("quote"), "why": s.get("why"),
                               "verified": s.get("verified")} for s in (r.get("spans") or [])],
                    "kept": r.get("kept"), "what_breaks": r.get("what_breaks"),
                    "depended_on_by": r.get("depended_on_by"),
                    "waste_chars": r.get("waste_chars"), "waste_share": r.get("waste_share")}
        tokens: dict[str, dict[int, dict]] = {}
        audited, notes = {}, []
        et = rdir / "exact_tokens.json"
        if et.exists():
            d = json.loads(et.read_text())
            audited = d.get("audited") or {}
            notes = [k for k, v in (d.get("notes") or {}).items() if v]
            for rd in d.get("rounds") or []:
                tokens.setdefault(rd["trace_id"], {})[int(rd["round"])] = rd.get("buckets") or {}
        shown = SHOWN.get(bench, bench) + ("" if dom == "all" else f"-{dom}")
        out_file = out_dir / f"{shown}.labeled.jsonl"
        n = 0
        grades = []
        with out_file.open("w", encoding="utf-8") as fh:
            for shard in sorted(cell_dir.glob("*.jsonl")):
                tid = shard.stem
                if tid not in labels:
                    continue          # not judged (outside --limit, or no usable verdict)
                turns = []
                for line in shard.open(encoding="utf-8"):
                    if not line.strip():
                        continue
                    t = json.loads(line)
                    rd = int(t["round"])
                    turns.append({"round": rd, "reasoning": t.get("reasoning_content") or "",
                                  "visible": t.get("visible_text") or "", "action": t.get("action") or "",
                                  "received": {"role": t.get("prev_role"), "text": t.get("prev_observation")},
                                  "output_tokens": t.get("output_tokens"), "reasoning_tokens": t.get("reasoning_tokens"),
                                  "labels": {"efficiency": labels[tid].get(rd, {})},
                                  "tokens": tokens.get(tid, {}).get(rd, {})})
                task_id = task_of.get(tid)
                s = solved.get((bench, dom, tid))
                if s is not None:
                    grades.append(s)
                fh.write(json.dumps({"trace_id": tid, "model": model, "harness": harness, "benchmark": SHOWN.get(bench, bench),
                                     "domain": dom, "task_id": task_id, "solved": s, "turns": turns,
                                     "problems": problems.get(tid, [])}, ensure_ascii=False) + "\n")
                n += 1
        by_kind = audited.get("waste_tokens_by_kind") or {}
        billed = audited.get("billed_tokens") or 0
        strict = sum(v for k, v in by_kind.items() if k != "detour")
        cell_summary = {"benchmark": shown, "n_trajectories": n, "audited_tokens": billed,
                        "waste_tokens": audited.get("waste_tokens"), "waste_ratio": audited.get("waste_share"),
                        "strict_waste_ratio": (strict / billed) if billed else None,
                        "waste_tokens_by_kind": by_kind, "waste_dollars": audited.get("waste_dollars"),
                        "success_rate": (sum(grades) / len(grades)) if grades else None, "graded": len(grades),
                        "notes": notes, "labeled_file": out_file.name}
        summary["cells"].append(cell_summary)
        entries.append({"date": dt.date.today().isoformat(), "model": model, "harness": harness, "benchmark": shown,
                        "success_rate": None if not grades else round(100 * sum(grades) / len(grades), 1),
                        "waste_ratio": None if audited.get("waste_share") is None else round(100 * audited["waste_share"], 1),
                        "strict_waste_ratio": None if not billed else round(100 * strict / billed, 1),
                        "n_trajectories": n, "judge": judge_id, "prompt_version": summary["prompt_version"],
                        "submitter": repo_id.split("/")[0], "url": f"https://huggingface.co/datasets/{repo_id}"})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "community_entry.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="<user>/<dataset> on Hugging Face")
    ap.add_argument("--judge", default="kimi-k3")
    ap.add_argument("--run", default=None, help="run id (default: today's date)")
    ap.add_argument("--limit", type=int, default=200, help="trajectories per cell (default 200: all of a submission)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--harness", default="native", help="how the entry is labeled (default native)")
    ap.add_argument("--deliverable", default=None, help="one sentence for a benchmark the paper did not run")
    ap.add_argument("--skip-download", action="store_true", help="reuse input/ from an earlier run")
    ap.add_argument("--no-judge", action="store_true", help="download, check and import only")
    ap.add_argument("--collect-only", action="store_true", help="rebuild output/ from an earlier judge run")
    a = ap.parse_args()
    a.run = a.run or dt.date.today().strftime("%m%d")
    load_env()
    user, _, name = a.dataset.partition("/")
    if not name:
        sys.exit("dataset must be <user>/<dataset>")
    base = ROOT / "artifacts" / "submissions" / user / name
    inp, work, out = base / "input", base / "work", base / "output"
    for d in (inp, work, out):
        d.mkdir(parents=True, exist_ok=True)
    log = (base / "log.txt").open("a", encoding="utf-8")
    log.write(f"\n==== {dt.datetime.now():%Y-%m-%d %H:%M} {a.dataset} judge={a.judge} run={a.run}\n")

    files = (sorted(p for p in inp.rglob("*.jsonl") if ".cache" not in p.parts)
             if (a.skip_download or a.collect_only) else download(a.dataset, inp))
    print(f"input   {len(files)} file(s): " + ", ".join(f.name for f in files))
    model, solved, benches = prepare(files, work, a.deliverable, log, do_import=not a.collect_only)
    print(f"model   {model}   cells " + ", ".join(f"{b}/{d} ({n})" for (b, d), n in benches.items())
          + f"   grades for {len(solved)} trajectories")
    if a.no_judge:
        print(f"imported under {work}; not judged (--no-judge)")
        return 0
    if not a.collect_only:
        judge(model, a, work, log)
    summary = collect(work, model, a.judge, a.run, solved, out, a.dataset, a.harness)
    for c in summary["cells"]:
        wr = c["waste_ratio"]; sr = c["success_rate"]
        print(f"{c['benchmark']:<16} {c['n_trajectories']:>3} judged   waste {wr * 100 if wr is not None else float('nan'):5.1f}%"
              f"   strict {((c['strict_waste_ratio'] or 0) * 100):5.1f}%   success "
              f"{'-' if sr is None else f'{sr * 100:.1f}%'}   -> output/{c['labeled_file']}")
    print(f"summary {out / 'summary.json'}\nentry   {out / 'community_entry.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
