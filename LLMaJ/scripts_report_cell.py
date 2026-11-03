#!/usr/bin/env python3
"""Print one cell's numbers as a submission entry for the project page.

    python3 scripts_report_cell.py <model alias> --benchmark swe_bench --run <run id> [--domain all] [--judge kimi-k3]

Reads the cell's exact_tokens.json (written by evaluate.py) and the benchmark's own grades.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llmaj import paths  # noqa: E402
from llmaj.preprocess import grade_sources  # noqa: E402

SHOWN = {"swe_bench": "swe_verified"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--benchmark", required=True); ap.add_argument("--domain", default="all")
    ap.add_argument("--run", required=True); ap.add_argument("--judge", default="kimi-k3"); ap.add_argument("--harness", default="native")
    a = ap.parse_args()
    cell = paths.cell_dir(paths.EFFICIENCY_TASK, a.judge, a.run, a.model, a.benchmark, a.domain)
    if not (cell / "exact_tokens.json").exists():
        sys.exit(f"no judged cell at {cell}\n"
                 f"run the judge first: ./run_llmaj.sh {a.model} --only efficiency --judge {a.judge} --run {a.run}"
                 f" --benchmark {a.benchmark}" + ("" if a.domain == "all" else f" --domain {a.domain}"))
    ex = json.loads((cell / "exact_tokens.json").read_text())
    au = ex["audited"]; kinds = au.get("waste_tokens_by_kind") or {}
    strict = au["waste_tokens"] - kinds.get("detour", 0)
    ids = [p.stem for p in cell.glob("*.jsonl")]
    g = grade_sources.grades(a.model, a.benchmark, a.domain) or {}
    tasks = {}
    for line in open(paths.TRACES_INDEX):
        r = json.loads(line)
        if r["trace_id"] in ids:
            tasks[r["trace_id"]] = str(r.get("task_id"))
    graded = [g[t] for t in tasks.values() if t in g]
    pv = None
    for p in cell.glob("*.jsonl"):
        for line in open(p):
            pv = json.loads(line).get("prompt_version"); break
        if pv: break
    entry = {"model": a.model, "harness": a.harness,
             "benchmark": SHOWN.get(a.benchmark, a.benchmark) + ("" if a.domain == "all" else f"-{a.domain}"),
             "n_trajectories": len(ids),
             "success_rate": round(100 * sum(graded) / len(graded), 1) if graded else None,
             "graded_tasks": len(graded),
             "waste_ratio": round(100 * au["waste_share"], 1),
             "strict_waste_ratio": round(100 * strict / au["billed_tokens"], 1) if au.get("billed_tokens") else None,
             "judge": a.judge, "prompt_version": pv}
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
