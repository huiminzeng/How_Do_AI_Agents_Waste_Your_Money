#!/usr/bin/env python3
"""Replay judged trajectories as the walkthrough page of the project page.

    python scripts_walkthrough.py artifacts/submissions/<user>/<dataset>/output/tau2-retail.labeled.jsonl
    python scripts_walkthrough.py data/kimi-k3-high/native/*.jsonl --limit 20 --out artifacts/walkthrough/kimi

Input: one or more `<benchmark>.labeled.jsonl` files, as `scripts_judge_submission.py`
writes them or as the paper's dataset on Hugging Face ships them. Output: a folder with
`index.html` (open it in a browser; the data is embedded) and `merged.json`.

The page shows one trajectory at a time, chosen from a menu: the transcript with every
removable span highlighted, the floor plan of the task with the agent's route, and the
token curves. It knows the paper's four benchmarks (SWE-bench Verified, tau2 retail and
airline, MCP-Atlas, ClawEval); any other benchmark gets a generic tool floor plan.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from llmaj.walkthrough import build as wt  # noqa: E402

ROOT = HERE.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labeled", nargs="+", type=Path, help="<benchmark>.labeled.jsonl file(s)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default artifacts/walkthrough/<name of the first file>)")
    ap.add_argument("--limit", type=int, default=None, help="at most this many trajectories (default: all)")
    a = ap.parse_args()
    for f in a.labeled:
        if not f.is_file():
            sys.exit(f"not a file: {f}")
    trajs = wt.build(a.labeled, a.limit)
    if not trajs:
        sys.exit("no trajectories in the input")
    name = a.labeled[0].name.replace(".labeled.jsonl", "").replace(".jsonl", "")
    out = a.out or (ROOT / "artifacts" / "walkthrough" / name)
    page = wt.write(trajs, out)
    by_bench = {}
    for t in trajs:
        by_bench[t["bench"]] = by_bench.get(t["bench"], 0) + 1
    print(f"{len(trajs)} trajectories: " + ", ".join(f"{k} {v}" for k, v in sorted(by_bench.items())))
    print(f"open {page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
