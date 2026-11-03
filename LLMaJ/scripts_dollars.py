"""Step 2 of 4 after a re-count: rebuild every cell's dollars.json from its exact_tokens.json.

    python3 scripts_dollars.py                        # every cell
    python3 scripts_dollars.py gpt-6-astra/mcp_atlas  # cells whose agent/bench/domain contains this

Order: scripts_exact.py (1) -> scripts_dollars.py (2).
The front page is not a file and needs no step: the server renders it from these artefacts on every request.

The counted tokens are the only input to the dollar figures. No judge, no token
counts, no network: it multiplies counts by the cell's own output rate and reads
the audited trajectories' recorded spend.
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llmaj import paths
from llmaj.aggregate import dollars

ONLY = sys.argv[1] if len(sys.argv) > 1 else None   # optional substring filter, e.g. gpt-6-astra/mcp_atlas
n = 0
for f in sorted(glob.glob('work/results/efficiency/*/*/*/*/exact_tokens.json')):
    d = os.path.dirname(f)
    ex = json.load(open(f))
    if ONLY and ONLY not in f"{ex['agent_model']}/{ex['benchmark']}/{ex['domain'] or ''}":
        continue
    judge = paths.strip_role(f.split('/')[3]); run = f.split('/')[4]
    ids = {'meta-' + x.split('meta-')[1].split('.')[0] for x in glob.glob(os.path.join(d, 'meta-*.jsonl'))}
    w = dollars.build(ex['agent_model'], ex['benchmark'], ex['domain'] or '', ids, exact=ex)
    dollars.save(w, judge, run, ex['agent_model'], ex['benchmark'], ex['domain'] or '')
    o = w.output.get('visible+action', 0.0); r = w.output.get('reasoning')
    print(f"  {ex['agent_model']}/{ex['benchmark']}/{ex['domain'] or '-'}  v+a waste ${o:.4f}" + (f"  reasoning ${r:.4f}" if r is not None else ""), flush=True)
    n += 1
print(f"\n{n} dollars.json rebuilt")
