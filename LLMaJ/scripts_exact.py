"""Step 1 of 4 after a code change: count exact waste tokens for every judged cell
and write exact_tokens.json.

    python3 scripts_exact.py                 # every cell
    python3 scripts_exact.py gpt-6-astra/mcp_atlas   # cells whose agent/bench/domain contains this

Then scripts_dollars.py (2).
The front page needs none of them: it renders from the files these write.

Reads the verdicts already on disk. No judge, no re-run: the only calls it can
make are token counts, and those are cached by text so a second pass is free.
"""
import os, sys, json, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llmaj import paths as _paths
for line in open(_paths.PROJECT_ROOT / '.env'):
    if '=' in line and not line.strip().startswith('#'):
        k, _, v = line.strip().partition('=')
        os.environ.setdefault(k, v.strip().strip('"\''))
from llmaj import paths
from llmaj.aggregate import exact_tokens as ex
from llmaj.evaluate.efficiency import verdicts_for
from llmaj.preprocess.sample import iter_index, load_turns

JUDGE = 'kimi-k3'
index = {r['trace_id']: r for r in iter_index()}
cells = []
for f in sorted(glob.glob('work/results/efficiency/*/*/*/*/summary.json')):
    s = json.load(open(f))
    cells.append((f.split('/')[4], s['agent_model'], s['benchmark'],
                  s['domain'] or '', os.path.dirname(f)))
counters = {}
ONLY = sys.argv[1] if len(sys.argv) > 1 else None   # optional substring filter, e.g. gpt-5.6-terra/claw_eval
for run, agent, bench, dom, d in cells:
    if ONLY and ONLY not in f"{agent}/{bench}/{dom}":
        continue
    ids = {'meta-' + x.split('meta-')[1].split('.')[0]
           for x in glob.glob(os.path.join(d, 'meta-*.jsonl'))}
    rows = [index[i] for i in ids if i in index]
    if len(rows) != len(ids):
        print(f"  ✗ {agent}/{bench}/{dom}: index 对不上，跳过"); continue
    # PER TOKEN, the unit `exact_tokens.build` takes and the unit
    # `dollars.Prices.output` is in. The recorded table is per million.
    price = json.load(open(os.path.join(d, 'dollars.json')))['prices']['per_million']['output'] / 1e6
    if agent not in counters:
        counters[agent] = ex.Counter(agent)
    c = counters[agent]
    before = c.calls
    t0 = time.time()
    groups = ((load_turns(r),
               verdicts_for(JUDGE, run, agent, bench, dom, r['trace_id']))
              for r in rows)
    cell = ex.build(agent, bench, dom, c, groups, price)
    p = ex.save(cell, JUDGE, run)
    c.save()
    o = cell.buckets.get(ex.OUTPUT); rr = cell.buckets.get(ex.REASONING)
    au = cell.audited_json()
    line = (f"  {agent}/{bench}/{dom or '-'}  {cell.n_traces} traj  "
            f"{c.calls - before} 次新调用  {time.time()-t0:.0f}s"
            f"\n      audited ({' + '.join(au['buckets'])})  "
            f"{au['waste_tokens']:,}/{au['billed_tokens']:,} = {au['waste_share']*100:.1f}%  ${au['waste_dollars']:.4f}")
    if o and o.billed:
        line += f"\n      v+a  {o.waste:,}/{o.billed:,} = {o.share*100:.1f}%  ${o.waste_dollars:.4f}"
    if rr and rr.billed:
        kind = ex.reasoning_kind(agent, has_text=bool(rr.text))
        if kind == "raw":
            line += f"\n      reasoning  {rr.waste:,}/{rr.billed:,} = {rr.share*100:.1f}%  (原文，已并入 audited)"
        elif kind == "summary":
            # Marked SUMMARY tokens over SUMMARY tokens. Never over the bill:
            # the summary is not the billed object.
            line += (f"\n      reasoning  {rr.billed:,} billed, 不计入；摘要 {rr.text:,} tokens，"
                     f"其中 {rr.share_of_text*100:.1f}% 被标")
        else:
            line += f"\n      reasoning  {rr.billed:,} billed, 无文本，不计入"
    print(line, flush=True)
for c in counters.values():
    c.save()
print("\n完成")
