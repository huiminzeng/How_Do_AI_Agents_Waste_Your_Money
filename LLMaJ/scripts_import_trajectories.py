#!/usr/bin/env python3
"""Bring trajectories recorded by someone else's harness into the judge's input tree.

    python3 scripts_import_trajectories.py my_trajectories.jsonl

One line per trajectory:

    {"trace_id": "t-0001", "task_id": "task-17", "model": "my-agent", "benchmark": "my_tasks",
     "domain": "all", "run_id": "run1", "system_prompt": "<the agent's standing instructions, if any>",
     "turns": [
       {"round": 1, "received": {"role": "user", "text": "<what the agent was given>"},
        "reasoning": "<chain of thought, or empty>", "visible": "<the reply the user sees>",
        "tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}}],
        "output_tokens": 312, "reasoning_tokens": 200}
     ]}

`received.role` is "user" or "tool" (what the turn was reacting to). `output_tokens` /
`reasoning_tokens` are the provider's billed counts; when absent they are estimated with
o200k. Everything else is optional. The trajectories land under

    <repo>/artifacts/trajectories/shared/no_key/<benchmark>/<domain>/<model>/trace_<trace_id>/calls.jsonl

in the rows the tracing proxy writes, so `evaluate.py <model>` finds them as the cell
`<model>/<benchmark>/<domain>`. The judge also needs one sentence saying what the agent was
producing: put it in <repo>/artifacts/llmaj/deliverables.json as {"<benchmark>/<domain>": "..."}.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llmaj import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="JSONL, one trajectory per line (see the module docstring)")
    ap.add_argument("--model", help="override the model name for every trajectory")
    ap.add_argument("--benchmark", help="override the benchmark name for every trajectory")
    a = ap.parse_args()
    enc = None
    n_traj = n_turns = 0
    for line in open(a.file):
        if not line.strip():
            continue
        t = json.loads(line)
        model = a.model or t.get("model") or "unknown-model"
        bench = (a.benchmark or t.get("benchmark") or "my_tasks").replace("-", "_")
        domain = t.get("domain") or "no_domain"
        tid = str(t.get("trace_id") or f"{bench}-{n_traj:05d}")
        d = paths.MONITOR / "shared" / "no_key" / bench / domain / model / f"trace_{tid}"
        d.mkdir(parents=True, exist_ok=True)
        sysmsg = [{"role": "system", "content": t["system_prompt"]}] if t.get("system_prompt") else []
        with (d / "calls.jsonl").open("w") as fh:
            for i, turn in enumerate(t.get("turns") or [], 1):
                calls = []
                for j, c in enumerate(turn.get("tool_calls") or []):
                    args = c.get("arguments")
                    calls.append({"id": f"call_{i}_{j}", "type": "function",
                                  "function": {"name": c.get("name") or "?",
                                               "arguments": args if isinstance(args, str) else json.dumps(args or {}, ensure_ascii=False)}})
                reasoning = turn.get("reasoning") or ""
                visible = turn.get("visible") or ""
                out_tok, rs_tok = turn.get("output_tokens"), turn.get("reasoning_tokens")
                if out_tok is None or rs_tok is None:
                    if enc is None:
                        import tiktoken
                        enc = tiktoken.get_encoding("o200k_base")
                    rs_tok = rs_tok if rs_tok is not None else len(enc.encode(reasoning))
                    out_tok = out_tok if out_tok is not None else rs_tok + len(enc.encode(visible)) + sum(len(enc.encode(c["function"]["arguments"])) for c in calls)
                rec = {"kind": "call", "trace_id": tid, "round": int(turn.get("round") or i),
                       "model": model, "requested_model": model,
                       "trace_meta": {"benchmark": bench, "domain": domain, "task_id": str(t.get("task_id") or tid),
                                      "run_id": t.get("run_id") or "imported"},
                       "request_messages": sysmsg + [{"role": (turn.get("received") or {}).get("role") or "user",
                                                      "content": (turn.get("received") or {}).get("text") or ""}],
                       "response_message": {"role": "assistant", "content": visible,
                                            "reasoning_content": reasoning, "tool_calls": calls},
                       "output_tokens": int(out_tok), "reasoning_tokens": int(rs_tok),
                       "tool_called": calls[0]["function"]["name"] if calls else None}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); n_turns += 1
        n_traj += 1
    print(f"{n_traj} trajectories, {n_turns} turns written under {paths.MONITOR}")
    print(f"next: put a deliverable sentence in {paths.WORK / 'deliverables.json'}, then run ./run_llmaj.sh <model> --re-extract")


if __name__ == "__main__":
    main()
