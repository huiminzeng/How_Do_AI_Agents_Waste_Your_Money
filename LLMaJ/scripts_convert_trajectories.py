#!/usr/bin/env python3
"""Turn trajectories from any of the supported sources into the judge's JSONL, and check them.

    python3 scripts_convert_trajectories.py <input> --model NAME [--benchmark NAME] [-o FILE]
    python3 scripts_convert_trajectories.py <input> --check          # report problems, write nothing

The output file defaults to <benchmark>.jsonl, or <benchmark>-<domain>.jsonl when the benchmark
has domains (tau2-retail.jsonl). --model is required unless the source names the model (τ²-bench
does); --benchmark is required for sources that do not name theirs (a plain messages list).

The input is recognised by its shape:

    the judge's JSONL          one trajectory per line, `turns: [...]` (passed through, checked)
    the tracing proxy's rows   calls.jsonl rows (`kind: "call"`), one directory per trajectory,
                               or any file / directory holding such rows
    τ²-bench results.json      `simulations: [...]`; one trajectory per task (the first trial when
                               there are several), `solved` from reward_info.reward, the agent policy
                               as system_prompt, reasoning text and reasoning tokens from the
                               provider's raw response, the scripted opening greeting dropped
    an OpenAI-style messages   a JSON list of {role, content, tool_calls?} (mini-swe-agent's
    list                       .traj.json and most harness logs), or an object with `messages`;
                               one trajectory per file

A directory is walked for *.json, *.jsonl and *.traj.json. The output is the format
`scripts_import_trajectories.py` reads, documented there and at the project page's skill.md:
one line per trajectory with trace_id, task_id, model, benchmark, domain, run_id,
system_prompt, solved (optional, 0-1) and turns[{round, received{role,text}, reasoning,
visible, tool_calls[{name, arguments}], output_tokens, reasoning_tokens}].

Nothing is sent anywhere; this only reads and writes local files.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SAFE = re.compile(r"[^A-Za-z0-9_.]+")
KEY = re.compile(r'"api_key"\s*:\s*"[^"]{8,}"|\bsk-(ant-)?[A-Za-z0-9_-]{16,}')


# ----------------------------------------------------------------- helpers

def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else (p.get("text") or p.get("content") or "") for p in content if p)
    return "" if content is None else str(content)


def args_of(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except ValueError:
            return a
    return a if a is not None else {}


def calls_of(tcs) -> list[dict]:
    out = []
    for tc in tcs or []:
        fn = tc.get("function") or {}
        out.append({"name": tc.get("name") or fn.get("name") or "?",
                    "arguments": args_of(tc.get("arguments", fn.get("arguments")))})
    return out


def is_msg_list(v) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(m, dict) and isinstance(m.get("role"), str) for m in v)


# -------------------------------------------------------------- converters

def from_messages(msgs: list[dict], meta: dict, n: int) -> dict:
    out = {"trace_id": str(meta.get("trace_id") or meta.get("id") or f"t-{n}"),
           "task_id": str(meta.get("task_id") or meta.get("instance_id") or meta.get("trace_id") or f"task-{n}"),
           "model": str(meta.get("model") or "my-model"), "benchmark": str(meta.get("benchmark") or "my_tasks"),
           "domain": meta.get("domain") or "all", "run_id": str(meta.get("run_id") or "imported"), "turns": []}
    if meta.get("solved") is not None:
        out["solved"] = meta["solved"]
    pending = None
    rnd = 0
    for m in msgs:
        role = m.get("role")
        if role == "system":
            out.setdefault("system_prompt", text_of(m.get("content")))
            continue
        if role == "assistant":
            raw = m.get("raw_data") if isinstance(m.get("raw_data"), dict) else {}
            raw_msg = ((raw.get("choices") or [{}])[0].get("message") or {}) if raw else {}
            if m.get("usage") is None and not raw and not m.get("tool_calls") and rnd == 0 and len(out["turns"]) == 0 \
                    and str(meta.get("benchmark")) == "tau2":
                continue          # τ²-bench's scripted opening line: not a model call, nothing to judge
            content = text_of(m.get("content"))
            rnd += 1
            reasoning = text_of(m.get("reasoning_content") or m.get("reasoning") or raw_msg.get("reasoning_content") or "")
            turn = {"round": rnd, "reasoning": reasoning,
                    "visible": "" if content == "None" else content, "tool_calls": calls_of(m.get("tool_calls"))}
            if pending:
                turn["received"] = pending
            u = m.get("usage") or raw.get("usage") or {}
            if u.get("completion_tokens") is not None:
                turn["output_tokens"] = u["completion_tokens"]
            if m.get("output_tokens") is not None:
                turn["output_tokens"] = m["output_tokens"]
            details = u.get("completion_tokens_details") or (raw.get("usage") or {}).get("completion_tokens_details") or {}
            rt = u.get("reasoning_tokens") if u.get("reasoning_tokens") is not None else details.get("reasoning_tokens")
            if rt is not None:
                turn["reasoning_tokens"] = rt
            out["turns"].append(turn)
            pending = None
            continue
        r = "tool" if role == "tool" else "user"
        t = text_of(m.get("content"))
        if pending:
            pending = {"role": "tool" if pending["role"] == r == "tool" else ("user" if pending["role"] != r else r),
                       "text": pending["text"] + "\n\n" + t}
        else:
            pending = {"role": r, "text": t}
    return out


def from_tau2(doc: dict) -> list[dict]:
    info = doc.get("info") or {}
    ai = info.get("agent_info") or {}
    ei = info.get("environment_info") or {}
    model = str(ai.get("llm") or ai.get("model") or (ai.get("llm_args") or {}).get("model") or "tau2-agent")
    model = model.split("/")[-1]          # "openai/<alias>" is the route, not the model
    domain = str(ei.get("domain_name") or ei.get("domain") or re.sub(r"_.*", "", ei.get("name") or "") or "all")
    policy = ei.get("policy") or ""
    sims = doc.get("simulations") or []
    # One trial per task: the lowest trial id of each task; the others are dropped and said so.
    first: dict = {}
    for sim in sims:
        key = str(sim.get("task_id"))
        if key not in first or (sim.get("trial") or 0) < (first[key].get("trial") or 0):
            first[key] = sim
    if len(first) < len(sims):
        print(f"  note: {len(sims)} simulations over {len(first)} tasks; keeping the first trial of each task")
    out = []
    for i, sim in enumerate(first.values(), 1):
        meta = {"trace_id": sim.get("id"), "task_id": sim.get("task_id"), "model": model, "benchmark": "tau2",
                "domain": domain, "run_id": doc.get("timestamp"),
                "solved": (sim.get("reward_info") or {}).get("reward")}
        t = from_messages(sim.get("messages") or [], meta, i)
        sp = sim.get("policy") or policy
        if sp and not t.get("system_prompt"):
            t["system_prompt"] = sp
        out.append(t)
    return out


def from_proxy(rows: list[dict]) -> list[dict]:
    by_trace: dict[str, list[dict]] = {}
    for r in rows:
        by_trace.setdefault(r.get("trace_id") or "trace", []).append(r)
    out = []
    for tid, rs in by_trace.items():
        rs.sort(key=lambda r: r.get("round") or 0)
        tm = rs[0].get("trace_meta") or {}
        t = {"trace_id": tid, "task_id": str(tm.get("task_id") or tm.get("unit_id") or tid),
             "model": str(rs[0].get("requested_model") or tm.get("model") or "my-model"),
             "benchmark": str(tm.get("benchmark") or "my_tasks"), "domain": tm.get("domain") or "all",
             "run_id": str(tm.get("run_id") or "imported"), "turns": []}
        first = (rs[0].get("request_messages") or [None])[0]
        if first and first.get("role") == "system":
            t["system_prompt"] = text_of(first.get("content"))
        for j, r in enumerate(rs, 1):
            rm = r.get("request_messages") or []
            last = rm[-1] if rm else None
            resp = r.get("response_message") or {}
            turn = {"round": r.get("round") or j, "reasoning": text_of(resp.get("reasoning_content") or ""),
                    "visible": text_of(resp.get("content") or ""), "tool_calls": calls_of(resp.get("tool_calls"))}
            if last:
                turn["received"] = {"role": "tool" if last.get("role") == "tool" else "user", "text": text_of(last.get("content"))}
            for k in ("output_tokens", "reasoning_tokens"):
                if r.get(k) is not None:
                    turn[k] = r[k]
            t["turns"].append(turn)
        out.append(t)
    return out


# ---------------------------------------------------------------- detection

def load_any(path: Path) -> tuple[str, list]:
    """(format, trajectories) for one file."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError("empty file")
    whole = None
    try:
        whole = json.loads(text)
    except ValueError:
        pass
    if whole is not None:
        if isinstance(whole, dict) and isinstance(whole.get("simulations"), list):
            return "tau2", from_tau2(whole)
        if is_msg_list(whole):
            return "messages", [from_messages(whole, {"trace_id": path.stem, "task_id": path.stem}, 1)]
        if isinstance(whole, list):
            if all(isinstance(x, dict) and x.get("kind") == "call" for x in whole):
                return "proxy", from_proxy(whole)
            if all(isinstance(x, dict) and isinstance(x.get("turns"), list) for x in whole):
                return "ours", whole
            if all(isinstance(x, dict) and is_msg_list(x.get("messages")) for x in whole):
                return "messages", [from_messages(x["messages"], x, i) for i, x in enumerate(whole, 1)]
            raise ValueError("a JSON array, but not of trajectories, proxy rows or messages")
        if isinstance(whole, dict):
            if isinstance(whole.get("turns"), list):
                return "ours", [whole]
            msgs = whole.get("messages") if is_msg_list(whole.get("messages")) else whole.get("trajectory")
            if is_msg_list(msgs):
                return "messages", [from_messages(msgs, {**whole, "trace_id": whole.get("trace_id") or path.stem}, 1)]
            if whole.get("kind") == "call":
                return "proxy", from_proxy([whole])
        raise ValueError("a JSON document, but not a trajectory: expected turns, messages or simulations")
    lines = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                lines.append(json.loads(line))
            except ValueError as e:
                raise ValueError(f"line {n}: not valid JSON: {e}")
    if all(isinstance(x, dict) and x.get("kind") == "call" for x in lines):
        return "proxy", from_proxy(lines)
    if all(isinstance(x, dict) and isinstance(x.get("turns"), list) for x in lines):
        return "ours", lines
    if all(isinstance(x, dict) and isinstance(x.get("role"), str) for x in lines):
        return "messages", [from_messages(lines, {"trace_id": path.stem, "task_id": path.stem}, 1)]
    if all(isinstance(x, dict) and is_msg_list(x.get("messages")) for x in lines):
        return "messages", [from_messages(x["messages"], x, i) for i, x in enumerate(lines, 1)]
    raise ValueError("JSONL lines of mixed or unknown shape; every line must be one trajectory")


# --------------------------------------------------------------------- check

def check(t: dict, where: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    w = f"{where}: "
    if not isinstance(t, dict):
        return [("bad", w + "must be a JSON object")]
    for k in ("trace_id", "task_id", "model", "benchmark"):
        if not isinstance(t.get(k), str) or not t[k]:
            out.append(("bad", w + f"{k}: a non-empty string"))
    if isinstance(t.get("benchmark"), str) and "-" in t["benchmark"]:
        out.append(("warn", w + "benchmark: a dash is replaced by an underscore on import"))
    if t.get("system_prompt") is not None and not isinstance(t["system_prompt"], str):
        out.append(("bad", w + "system_prompt must be a string"))
    if t.get("solved") is not None and not (isinstance(t["solved"], (int, float)) and 0 <= t["solved"] <= 1):
        out.append(("bad", w + "solved must be a number between 0 and 1"))
    turns = t.get("turns")
    if not isinstance(turns, list) or not turns:
        out.append(("bad", w + "turns: a non-empty array"))
        return out
    no_reasoning = no_tokens = no_received = 0
    for i, u in enumerate(turns, 1):
        p = w + f"turn {i}: "
        if not isinstance(u, dict):
            out.append(("bad", p + "must be an object")); continue
        if u.get("round") is not None and not isinstance(u["round"], int):
            out.append(("bad", p + "round must be an integer"))
        rec = u.get("received")
        if rec is not None:
            if not isinstance(rec, dict) or not isinstance(rec.get("text"), str):
                out.append(("bad", p + "received must be {role, text}"))
            elif rec.get("role") and rec["role"] not in ("user", "tool", "system"):
                out.append(("warn", p + f"received.role {rec['role']!r} (expected user or tool)"))
        else:
            no_received += 1
        for k in ("reasoning", "visible"):
            if u.get(k) is not None and not isinstance(u[k], str):
                out.append(("bad", p + f"{k} must be a string"))
        calls = u.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list):
                out.append(("bad", p + "tool_calls must be an array"))
            else:
                for j, c in enumerate(calls):
                    if not isinstance(c, dict) or not isinstance(c.get("name"), str):
                        out.append(("bad", p + f"tool_calls[{j}].name must be a string"))
        if not (u.get("reasoning") or u.get("visible") or calls):
            out.append(("warn", p + "has no reasoning, visible text or tool calls; nothing to judge"))
        for k in ("output_tokens", "reasoning_tokens"):
            if u.get(k) is not None and not isinstance(u[k], (int, float)):
                out.append(("bad", p + f"{k} must be a number"))
        if not u.get("reasoning"):
            no_reasoning += 1
        if u.get("output_tokens") is None:
            no_tokens += 1
    if no_received:
        out.append(("warn", w + f"{no_received} turn(s) without received text; the judge cannot tell restated or re-fetched content there"))
    if no_reasoning == len(turns):
        out.append(("warn", w + "no reasoning text in any turn; only the replies and tool calls will be judged"))
    if no_tokens:
        out.append(("warn", w + f"{no_tokens} turn(s) without output_tokens; those counts will be estimated with o200k"))
    return out


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="files or directories")
    ap.add_argument("-o", "--out", help="write the trajectories here as JSONL (default: <benchmark>.jsonl)")
    ap.add_argument("--model", help="set the model name on every trajectory")
    ap.add_argument("--benchmark", help="set the benchmark name on every trajectory")
    ap.add_argument("--check", action="store_true", help="only report problems; write nothing")
    a = ap.parse_args()

    files: list[Path] = []
    for inp in a.inputs:
        p = Path(inp)
        if p.is_dir():
            files += sorted(q for q in p.rglob("*") if q.is_file() and q.suffix in (".json", ".jsonl") or q.name.endswith(".traj.json"))
        else:
            files.append(p)
    if not files:
        sys.exit("no input files")

    trajs: list[dict] = []
    formats: dict[str, int] = {}
    for f in files:
        try:
            fmt, ts = load_any(f)
        except ValueError as e:
            print(f"  skip {f}: {e}")
            continue
        if KEY.search(f.read_text(encoding="utf-8", errors="replace")):
            print(f"  warning: {f} contains an API key; the converted output leaves it out, do not share the raw file")
        formats[fmt] = formats.get(fmt, 0) + len(ts)
        trajs += ts
    if not trajs:
        sys.exit("nothing recognised")
    for t in trajs:
        if a.model:
            t["model"] = a.model
        if a.benchmark:
            t["benchmark"] = a.benchmark
        t["benchmark"] = SAFE.sub("_", str(t.get("benchmark") or "my_tasks")).strip("_") or "my_tasks"
    if any(t.get("model") == "my-model" for t in trajs):
        sys.exit("the source does not name the model: pass --model <name> (the model as it should be shown, e.g. my-model-7b-high)")
    if any(t.get("benchmark") == "my_tasks" for t in trajs) and not a.benchmark:
        sys.exit("the source does not name the benchmark: pass --benchmark (tau2, claw_eval, swe_verified, mcp_atlas, or your own name)")

    problems: list[tuple[str, str]] = []
    for i, t in enumerate(trajs, 1):
        problems += check(t, f"trajectory {i} ({t.get('trace_id', '?')})")
    bad = [m for k, m in problems if k == "bad"]
    warn = [m for k, m in problems if k == "warn"]
    print(f"{len(trajs)} trajectories, {sum(len(t.get('turns') or []) for t in trajs)} turns, from "
          + ", ".join(f"{n} {f}" for f, n in formats.items()))
    for m in bad:
        print(f"  ERROR  {m}")
    for m in warn[:12]:
        print(f"  note   {m}")
    if len(warn) > 12:
        print(f"  note   ... {len(warn) - 12} more")
    if bad:
        print("not written: fix the errors above")
        return 1
    if a.check:
        print("ok: ready for the judge")
        return 0
    cells = sorted({(t["benchmark"], t.get("domain") or "all") for t in trajs})
    if len(cells) > 1 and not a.out:
        sys.exit("several benchmarks or domains in the input (" + ", ".join("/".join(c) for c in cells)
                 + "): pass -o, or convert them separately")
    bench, dom = cells[0]
    out = Path(a.out) if a.out else Path(f"{bench}.jsonl" if dom == "all" else f"{bench}-{dom}.jsonl")
    text = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in trajs)
    if KEY.search(text):
        sys.exit("the converted output still contains an API key; remove it from the source and convert again")
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(trajs)} trajectories): ready for the judge")
    return 0


if __name__ == "__main__":
    sys.exit(main())
