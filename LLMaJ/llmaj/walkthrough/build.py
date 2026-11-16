"""From labeled trajectories to what the walkthrough page draws.

The page (index.html) wants, per trajectory, the four texts of every turn, the
judge's spans as character offsets inside those texts, the "rooms" the turn
visited so the floor plan can route the agent, and the tokens the turn cost.
The labeled file (what `scripts_judge_submission.py` writes, and what the
paper's dataset on Hugging Face holds) has the texts, the spans as verbatim
quotes, and per-turn token counts. This module bridges the two.

Rooms are benchmark-specific, because the floor plan is: on SWE-bench a room is
a file in the repository, on tau2 a desk per tool, on MCP-Atlas and ClawEval a
tool of an app or server. A benchmark the page does not know gets the generic
tool treatment and the SWE-bench floor plan, which is wrong but not broken.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

#: What the page calls each benchmark (its BENCH table keys).
BENCH_KEY = {"swe_bench": "swe_verified", "swe_verified": "swe_verified", "mcp_atlas": "mcp_atlas",
             "claw_eval": "claw_eval", "tau2": "tau2"}

EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")

# ----------------------------------------------------------------------------- texts and spans


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _fold(text: str) -> tuple[str, list[int]]:
    """`text` with every whitespace run one space and every backslash gone, plus
    the original index of each kept character."""
    out, idx, space = [], [], False
    for i, ch in enumerate(text):
        if ch == "\\":
            continue
        if ch.isspace():
            if not space and out:
                out.append(" "); idx.append(i); space = True
            continue
        out.append(ch); idx.append(i); space = False
    return "".join(out), idx


def locate(text: str, quote: str, start: int = 0) -> tuple[int, int] | None:
    """Where `quote` sits in `text` at or after `start`: exact first, then
    ignoring how whitespace is run and how quotes are escaped (the judge quotes
    verbatim, but a tool call reaches it JSON-encoded and a reasoning block may
    be re-wrapped)."""
    if not quote or not text:
        return None
    i = text.find(quote, start)
    if i >= 0:
        return i, i + len(quote)
    ft, idx = _fold(text)
    fq, _ = _fold(quote)
    fq = fq.strip()
    if not fq:
        return None
    k = next((n for n, orig in enumerate(idx) if orig >= start), len(idx))
    j = ft.find(fq, k)
    if j < 0:
        return None
    return idx[j], idx[j + len(fq) - 1] + 1


def _js_index(text: str) -> "callable":
    """Python indexes a string by code point, a browser by UTF-16 unit, so after
    an emoji the two disagree; the page gets browser offsets."""
    if text.isascii():
        return lambda i: i
    cum, n = [0], 0
    for ch in text:
        n += 2 if ord(ch) > 0xFFFF else 1
        cum.append(n)
    return lambda i: cum[i]


def marks_of(turn: dict) -> tuple[dict, dict | None]:
    """The judge's spans per stream as [start, end, kind, why] in the page's
    (UTF-16) offsets, and the single span the floor plan colours a wasteful
    turn by: the longest one."""
    eff = (turn.get("labels") or {}).get("efficiency") or {}
    texts = {"reasoning": turn.get("reasoning") or "", "visible": turn.get("visible") or "",
             "action": turn.get("action") or ""}
    out = {k: [] for k in texts}
    top, top_len = None, -1
    for stream, text in texts.items():
        seen: dict[str, int] = {}   # a quote the judge marked twice is two occurrences
        js = _js_index(text)
        for s in (eff.get(stream) or {}).get("spans") or []:
            q = s.get("quote") or ""
            at = locate(text, q, seen.get(q, 0))
            if not at:
                continue
            seen[q] = at[1]
            kind = s.get("kind") or "detour"
            out[stream].append([js(at[0]), js(at[1]), kind, s.get("why") or ""])
            if at[1] - at[0] > top_len:
                top, top_len = {"kind": kind, "why": s.get("why") or ""}, at[1] - at[0]
    return out, top


def tokens_of(turn: dict, marks: dict) -> tuple[int, int]:
    """(billed, removable) tokens for the turn. From the exact counts when the
    labeled file carries them; otherwise the billed output tokens with the
    removable share estimated from characters, so a dataset without the
    tokenizer pass still animates."""
    buckets = turn.get("tokens") or {}
    if buckets:
        b = sum(int(v.get("billed_tokens") or 0) for v in buckets.values())
        w = sum(int(v.get("waste_tokens") or 0) for v in buckets.values())
        return b, w
    b = int(turn.get("output_tokens") or 0)
    total = sum(len(turn.get(k) or "") for k in ("reasoning", "visible", "action"))
    waste = sum(e - s for ms in marks.values() for s, e, _k, _w in ms)   # UTF-16 units, near enough
    return b, min(b, round(b * waste / total)) if total else 0


# ----------------------------------------------------------------------------- tool calls

CALL = re.compile(r"([\w.\-]+)\((\{.*?\})\)(?=\s*[\w.\-]+\(\{|\s*$)", re.S)


def calls_of(action: str) -> list[tuple[str, dict]]:
    """The `name({json})` calls in an action, with their arguments parsed when
    they parse. A call whose JSON does not parse keeps an empty argument dict."""
    out = []
    for m in CALL.finditer(action or ""):
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {}
        out.append((m.group(1), args if isinstance(args, dict) else {}))
    return out


def _first_arg(args: dict) -> tuple[str, str] | None:
    for k, v in args.items():
        if isinstance(v, (str, int, float)) and str(v).strip():
            return k, str(v)
    return None


def _short(v: str, n: int = 40) -> str:
    return _norm(v)[:n]


# ----------------------------------------------------------------------------- SWE-bench: files

EXTS = "py|pyx|pxd|rst|txt|cfg|toml|ini|html|js|ts|c|h|cpp|yml|yaml|json|md|tex|sh|csv"
PATH = re.compile(r"(?<![\w@=:.\-])/?(?:testbed/)?((?:[\w.\-]+/)*[\w\-]+\.(?:" + EXTS + r"))(?![\w/])")
SED_SCRIPT = re.compile(r"""(['"])\d*(?:,\d+)?[sy]/.*?\1""", re.S)


def swe_files(cmd: str) -> list[str]:
    """Files a shell command names, repository-relative, in order of mention, at
    most three. Only files: a directory is not a room. A sed script's s/old/new/
    is not a path either."""
    out = []
    for m in PATH.finditer(SED_SCRIPT.sub("''", cmd)):
        p = m.group(1)
        if p.startswith(("dev/", "usr/", "bin/", "etc/", "proc/")) or "*" in p:
            continue
        if p not in out:
            out.append(p)
    return out[:3]


def swe_op(cmd: str) -> str:
    """What the first shell command of a turn did, in the plan's vocabulary:
    submit, test, edit, git, run, search or read."""
    c = cmd.strip()
    head = re.sub(r"^(cd\s+\S+\s*(&&|;)\s*)+", "", c)
    if "COMPLETE_TASK_AND_SUBMIT" in c or re.match(r"submit\b", head):
        return "submit"
    if re.search(r"\b(pytest|runtests\.py|bin/test|unittest|tox|nosetests)\b", c):
        return "test"
    writes = re.search(r"\bsed\s+-i\b|\bcat\s*>+|\btee\b|>>?\s*/?[\w./-]+\.\w+|\bapplypatch\b|\bpatch\b\s+-p", c)
    heredoc_writes = re.search(r"\bpython3?\s+-\s*<<", c) and re.search(r"open\([^)]*['\"]w|\.write\(|write_text\(", c)
    if writes or heredoc_writes:
        return "edit"
    if re.match(r"git\b", head):
        return "git"
    if re.match(r"(python3?|pip3?|make|bash|sh|node|npm)\b", head):
        return "run"
    if re.match(r"(grep|rg|find|ls|tree|ack|locate|which|wc)\b", head):
        return "search"
    if re.match(r"(cat|sed|head|tail|less|more|awk|nl|od|file|stat|diff)\b", head):
        return "read"
    return "search"


def swe_round(turn: dict, is_last: bool) -> tuple[str, list[str], str]:
    calls = calls_of(turn.get("action") or "")
    if not calls:
        text = turn.get("visible") or ""
        return ("submit", ["patch.txt"], _short(text, 60)) if is_last and text else ("note", [], _short(text, 60))
    name, args = calls[0]
    cmd = args.get("command") or args.get("cmd") or ""
    if name in ("bash", "shell", "execute_bash", "run_command", "terminal", "run") and isinstance(cmd, str) and cmd:
        return swe_op(cmd), swe_files(cmd), cmd[:60]
    # a harness with file tools: the tool name says what it did, the path argument where
    n = name.lower()
    path = next((str(v) for k, v in args.items() if isinstance(v, str) and k in ("path", "file", "file_path", "filename", "target_file")), "")
    files = swe_files(path) if path else []
    if "submit" in n:
        op = "submit"
    elif any(w in n for w in ("edit", "write", "create", "replace", "patch", "insert")):
        op = "edit"
    elif any(w in n for w in ("search", "grep", "find", "glob", "list", "ls")):
        op = "search"
    elif any(w in n for w in ("read", "view", "open", "cat")):
        op = "read"
    elif any(w in n for w in ("test", "pytest")):
        op = "test"
    else:
        op = "read"
    cmd = f"{name}({path or _short(json.dumps(args, ensure_ascii=False), 50)})"
    return op, files, cmd[:60]


# ----------------------------------------------------------------------------- tau2: desks

TAU2_WING = [("verify", ("find_user_id", "get_user_details")),
             ("orders", ("get_order_details",)),
             ("products", ("get_product_details", "list_all_product_types")),
             ("reservations", ("get_reservation_details",)),
             ("flights", ("search_direct_flight", "search_onestop_flight", "get_flight_status")),
             ("calculator", ("calculate",)),
             ("transfer", ("transfer_to_human_agents",))]
LOOKUP_WINGS = {"verify", "orders", "products", "reservations", "flights", "calculator"}


def tau2_wing(tool: str) -> str:
    for wing, names in TAU2_WING:
        if any(tool.startswith(n) for n in names):
            return wing
    return "changes"


def tau2_key(tool: str, args: dict) -> str:
    """What identifies the record a desk was asked about: an id argument when
    there is one, else the first argument; for a flight search the destination
    and the date, so two days are two visits."""
    items = [(k, str(v)) for k, v in args.items() if isinstance(v, (str, int, float)) and str(v).strip()]
    if not items:
        return ""
    vals = [v for _k, v in items]
    if tool.startswith(("search_", "get_flight_status")):
        date = next((v for v in vals if re.match(r"\d{4}-\d{2}-\d{2}", v)), "")
        rest = [v for v in vals if v != date]
        head = rest[-1] if tool.startswith("search_") and rest else (rest[0] if rest else "")
        return _short(f"{head} {date}".strip())
    ident = next((v for k, v in items if k == "id" or k.endswith("_id")), None)
    return _short(ident if ident is not None else vals[0])


def tau2_round(turn: dict, is_last: bool) -> tuple[str, list[str], str]:
    calls = calls_of(turn.get("action") or "")
    if not calls:
        return "reply", ["customer/reply"], _short(turn.get("visible") or "", 60)
    files, op = [], "lookup"
    for tool, args in calls:
        if tool == "think":
            continue
        wing = tau2_wing(tool)
        files.append(f"{wing}/{tool}:{tau2_key(tool, args)}".rstrip(":"))
        if wing not in LOOKUP_WINGS:
            op = "action"
    if not files:
        return "note", [], _short(turn.get("visible") or "", 60)
    cmd = " ".join(f"{t}({', '.join(_short(str(v), 30) for v in a.values())})" for t, a in calls if t != "think")
    return op, files[:3], cmd[:60]


# ----------------------------------------------------------------------------- mcp_atlas, claw_eval: tools

SERVER = re.compile(r"^([A-Za-z0-9\-]+?)_")


def tool_round(turn: dict, is_last: bool) -> tuple[str, list[str], str]:
    """One room per tool of a server (`<server>/<tool>:<first argument>`); the
    final reply is the exit, text between calls a note at the desk."""
    calls = calls_of(turn.get("action") or "")
    if not calls:
        text = turn.get("visible") or ""
        return ("answer", ["answer/final"], _short(text, 60)) if is_last and text else ("note", [], _short(text, 60))
    files, parts = [], []
    for tool, args in calls:
        m = SERVER.match(tool)
        server = m.group(1) if m else tool
        fa = _first_arg(args)
        key = f"{fa[0]}={_short(fa[1])}" if fa else ""
        files.append(f"{server}/{tool}" + (f":{key}" if key else ""))
        parts.append(f"{tool}({_short(key, 30)})")
    return "tool", files[:3], " ".join(parts)[:60]


# ----------------------------------------------------------------------------- the trajectory


def bench_key(traj: dict) -> str:
    b = (traj.get("benchmark") or "").lower()
    dom = (traj.get("domain") or "").lower()
    if b == "tau2":
        return f"tau2-{dom}" if dom and dom != "all" else "tau2-retail"
    return BENCH_KEY.get(b, b or "swe_verified")


def round_of(bench: str, turn: dict, is_last: bool) -> tuple[str, list[str], str]:
    if bench.startswith("tau2"):
        return tau2_round(turn, is_last)
    if bench in ("mcp_atlas", "claw_eval"):
        return tool_round(turn, is_last)
    return swe_round(turn, is_last)


def split_model(model: str) -> tuple[str, str]:
    """`kimi-k3-high` is the model `kimi-k3` at effort `high`; a model without
    a known effort suffix has no effort."""
    for e in EFFORTS:
        if model.endswith("-" + e):
            return model[:-len(e) - 1], e
    return model, ""


def trajectory(traj: dict) -> dict:
    bench = bench_key(traj)
    short, effort = split_model(traj.get("model") or "agent")
    harness = traj.get("harness") or "native"
    turns = sorted(traj.get("turns") or [], key=lambda t: int(t.get("round") or 0))
    rounds = []
    for i, t in enumerate(turns):
        marks, top = marks_of(t)
        b, w = tokens_of(t, marks)
        op, files, cmd = round_of(bench, t, i == len(turns) - 1)
        rec = t.get("received") or {}
        labels = t.get("labels") or {}
        rd = {"r": int(t.get("round") or i + 1), "reasoning": t.get("reasoning") or "",
              "text": t.get("visible") or "", "action": t.get("action") or "",
              "obs": rec.get("text") or "", "obs_role": rec.get("role") or "",
              "out": int(t.get("output_tokens") or 0), "reas": int(t.get("reasoning_tokens") or 0),
              "b": b, "w": w, "marks": marks, "top": top, "files": files, "op": op, "cmd": cmd}
        # the action categories, when the file carries them (the paper's dataset does)
        a, u = labels.get("action") or {}, labels.get("utility") or {}
        if a.get("state"):
            rd["state"], rd["state_why"] = a["state"], a.get("rationale") or ""
        if u.get("state"):
            rd["tstate"], rd["tstate_why"] = u["state"], u.get("rationale") or ""
        rounds.append(rd)
    agent = traj.get("model") or "agent"
    if harness != "native":
        agent += f" · {harness}"
    return {"agent": agent, "short": short, "effort": effort, "bench": bench,
            "task": str(traj.get("task_id") or traj.get("trace_id") or "?"), "tid": traj.get("trace_id") or "",
            "score": traj.get("solved"), "rounds": rounds}


def build(files: list[Path], limit: int | None = None) -> list[dict]:
    """Every trajectory of the labeled files, in file order, as the page's records."""
    out = []
    for f in files:
        with Path(f).open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                out.append(trajectory(json.loads(line)))
                if limit and len(out) >= limit:
                    return out
    return out


HERE = Path(__file__).resolve().parent
PAGE = HERE / "index.html"


def write(trajs: list[dict], out_dir: Path) -> Path:
    """The folder a browser opens: index.html with the data embedded (so a
    double-click works, file:// forbids fetch in Chrome) and merged.json beside
    it for anything else."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(trajs, ensure_ascii=False)
    (out_dir / "merged.json").write_text(data, encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    embed = '<script id="data" type="application/json">' + data.replace("</", "<\\/") + "</script>\n"
    page = page.replace("<script>\nlet DATA=[]", embed + "<script>\nlet DATA=[]", 1)
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    return out_dir / "index.html"
