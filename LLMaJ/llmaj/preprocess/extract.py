"""Stage 1: traces -> one shard per trajectory, plus an index.

The index carries per-trajectory counts so the sample-size table and stratified
sampling never need to open a shard.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

from .. import paths
from ..schema import Trace
from .sources import Cell, cells_for, iter_traces


@dataclass
class CellStats:
    cell_key: str
    traces: int = 0
    turns: int = 0
    labellable: int = 0
    transitions: int = 0

    def add(self, tr: Trace) -> None:
        self.traces += 1
        self.turns += tr.n_turns
        self.labellable += tr.n_labellable
        self.transitions += tr.n_transitions


def write_trace(tr: Trace) -> None:
    path = paths.ensure_parent(
        paths.trace_shard(tr.agent_model, tr.benchmark, tr.domain, tr.trace_id)
    )
    with path.open("w") as fh:
        for row in tr.iter_json():
            fh.write(json.dumps(row) + "\n")


def extract_cell(cell: Cell, index_fh) -> CellStats:
    stats = CellStats(cell_key=cell.key)
    for tr in iter_traces(cell):
        if not tr.trace_id:
            continue
        write_trace(tr)
        index_fh.write(json.dumps(tr.index_row()) + "\n")
        stats.add(tr)
    return stats


#: One process extracts; the rest wait and then use the result.
#:
#: Five sessions launched together all find no index, all call extract(), and all
#: rmtree the tree the others are writing into. Nothing fails - the survivors
#: just get a partial index, and every cell downstream reports fewer
#: trajectories than it has. Telling the operator "start one first, then the
#: others" is not a fix; the code should hold the door.
LOCK = "extract.lock"


def extract_locked(agent_model=None, wait: float = 600.0, cells=None):
    """Extract under an exclusive lock, or wait for whoever holds it.

    The waiter does NOT extract again: by the time the lock frees, the tree is
    complete, and re-doing it would delete what the others are already reading.
    """
    paths.WORK.mkdir(parents=True, exist_ok=True)
    lock_path = paths.WORK / LOCK
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("  another session is extracting; waiting for it", flush=True)
            deadline = time.time() + wait
            while time.time() < deadline:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(2)
            else:
                raise TimeoutError(
                    f"waited {wait:.0f}s for {lock_path}; if no session is "
                    f"extracting, delete it and retry")
            if paths.TRACES_INDEX.exists() and cells is None and agent_model is None:
                print("  extraction finished elsewhere; using it\n", flush=True)
                return []
        return extract(agent_model, cells=cells)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


JUDGE_LOCK = "judge.lock"
#: One file per running judge, `work/judge_locks/<pid>.json`, naming the cells
#: it reads. One file per judge rather than a single lock, so two judges cannot
#: overwrite each other's pid and an extraction of an unrelated cell is not
#: refused along with the dangerous kind.
JUDGE_LOCK_DIR = "judge_locks"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def judge_locks() -> list[dict]:
    """Every live judging run: [{pid, cells}], `cells` None when unknown.

    Reads the per-pid files and, for a session started before they existed,
    the legacy single `judge.lock` (pid only, cells unknown). Dead pids are
    ignored and their files removed.
    """
    out = []
    d = paths.WORK / JUDGE_LOCK_DIR
    if d.exists():
        for f in sorted(d.glob("*.json")):
            try:
                rec = json.loads(f.read_text())
                pid = int(rec["pid"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if _alive(pid):
                out.append({"pid": pid, "cells": rec.get("cells")})
            else:
                try:
                    f.unlink()
                except OSError:
                    pass
    legacy = paths.WORK / JUDGE_LOCK
    try:
        pid = int(legacy.read_text().strip())
        if _alive(pid) and all(x["pid"] != pid for x in out):
            out.append({"pid": pid, "cells": None})
    except (OSError, ValueError):
        pass
    return out


def hold_judge_lock(cell_keys) -> "callable":
    """Record this process as a judge of `cell_keys`; returns a release()."""
    d = paths.WORK / JUDGE_LOCK_DIR
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{os.getpid()}.json"
    f.write_text(json.dumps({"pid": os.getpid(), "cells": sorted(cell_keys),
                             "started": datetime.now(timezone.utc).isoformat()}))

    def release():
        try:
            f.unlink()
        except OSError:
            pass
    return release


def _refuse_if_judged(cell_keys=None) -> None:
    """Raise when a live judge is reading what is about to be rewritten.

    `cell_keys` None means the whole tree. A judge whose cells are unknown
    (legacy lock) blocks a whole-tree rewrite but not an incremental one of
    other cells - the incremental path never touches a cell it was not asked
    for, and the index is swapped atomically.
    """
    for j in judge_locks():
        if cell_keys is None:
            raise RuntimeError(
                f"a judging run (evaluate.py, pid {j['pid']}) is reading the trace tree; "
                f"rebuilding it now would pull its shards out from under it. Wait for it, "
                f"or extract only the cells you need: extract(cells=[...]).")
        if j["cells"] is None:
            print(f"  note: judge pid {j['pid']} holds the legacy lock (cells unknown); "
                  f"proceeding because only {sorted(cell_keys)} will be rewritten", flush=True)
            continue
        hit = sorted(set(cell_keys) & set(j["cells"]))
        if hit:
            raise RuntimeError(
                f"a judging run (evaluate.py, pid {j['pid']}) is reading {hit}; "
                f"re-extracting those cells would pull its shards out from under it. "
                f"Wait for it to finish.")


def extract(agent_model: str | None = None, cells=None) -> list[CellStats]:
    """Extract cells into the trace tree and the index.

    Whole tree (`agent_model` and `cells` both None): clear first, rewrite
    everything, so a re-run cannot leave two rows for one trajectory or a
    stale layout beside the current one.

    Incremental (`cells` given, or `agent_model` given): ONLY those cells'
    directories are removed and rewritten; the index keeps every other row
    and is replaced atomically; the manifest records a per-cell
    `extracted_at`. This is how a cell that has just finished generating is
    brought in while judges run on other cells.
    """
    from .sources import CELLS
    if cells is None and agent_model is not None:
        cells = [c for c in CELLS if c.agent_model == agent_model]
    if cells is None:
        _refuse_if_judged(None)
        if paths.TRACES.exists():
            shutil.rmtree(paths.TRACES)
        index_path = paths.ensure_parent(paths.TRACES_INDEX)
        out: list[CellStats] = []
        with index_path.open("w") as fh:
            for cell in cells_for(agent_model):
                out.append(extract_cell(cell, fh))
        _write_manifest(out)
        return out

    cells = list(cells)
    keys = {c.key for c in cells}
    _refuse_if_judged(keys)
    # Old rows of these cells go; every other row stays byte for byte.
    keep = [r for r in iter_index()
            if f"{r['agent_model']}/{r['benchmark']}/{r['domain']}" not in keys]
    out = []
    tmp = paths.ensure_parent(paths.TRACES_INDEX).with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")
        for cell in cells:
            d = paths.trace_shard(cell.agent_model, cell.benchmark, cell.domain, "x").parent
            if d.exists():
                shutil.rmtree(d)
            out.append(extract_cell(cell, fh))
    os.replace(tmp, paths.TRACES_INDEX)
    _write_manifest(out, merge=True)
    return out


def _write_manifest(stats: list[CellStats], merge: bool = False) -> None:
    """Records when extraction ran and what it saw, per cell.

    Staleness matters: cells are still being produced, so a later run must be
    able to tell which cell moved. Each cell carries its own `extracted_at`;
    the top-level one is the time of the last WHOLE-tree extraction and is the
    fallback for a cell record written before per-cell times existed.
    `merge` keeps the records of cells not in `stats` (incremental extraction).
    """
    path = paths.ensure_parent(paths.TRACES / "manifest.json")
    now = datetime.now(timezone.utc).isoformat()
    cells = {}
    if merge and path.exists():
        try:
            old = json.loads(path.read_text())
            cells = dict(old.get("cells", {}))
            top = old.get("extracted_at", now)
        except (OSError, ValueError):
            top = now
    else:
        top = now
    for s in stats:
        cells[s.cell_key] = dict(s.__dict__, extracted_at=now)
    path.write_text(json.dumps({"extracted_at": top, "cells": cells}, indent=2) + "\n")


def cell_extracted_at(manifest: dict, cell_key: str) -> float | None:
    """When `cell_key` was last extracted (epoch seconds), or None if never."""
    rec = (manifest.get("cells") or {}).get(cell_key)
    # A record with no trajectories is a cell that was LISTED but had nothing
    # to extract yet (its run was still going); treat it as never extracted,
    # or a finished run reads as "stale" instead of "new".
    if rec is None or not rec.get("traces"):
        return None
    ts = rec.get("extracted_at") or manifest.get("extracted_at")
    return datetime.fromisoformat(ts).timestamp() if ts else None


def iter_index() -> Iterator[dict]:
    if not paths.TRACES_INDEX.exists():
        return
    with paths.TRACES_INDEX.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
