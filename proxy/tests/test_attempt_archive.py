"""A superseded attempt's log must survive, because its calls cost money.

A retry re-enters the same trace scope with a fresh scope id but the same trace
id, so it resolves to the same file. Deleting there discarded the earlier
attempt's calls and under-reported spend.
"""

import os

from llm_cost_tracker.proxy.trace_logger import ATTEMPT_SUFFIX, _archive_superseded_log


def test_superseded_log_is_renamed_not_deleted(tmp_path):
    log = tmp_path / "calls.jsonl"
    log.write_text('{"kind":"call","cost_usd":0.25}\n')

    archived = _archive_superseded_log(str(log))

    assert not log.exists()
    assert os.path.basename(archived) == f"calls.jsonl{ATTEMPT_SUFFIX}1"
    assert '"cost_usd":0.25' in open(archived).read()


def test_each_retry_gets_its_own_archive(tmp_path):
    log = tmp_path / "calls.jsonl"
    for attempt in range(1, 4):
        log.write_text(f'{{"attempt":{attempt}}}\n')
        _archive_superseded_log(str(log))

    archives = sorted(p.name for p in tmp_path.glob("calls.jsonl.attempt*"))
    assert archives == ["calls.jsonl.attempt1", "calls.jsonl.attempt2", "calls.jsonl.attempt3"]
    # Nothing overwrote anything: each attempt's payload is still distinct.
    bodies = {(tmp_path / name).read_text().strip() for name in archives}
    assert bodies == {'{"attempt":1}', '{"attempt":2}', '{"attempt":3}'}
