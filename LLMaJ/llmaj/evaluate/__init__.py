"""Running a judge. One module per task, plus the call itself.

    judge.py        one request to one judge, with retries and the trace header
    utility.py      drives the per-turn pass    -> out/utility/
    efficiency.py   drives the per-trajectory pass -> out/efficiency/   (not built)

Both runners are resumable the same way, and structurally rather than by
bookkeeping: a shard that exists is done, so an interrupted run resumes with no
progress file to keep in sync. Both also snapshot their prompt before the first
call, so a run that dies halfway still leaves behind what it was asking.
"""

from .judge import DEFAULT_BASE_URL, Judge, Verdict  # noqa: F401
from .utility import iter_labels, run  # noqa: F401
