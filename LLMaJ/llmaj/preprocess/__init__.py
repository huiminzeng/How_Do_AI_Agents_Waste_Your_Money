"""Traces on disk -> Turn/Trace records under out/turns/.

`sources.py` is the only module that knows the proxy's record format; everything
downstream sees Turn and Trace and never a raw record.
"""

from .extract import extract, iter_index  # noqa: F401
from .sample import iter_selected, load_turns, select  # noqa: F401
from .sources import CELLS, cells_for  # noqa: F401
