"""The two classification tasks: what their labels ARE, and what we ask a judge.

Two files per task, paired by prefix:

    utility_states.py       the eight states, their definitions, the tie-breaks
    utility_prompt.py       one turn, action withheld  -> one state

    efficiency_classes.py   the three classes, their definitions, the tie-breaks
    efficiency_prompt.py    one trajectory, action shown -> three classes per
                            output stream, plus verbatim spans of removable text

The `*_states` / `*_classes` files are the specification and are imported well
beyond the prompts: `matrix.py` needs the state list to build a transition
matrix needs the spine order. Changing a definition
there changes what the judge is asked, so both files carry a dated revision log
and the prompt that renders them carries a version that must be bumped with it.
"""

from .efficiency_classes import CLASSES, MAX_SPANS, STREAMS  # noqa: F401
from .efficiency_prompt import EFF_PROMPT_VERSION  # noqa: F401
from .utility_prompt import PROMPT_VERSION, PromptVersionConflict  # noqa: F401
from .utility_states import STATES  # noqa: F401
