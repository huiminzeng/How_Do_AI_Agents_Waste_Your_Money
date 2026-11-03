"""LLM-as-a-judge over recorded agent trajectories.

Two independent classifications of the same turns:

    utility      what activity is this turn?        one turn per call, action hidden
    efficiency   was this output worth producing?   one trajectory per call, action shown

Layout:

    preprocess/   traces on disk -> Turn/Trace records under out/turns/
    prompts/      each task's label set AND the question put to a judge
    evaluate/     running a judge: one module per task, plus the call itself
    aggregate/    verdicts -> the numbers a figure is drawn from
    paths.py      the only module holding a path literal, and it refuses to
                  write anywhere outside out/
"""

from . import aggregate, evaluate, paths, preprocess, prompts  # noqa: F401
