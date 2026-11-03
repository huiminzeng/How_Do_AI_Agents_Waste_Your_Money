"""Verdicts -> the numbers a figure is drawn from. One module per task.

    utility_matrix.py     labels -> transition counts, with the turns behind
                          every edge kept beside the count
    efficiency_spend.py   verdicts -> share per class per stream,
                          not share of turns  (not built)

Both weight by size rather than by count, for the same reason: a 5,000-token
turn and a 50-token turn are not one observation each when the question is
where the tokens went.
"""

from .utility_matrix import MIN_EDGE_N, TransitionMatrix, build, save  # noqa: F401
