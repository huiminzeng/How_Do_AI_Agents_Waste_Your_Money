"""The walkthrough: a judged trajectory replayed as a page.

`scripts_walkthrough.py` turns the labeled trajectories the judge returns
(`<benchmark>.labeled.jsonl`) into one folder a browser opens from disk:
`index.html`, the page, with the trajectories embedded, and `merged.json`,
the same data on its own. The page is the one on the paper's project page:
the transcript on the left with every removable span highlighted, the floor
plan of the task on the right with the agent's route drawn across it, and
the token curves under it.

`build.py` is the only logic: it maps each labeled turn to what the page
draws (the spans located in the text, the rooms the turn visited, the
tokens it cost). The page itself is static and knows the four benchmarks of
the paper.
"""
