"""Bucket names shared by the token stages. Nothing is estimated here.

Every token and dollar figure in LLMaJ comes from `exact_tokens`, which
counts with the model's own tokenizer: the full text, the text with all
marks stripped, and the text with one kind's marks stripped at a time.
"""

REASONING = "reasoning"
OUTPUT = "visible+action"
BUCKETS: tuple[str, ...] = (REASONING, OUTPUT)

#: Which streams fall in which bucket.
BUCKET_STREAMS: dict[str, tuple[str, ...]] = {
    REASONING: ("reasoning",),
    OUTPUT: ("visible", "action"),
}
