"""Models whose reasoning arrives INSIDE `content`, delimited by `</think>`.

Qwen3-30B-A3B-Thinking-2507-tau2-consecutive-tool-penalty was served
by SGLang WITHOUT a reasoning parser, so the proxy recorded no `reasoning_content`
and no `reasoning_tokens`: every response is one string, `<reasoning>\\n</think>\\n\\n
<reply>`. (The `<think>` opener is prefilled by the chat template, so it is not in
the output.) sentinel-qwen3-4b-thinking-rl, the other self-served Qwen, was served
WITH the parser and needs nothing here.

Two rules, both copied from what tau2 itself did with the same string
(tau2/utils/llm_utils.py): text after the FIRST `</think>`, left-stripped, is the
reply; with no `</think>` at all the whole string is the reply. The second case
is 182 of 5,251 rows: the model wrote its answer inside the thinking block and
never closed it, and tau2 showed the user all of it. Treating it as reasoning
here would describe a turn the user never saw; the judge sees what tau2 saw.

`reasoning_tokens` is reconstructed with the model's own tokenizer: the proxy
figure for sentinel-qwen3-4b (SGLang's parser) is the tokenizer count of the
reasoning text + 2 on 77/77 rows checked (p10 = p90 = -2), the two
being `</think>` and the newline after it, so the same constant is added here.
"""

from __future__ import annotations

import threading
from typing import Any

#: canonical model id -> HF repo whose tokenizer.json reproduces its bill.
MODELS: dict[str, str] = {
    "qwen3-30b-a3b-thinking-2507-tau2-consecutive-tool-penalty":
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
}
CLOSE = "</think>"
#: billed reasoning tokens = tokenizer count of the text + this (see above).
ENVELOPE = 2


def _canonical(model: str | None) -> str:
    return (model or "").split("/")[-1].split("__")[0].lower()


def is_inline(model: str | None) -> bool:
    return _canonical(model) in MODELS


def split(content: str | None) -> tuple[str, str]:
    """(reasoning, reply) by tau2's rule. Reasoning is the raw text before the
    first `</think>` (unstripped: its token count is what is billed)."""
    if not isinstance(content, str):
        return "", ""
    if CLOSE not in content:
        return "", content
    before, after = content.split(CLOSE, 1)
    return before, after.lstrip()


_lock = threading.Lock()
_counters: dict[str, Any] = {}


def counter(model: str):
    """len(tokenizer.encode(text)) for `model`, loaded once per process."""
    key = _canonical(model)
    with _lock:
        c = _counters.get(key)
        if c is None:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(hf_hub_download(MODELS[key], "tokenizer.json"))
            c = _counters[key] = (
                lambda t: len(tok.encode(t, add_special_tokens=False).ids))
    return c


def reasoning_tokens(rec: dict[str, Any]) -> int | None:
    """The record's own `reasoning_tokens` when the proxy set one; for an inline
    model with none, the reconstructed figure; otherwise None."""
    if rec.get("reasoning_tokens") is not None or not is_inline(rec.get("model")):
        return rec.get("reasoning_tokens")
    msg = rec.get("response_message") or {}
    reasoning, _ = split(msg.get("content") if isinstance(msg, dict) else None)
    if not reasoning:
        return 0
    n = counter(rec.get("model"))(reasoning) + ENVELOPE
    out = rec.get("output_tokens")
    return min(n, int(out)) if out else n
