"""One call to one judge.

Judges are reached through the local tracing proxy, so their spend is recorded
like every other call. What it is recorded *as* is set by `x-trace-meta`: without
that header the proxy has no identity to file the call under and drops
everything into one shared `untraced/` log, where four labelling runs across five
prompt versions piled up indistinguishable from each other and from unrelated
smoke tests. See `trace_meta_for`.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import httpx

from ..prompts.utility_states import STATES

#: LLMaJ's OWN proxy, on 4500. 4000 serves the agents; a judge pointed there
#: would land in the agents' cost tree and be counted as agent spend. The default
#: has to be right when `evaluate.py` is run without the wrapper, because a
#: default that is wrong when the wrapper is skipped is not a
#: safe default.
DEFAULT_BASE_URL = os.environ.get("LLMAJ_BASE_URL", "http://127.0.0.1:4500/v1")

#: Structured-output schema. Verified that the proxy passes
#: json_schema through to Gemini and returns valid JSON.
#: Sent whenever the caller does not name a limit. Without it the server applies
#: its own default, which on the kimi-k3 endpoint we used is 128 tokens - and
#: kimi at `reasoning_effort: max` spends 600-700 characters on the chain of
#: thought before writing anything, so `content` came back EMPTY on 56 of 85
#: calls and truncated on 78, leaving 37 of 48 utility labels unusable. A one-state answer needs a few dozen tokens; the
#: reasoning in front of it needs whatever the model wants.
DEFAULT_MAX_TOKENS = 4000

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "state": {"type": "string", "enum": list(STATES)},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["state", "confidence", "rationale"],
}


#: The proxy builds its path as
#:   <owner>/<credential>/<benchmark>/<domain>/<model>/trace_<id>/calls.jsonl
#: and only <benchmark> and <domain> are ours to set. Four axes have to fit in
#: two segments, so they are packed: the task in the first, and the run plus the
#: cell being judged in the second.
#:
#: The run id alone in <domain> would leave the path unable to say WHICH agent
#: model or benchmark a given cost belonged to (that is hashed into trace_id and
#: therefore unreadable). Cost you cannot attribute is not cost tracking.
#:
#: "llmaj-" prefixes the task so the segment can never be mistaken for one of
#: the four benchmark names that `sources.py` globs on.
TRACE_BUCKET = "llmaj"


def trace_meta_for(
    judge_id: str,
    judge_run_id: str,
    prompt_version: str,
    agent_model: str,
    benchmark: str,
    domain: str,
    task: str = "utility",
) -> dict[str, str]:
    """Identity the proxy files a judge call under.

    Produces, for example:

        judge_calls/shared/no_key/llmaj-efficiency/
            e001__kimi-k3__tau2-airline/gemini-3.1-pro-preview/trace_.../calls.jsonl

    The model segment is the RESOLVED model, filled in by the proxy - so the path
    records what the alias actually was, which the alias itself does not.

    `reset` is never set: it would truncate the log.
    """
    return {
        "benchmark": f"{TRACE_BUCKET}-{task}",
        "domain": f"{judge_run_id}__{agent_model}__{benchmark}-{domain}",
        "workflow": TRACE_BUCKET,
        "task": task,
        "judge": judge_id,
        "judge_run_id": judge_run_id,
        "prompt_version": prompt_version,
        "labels_agent_model": agent_model,
        "labels_cell": f"{benchmark}/{domain}",
    }


@dataclass
class Verdict:
    state: str | None
    confidence: float | None
    rationale: str
    error: str | None = None


def _from_dict(d: dict) -> Verdict:
    """One utility verdict out of an already-parsed object."""
    state = str(d.get("state", "")).strip().upper()
    if state not in STATES:
        return Verdict(None, None, str(d.get("rationale", ""))[:200],
                       f"unknown state {state!r}")
    conf = d.get("confidence")
    return Verdict(
        state=state,
        confidence=float(conf) if isinstance(conf, (int, float)) else None,
        rationale=str(d.get("rationale", ""))[:500],
    )


class Judge:
    """A single judge model, called synchronously."""

    #: Judges that reject the parameter outright. opus-5 and sonnet-5 answer
    #: `temperature is deprecated for this model` with HTTP 400 and produce
    #: nothing, so the parameter must be absent rather than forced to a value.
    NO_TEMPERATURE = ("opus-5", "sonnet-5", "claude")

    @classmethod
    def _sends_temperature(cls, judge_id: str) -> bool:
        return not any(t in judge_id for t in cls.NO_TEMPERATURE)

    def __init__(
        self,
        judge_id: str,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_attempts: int = 4,
    ) -> None:
        self.judge_id = judge_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("PROXY_API_KEY", "")
        if not self.api_key:
            # Sending "Bearer " with nothing after it fails on every call with
            # LocalProtocolError, and a whole cell then reports "109 turns, 109
            # errors" as if the judge had misbehaved. Refuse up front instead.
            raise RuntimeError("PROXY_API_KEY is empty: source .env (or pass "
                               "api_key) before judging")
        self.temperature = temperature
        self.send_temperature = self._sends_temperature(judge_id)
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._client = httpx.Client(timeout=timeout)

    #: judge id -> the largest max_tokens its provider accepts, learned from a 400.
    _max_tokens_cap: dict[str, int] = {}

    def ask(self, messages: list[dict[str, str]], schema: dict,
            schema_name: str, trace_meta: dict[str, str] | None = None,
            max_tokens: int | None = None) -> tuple[dict | None, str | None]:
        """One structured-output call. Returns (parsed, error); exactly one is None.

        Generic because the two tasks want different shapes: the utility pass
        wants one small object per turn, the efficiency pass one large object per
        trajectory. Keeping the transport in one place means retries, the trace
        header and the JSON salvage below are not implemented twice.
        """
        payload: dict = {
            "model": self.judge_id,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema},
            },
        }
        if self.send_temperature:
            payload["temperature"] = self.temperature
        payload["max_tokens"] = max_tokens or DEFAULT_MAX_TOKENS
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if trace_meta:
            headers["x-trace-meta"] = json.dumps(trace_meta, sort_keys=True)

        # The efficiency pass asks for 262,144 output tokens, which kimi-k3 (the
        # default judge) accepts and Anthropic/Gemini judges refuse outright with
        # "max_tokens: N > M". The cap is learned from that refusal once per judge
        # and applied to every later call, so a different judge needs no table.
        cap = self._max_tokens_cap.get(self.judge_id)
        if cap and payload["max_tokens"] > cap:
            payload["max_tokens"] = cap

        last = "not attempted"
        for attempt in range(self.max_attempts):
            try:
                r = self._client.post(f"{self.base_url}/chat/completions",
                                      headers=headers, json=payload)
                if r.status_code == 400 and (m := re.search(
                        r"max_tokens: (\d+) > (\d+)", r.text)):
                    cap = int(m.group(2))
                    self._max_tokens_cap[self.judge_id] = cap
                    payload["max_tokens"] = cap
                    last = f"max_tokens capped at {cap} by the provider, retrying"
                    continue
                if r.status_code != 200:
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                else:
                    body = r.json()
                    msg = body["choices"][0]["message"]
                    text = (msg.get("content") or "").strip()
                    stop = (body["choices"][0].get("finish_reason")
                            or body["choices"][0].get("stop_reason") or "")
                    # Truncation and a malformed answer need different responses:
                    # one wants a bigger budget, the other is broken. Reporting
                    # both as "no JSON object" hid 78 truncated calls behind a
                    # message that suggested the model had misbehaved.
                    if stop == "length":
                        reasoning = len(msg.get("reasoning_content") or "")
                        last = (f"truncated at max_tokens="
                                f"{payload['max_tokens']} "
                                f"({reasoning} chars of reasoning, "
                                f"{len(text)} of content)")
                    elif not text:
                        last = "empty response"
                    else:
                        # Structured output is requested but not every provider
                        # honours it strictly, and a hard parse failure would
                        # throw away a call already paid for.
                        m = re.search(r"\{.*\}", text, re.DOTALL)
                        if not m:
                            last = f"no JSON object: {text[:120]}"
                        else:
                            try:
                                return json.loads(m.group(0)), None
                            except json.JSONDecodeError as e:
                                last = f"bad JSON: {e}"
            except Exception as e:  # network, timeout, malformed body
                last = f"{e.__class__.__name__}: {e}"
            time.sleep(2 ** attempt)
        return None, last

    def label(self, messages: list[dict[str, str]],
              trace_meta: dict[str, str] | None = None) -> Verdict:
        d, err = self.ask(messages, RESPONSE_SCHEMA, "state_label", trace_meta)
        if d is None:
            return Verdict(None, None, "", err)
        return _from_dict(d)

    def close(self) -> None:
        self._client.close()
