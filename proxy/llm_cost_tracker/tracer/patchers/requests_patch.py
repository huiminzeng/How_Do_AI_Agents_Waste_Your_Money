"""Patch requests Session.send."""

from __future__ import annotations

import functools

from ..config import header_for
from .base import BasePatcher

__all__ = ["Patcher"]


class Patcher(BasePatcher):
    name = "requests"
    modules = ("requests",)

    def __init__(self) -> None:
        self._orig = None

    def patch(self) -> list[str]:
        mod = self._import("requests.sessions")
        if mod is None:
            return []
        orig = mod.Session.send
        if getattr(orig, "_llm_cost_tracker_tracer_patched", False):
            return []

        @functools.wraps(orig)
        def send(self, request, **kwargs):  # type: ignore[no-untyped-def]
            info = header_for(request.url)
            if info is not None:
                name, value = info
                request.headers[name] = value
            return orig(self, request, **kwargs)

        send._llm_cost_tracker_tracer_patched = True  # type: ignore[attr-defined]
        self._orig = orig
        mod.Session.send = send
        return ["requests.Session.send"]

    def unpatch(self) -> None:
        if self._orig is not None:
            mod = self._import("requests.sessions")
            if mod is not None:
                mod.Session.send = self._orig
            self._orig = None

