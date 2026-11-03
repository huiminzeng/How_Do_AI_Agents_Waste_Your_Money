"""Patch stdlib urllib OpenerDirector.open."""

from __future__ import annotations

import functools

from ..config import header_for
from .base import BasePatcher

__all__ = ["Patcher"]


class Patcher(BasePatcher):
    name = "urllib"
    modules = ("urllib.request",)

    def __init__(self) -> None:
        self._orig = None

    def available(self) -> bool:
        return True

    def patch(self) -> list[str]:
        import urllib.request as u

        orig = u.OpenerDirector.open
        if getattr(orig, "_llm_cost_tracker_tracer_patched", False):
            return []

        @functools.wraps(orig)
        def open(self, fullurl, *args, **kwargs):  # type: ignore[no-untyped-def]
            url = fullurl if isinstance(fullurl, str) else fullurl.full_url
            info = header_for(url)
            if info is not None:
                name, value = info
                if isinstance(fullurl, str):
                    fullurl = u.Request(fullurl)
                fullurl.add_header(name, value)
            return orig(self, fullurl, *args, **kwargs)

        open._llm_cost_tracker_tracer_patched = True  # type: ignore[attr-defined]
        self._orig = orig
        u.OpenerDirector.open = open
        return ["urllib.request.OpenerDirector.open"]

    def unpatch(self) -> None:
        if self._orig is not None:
            import urllib.request as u

            u.OpenerDirector.open = self._orig
            self._orig = None

