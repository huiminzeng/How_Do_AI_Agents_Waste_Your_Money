"""Patch urllib3 HTTPConnectionPool.urlopen."""

from __future__ import annotations

import functools

from ..config import header_for
from .base import BasePatcher

__all__ = ["Patcher"]


class Patcher(BasePatcher):
    name = "urllib3"
    modules = ("urllib3",)

    def __init__(self) -> None:
        self._orig = None

    def patch(self) -> list[str]:
        mod = self._import("urllib3.connectionpool")
        if mod is None:
            return []
        orig = mod.HTTPConnectionPool.urlopen
        if getattr(orig, "_llm_cost_tracker_tracer_patched", False):
            return []

        @functools.wraps(orig)
        def urlopen(self, method, url, *args, **kwargs):  # type: ignore[no-untyped-def]
            origin = f"{self.scheme}://{self.host}:{self.port}" if self.port else f"{self.scheme}://{self.host}"
            info = header_for(origin)
            if info is not None:
                name, value = info
                headers = kwargs.get("headers")
                if headers is None:
                    headers = self.headers
                headers = dict(headers) if headers else {}
                headers[name] = value
                kwargs["headers"] = headers
            return orig(self, method, url, *args, **kwargs)

        urlopen._llm_cost_tracker_tracer_patched = True  # type: ignore[attr-defined]
        self._orig = orig
        mod.HTTPConnectionPool.urlopen = urlopen
        return ["urllib3.HTTPConnectionPool.urlopen"]

    def unpatch(self) -> None:
        if self._orig is not None:
            mod = self._import("urllib3.connectionpool")
            if mod is not None:
                mod.HTTPConnectionPool.urlopen = self._orig
            self._orig = None

