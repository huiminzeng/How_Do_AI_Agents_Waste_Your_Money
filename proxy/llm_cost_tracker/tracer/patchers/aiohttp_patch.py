"""Patch aiohttp ClientSession._request."""

from __future__ import annotations

import functools

from ..config import header_for
from .base import BasePatcher

__all__ = ["Patcher"]


class Patcher(BasePatcher):
    name = "aiohttp"
    modules = ("aiohttp",)

    def __init__(self) -> None:
        self._orig = None

    def patch(self) -> list[str]:
        mod = self._import("aiohttp.client")
        if mod is None:
            return []
        orig = mod.ClientSession._request
        if getattr(orig, "_llm_cost_tracker_tracer_patched", False):
            return []

        @functools.wraps(orig)
        async def _request(self, method, str_or_url, **kwargs):  # type: ignore[no-untyped-def]
            info = header_for(str(str_or_url))
            if info is not None:
                name, value = info
                from multidict import CIMultiDict

                headers = kwargs.get("headers")
                merged = CIMultiDict(headers) if headers else CIMultiDict()
                merged[name] = value
                kwargs["headers"] = merged
            return await orig(self, method, str_or_url, **kwargs)

        _request._llm_cost_tracker_tracer_patched = True  # type: ignore[attr-defined]
        self._orig = orig
        mod.ClientSession._request = _request
        return ["aiohttp.ClientSession._request"]

    def unpatch(self) -> None:
        if self._orig is not None:
            mod = self._import("aiohttp.client")
            if mod is not None:
                mod.ClientSession._request = self._orig
            self._orig = None

