"""Patch httpx Client.send / AsyncClient.send."""

from __future__ import annotations

import functools

from ..config import header_for
from .base import BasePatcher

__all__ = ["Patcher"]


def _wrap_send(cls: type) -> object | None:
    orig = cls.send
    if getattr(orig, "_llm_cost_tracker_tracer_patched", False):
        return None

    @functools.wraps(orig)
    def send(self, request, *args, **kwargs):  # type: ignore[no-untyped-def]
        info = header_for(str(request.url))
        if info is not None:
            name, value = info
            request.headers[name] = value
        return orig(self, request, *args, **kwargs)

    send._llm_cost_tracker_tracer_patched = True  # type: ignore[attr-defined]
    cls.send = send
    return orig


class Patcher(BasePatcher):
    name = "httpx"
    modules = ("httpx", "httpxyz")

    def __init__(self) -> None:
        self._originals: list[tuple[type, object]] = []

    def patch(self) -> list[str]:
        labels: list[str] = []
        for mod_name in self.modules:
            mod = self._import(mod_name)
            if mod is None:
                continue
            for cls_name in ("Client", "AsyncClient"):
                cls = getattr(mod, cls_name, None)
                if cls is None:
                    continue
                orig = _wrap_send(cls)
                if orig is not None:
                    self._originals.append((cls, orig))
                    labels.append(f"{mod_name}.{cls_name}.send")
        return labels

    def unpatch(self) -> None:
        for cls, orig in self._originals:
            cls.send = orig  # type: ignore[assignment]
        self._originals.clear()

