"""Base patcher class."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

__all__ = ["BasePatcher"]


class BasePatcher:
    name: str = ""
    modules: tuple[str, ...] = ()

    def available(self) -> bool:
        for mod in self.modules:
            try:
                if importlib.util.find_spec(mod) is not None:
                    return True
            except (ImportError, ValueError):
                continue
        return False

    def patch(self) -> list[str]:
        raise NotImplementedError

    def unpatch(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _import(name: str) -> ModuleType | None:
        try:
            return importlib.import_module(name)
        except ImportError:
            return None

