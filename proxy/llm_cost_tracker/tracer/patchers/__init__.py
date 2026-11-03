"""Patcher registry and install/uninstall orchestration."""

from __future__ import annotations

from .. import config as _config
from . import aiohttp_patch, httpx_patch, requests_patch, urllib3_patch, urllib_patch
from .base import BasePatcher

__all__ = ["install", "uninstall", "active_patchers"]

_REGISTRY: list[BasePatcher] = [
    httpx_patch.Patcher(),
    requests_patch.Patcher(),
    urllib3_patch.Patcher(),
    urllib_patch.Patcher(),
    aiohttp_patch.Patcher(),
]

_installed: list[BasePatcher] = []


def install(
    proxy_base_url: str | None = None,
    *,
    header_name: str | None = None,
    size_warn_bytes: int | None = None,
    env_var: str | None = None,
    modules: list[str] | None = None,
) -> dict[str, list[str]]:
    cfg_kwargs: dict[str, object] = {}
    if header_name is not None:
        cfg_kwargs["header_name"] = header_name
    if size_warn_bytes is not None:
        cfg_kwargs["size_warn_bytes"] = size_warn_bytes
    if env_var is not None:
        cfg_kwargs["env_var"] = env_var

    cfg = _config.build_config(proxy_base_url=proxy_base_url, **cfg_kwargs)
    _config.set_config(cfg)

    selected = set(modules) if modules else None
    patched: dict[str, list[str]] = {}
    for patcher in _REGISTRY:
        if selected is not None and patcher.name not in selected:
            continue
        if not patcher.available():
            continue
        labels = patcher.patch()
        if labels:
            patched[patcher.name] = labels
        if patcher not in _installed:
            _installed.append(patcher)
    return patched


def uninstall() -> None:
    while _installed:
        patcher = _installed.pop()
        try:
            patcher.unpatch()
        except Exception:
            pass
    _config.clear_config()


def active_patchers() -> list[str]:
    return [p.name for p in _installed]

