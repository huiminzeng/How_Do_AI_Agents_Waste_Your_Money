"""Runtime tracer config and header encoding."""

from __future__ import annotations

import base64
import json
import os
import warnings
from urllib.parse import urlsplit

from .context import current

__all__ = ["Config", "build_config", "header_for", "set_config", "clear_config"]

DEFAULT_HEADER_NAME = "X-Trace-Context"
DEFAULT_ENV_VAR = "TRACE_PROXY_BASE_URL"
DEFAULT_SIZE_WARN_BYTES = 8192

_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    scheme = (parts.scheme or "").lower()
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is None:
        port = _DEFAULT_PORTS.get(scheme)
    return (scheme, host, port)


class Config:
    def __init__(
        self,
        proxy_base_url: str,
        header_name: str = DEFAULT_HEADER_NAME,
        size_warn_bytes: int = DEFAULT_SIZE_WARN_BYTES,
    ) -> None:
        if not proxy_base_url:
            raise ValueError("proxy_base_url is required")
        self.proxy_base_url = proxy_base_url
        self.header_name = header_name
        self.size_warn_bytes = size_warn_bytes
        self._proxy_origin = _origin(proxy_base_url)

    def matches(self, url: str) -> bool:
        try:
            return _origin(url) == self._proxy_origin
        except Exception:
            return False

    def encode(self, ctx: dict[str, str]) -> str:
        raw = json.dumps(ctx, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        out = base64.b64encode(raw).decode("ascii")
        if len(out) > self.size_warn_bytes:
            warnings.warn(
                f"llm_cost_tracker.tracer: encoded trace context is {len(out)} bytes; "
                f"this may exceed proxy header limits.",
                stacklevel=2,
            )
        return out


_config: Config | None = None


def set_config(cfg: Config) -> None:
    global _config
    _config = cfg


def clear_config() -> None:
    global _config
    _config = None


def build_config(
    proxy_base_url: str | None = None,
    header_name: str = DEFAULT_HEADER_NAME,
    size_warn_bytes: int = DEFAULT_SIZE_WARN_BYTES,
    env_var: str = DEFAULT_ENV_VAR,
) -> Config:
    if proxy_base_url is None:
        proxy_base_url = os.environ.get(env_var)
    if not proxy_base_url:
        raise ValueError(
            f"No proxy URL provided and ${env_var} is not set. "
            f"Pass install(proxy_base_url=...) or set {env_var}."
        )
    return Config(
        proxy_base_url=proxy_base_url,
        header_name=header_name,
        size_warn_bytes=size_warn_bytes,
    )


def header_for(url: str) -> tuple[str, str] | None:
    cfg = _config
    if cfg is None:
        return None
    ctx = current()
    if not ctx:
        return None
    if not cfg.matches(url):
        return None
    return cfg.header_name, cfg.encode(ctx)

