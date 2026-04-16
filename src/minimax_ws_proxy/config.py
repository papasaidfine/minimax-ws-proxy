from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

PASSTHRU_KEY_SENTINEL = "$PASSTHRU_API_KEY"

_ENV_REF_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$|^\$([A-Za-z_][A-Za-z0-9_]*)$")


def _resolve(value: str, *, allow_passthru: bool) -> str:
    """Interpolate magic strings.

    - `$PASSTHRU_API_KEY` -> preserved as sentinel; resolved per-request to
      the client's `x-api-key`. Only valid for REST backend `api_key`.
    - `$VAR` / `${VAR}` -> looked up in os.environ at load time (`.env` is
      loaded into os.environ at startup). Missing var raises ValueError.
    - anything else -> returned as-is (literal).
    """
    if value == PASSTHRU_KEY_SENTINEL:
        if not allow_passthru:
            raise ValueError(
                f"{PASSTHRU_KEY_SENTINEL!r} is only valid for REST backend api_key; "
                "MCP env values are fixed at subprocess spawn time"
            )
        return value
    m = _ENV_REF_RE.match(value)
    if not m:
        return value
    name = m.group(1) or m.group(2)
    if name not in os.environ:
        raise ValueError(f"config references env var ${name} which is not set")
    return os.environ[name]


@dataclass
class BackendConfig:
    name: str
    type: str  # "rest" or "mcp"
    # rest
    url: str = ""
    api_key: str | None = None  # None = no auth header; $PASSTHRU_API_KEY = client key
    # mcp
    command: str = "uvx"
    args: list[str] = field(default_factory=lambda: ["minimax-coding-plan-mcp", "-y"])
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ProxyConfig:
    name: str
    listen: str  # "host:port"
    upstream: str  # base URL, e.g. "https://api.minimaxi.com/anthropic"
    search_backend: str | None = None  # key into backends; None = pure transparent proxy

    @property
    def host(self) -> str:
        return self.listen.rsplit(":", 1)[0]

    @property
    def port(self) -> int:
        return int(self.listen.rsplit(":", 1)[1])


@dataclass
class AppConfig:
    backends: dict[str, BackendConfig]
    proxies: list[ProxyConfig]


def _build_backend(name: str, cfg: dict) -> BackendConfig:
    btype = cfg.get("type")
    if btype == "rest":
        raw_key = cfg.get("api_key")
        api_key = _resolve(raw_key, allow_passthru=True) if isinstance(raw_key, str) else None
        return BackendConfig(
            name=name,
            type="rest",
            url=cfg.get("url", ""),
            api_key=api_key,
        )
    if btype == "mcp":
        env = {k: _resolve(v, allow_passthru=False) for k, v in (cfg.get("env") or {}).items()}
        return BackendConfig(
            name=name,
            type="mcp",
            command=cfg.get("command", "uvx"),
            args=cfg.get("args") or ["minimax-coding-plan-mcp", "-y"],
            env=env,
        )
    raise ValueError(f"backend {name!r}: unknown type {btype!r} (expected 'rest' or 'mcp')")


def load_config(path: str) -> AppConfig:
    with open(path) as f:
        raw = json.load(f)

    backends: dict[str, BackendConfig] = {
        name: _build_backend(name, cfg) for name, cfg in raw.get("backends", {}).items()
    }

    proxies: list[ProxyConfig] = []
    for entry in raw.get("proxies", []):
        ref = entry.get("search_backend")
        if ref and ref not in backends:
            raise ValueError(f"proxy {entry['name']!r} references unknown backend {ref!r}")
        proxies.append(ProxyConfig(**entry))

    if not proxies:
        raise ValueError("config.json must define at least one proxy")

    return AppConfig(backends=backends, proxies=proxies)
