"""Pluggable search backends.

All backends return data shaped as MiniMax's REST search response:
    {"organic": [{"title": ..., "link": ..., "snippet": ..., "date": ...}, ...]}

so the rest of the proxy can treat them uniformly. Backends that speak a
different wire format adapt inside `search()`.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Protocol

import aiohttp

if TYPE_CHECKING:
    from .config import BackendConfig

from .config import PASSTHRU_KEY_SENTINEL

log = logging.getLogger("minimax-ws-proxy.backends")


class SearchBackend(Protocol):
    async def search(self, query: str, api_key: str) -> dict: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


class RestBackend:
    """MiniMax-shaped REST search.

    Wire format (request): POST {url} with JSON body `{"q": query}` and
    `Authorization: Bearer <key>`.
    Wire format (response): `{"organic": [{title, link, snippet, date}, ...]}`.

    Other vendors (Brave, Serper, SerpAPI, Google CSE) use different shapes
    and need their own backend class — this one is not generic REST.

    `api_key` controls auth:
      - `$PASSTHRU_API_KEY` sentinel -> use the per-request client key
      - any other non-empty string -> literal Bearer token
      - None / empty -> no Authorization header
    """

    def __init__(self, url: str, api_key: str | None = PASSTHRU_KEY_SENTINEL):
        self._url = url
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def _resolve_key(self, client_key: str) -> str:
        if self._api_key == PASSTHRU_KEY_SENTINEL:
            return client_key
        return self._api_key or ""

    async def search(self, query: str, api_key: str) -> dict:
        if self._session is None:
            raise RuntimeError("RestBackend.search called before start()")
        headers = {"Content-Type": "application/json"}
        key = self._resolve_key(api_key)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        async with self._session.post(
            self._url,
            json={"q": query},
            headers=headers,
        ) as r:
            return await r.json()


class McpBackend:
    """Call an MCP server's web_search tool over stdio.

    The subprocess is spawned once at proxy startup and held open; a single
    MCP session services all concurrent requests. API key is fixed at spawn
    time via env vars — the client's per-request api_key is ignored.
    """

    def __init__(self, command: str = "uvx",
                 args: list[str] | None = None,
                 env: dict[str, str] | None = None):
        self._command = command
        self._args = args or ["minimax-coding-plan-mcp", "-y"]
        self._env = env or {}
        self._stack: AsyncExitStack | None = None
        self._session = None

    async def start(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        stack = AsyncExitStack()
        try:
            env = {**os.environ, **self._env}
            params = StdioServerParameters(
                command=self._command,
                args=self._args,
                env=env,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session
        log.info("MCP backend ready  cmd=%s %s", self._command, " ".join(self._args))

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    async def search(self, query: str, api_key: str) -> dict:
        del api_key
        if self._session is None:
            raise RuntimeError("McpBackend.search called before start()")
        result = await self._session.call_tool("web_search", {"query": query})
        return _mcp_to_organic(result)


def _mcp_to_organic(result) -> dict:
    if getattr(result, "isError", False):
        raise RuntimeError(f"MCP tool returned error: {_mcp_text(result)}")

    text = _mcp_text(result)
    if not text:
        return {"organic": []}

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"organic": [{"title": "search_result", "link": "", "snippet": text, "date": ""}]}

    if isinstance(data, dict) and "organic" in data:
        return data

    items = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("results", "organic_results", "web_search_results", "data"):
            if isinstance(data.get(key), list):
                items = data[key]
                break

    if items is None:
        return {"organic": [{"title": "search_result", "link": "", "snippet": text, "date": ""}]}

    organic = []
    for r in items:
        if not isinstance(r, dict):
            organic.append({"title": "", "link": "", "snippet": str(r), "date": ""})
            continue
        organic.append({
            "title": r.get("title") or r.get("name") or "",
            "link": r.get("link") or r.get("url") or r.get("href") or "",
            "snippet": r.get("snippet") or r.get("description") or r.get("content") or "",
            "date": r.get("date") or r.get("published") or "",
        })
    return {"organic": organic}


def _mcp_text(result) -> str:
    parts: list[str] = []
    for c in getattr(result, "content", []) or []:
        t = getattr(c, "text", None)
        if t:
            parts.append(t)
    return "\n".join(parts)


def build_backend(cfg: BackendConfig) -> SearchBackend:
    if cfg.type == "rest":
        return RestBackend(cfg.url, api_key=cfg.api_key)
    if cfg.type == "mcp":
        return McpBackend(command=cfg.command, args=cfg.args, env=cfg.env)
    raise RuntimeError(f"Unknown backend type {cfg.type!r}; expected 'rest' or 'mcp'")
