"""
minimax-ws-proxy: Lightweight Anthropic API proxy for MiniMax with web_search support.

API key is passed through from the client (x-api-key header).
The proxy only needs to know where MiniMax is.
"""

import json
import logging
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

import aiohttp
from aiohttp import web

log = logging.getLogger("minimax-ws-proxy")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MINIMAX_API_HOST = os.environ.get("MINIMAX_API_HOST", "https://api.minimaxi.com")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8082"))

INTERNAL_TOOL = "__proxy_web_search__"
ANTHROPIC_WS_TYPES = {"web_search_20250305", "web_search_20260209"}

# ---------------------------------------------------------------------------
# MiniMax search API
# ---------------------------------------------------------------------------


async def do_search(session: aiohttp.ClientSession, query: str, api_key: str) -> dict:
    """POST /v1/coding_plan/search -> {organic: [...], related_searches: [...]}"""
    async with session.post(
        f"{MINIMAX_API_HOST}/v1/coding_plan/search",
        json={"q": query},
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    ) as r:
        return await r.json()


# ---------------------------------------------------------------------------
# Request transformation
# ---------------------------------------------------------------------------


def _pick_internal_name(tools: list[dict]) -> str:
    """Avoid name collision with user-defined tools."""
    names = {t.get("name") for t in tools}
    if "web_search" not in names:
        return "web_search"
    return INTERNAL_TOOL


def strip_ws_tool(body: dict) -> tuple[dict, dict | None, str]:
    """Remove web_search_20250305 from tools, inject a regular tool.

    Returns (modified_body, ws_config_or_None, internal_tool_name).
    """
    tools = body.get("tools")
    if not tools:
        return body, None, ""

    ws_config = None
    kept: list[dict] = []
    for t in tools:
        if t.get("type") in ANTHROPIC_WS_TYPES:
            ws_config = t
        else:
            kept.append(t)

    if ws_config is None:
        return body, None, ""

    name = _pick_internal_name(kept)
    kept.append(
        {
            "name": name,
            "description": "Search the web for current information. Use this when you need up-to-date facts.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        }
    )
    return {**body, "tools": kept}, ws_config, name


def clean_history(messages: list[dict]) -> list[dict]:
    """Convert server_tool_use / web_search_tool_result blocks to text so MiniMax can digest them."""
    out: list[dict] = []
    for msg in messages:
        if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            out.append(msg)
            continue
        blocks: list[dict] = []
        for b in msg["content"]:
            bt = b.get("type", "")
            if bt == "server_tool_use":
                q = b.get("input", {}).get("query", "")
                blocks.append({"type": "text", "text": f"[Web search: {q}]"})
            elif bt == "web_search_tool_result":
                lines = ["[Search results:]"]
                for r in b.get("content", []):
                    if r.get("type") == "web_search_result":
                        lines.append(f"- {r.get('title', '')}: {r.get('url', '')}")
                        snip = r.get("encrypted_content", "")
                        if snip:
                            lines.append(f"  {snip}")
                blocks.append({"type": "text", "text": "\n".join(lines)})
            else:
                blocks.append(b)
        out.append({**msg, "content": blocks})
    return out


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------


def _fwd_headers(req_headers: dict, api_key: str) -> dict:
    h = {"content-type": "application/json", "x-api-key": api_key}
    for k in ("anthropic-version", "anthropic-beta"):
        v = req_headers.get(k)
        if v:
            h[k] = v
    return h


async def _call(session: aiohttp.ClientSession, body: dict, api_key: str, headers: dict) -> dict:
    """Non-streaming POST to MiniMax Anthropic endpoint."""
    async with session.post(
        f"{MINIMAX_API_HOST}/anthropic/v1/messages",
        json={**body, "stream": False},
        headers=_fwd_headers(headers, api_key),
    ) as r:
        return await r.json()


# ---------------------------------------------------------------------------
# Search resolution loop
# ---------------------------------------------------------------------------


async def resolve(
    session: aiohttp.ClientSession,
    body: dict,
    api_key: str,
    headers: dict,
    tool_name: str,
    max_uses: int = 5,
) -> dict:
    """Call MiniMax, resolve web-search tool calls in a loop, return final response."""
    used = 0
    collected: list[dict] = []  # accumulated content blocks
    total_in = 0
    total_out = 0

    while True:
        resp = await _call(session, body, api_key, headers)

        # Accumulate usage
        usage = resp.get("usage", {})
        total_in += usage.get("input_tokens", 0)
        total_out += usage.get("output_tokens", 0)

        # Check for errors from MiniMax
        if resp.get("type") == "error" or "error" in resp:
            resp["content"] = collected + resp.get("content", [])
            return resp

        content = resp.get("content", [])

        # Find search calls in this round
        searches = [
            b
            for b in content
            if b.get("type") == "tool_use" and b.get("name") == tool_name and used < max_uses
        ]

        if not searches:
            # Done - merge and return
            collected.extend(content)
            resp["content"] = collected
            resp["usage"] = {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "server_tool_use": {"web_search_requests": used},
            }
            return resp

        # Execute searches
        results_map: dict[str, tuple[str, dict]] = {}
        for sc in searches:
            used += 1
            q = sc.get("input", {}).get("query", "")
            log.info("Web search [%d/%d]: %s", used, max_uses, q)
            try:
                data = await do_search(session, q, api_key)
            except Exception as exc:
                log.warning("Search failed for %r: %s", q, exc)
                data = {"organic": []}
            results_map[sc["id"]] = (q, data)

        # Build transformed content for this round (preserve block order)
        has_other_tools = False
        for b in content:
            bid = b.get("id", "")
            if b.get("type") == "tool_use" and bid in results_map:
                q, data = results_map[bid]
                sid = f"srvtoolu_{uuid.uuid4().hex[:20]}"
                collected.append(
                    {"type": "server_tool_use", "id": sid, "name": "web_search", "input": {"query": q}}
                )
                collected.append(
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": sid,
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": r.get("link", ""),
                                "title": r.get("title", ""),
                                "encrypted_content": r.get("snippet", ""),
                                "page_age": r.get("date", ""),
                            }
                            for r in data.get("organic", [])
                        ],
                    }
                )
            elif b.get("type") == "tool_use":
                has_other_tools = True
                collected.append(b)
            else:
                collected.append(b)

        if has_other_tools:
            # Mixed: return now, let Claude Code handle the regular tool calls
            resp["content"] = collected
            resp["stop_reason"] = "tool_use"
            resp["usage"] = {"input_tokens": total_in, "output_tokens": total_out}
            return resp

        # Only search calls - build next round messages
        msgs = list(body.get("messages", []))
        msgs.append({"role": "assistant", "content": content})
        msgs.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": sc["id"],
                        "content": json.dumps(
                            results_map[sc["id"]][1].get("organic", []), ensure_ascii=False
                        ),
                    }
                    for sc in searches
                ],
            }
        )
        body = {**body, "messages": msgs}


# ---------------------------------------------------------------------------
# SSE re-streaming
# ---------------------------------------------------------------------------


def to_sse(resp: dict) -> bytes:
    """Convert a buffered JSON response into Anthropic SSE stream bytes."""
    chunks: list[str] = []

    def ev(event: str, data: dict):
        chunks.append(f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n")

    msg_id = resp.get("id", f"msg_{uuid.uuid4().hex[:20]}")
    ev(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": resp.get("model", ""),
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": resp.get("usage", {}).get("input_tokens", 0),
                    "output_tokens": 0,
                },
            },
        },
    )

    for i, block in enumerate(resp.get("content", [])):
        bt = block.get("type", "")

        if bt == "text":
            ev(
                "content_block_start",
                {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
            )
            text = block.get("text", "")
            for j in range(0, max(len(text), 1), 64):
                chunk = text[j : j + 64]
                if chunk:
                    ev(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": i,
                            "delta": {"type": "text_delta", "text": chunk},
                        },
                    )
            ev("content_block_stop", {"type": "content_block_stop", "index": i})

        elif bt == "thinking":
            ev(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            )
            text = block.get("thinking", "")
            for j in range(0, max(len(text), 1), 64):
                chunk = text[j : j + 64]
                if chunk:
                    ev(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": i,
                            "delta": {"type": "thinking_delta", "thinking": chunk},
                        },
                    )
            ev("content_block_stop", {"type": "content_block_stop", "index": i})

        elif bt == "server_tool_use":
            ev(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": {},
                    },
                },
            )
            ev(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}))},
                },
            )
            ev("content_block_stop", {"type": "content_block_stop", "index": i})

        elif bt == "web_search_tool_result":
            ev("content_block_start", {"type": "content_block_start", "index": i, "content_block": block})
            ev("content_block_stop", {"type": "content_block_stop", "index": i})

        elif bt == "tool_use":
            ev(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": {},
                    },
                },
            )
            ev(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}))},
                },
            )
            ev("content_block_stop", {"type": "content_block_stop", "index": i})

    ev(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": resp.get("stop_reason", "end_turn"), "stop_sequence": None},
            "usage": resp.get("usage", {"output_tokens": 0}),
        },
    )
    ev("message_stop", {"type": "message_stop"})

    return "".join(chunks).encode("utf-8")


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------


async def handle_messages(req: web.Request) -> web.StreamResponse:
    """POST /v1/messages - main proxy handler."""
    body = await req.json()
    api_key = req.headers["x-api-key"]
    hdrs = {k.lower(): v for k, v in req.headers.items()}
    streaming = body.get("stream", False)

    # Clean history
    if "messages" in body:
        body["messages"] = clean_history(body["messages"])

    # Extract web_search_20250305
    body, ws_cfg, tool_name = strip_ws_tool(body)
    session: aiohttp.ClientSession = req.app["http"]

    # --- No web search: transparent proxy ---
    if ws_cfg is None:
        if streaming:
            resp = web.StreamResponse(
                headers={"content-type": "text/event-stream", "cache-control": "no-cache"}
            )
            await resp.prepare(req)
            async with session.post(
                f"{MINIMAX_API_HOST}/anthropic/v1/messages",
                json=body,
                headers=_fwd_headers(hdrs, api_key),
            ) as upstream:
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)
            return resp
        else:
            data = await _call(session, body, api_key, hdrs)
            return web.json_response(data)

    # --- Has web search: resolve loop ---
    max_uses = ws_cfg.get("max_uses", 5)
    result = await resolve(session, body, api_key, hdrs, tool_name, max_uses)

    if streaming:
        resp = web.StreamResponse(
            headers={"content-type": "text/event-stream", "cache-control": "no-cache"}
        )
        await resp.prepare(req)
        await resp.write(to_sse(result))
        return resp
    else:
        return web.json_response(result)


async def handle_other(req: web.Request) -> web.StreamResponse:
    """Proxy any other path transparently."""
    raw = await req.read()
    api_key = req.headers["x-api-key"]
    hdrs = _fwd_headers({k.lower(): v for k, v in req.headers.items()}, api_key)
    session: aiohttp.ClientSession = req.app["http"]

    async with session.request(
        req.method,
        f"{MINIMAX_API_HOST}/anthropic{req.path}",
        data=raw,
        headers=hdrs,
    ) as upstream:
        return web.Response(
            status=upstream.status,
            body=await upstream.read(),
            content_type=upstream.content_type,
        )


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


async def on_startup(app: web.Application):
    app["http"] = aiohttp.ClientSession()


async def on_cleanup(app: web.Application):
    await app["http"].close()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/v1/messages", handle_messages)
    app.router.add_route("*", "/{path:.*}", handle_other)

    log.info("minimax-ws-proxy  %s:%d -> %s", HOST, PORT, MINIMAX_API_HOST)
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
