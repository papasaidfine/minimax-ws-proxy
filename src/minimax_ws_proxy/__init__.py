"""
minimax-ws-proxy: Lightweight Anthropic API proxy for MiniMax with web_search_20250305 support.

API key and model are passed through from the client (x-api-key header).
The proxy only needs to know where MiniMax is.

Usage:
    uv run proxy.py
    # Then in Claude Code:
    #   ANTHROPIC_BASE_URL=http://127.0.0.1:8082
    #   ANTHROPIC_API_KEY=your_minimax_key

Environment variables (proxy-side):
    MINIMAX_API_HOST  - MiniMax API host (default: https://api.minimaxi.com)
    HOST              - Proxy listen host (default: 127.0.0.1)
    PORT              - Proxy listen port (default: 8082)
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


def _summarize_blocks(blocks: list) -> str:
    parts = []
    for b in blocks or []:
        bt = b.get("type", "?")
        if bt == "text":
            t = b.get("text", "")
            parts.append(f"text({len(t)}ch:{t[:60]!r})")
        elif bt == "thinking":
            parts.append(f"thinking({len(b.get('thinking',''))}ch)")
        elif bt in ("tool_use", "server_tool_use"):
            q = b.get("input", {}).get("query", "")
            parts.append(f"{bt}({b.get('name','?')},{b.get('id','')[:12]},q={q!r})")
        elif bt == "web_search_tool_result":
            c = b.get("content")
            n = len(c) if isinstance(c, list) else f"err:{c.get('error_code','?')}" if isinstance(c, dict) else "?"
            parts.append(f"ws_result(n={n})")
        elif bt == "tool_result":
            parts.append(f"tool_result({b.get('tool_use_id','')[:12]})")
        else:
            parts.append(bt)
    return "[" + ", ".join(parts) + "]"


def _usage(in_t: int, out_t: int, used: int) -> dict:
    return {
        "input_tokens": in_t,
        "output_tokens": out_t,
        "server_tool_use": {"web_search_requests": used, "web_fetch_requests": 0},
    }


# Anthropic-standard error types. Anything else gets bucketed into "api_error".
_SAFE_ERROR_TYPES = {
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "request_too_large",
    "rate_limit_error",
    "api_error",
    "overloaded_error",
}

_GENERIC_ERROR_MESSAGES = {
    "invalid_request_error": "Invalid request",
    "authentication_error": "Authentication failed",
    "permission_error": "Permission denied",
    "not_found_error": "Not found",
    "request_too_large": "Request too large",
    "rate_limit_error": "Rate limit exceeded",
    "api_error": "Upstream error",
    "overloaded_error": "Upstream overloaded",
}

_ERROR_HTTP_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "overloaded_error": 529,
}


def _sanitize_error(raw) -> dict:
    """Map any upstream error shape to a generic Anthropic error object.

    Only the error *type* is preserved (from a whitelist); the message is
    always replaced with a canned string so upstream vendor names, internal
    field identifiers, or backend URLs never reach the client.
    """
    inner: dict = {}
    if isinstance(raw, dict):
        candidate = raw.get("error", raw)
        if isinstance(candidate, dict):
            inner = candidate
    etype = inner.get("type") if isinstance(inner.get("type"), str) else "api_error"
    if etype not in _SAFE_ERROR_TYPES:
        etype = "api_error"
    return {"type": etype, "message": _GENERIC_ERROR_MESSAGES[etype]}


def _error_json_response(raw) -> web.Response:
    """Build a non-streaming Anthropic-spec error response with sanitized body."""
    sanitized = _sanitize_error(raw)
    status = _ERROR_HTTP_STATUS.get(sanitized["type"], 500)
    return web.json_response(
        {"type": "error", "error": sanitized},
        status=status,
    )


def _ws_result(sid: str, kind: str, payload) -> dict:
    if kind == "error":
        return {
            "type": "web_search_tool_result",
            "tool_use_id": sid,
            "content": {"type": "web_search_tool_result_error", "error_code": payload},
        }
    return {
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
            for r in payload.get("organic", [])
        ],
    }

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
    unsupported = [k for k in ("allowed_domains", "blocked_domains", "user_location") if k in ws_config]
    if unsupported:
        log.warning("web_search tool specified %s; MiniMax search API ignores these", unsupported)
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
    msgs = body.get("messages", [])
    tools = [t.get("name", t.get("type", "?")) for t in body.get("tools", [])]
    log.info("→ MiniMax: model=%s msgs=%d tools=%s", body.get("model"), len(msgs), tools)
    async with session.post(
        f"{MINIMAX_API_HOST}/anthropic/v1/messages",
        json={**body, "stream": False},
        headers=_fwd_headers(headers, api_key),
    ) as r:
        resp = await r.json()
    if resp.get("type") == "error" or "error" in resp:
        log.warning("← MiniMax ERROR: %s", resp)
    else:
        log.info(
            "← MiniMax: stop=%s usage=%s blocks=%s",
            resp.get("stop_reason"),
            resp.get("usage"),
            _summarize_blocks(resp.get("content", [])),
        )
    return resp


# ---------------------------------------------------------------------------
# Search resolution loop
# ---------------------------------------------------------------------------


def _is_ws_call(b: dict, tool_name: str) -> bool:
    return (
        b.get("type") in ("tool_use", "server_tool_use")
        and b.get("name") in (tool_name, "web_search")
    )


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
    collected: list[dict] = []
    total_in = 0
    total_out = 0
    round_idx = 0

    while True:
        round_idx += 1
        try:
            resp = await _call(session, body, api_key, headers)
        except Exception as exc:
            log.warning("resolve: upstream call failed at round %d: %r", round_idx, exc)
            return {"__error__": {"type": "api_error"}}
        usage = resp.get("usage", {})
        total_in += usage.get("input_tokens", 0)
        total_out += usage.get("output_tokens", 0)

        if resp.get("type") == "error" or "error" in resp:
            log.warning("resolve: upstream returned error at round %d", round_idx)
            return {"__error__": resp}

        content = resp.get("content", [])
        searches = [b for b in content if _is_ws_call(b, tool_name)]

        if not searches:
            collected.extend(content)
            resp["content"] = collected
            resp["usage"] = _usage(total_in, total_out, used)
            log.info(
                "✓ resolve done  rounds=%d searches=%d blocks=%s",
                round_idx, used, _summarize_blocks(collected),
            )
            return resp

        # results_map[id] = (query, (kind, payload))
        #   kind="ok"    -> payload is search data dict
        #   kind="error" -> payload is web_search error_code
        results_map: dict = {}
        for sc in searches:
            q = sc.get("input", {}).get("query", "")
            if used >= max_uses:
                log.info("  Web search SKIPPED max_uses_exceeded: %s", q)
                results_map[sc["id"]] = (q, ("error", "max_uses_exceeded"))
                continue
            used += 1
            try:
                data = await do_search(session, q, api_key)
            except Exception as exc:
                log.warning("  Web search [%d/%d] FAILED: %s (%s)", used, max_uses, q, exc)
                results_map[sc["id"]] = (q, ("error", "unavailable"))
                continue
            log.info(
                "  Web search [%d/%d]: %s → %d results",
                used, max_uses, q, len(data.get("organic", [])),
            )
            results_map[sc["id"]] = (q, ("ok", data))

        # Build this round's content (preserve block order)
        has_other_tools = False
        for b in content:
            bid = b.get("id", "")
            if b.get("type") in ("tool_use", "server_tool_use") and bid in results_map:
                q, (kind, payload) = results_map[bid]
                sid = f"srvtoolu_{uuid.uuid4().hex[:20]}"
                collected.append(
                    {"type": "server_tool_use", "id": sid, "name": "web_search", "input": {"query": q}}
                )
                collected.append(_ws_result(sid, kind, payload))
            elif b.get("type") == "tool_use":
                has_other_tools = True
                collected.append(b)
            else:
                collected.append(b)

        if has_other_tools:
            # Mixed: let Claude Code handle regular tool calls
            resp["content"] = collected
            resp["stop_reason"] = "tool_use"
            resp["usage"] = _usage(total_in, total_out, used)
            log.info("✓ resolve mixed-exit  rounds=%d searches=%d", round_idx, used)
            return resp

        # Normalize server_tool_use → tool_use for MiniMax's next turn
        normalized = [
            {**b, "type": "tool_use"} if b.get("type") == "server_tool_use" and b.get("id") in results_map else b
            for b in content
        ]

        tool_results = []
        for sc in searches:
            _, (kind, payload) = results_map[sc["id"]]
            tr = {"error": payload} if kind == "error" else payload.get("organic", [])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": sc["id"],
                "content": json.dumps(tr, ensure_ascii=False),
            })

        msgs = list(body.get("messages", []))
        msgs.append({"role": "assistant", "content": normalized})
        msgs.append({"role": "user", "content": tool_results})
        body = {**body, "messages": msgs}

        if round_idx > max_uses + 3:
            log.warning("resolve loop exceeded safety cap at round %d; bailing", round_idx)
            resp["content"] = collected
            resp["stop_reason"] = "end_turn"
            resp["usage"] = _usage(total_in, total_out, used)
            return resp


# ---------------------------------------------------------------------------
# SSE emission helpers
# ---------------------------------------------------------------------------


async def _sse(resp: web.StreamResponse, event: str, data: dict) -> None:
    await resp.write(
        f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()
    )


async def _sse_error(resp: web.StreamResponse, raw) -> None:
    """Emit an Anthropic-spec `error` SSE event with a sanitized payload.

    After this event the stream is terminal — Anthropic SDK will raise.
    Caller should return immediately without emitting message_stop.
    """
    await _sse(resp, "error", {"type": "error", "error": _sanitize_error(raw)})


async def _sse_block(resp: web.StreamResponse, idx: int, block: dict) -> None:
    bt = block.get("type", "")
    if bt in ("text", "thinking"):
        field = "text" if bt == "text" else "thinking"
        delta_type = "text_delta" if bt == "text" else "thinking_delta"
        await _sse(resp, "content_block_start",
                   {"type": "content_block_start", "index": idx,
                    "content_block": {"type": bt, field: ""}})
        text = block.get(field, "")
        for j in range(0, max(len(text), 1), 64):
            chunk = text[j : j + 64]
            if chunk:
                await _sse(resp, "content_block_delta",
                           {"type": "content_block_delta", "index": idx,
                            "delta": {"type": delta_type, field: chunk}})
        await _sse(resp, "content_block_stop", {"type": "content_block_stop", "index": idx})
    elif bt in ("tool_use", "server_tool_use"):
        await _sse(resp, "content_block_start",
                   {"type": "content_block_start", "index": idx,
                    "content_block": {"type": bt, "id": block["id"],
                                      "name": block["name"], "input": {}}})
        await _sse(resp, "content_block_delta",
                   {"type": "content_block_delta", "index": idx,
                    "delta": {"type": "input_json_delta",
                              "partial_json": json.dumps(block.get("input", {}))}})
        await _sse(resp, "content_block_stop", {"type": "content_block_stop", "index": idx})
    elif bt == "web_search_tool_result":
        await _sse(resp, "content_block_start",
                   {"type": "content_block_start", "index": idx, "content_block": block})
        await _sse(resp, "content_block_stop", {"type": "content_block_stop", "index": idx})


async def resolve_streaming(
    session: aiohttp.ClientSession,
    body: dict,
    api_key: str,
    headers: dict,
    tool_name: str,
    resp: web.StreamResponse,
    max_uses: int = 5,
) -> None:
    """Resolve web_search loop while streaming SSE to the client as each block becomes available."""
    msg_id = f"msg_{uuid.uuid4().hex[:20]}"
    await _sse(resp, "message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": body.get("model", ""),
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    used = 0
    idx = 0
    total_in = 0
    total_out = 0
    round_idx = 0
    stop_reason = "end_turn"

    while True:
        round_idx += 1
        try:
            r = await _call(session, body, api_key, headers)
        except Exception as exc:
            log.warning("resolve_streaming: upstream call failed at round %d: %r", round_idx, exc)
            await _sse_error(resp, {"type": "api_error"})
            return
        usage = r.get("usage", {})
        total_in += usage.get("input_tokens", 0)
        total_out += usage.get("output_tokens", 0)

        if r.get("type") == "error" or "error" in r:
            log.warning("resolve_streaming: upstream returned error at round %d", round_idx)
            await _sse_error(resp, r)
            return

        content = r.get("content", [])
        searches = [b for b in content if _is_ws_call(b, tool_name)]

        if not searches:
            for b in content:
                await _sse_block(resp, idx, b)
                idx += 1
            stop_reason = r.get("stop_reason", "end_turn")
            break

        results_map: dict = {}
        has_other_tools = False
        for b in content:
            bid = b.get("id", "")
            if _is_ws_call(b, tool_name):
                q = b.get("input", {}).get("query", "")
                sid = f"srvtoolu_{uuid.uuid4().hex[:20]}"
                # Emit server_tool_use first so UI shows "Searching: <query>" during the wait.
                await _sse_block(resp, idx,
                                 {"type": "server_tool_use", "id": sid, "name": "web_search",
                                  "input": {"query": q}})
                idx += 1
                if used >= max_uses:
                    log.info("  Web search SKIPPED max_uses_exceeded: %s", q)
                    kind, payload = "error", "max_uses_exceeded"
                else:
                    used += 1
                    try:
                        data = await do_search(session, q, api_key)
                        kind, payload = "ok", data
                        log.info("  Web search [%d/%d]: %s → %d results",
                                 used, max_uses, q, len(data.get("organic", [])))
                    except Exception as exc:
                        log.warning("  Web search [%d/%d] FAILED: %s (%s)",
                                    used, max_uses, q, exc)
                        kind, payload = "error", "unavailable"
                await _sse_block(resp, idx, _ws_result(sid, kind, payload))
                idx += 1
                results_map[bid] = (q, kind, payload)
            elif b.get("type") == "tool_use":
                has_other_tools = True
                await _sse_block(resp, idx, b)
                idx += 1
            else:
                await _sse_block(resp, idx, b)
                idx += 1

        if has_other_tools:
            stop_reason = "tool_use"
            break

        normalized = [
            {**b, "type": "tool_use"} if b.get("type") == "server_tool_use" and b.get("id") in results_map else b
            for b in content
        ]
        tool_results = []
        for sc in searches:
            _, kind, payload = results_map[sc["id"]]
            tr = {"error": payload} if kind == "error" else payload.get("organic", [])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": sc["id"],
                "content": json.dumps(tr, ensure_ascii=False),
            })
        msgs = list(body.get("messages", []))
        msgs.append({"role": "assistant", "content": normalized})
        msgs.append({"role": "user", "content": tool_results})
        body = {**body, "messages": msgs}

        if round_idx > max_uses + 3:
            log.warning("resolve_streaming loop exceeded safety cap at round %d; bailing", round_idx)
            break

    await _sse(resp, "message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": _usage(total_in, total_out, used),
    })
    await _sse(resp, "message_stop", {"type": "message_stop"})
    log.info("✓ resolve_streaming done  rounds=%d searches=%d", round_idx, used)


# ---------------------------------------------------------------------------
# SSE re-streaming (non-streaming fallback → SSE)
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

    # Debug: log incoming tool types
    if body.get("tools"):
        tool_types = [t.get("type", t.get("name", "?")) for t in body["tools"]]
        log.info("Incoming tools: %s  stream=%s", tool_types, streaming)

    # Clean history
    if "messages" in body:
        body["messages"] = clean_history(body["messages"])

    # Extract web_search_20250305
    body, ws_cfg, tool_name = strip_ws_tool(body)
    if ws_cfg:
        log.info("Stripped %s -> injected tool '%s'", ws_cfg.get("type"), tool_name)
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
            try:
                data = await _call(session, body, api_key, hdrs)
            except Exception as exc:
                log.warning("transparent proxy: upstream call failed: %r", exc)
                return _error_json_response({"type": "api_error"})
            if data.get("type") == "error" or "error" in data:
                return _error_json_response(data)
            return web.json_response(data)

    # --- Has web search: resolve loop ---
    max_uses = ws_cfg.get("max_uses", 5)

    if streaming:
        resp = web.StreamResponse(
            headers={"content-type": "text/event-stream", "cache-control": "no-cache"}
        )
        await resp.prepare(req)
        await resolve_streaming(session, body, api_key, hdrs, tool_name, resp, max_uses)
        return resp
    else:
        result = await resolve(session, body, api_key, hdrs, tool_name, max_uses)
        if "__error__" in result:
            return _error_json_response(result["__error__"])
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
