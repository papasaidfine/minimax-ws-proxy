"""
minimax-ws-proxy: Lightweight Anthropic API proxy with web_search_20250305 support.

Reads everything from config.json (host/port/upstream/backends). A `.env`
file is loaded into the process environment so config.json can reference
secrets via `$VAR` / `${VAR}`.

Usage:
    uv run minimax-ws-proxy                       # uses ./config.json
    uv run minimax-ws-proxy --config /path/to.json
"""

import asyncio
import json
import logging
import os
import secrets
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()  # populate os.environ from .env so $VAR refs in config resolve

import aiohttp
from aiohttp import web

from .backends import SearchBackend, build_backend
from .config import AppConfig, load_config

log = logging.getLogger("minimax-ws-proxy")

INTERNAL_TOOL = "__proxy_web_search__"
ANTHROPIC_WS_TYPES = {"web_search_20250305", "web_search_20260209"}

# Hop-by-hop + length headers we must not blindly forward when mirroring an
# upstream response. aiohttp handles framing itself.
_HOP_BY_HOP_HEADERS = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
})


def _anthropic_msg_id() -> str:
    """Mint a message id that matches the Anthropic `msg_01<22hex>` shape."""
    return "msg_01" + secrets.token_hex(11)


def _scrub_claude_message(obj: dict, client_model: str) -> None:
    """Rewrite MiniMax-specific fields in a Claude `message` object in-place.

    MiniMax's Anthropic-compatible endpoint returns `model` as its internal
    name (e.g. `MiniMax-M2.7`), a 32-hex `id`, and a `base_resp` envelope —
    all of which fingerprint the upstream. Replace them so consumers see
    something shaped like a real Anthropic response.
    """
    if not isinstance(obj, dict):
        return
    if client_model:
        obj["model"] = client_model
    cur_id = obj.get("id")
    if isinstance(cur_id, str) and not cur_id.startswith("msg_"):
        obj["id"] = _anthropic_msg_id()
    obj.pop("base_resp", None)


def _scrub_response_body(raw: bytes, client_model: str) -> bytes:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return raw
    if isinstance(data, dict) and data.get("type") == "message":
        _scrub_claude_message(data, client_model)
        return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return raw


def _scrub_message_start_event(event_bytes: bytes, client_model: str) -> bytes:
    """Rewrite the `message_start` SSE event's embedded message object.

    Works on a single complete event (one or more lines terminated by a blank
    line). Returns the bytes unchanged if parsing fails or if this isn't a
    message_start event.
    """
    if b"message_start" not in event_bytes:
        return event_bytes
    try:
        text = event_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return event_bytes
    out_lines: list[str] = []
    changed = False
    for line in text.split("\n"):
        if line.startswith("data:"):
            payload = line[len("data:"):].lstrip()
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                out_lines.append(line)
                continue
            if isinstance(data, dict) and data.get("type") == "message_start":
                msg = data.get("message")
                if isinstance(msg, dict):
                    _scrub_claude_message(msg, client_model)
                    out_lines.append("data: " + json.dumps(data, ensure_ascii=False, separators=(",", ":")))
                    changed = True
                    continue
        out_lines.append(line)
    if not changed:
        return event_bytes
    return "\n".join(out_lines).encode("utf-8")


# ---------------------------------------------------------------------------
# Helpers (unchanged)
# ---------------------------------------------------------------------------


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
# Request transformation
# ---------------------------------------------------------------------------


def _pick_internal_name(tools: list[dict]) -> str:
    names = {t.get("name") for t in tools}
    if "web_search" not in names:
        return "web_search"
    return INTERNAL_TOOL


def strip_ws_tool(body: dict) -> tuple[dict, dict | None, str]:
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
        log.warning("web_search tool specified %s; search backend ignores these", unsupported)
    return {**body, "tools": kept}, ws_config, name


def clean_history(messages: list[dict]) -> list[dict]:
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


async def _call(
    session: aiohttp.ClientSession,
    upstream: str,
    body: dict,
    api_key: str,
    headers: dict,
) -> dict:
    msgs = body.get("messages", [])
    tools = [t.get("name", t.get("type", "?")) for t in body.get("tools", [])]
    log.info("→ upstream: model=%s msgs=%d tools=%s", body.get("model"), len(msgs), tools)
    async with session.post(
        f"{upstream}/v1/messages",
        json={**body, "stream": False},
        headers=_fwd_headers(headers, api_key),
    ) as r:
        resp = await r.json()
    if resp.get("type") == "error" or "error" in resp:
        log.warning("← upstream ERROR: %s", resp)
    else:
        log.info(
            "← upstream: stop=%s usage=%s blocks=%s",
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
    upstream: str,
    backend: SearchBackend,
    body: dict,
    api_key: str,
    headers: dict,
    tool_name: str,
    max_uses: int = 5,
) -> dict:
    used = 0
    collected: list[dict] = []
    total_in = 0
    total_out = 0
    round_idx = 0

    while True:
        round_idx += 1
        try:
            resp = await _call(session, upstream, body, api_key, headers)
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

        results_map: dict = {}
        for sc in searches:
            q = sc.get("input", {}).get("query", "")
            if used >= max_uses:
                log.info("  Web search SKIPPED max_uses_exceeded: %s", q)
                results_map[sc["id"]] = (q, ("error", "max_uses_exceeded"))
                continue
            used += 1
            try:
                data = await backend.search(q, api_key)
            except Exception as exc:
                log.warning("  Web search [%d/%d] FAILED: %s (%s)", used, max_uses, q, exc)
                results_map[sc["id"]] = (q, ("error", "unavailable"))
                continue
            log.info(
                "  Web search [%d/%d]: %s → %d results",
                used, max_uses, q, len(data.get("organic", [])),
            )
            results_map[sc["id"]] = (q, ("ok", data))

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
            resp["content"] = collected
            resp["stop_reason"] = "tool_use"
            resp["usage"] = _usage(total_in, total_out, used)
            log.info("✓ resolve mixed-exit  rounds=%d searches=%d", round_idx, used)
            return resp

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
    upstream: str,
    backend: SearchBackend,
    body: dict,
    api_key: str,
    headers: dict,
    tool_name: str,
    resp: web.StreamResponse,
    max_uses: int = 5,
) -> None:
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
            r = await _call(session, upstream, body, api_key, headers)
        except Exception as exc:
            log.warning("resolve_streaming: upstream call failed at round %d: %r", round_idx, exc)
            await _sse_error(resp, {"type": "api_error"})
            # After event: error, emit message_stop (needed by strict consumers
            # like sub2api) but skip the synthetic message_delta — a trailing
            # stop_reason=end_turn would contradict the error we just sent.
            await _sse(resp, "message_stop", {"type": "message_stop"})
            return
        usage = r.get("usage", {})
        total_in += usage.get("input_tokens", 0)
        total_out += usage.get("output_tokens", 0)

        if r.get("type") == "error" or "error" in r:
            log.warning("resolve_streaming: upstream returned error at round %d", round_idx)
            await _sse_error(resp, r)
            await _sse(resp, "message_stop", {"type": "message_stop"})
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
                        data = await backend.search(q, api_key)
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
# HTTP handlers — read upstream/backend from app context
# ---------------------------------------------------------------------------


async def handle_messages(req: web.Request) -> web.StreamResponse:
    body = await req.json()
    api_key = req.headers["x-api-key"]
    hdrs = {k.lower(): v for k, v in req.headers.items()}
    streaming = body.get("stream", False)
    upstream: str = req.app["upstream"]

    if body.get("tools"):
        tool_types = [t.get("type", t.get("name", "?")) for t in body["tools"]]
        log.info("Incoming tools: %s  stream=%s", tool_types, streaming)

    if "messages" in body:
        body["messages"] = clean_history(body["messages"])

    body, ws_cfg, tool_name = strip_ws_tool(body)
    if ws_cfg:
        log.info("Stripped %s -> injected tool '%s'", ws_cfg.get("type"), tool_name)
    session: aiohttp.ClientSession = req.app["http"]

    client_model = body.get("model") or ""

    # --- No web search: transparent proxy ---
    if ws_cfg is None:
        if streaming:
            async with session.post(
                f"{upstream}/v1/messages",
                json=body,
                headers=_fwd_headers(hdrs, api_key),
            ) as up:
                # Mirror upstream status + headers verbatim. Rewrite only the
                # `message_start` event's embedded message object to scrub
                # MiniMax-specific fingerprints (`model`, `id`, `base_resp`);
                # everything else streams unchanged.
                fwd = {k: v for k, v in up.headers.items()
                       if k.lower() not in _HOP_BY_HOP_HEADERS}
                resp = web.StreamResponse(status=up.status, headers=fwd)
                await resp.prepare(req)
                is_sse = up.status == 200 and "text/event-stream" in (up.content_type or "")
                saw_stop = False
                tail = b""
                # Buffer only until we've emitted the first SSE event
                # (message_start). After that, pipe bytes directly.
                first_event_buf = b""
                first_event_done = not is_sse  # non-SSE → no rewriting, flush as-is
                try:
                    async for chunk in up.content.iter_any():
                        out: bytes
                        if first_event_done:
                            out = chunk
                        else:
                            first_event_buf += chunk
                            sep = first_event_buf.find(b"\n\n")
                            if sep < 0:
                                # Need more bytes to see the end of the first event.
                                continue
                            first = first_event_buf[:sep + 2]
                            rest = first_event_buf[sep + 2:]
                            first = _scrub_message_start_event(first, client_model)
                            out = first + rest
                            first_event_done = True
                            first_event_buf = b""
                        if is_sse and not saw_stop:
                            combined = tail + out
                            if b"message_stop" in combined:
                                saw_stop = True
                            tail = combined[-32:]
                        await resp.write(out)
                    # Upstream ended before first event boundary — flush remaining buf.
                    if not first_event_done and first_event_buf:
                        await resp.write(first_event_buf)
                except Exception as exc:
                    log.warning("transparent stream: read failed mid-stream: %r", exc)
                # If upstream promised an SSE stream but ended without
                # message_stop (mid-stream truncation), append a bare
                # message_stop so strict downstream parsers can terminate.
                if is_sse and not saw_stop:
                    log.warning("transparent stream: SSE ended without message_stop; emitting synthetic")
                    await resp.write(
                        b"event: message_stop\n"
                        b'data: {"type":"message_stop"}\n\n'
                    )
            return resp
        else:
            # Non-streaming transparent path: mirror upstream status + body but
            # scrub MiniMax fingerprints from successful message responses.
            async with session.post(
                f"{upstream}/v1/messages",
                json=body,
                headers=_fwd_headers(hdrs, api_key),
            ) as up:
                raw = await up.read()
                fwd = {k: v for k, v in up.headers.items()
                       if k.lower() not in _HOP_BY_HOP_HEADERS}
                if up.status == 200 and "application/json" in (up.content_type or ""):
                    raw = _scrub_response_body(raw, client_model)
                resp = web.Response(status=up.status, body=raw, headers=fwd)
                return resp

    # --- Has web search: resolve loop ---
    max_uses = ws_cfg.get("max_uses", 5)
    backend: SearchBackend = req.app["backend"]

    if streaming:
        resp = web.StreamResponse(
            headers={"content-type": "text/event-stream", "cache-control": "no-cache"}
        )
        await resp.prepare(req)
        await resolve_streaming(session, upstream, backend, body, api_key, hdrs, tool_name, resp, max_uses)
        return resp
    else:
        result = await resolve(session, upstream, backend, body, api_key, hdrs, tool_name, max_uses)
        if "__error__" in result:
            return _error_json_response(result["__error__"])
        _scrub_claude_message(result, client_model)
        return web.json_response(result)


async def handle_other(req: web.Request) -> web.StreamResponse:
    raw = await req.read()
    api_key = req.headers["x-api-key"]
    hdrs = _fwd_headers({k.lower(): v for k, v in req.headers.items()}, api_key)
    session: aiohttp.ClientSession = req.app["http"]
    upstream: str = req.app["upstream"]

    async with session.request(
        req.method,
        f"{upstream}{req.path}",
        data=raw,
        headers=hdrs,
    ) as up:
        return web.Response(
            status=up.status,
            body=await up.read(),
            content_type=up.content_type,
        )


# ---------------------------------------------------------------------------
# App factory & multi-proxy runner
# ---------------------------------------------------------------------------


def create_app(upstream: str, backend: SearchBackend | None) -> web.Application:
    app = web.Application()
    app["upstream"] = upstream
    if backend is not None:
        app["backend"] = backend

    async def _on_startup(app: web.Application):
        app["http"] = aiohttp.ClientSession()

    async def _on_cleanup(app: web.Application):
        await app["http"].close()

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    app.router.add_post("/v1/messages", handle_messages)
    app.router.add_route("*", "/{path:.*}", handle_other)
    return app


async def _run_multi(config: AppConfig) -> None:
    backends: dict[str, SearchBackend] = {}
    for name, bcfg in config.backends.items():
        b = build_backend(bcfg)
        await b.start()
        backends[name] = b
        log.info("Backend '%s' (%s) started", name, bcfg.type)

    runners: list[web.AppRunner] = []
    sites: list[web.TCPSite] = []

    try:
        for pcfg in config.proxies:
            backend = backends.get(pcfg.search_backend) if pcfg.search_backend else None
            app = create_app(pcfg.upstream, backend)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, pcfg.host, pcfg.port)
            await site.start()
            runners.append(runner)
            sites.append(site)
            log.info("Proxy '%s' listening on %s -> %s  backend=%s",
                     pcfg.name, pcfg.listen, pcfg.upstream, pcfg.search_backend or "(none)")

        await asyncio.Event().wait()  # run forever
    finally:
        for runner in runners:
            await runner.cleanup()
        for b in backends.values():
            await b.stop()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        config_path = sys.argv[idx + 1]
    elif os.environ.get("CONFIG_PATH"):
        config_path = os.environ["CONFIG_PATH"]
    else:
        config_path = os.path.join(os.getcwd(), "config.json")

    if not os.path.isfile(config_path):
        sys.exit(f"config not found: {config_path}\n"
                 "create one (see config.example.json) or pass --config <path>.")

    log.info("Loading config from %s", config_path)
    config = load_config(config_path)

    if len(config.proxies) == 1:
        p = config.proxies[0]
        log.info("minimax-ws-proxy  %s -> %s  backend=%s",
                 p.listen, p.upstream, p.search_backend or "(none)")

    asyncio.run(_run_multi(config))


if __name__ == "__main__":
    main()
