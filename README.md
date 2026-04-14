# minimax-ws-proxy

MiniMax Anthropic 兼容端点的轻量代理，补全 `web_search_20250305` 支持。

## 它做什么

MiniMax 的 Anthropic 兼容接口不支持 `web_search_20250305` 服务端工具。本代理透明代理所有请求，拦截 web search 调用并通过 MiniMax Coding Plan 搜索 API 执行，返回 Anthropic 原生格式。

```
Claude Code ──► proxy (localhost) ──► MiniMax /anthropic/v1/messages
                     │
                     └──► MiniMax /v1/coding_plan/search （按需）
```

API key 和模型由客户端透传（`x-api-key` 头），代理本身不存储密钥。

## 快速开始

```bash
cp .env.example .env
# 编辑 .env（海外版改为 https://api.minimax.io）

./run.sh
```

Claude Code 侧：

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8082
export ANTHROPIC_API_KEY=your_minimax_key
```

## .env 配置

```env
# 仅代理自身配置，key 由客户端透传
MINIMAX_API_HOST=https://api.minimaxi.com   # 海外版: https://api.minimax.io
HOST=127.0.0.1
PORT=8082
```

## 工作原理

1. 请求进来 → 从 `tools` 中剥离 `web_search_20250305`，注入等效普通 tool
2. 转发到 MiniMax（key 原样透传）
3. 模型调用搜索 → 用 MiniMax `/v1/coding_plan/search` 执行 → 喂回结果 → 循环
4. 搜索过程转换为 `server_tool_use` + `web_search_tool_result` 原生格式返回
5. 无搜索的请求 → 纯透明代理，streaming 直通

## 前提条件

- Python >= 3.10, [uv](https://docs.astral.sh/uv/)
- MiniMax Coding Plan / Token Plan（搜索 API 需要）

## 已知限制

- 含 `web_search_20250305` 的请求走缓冲模式（需拦截搜索调用），额外延迟很小
- 搜索后端仅使用 MiniMax 自身搜索 API

## 许可

MIT
