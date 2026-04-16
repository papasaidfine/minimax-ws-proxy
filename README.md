# minimax-ws-proxy

MiniMax Anthropic 兼容端点的轻量代理，补全 `web_search_20250305` 支持。

## 它做什么

MiniMax 的 Anthropic 兼容接口不支持 `web_search_20250305` 服务端工具。本代理透明代理所有请求，拦截 web search 调用并通过可插拔的搜索后端执行，返回 Anthropic 原生格式。

```
Claude Code ──► proxy (localhost:8082) ──► MiniMax /anthropic/v1/messages
                     │
                     ├──► REST: MiniMax /v1/coding_plan/search
                     └──► MCP:  minimax-coding-plan-mcp (stdio)
```

支持一个进程起多个代理实例，各自独立端口、上游、搜索后端。

## 快速开始

```bash
cp config.example.json config.json
# 编辑 config.json

# 如有需要引用的密钥（如 MCP API key），放进 .env：
cp .env.example .env

./run.sh
# 或指定路径: uv run minimax-ws-proxy --config /path/to/config.json
```

启动时按以下顺序查找配置：`--config <path>` → `CONFIG_PATH` 环境变量 → 当前目录 `config.json`。找不到则报错退出。

Claude Code 侧：

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8082
export ANTHROPIC_API_KEY=your_minimax_key
```

## 配置

`config.json` 是唯一的配置入口 —— 监听地址、上游、搜索后端全部在这里声明。`.env` 仅用于存放 `config.json` 通过 `$VAR` 引用的密钥。

```json
{
  "backends": {
    "minimax-search": {
      "type": "rest",
      "url": "https://api.minimaxi.com/v1/coding_plan/search",
      "api_key": "$PASSTHRU_API_KEY"
    },
    "cross-provider-search": {
      "type": "rest",
      "url": "https://api.minimaxi.com/v1/coding_plan/search",
      "api_key": "$MINIMAX_SEARCH_KEY"
    },
    "minimax-mcp-a": {
      "type": "mcp",
      "command": "uvx",
      "args": ["minimax-coding-plan-mcp", "-y"],
      "env": { "MINIMAX_API_KEY": "$MM_KEY_A", "MINIMAX_API_HOST": "https://api.minimaxi.com" }
    }
  },
  "proxies": [
    {
      "name": "team-a",
      "listen": "127.0.0.1:8082",
      "upstream": "https://api.minimaxi.com/anthropic",
      "search_backend": "minimax-search"
    },
    {
      "name": "team-b",
      "listen": "127.0.0.1:8083",
      "upstream": "https://open.bigmodel.cn/api/anthropic",
      "search_backend": "cross-provider-search"
    }
  ]
}
```

**backends**：搜索后端，按名称声明，可被多个 proxy 共享。

**proxies**：代理实例列表，每个有独立端口和上游。`search_backend` 引用 backends 中的名称；省略则为纯透明代理（不拦截 web_search）。

### API key 解析（魔法字符串）

`api_key` 与 MCP `env` 值支持以下写法，让密钥可以待在 `.env`（gitignore）里而不进 `config.json`：

| 写法 | 含义 |
|---|---|
| `$PASSTHRU_API_KEY` | 透传客户端请求的 `x-api-key`（仅 REST `api_key` 可用） |
| `$VAR` 或 `${VAR}` | 启动时从环境变量（`.env`）读取；变量缺失会报错 |
| 省略 / `null` | 不发送 `Authorization` 头（无需鉴权的搜索接口） |
| 其他字符串 | 字面量 |

`$PASSTHRU_API_KEY` 仅在 LLM 上游与搜索后端使用同一密钥（如 MiniMax 自带 search）时适用；跨提供商场景请用 `$VAR` 从 `.env` 取静态密钥。

### 搜索后端类型

| 后端 | 传输方式 | API key 来源 | 说明 |
|---|---|---|---|
| `rest` | HTTP POST 到 `url` | `api_key` 字段 | MiniMax `/v1/coding_plan/search` |
| `mcp` | stdio 子进程 | `env` 中的变量（spawn 时固定） | 使用 MCP server，如 MiniMax 官方 |

**MCP 模式**：代理启动时 spawn 子进程（通过 `uvx`），API key 在 spawn 时固定（无法 per-request 透传，因此 `$PASSTHRU_API_KEY` 在 MCP env 中会报错）。不同 API key 需要不同的 MCP 后端实例。MCP 子进程在代理生命周期内保活，所有请求共享同一个 session。

## 工作原理

1. 请求进来 → 从 `tools` 中剥离 `web_search_20250305`，注入等效普通 tool
2. 转发到上游 LLM（key 原样透传）
3. 模型调用搜索 → 通过配置的搜索后端（REST 或 MCP）执行 → 喂回结果 → 循环
4. 搜索过程转换为 `server_tool_use` + `web_search_tool_result` 原生格式返回
5. 无搜索的请求 → 纯透明代理，streaming 直通

## 前提条件

- Python >= 3.10, [uv](https://docs.astral.sh/uv/)
- MiniMax Coding Plan / Token Plan（搜索 API 需要）

## 已知限制

- 含 `web_search_20250305` 的请求走缓冲模式（需拦截搜索调用），额外延迟很小
- MCP 模式首次启动时 `uvx` 需拉取 `minimax-coding-plan-mcp` 包，冷启动可能需几秒

## 许可

MIT
