# LLM Inference Proxy

HTTP reverse proxy that sits between your apps and LLM providers, logging every request's token usage and cost to SQLite. Includes a JSON stats API and a web dashboard.

Designed for API key (pay-per-use) traffic from tools like Cline, VS Code extensions, and other LLM clients.

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

The proxy starts on `http://0.0.0.0:8080` by default.

> **Note**: If your system sets `PYTHONPATH` to include system site-packages (e.g. `/usr/lib/python3/dist-packages`), run with `PYTHONPATH= python main.py` to avoid version conflicts.

## Configuration

Copy and edit `.env` in the project root:

```
PROXY_PORT=8080
PROXY_HOST=0.0.0.0
PROXY_DB_PATH=./usage.db
OLLAMA_BASE_URL=http://spark-2448:11434
```

Environment variables override `.env` values. The `.env` file is loaded via `python-dotenv` at startup.

## How It Works

Point your LLM client at `http://localhost:8080/{provider}/` instead of the provider's API directly. The proxy forwards all requests transparently, parses token usage from responses, calculates cost, and logs everything to SQLite.

### Routing

Requests are routed by the first path segment:

```
http://localhost:8080/openai/v1/chat/completions
                     ^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^
                     provider   forwarded path
```

This becomes a request to `https://api.openai.com/v1/chat/completions`.

### Authentication

Passthrough — clients send their own API keys in headers as usual. The proxy does not inject, modify, or store any credentials.

## Supported Providers

| Slug | Base URL | Format |
|------|----------|--------|
| `openai` | `https://api.openai.com` | OpenAI |
| `anthropic` | `https://api.anthropic.com` | Anthropic |
| `gemini` | `https://generativelanguage.googleapis.com` | Gemini |
| `nvidia` | `https://integrate.api.nvidia.com` | OpenAI-compat |
| `openrouter` | `https://openrouter.ai` | OpenAI-compat |
| `moonshot` | `https://api.moonshot.ai` | OpenAI-compat |
| `zai` | `https://api.z.ai` | OpenAI-compat |
| `minimax` | `https://api.minimax.io` | OpenAI-compat |
| `ollama` | Configurable (`OLLAMA_BASE_URL`) | OpenAI-compat + native |
| `ollama-cloud` | `https://api.ollama.com` | OpenAI-compat |

7 of 10 providers use OpenAI-compatible format, so only 3 parsers are needed (OpenAI, Anthropic, Gemini) plus one for Ollama's native `/api/` endpoints.

## Client Configuration Examples

### Cline / VS Code Extensions

Set the base URL to `http://localhost:8080/openai` (or whichever provider) and configure your API key as usual.

### curl

```bash
# OpenAI (non-streaming)
curl http://localhost:8080/openai/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}'

# Anthropic
curl http://localhost:8080/anthropic/v1/messages \
  -H "x-api-key: sk-ant-..." \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model": "claude-sonnet-4-20250514", "max_tokens": 100, "messages": [{"role": "user", "content": "hello"}]}'

# Local Ollama
curl http://localhost:8080/ollama/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "messages": [{"role": "user", "content": "hello"}]}'

# Streaming
curl http://localhost:8080/openai/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}], "stream": true}'
```

## Streaming

Both streaming and non-streaming requests are supported. The proxy:

1. Detects streaming from `"stream": true` in the request body (or `streamGenerateContent` in Gemini URLs)
2. Forwards chunks to the client in real-time
3. Parses token usage from the stream as it arrives
4. Logs usage after the stream completes

For OpenAI-compatible providers, the proxy auto-injects `stream_options.include_usage` to get token counts in the final SSE chunk.

## Stats API

### `GET /stats/summary`

Overall totals and per-provider breakdown.

```json
{
  "totals": {
    "total_requests": 142,
    "total_input_tokens": 58320,
    "total_output_tokens": 23100,
    "total_tokens": 81420,
    "total_cost_usd": 0.4821
  },
  "by_provider": [
    {"provider": "openai", "requests": 80, "input_tokens": 40000, ...},
    {"provider": "anthropic", "requests": 62, ...}
  ]
}
```

### `GET /stats/by-model`

Per-model breakdown. Optional query parameters:
- `provider` — filter to a single provider
- `days` — limit to last N days

```
GET /stats/by-model?provider=openai&days=7
```

### `GET /stats/requests`

Paginated request log. Query parameters:
- `limit` (default 50)
- `offset` (default 0)

```
GET /stats/requests?limit=20&offset=0
```

### `GET /stats/timeseries`

Time-bucketed data for charts. Query parameters:
- `days` (default 30)
- `bucket` — `day` (default) or `hour`

```
GET /stats/timeseries?days=7&bucket=hour
```

## Dashboard

Open `http://localhost:8080/dashboard` in a browser. The dashboard is a self-contained HTML page with:

- Summary cards (total requests, tokens, cost)
- Time-series chart (tokens and cost over time)
- Provider breakdown table
- Model breakdown table with filters
- Recent request log with pagination

Auto-refreshes every 30 seconds.

## Pricing

Cost is calculated using a built-in pricing table in `pricing.py`, keyed by model name prefix with longest-prefix matching. Prices are in USD per million tokens.

To override or extend pricing, create a `pricing.json` file in the project root:

```json
{
  "my-custom-model": {"input": 1.00, "output": 2.00}
}
```

Local/Ollama models default to zero cost. Unknown models also default to zero cost.

## Database

SQLite database (`usage.db` by default) with a single `requests` table:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `timestamp` | REAL | Unix timestamp |
| `provider` | TEXT | Provider slug |
| `model` | TEXT | Model name |
| `endpoint` | TEXT | API path |
| `method` | TEXT | HTTP method |
| `input_tokens` | INTEGER | Prompt/input token count |
| `output_tokens` | INTEGER | Completion/output token count |
| `total_tokens` | INTEGER | Combined total |
| `cost_usd` | REAL | Calculated cost in USD |
| `duration_ms` | REAL | Request duration |
| `streaming` | INTEGER | 0 or 1 |
| `status_code` | INTEGER | HTTP status from upstream |
| `error` | TEXT | Error message if any |

Indexed on `timestamp`, `provider`, and `model`. No request/response bodies are stored.

## File Structure

```
inference_proxy/
  main.py           # FastAPI app, proxy route, stats endpoints
  providers.py      # Provider registry, response/stream parsers
  db.py             # SQLite schema, async insert/query helpers
  pricing.py        # Pricing table + calculate_cost()
  dashboard.html    # Self-contained HTML/JS/CSS dashboard
  requirements.txt  # Python dependencies
  .env              # Configuration
  .gitignore        # Excludes .env, venv/, usage.db, __pycache__/
```

## Disclaimer

This proxy is intended **exclusively for use with API key (pay-per-use) traffic**. This includes requests made by tools such as Cline, Continue, VS Code extensions, and other applications that authenticate via provider-issued API keys.

**Do not route subscription-based or OAuth-authenticated traffic through this proxy.** In particular, Anthropic's Claude Pro, Team, and Max subscription plans — including Claude Code when used under a Max subscription — authenticate via OAuth and are governed by Anthropic's [Consumer Terms of Service](https://www.anthropic.com/legal/consumer-terms) and [Acceptable Use Policy](https://www.anthropic.com/legal/aup). As of February 2025, those terms prohibit using subscription access to operate proxy or gateway services, and routing such traffic through an intermediary may constitute a violation regardless of intent. This applies to any provider whose terms of service restrict proxying of subscription-authenticated requests.

This software proxies only what the user explicitly configures it to proxy. It performs no credential injection, storage, or sharing. It is the user's responsibility to ensure that their use of this proxy complies with the terms of service of each upstream provider. The authors assume no liability for misuse.

When in doubt, consult the relevant provider's terms of service directly.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
