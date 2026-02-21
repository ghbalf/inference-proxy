from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import db
from pricing import calculate_cost
from providers import (
    PROVIDERS,
    get_provider,
    set_ollama_base_url,
    parse_response,
    get_stream_parser,
    detect_streaming,
    inject_stream_options,
)

load_dotenv()

PROXY_PORT = int(os.getenv("PROXY_PORT", "8080"))
PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
DB_PATH = os.getenv("PROXY_DB_PATH", "./usage.db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://spark-2448:11434")

# Configure
db.DB_PATH = DB_PATH
set_ollama_base_url(OLLAMA_BASE_URL)

http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    await db.init_db(DB_PATH)
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    yield
    if http_client:
        await http_client.aclose()


app = FastAPI(title="LLM Inference Proxy", lifespan=lifespan)


# ---------- Dashboard ----------


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html_path = Path(__file__).parent / "dashboard.html"
    return HTMLResponse(html_path.read_text())


# ---------- Stats API ----------


@app.get("/stats/summary")
async def stats_summary():
    return await db.query_summary(DB_PATH)


@app.get("/stats/by-model")
async def stats_by_model(provider: str | None = None, days: int | None = None):
    return await db.query_by_model(DB_PATH, provider=provider, days=days)


@app.get("/stats/requests")
async def stats_requests(limit: int = 50, offset: int = 0):
    return await db.query_requests(DB_PATH, limit=limit, offset=offset)


@app.get("/stats/timeseries")
async def stats_timeseries(days: int = 30, bucket: str = "day"):
    return await db.query_timeseries(DB_PATH, days=days, bucket=bucket)


# ---------- Proxy ----------

# Headers to never forward
STRIP_HEADERS = frozenset(
    {"host", "content-length", "transfer-encoding", "connection", "keep-alive"}
)


def _forward_headers(request: Request) -> dict[str, str]:
    """Extract headers to forward, stripping hop-by-hop and host."""
    return {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in STRIP_HEADERS
    }


def _extract_model_from_path(path: str) -> str:
    """Try to extract model name from Gemini-style URL paths."""
    # e.g. /v1beta/models/gemini-2.0-flash:generateContent
    for segment in path.split("/"):
        if ":" in segment and any(
            kw in segment for kw in ("generate", "Generate", "stream", "Stream", "embed")
        ):
            return segment.split(":")[0]
    return ""


@app.api_route(
    "/{provider_slug}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy(
    provider_slug: str,
    path: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    provider = get_provider(provider_slug)
    if not provider:
        return JSONResponse(
            {"error": f"Unknown provider: {provider_slug}"},
            status_code=404,
        )

    # Build target URL
    endpoint = f"/{path}"
    query = str(request.url.query)
    target_url = f"{provider.base_url}{endpoint}"
    if query:
        target_url += f"?{query}"

    # Read and parse body
    raw_body = await request.body()
    body_dict = None
    if raw_body:
        try:
            body_dict = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    # Detect streaming
    is_streaming = detect_streaming(provider.parser_type, body_dict, endpoint)

    # For streaming OpenAI-compat, inject stream_options
    if is_streaming and body_dict is not None:
        body_dict = inject_stream_options(provider.parser_type, body_dict, endpoint)
        raw_body = json.dumps(body_dict).encode()

    headers = _forward_headers(request)
    start_time = time.time()

    if is_streaming:
        return await _handle_streaming(
            provider_slug,
            provider.parser_type,
            endpoint,
            request.method,
            target_url,
            headers,
            raw_body,
            body_dict,
            start_time,
            background_tasks,
        )
    else:
        return await _handle_non_streaming(
            provider_slug,
            provider.parser_type,
            endpoint,
            request.method,
            target_url,
            headers,
            raw_body,
            body_dict,
            start_time,
        )


async def _handle_non_streaming(
    provider_slug: str,
    parser_type: str,
    endpoint: str,
    method: str,
    target_url: str,
    headers: dict,
    raw_body: bytes,
    body_dict: dict | None,
    start_time: float,
):
    try:
        resp = await http_client.request(
            method=method,
            url=target_url,
            headers=headers,
            content=raw_body,
        )
    except httpx.RequestError as e:
        duration_ms = (time.time() - start_time) * 1000
        await db.log_request(
            DB_PATH,
            provider=provider_slug,
            model=body_dict.get("model", "") if body_dict else "",
            endpoint=endpoint,
            method=method,
            duration_ms=duration_ms,
            error=str(e),
            status_code=502,
        )
        return JSONResponse(
            {"error": f"Upstream request failed: {e}"},
            status_code=502,
        )

    duration_ms = (time.time() - start_time) * 1000

    # Parse usage from response
    model = ""
    input_tokens = output_tokens = total_tokens = 0
    error = ""

    if resp.status_code >= 400:
        error = resp.text[:500]

    try:
        resp_body = resp.json()
        usage = parse_response(parser_type, resp_body, endpoint)
        model = usage.model
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        total_tokens = usage.total_tokens
    except Exception:
        resp_body = None

    # Fallback: model from request body or URL
    if not model:
        if body_dict and body_dict.get("model"):
            model = body_dict["model"]
        else:
            model = _extract_model_from_path(endpoint)

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    cost = calculate_cost(model, input_tokens, output_tokens)

    await db.log_request(
        DB_PATH,
        provider=provider_slug,
        model=model,
        endpoint=endpoint,
        method=method,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost,
        duration_ms=duration_ms,
        streaming=False,
        status_code=resp.status_code,
        error=error,
    )

    # Forward response
    response_headers = {
        k: v
        for k, v in resp.headers.items()
        if k.lower() not in STRIP_HEADERS
    }
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=response_headers,
    )


async def _handle_streaming(
    provider_slug: str,
    parser_type: str,
    endpoint: str,
    method: str,
    target_url: str,
    headers: dict,
    raw_body: bytes,
    body_dict: dict | None,
    start_time: float,
    background_tasks: BackgroundTasks,
):
    stream_parser = get_stream_parser(parser_type, endpoint)

    try:
        req = http_client.build_request(
            method=method,
            url=target_url,
            headers=headers,
            content=raw_body,
        )
        response = await http_client.send(req, stream=True)
    except httpx.RequestError as e:
        duration_ms = (time.time() - start_time) * 1000
        await db.log_request(
            DB_PATH,
            provider=provider_slug,
            model=body_dict.get("model", "") if body_dict else "",
            endpoint=endpoint,
            method=method,
            duration_ms=duration_ms,
            error=str(e),
            status_code=502,
            streaming=True,
        )
        return JSONResponse(
            {"error": f"Upstream request failed: {e}"},
            status_code=502,
        )

    response_headers = {
        k: v
        for k, v in response.headers.items()
        if k.lower() not in STRIP_HEADERS
    }

    async def stream_generator():
        try:
            async for chunk in response.aiter_bytes():
                stream_parser.feed_chunk(chunk)
                yield chunk
        finally:
            stream_parser.finalize()
            await response.aclose()

            duration_ms = (time.time() - start_time) * 1000
            usage = stream_parser.usage

            model = usage.model
            if not model:
                if body_dict and body_dict.get("model"):
                    model = body_dict["model"]
                else:
                    model = _extract_model_from_path(endpoint)

            total = usage.total_tokens or (usage.input_tokens + usage.output_tokens)
            cost = calculate_cost(model, usage.input_tokens, usage.output_tokens)

            # Log in background to not delay the response teardown
            background_tasks.add_task(
                db.log_request,
                DB_PATH,
                provider=provider_slug,
                model=model,
                endpoint=endpoint,
                method=method,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=total,
                cost_usd=cost,
                duration_ms=duration_ms,
                streaming=True,
                status_code=response.status_code,
            )

    return StreamingResponse(
        stream_generator(),
        status_code=response.status_code,
        headers=response_headers,
    )


# ---------- Entrypoint ----------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=PROXY_HOST,
        port=PROXY_PORT,
        log_level="info",
    )
