from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UsageResult:
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def merge(self, other: UsageResult):
        if other.model:
            self.model = other.model
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens


@dataclass
class ProviderConfig:
    slug: str
    base_url: str
    parser_type: str  # "openai", "anthropic", "gemini", "ollama_native"
    # Headers to strip before forwarding (lowercase)
    strip_headers: tuple[str, ...] = ("host", "content-length", "transfer-encoding")


# ---------- Provider Registry ----------

PROVIDERS: dict[str, ProviderConfig] = {}


def _register(slug: str, base_url: str, parser_type: str = "openai"):
    PROVIDERS[slug] = ProviderConfig(slug=slug, base_url=base_url, parser_type=parser_type)


_register("openai", "https://api.openai.com")
_register("anthropic", "https://api.anthropic.com", "anthropic")
_register("gemini", "https://generativelanguage.googleapis.com", "gemini")
_register("nvidia", "https://integrate.api.nvidia.com")
_register("openrouter", "https://openrouter.ai")
_register("moonshot", "https://api.moonshot.ai")
_register("zai", "https://api.z.ai")
_register("minimax", "https://api.minimax.io")
# Local Ollama — base_url set from env at startup
_register("ollama", "http://spark-2448:11434")
_register("ollama-cloud", "https://api.ollama.com")


def get_provider(slug: str) -> ProviderConfig | None:
    return PROVIDERS.get(slug)


def set_ollama_base_url(url: str):
    if "ollama" in PROVIDERS:
        PROVIDERS["ollama"].base_url = url.rstrip("/")


# ---------- Response Parsers (non-streaming) ----------


def parse_openai_response(body: dict) -> UsageResult:
    """Parse OpenAI-compatible (and Ollama /v1/) response."""
    result = UsageResult()
    result.model = body.get("model", "")
    usage = body.get("usage", {})
    if usage:
        result.input_tokens = usage.get("prompt_tokens", 0)
        result.output_tokens = usage.get("completion_tokens", 0)
        result.total_tokens = usage.get("total_tokens", 0)
    return result


def parse_anthropic_response(body: dict) -> UsageResult:
    result = UsageResult()
    result.model = body.get("model", "")
    usage = body.get("usage", {})
    if usage:
        result.input_tokens = usage.get("input_tokens", 0)
        result.output_tokens = usage.get("output_tokens", 0)
        result.total_tokens = result.input_tokens + result.output_tokens
    return result


def parse_gemini_response(body: dict) -> UsageResult:
    result = UsageResult()
    # Model is in the URL for Gemini, not always in body
    meta = body.get("usageMetadata", {})
    if meta:
        result.input_tokens = meta.get("promptTokenCount", 0)
        result.output_tokens = meta.get("candidatesTokenCount", 0)
        result.total_tokens = meta.get("totalTokenCount", 0)
    return result


def parse_ollama_native_response(body: dict) -> UsageResult:
    """Parse Ollama native /api/generate or /api/chat response."""
    result = UsageResult()
    result.model = body.get("model", "")
    result.input_tokens = body.get("prompt_eval_count", 0)
    result.output_tokens = body.get("eval_count", 0)
    result.total_tokens = result.input_tokens + result.output_tokens
    return result


def parse_response(parser_type: str, body: dict, endpoint: str = "") -> UsageResult:
    """Dispatch to the right parser based on type and endpoint."""
    # Ollama native endpoints use a different format
    if parser_type == "openai" and endpoint.startswith("/api/"):
        return parse_ollama_native_response(body)
    parsers = {
        "openai": parse_openai_response,
        "anthropic": parse_anthropic_response,
        "gemini": parse_gemini_response,
        "ollama_native": parse_ollama_native_response,
    }
    parser = parsers.get(parser_type, parse_openai_response)
    return parser(body)


# ---------- Stream Parsers ----------


class StreamParser:
    """Base class for stream parsers. Handles line buffering across chunks."""

    def __init__(self):
        self.usage = UsageResult()
        self._buffer = ""

    def feed_chunk(self, chunk: bytes):
        """Feed a raw chunk; parse complete lines."""
        text = self._buffer + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        # Last element is either empty (if chunk ended with \n) or partial
        self._buffer = lines[-1]
        for line in lines[:-1]:
            self._parse_line(line)

    def finalize(self):
        """Call after stream ends to flush any remaining buffer."""
        if self._buffer.strip():
            self._parse_line(self._buffer)
            self._buffer = ""
        if self.usage.total_tokens == 0:
            self.usage.total_tokens = self.usage.input_tokens + self.usage.output_tokens

    def _parse_line(self, line: str):
        raise NotImplementedError


class OpenAIStreamParser(StreamParser):
    """Parses SSE stream from OpenAI-compatible APIs.

    Expects `stream_options.include_usage` to be injected into request.
    The final data chunk contains a usage object.
    """

    def _parse_line(self, line: str):
        if not line.startswith("data: "):
            return
        data = line[6:].strip()
        if data == "[DONE]":
            return
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return
        # Capture model from first chunk
        if not self.usage.model and obj.get("model"):
            self.usage.model = obj["model"]
        # Usage in final chunk
        usage = obj.get("usage")
        if usage:
            self.usage.input_tokens = usage.get("prompt_tokens", 0)
            self.usage.output_tokens = usage.get("completion_tokens", 0)
            self.usage.total_tokens = usage.get("total_tokens", 0)


class AnthropicStreamParser(StreamParser):
    """Parses SSE stream from Anthropic API.

    Captures input_tokens from message_start and output_tokens from message_delta.
    """

    def __init__(self):
        super().__init__()
        self._event_type = ""

    def _parse_line(self, line: str):
        if line.startswith("event: "):
            self._event_type = line[7:].strip()
            return
        if not line.startswith("data: "):
            return
        data = line[6:].strip()
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return

        if self._event_type == "message_start":
            msg = obj.get("message", {})
            if msg.get("model"):
                self.usage.model = msg["model"]
            usage = msg.get("usage", {})
            self.usage.input_tokens = usage.get("input_tokens", 0)
        elif self._event_type == "message_delta":
            usage = obj.get("usage", {})
            self.usage.output_tokens = usage.get("output_tokens", 0)


class GeminiStreamParser(StreamParser):
    """Parses SSE stream from Gemini API.

    Each SSE data chunk may contain usageMetadata; the last one is authoritative.
    """

    def _parse_line(self, line: str):
        if not line.startswith("data: "):
            return
        data = line[6:].strip()
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return
        meta = obj.get("usageMetadata")
        if meta:
            self.usage.input_tokens = meta.get("promptTokenCount", 0)
            self.usage.output_tokens = meta.get("candidatesTokenCount", 0)
            self.usage.total_tokens = meta.get("totalTokenCount", 0)


class OllamaNativeStreamParser(StreamParser):
    """Parses newline-delimited JSON stream from Ollama native /api/ endpoints.

    The final line (where done=true) contains eval_count and prompt_eval_count.
    """

    def _parse_line(self, line: str):
        line = line.strip()
        if not line:
            return
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return
        if not self.usage.model and obj.get("model"):
            self.usage.model = obj["model"]
        if obj.get("done"):
            self.usage.input_tokens = obj.get("prompt_eval_count", 0)
            self.usage.output_tokens = obj.get("eval_count", 0)


def get_stream_parser(parser_type: str, endpoint: str = "") -> StreamParser:
    """Get the appropriate stream parser for the given provider/endpoint."""
    # Ollama native endpoints
    if parser_type == "openai" and endpoint.startswith("/api/"):
        return OllamaNativeStreamParser()
    parsers = {
        "openai": OpenAIStreamParser,
        "anthropic": AnthropicStreamParser,
        "gemini": GeminiStreamParser,
        "ollama_native": OllamaNativeStreamParser,
    }
    cls = parsers.get(parser_type, OpenAIStreamParser)
    return cls()


def detect_streaming(parser_type: str, body: dict | None, path: str) -> bool:
    """Detect if the request is a streaming request."""
    # Gemini: streaming is indicated by endpoint name
    if parser_type == "gemini" and "streamGenerateContent" in path:
        return True
    # All others: check body for stream flag
    if body and body.get("stream") is True:
        return True
    return False


def inject_stream_options(parser_type: str, body: dict, endpoint: str = "") -> dict:
    """Inject options to get usage info in streaming responses."""
    if parser_type == "openai" and not endpoint.startswith("/api/"):
        # Tell OpenAI-compatible APIs to include usage in stream
        if "stream_options" not in body:
            body["stream_options"] = {}
        body["stream_options"]["include_usage"] = True
    return body
