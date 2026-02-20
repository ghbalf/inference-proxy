import json
from pathlib import Path

# Prices in USD per million tokens: {"input": ..., "output": ...}
# Sorted by provider/model prefix for clarity.
# Uses longest-prefix matching — more specific entries win.
PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-2024": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o1-pro": {"input": 150.00, "output": 600.00},
    "o3": {"input": 10.00, "output": 40.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o3-pro": {"input": 20.00, "output": 80.00},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # Anthropic
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-3-7-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    # Google Gemini
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # NVIDIA — mostly hosting open models, prices vary
    "meta/llama-3.1-405b": {"input": 5.00, "output": 16.00},
    "meta/llama-3.1-70b": {"input": 0.88, "output": 0.88},
    "meta/llama-3.1-8b": {"input": 0.30, "output": 0.30},
    # Moonshot (Kimi)
    "moonshot-v1-8k": {"input": 0.84, "output": 0.84},
    "moonshot-v1-32k": {"input": 1.68, "output": 1.68},
    "moonshot-v1-128k": {"input": 8.40, "output": 8.40},
    # Z.ai
    "z1": {"input": 1.00, "output": 2.00},
    # MiniMax
    "MiniMax-Text-01": {"input": 1.00, "output": 5.50},
    "abab6.5s": {"input": 0.28, "output": 0.28},
    # Ollama / local — free
    "llama": {"input": 0.0, "output": 0.0},
    "qwen": {"input": 0.0, "output": 0.0},
    "deepseek": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
    "gemma": {"input": 0.0, "output": 0.0},
    "phi": {"input": 0.0, "output": 0.0},
    "codellama": {"input": 0.0, "output": 0.0},
    "nomic": {"input": 0.0, "output": 0.0},
}

# Load optional pricing.json override
_override_path = Path(__file__).parent / "pricing.json"
if _override_path.exists():
    with open(_override_path) as f:
        PRICING.update(json.load(f))


def calculate_cost(
    model: str, input_tokens: int, output_tokens: int
) -> float:
    """Calculate cost in USD using longest-prefix matching on model name."""
    if not model:
        return 0.0

    model_lower = model.lower()
    best_match = ""
    best_prices = None

    for prefix, prices in PRICING.items():
        prefix_lower = prefix.lower()
        if model_lower.startswith(prefix_lower) and len(prefix_lower) > len(best_match):
            best_match = prefix_lower
            best_prices = prices

    if best_prices is None:
        return 0.0

    cost = (
        input_tokens * best_prices["input"] / 1_000_000
        + output_tokens * best_prices["output"] / 1_000_000
    )
    return round(cost, 8)
