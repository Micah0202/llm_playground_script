"""
config.py -- All configurable constants for the LLM Playground.
"""

from pathlib import Path

# --------------- Paths ---------------
PROJECT_DIR = Path(__file__).parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_FILE = LOG_DIR / "responses.jsonl"

# --------------- OpenAI Settings ---------------
OPENAI_MODEL = "gpt-4o-mini"

# Pricing per token (USD)
# Source: https://openai.com/api/pricing/
OPENAI_COST_PER_INPUT_TOKEN = 0.15 / 1_000_000    # $0.15 per 1M input tokens
OPENAI_COST_PER_OUTPUT_TOKEN = 0.60 / 1_000_000   # $0.60 per 1M output tokens

# --------------- Ollama Settings ---------------
OLLAMA_MODEL = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"

# --------------- Display ---------------
DISPLAY_WIDTH = 40  # Column width for side-by-side display
