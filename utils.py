"""
utils.py -- API calls, logging, and display helpers for the LLM Playground.
"""

import json
import textwrap
import time
import datetime

import requests
from openai import OpenAI, AuthenticationError, APIConnectionError

from config import (
    OPENAI_MODEL,
    OPENAI_COST_PER_INPUT_TOKEN,
    OPENAI_COST_PER_OUTPUT_TOKEN,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    LOG_DIR,
    LOG_FILE,
    DISPLAY_WIDTH,
)


# ────────────────────────────────────────────
#  API Query Functions
# ────────────────────────────────────────────

def query_openai(prompt: str, api_key: str) -> dict:
    """Send a prompt to the OpenAI chat completions API."""
    result = {
        "response": None,
        "model": OPENAI_MODEL,
        "input_tokens": 0,
        "output_tokens": 0,
        "response_time": 0.0,
        "cost_usd": 0.0,
        "error": None,
    }
    try:
        client = OpenAI(api_key=api_key)
        start = time.time()
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        result["response_time"] = round(time.time() - start, 3)
        result["response"] = completion.choices[0].message.content
        usage = completion.usage
        result["input_tokens"] = usage.prompt_tokens
        result["output_tokens"] = usage.completion_tokens
        result["cost_usd"] = round(
            usage.prompt_tokens * OPENAI_COST_PER_INPUT_TOKEN
            + usage.completion_tokens * OPENAI_COST_PER_OUTPUT_TOKEN,
            6,
        )
    except AuthenticationError:
        result["error"] = "Invalid OpenAI API key."
    except APIConnectionError:
        result["error"] = "Could not connect to OpenAI API."
    except Exception as e:
        result["error"] = f"OpenAI error: {e}"
    return result


def query_ollama(prompt: str) -> dict:
    """Send a prompt to the local Ollama REST API."""
    result = {
        "response": None,
        "model": OLLAMA_MODEL,
        "input_tokens": 0,
        "output_tokens": 0,
        "response_time": 0.0,
        "cost_usd": 0.0,
        "error": None,
    }
    try:
        start = time.time()
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        result["response_time"] = round(time.time() - start, 3)
        resp.raise_for_status()
        data = resp.json()
        result["response"] = data["message"]["content"]
        result["input_tokens"] = data.get("prompt_eval_count", 0)
        result["output_tokens"] = data.get("eval_count", 0)
    except requests.ConnectionError:
        result["error"] = "Could not connect to Ollama. Is it running? Try: ollama serve"
    except requests.Timeout:
        result["error"] = "Ollama request timed out (120s)."
    except Exception as e:
        result["error"] = f"Ollama error: {e}"
    return result


# ────────────────────────────────────────────
#  Display
# ────────────────────────────────────────────

def _wrap_or_error(result: dict, width: int) -> list[str]:
    """Word-wrap a successful response or format an error message."""
    if result["error"]:
        return textwrap.wrap(f"[ERROR] {result['error']}", width)
    if result["response"]:
        return textwrap.wrap(result["response"], width)
    return ["(no response)"]


def _format_stat(result: dict, key: str, suffix: str) -> str:
    """Format a single stat value from a result dict."""
    if result["error"] and key != "response_time":
        return "N/A"
    value = result[key]
    if isinstance(value, float):
        return f"{value} {suffix}".strip()
    return f"{value} {suffix}".strip()


def display_results(prompt: str, openai_result: dict, ollama_result: dict) -> None:
    """Print both results side-by-side in a formatted terminal table."""
    width = DISPLAY_WIDTH
    sep = " | "
    full_width = width * 2 + len(sep)

    print("\n" + "=" * full_width)
    print(f"PROMPT: {prompt}")
    print("=" * full_width)

    # Column headers
    left_header = f"OpenAI ({openai_result['model']})"
    right_header = f"Ollama ({ollama_result['model']})"
    print(f"{left_header:<{width}}{sep}{right_header:<{width}}")
    print("-" * width + sep + "-" * width)

    # Response text
    left_lines = _wrap_or_error(openai_result, width)
    right_lines = _wrap_or_error(ollama_result, width)
    max_lines = max(len(left_lines), len(right_lines))

    for i in range(max_lines):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""
        print(f"{left:<{width}}{sep}{right:<{width}}")

    # Stats
    print("-" * width + sep + "-" * width)
    stats = [
        ("Time", "response_time", "s"),
        ("In tokens", "input_tokens", ""),
        ("Out tokens", "output_tokens", ""),
        ("Cost", "cost_usd", "USD"),
    ]
    for label, key, suffix in stats:
        left_val = _format_stat(openai_result, key, suffix)
        right_val = _format_stat(ollama_result, key, suffix)
        print(f"{label + ': ' + left_val:<{width}}{sep}{label + ': ' + right_val:<{width}}")

    print("=" * full_width + "\n")


# ────────────────────────────────────────────
#  Logging
# ────────────────────────────────────────────

def log_results(prompt: str, openai_result: dict, ollama_result: dict) -> None:
    """Append one JSON object (one line) to the log file."""
    LOG_DIR.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "prompt": prompt,
        "openai": openai_result,
        "ollama": ollama_result,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
