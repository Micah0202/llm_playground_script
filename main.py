"""
main.py -- LLM Playground: compare OpenAI and Ollama responses side by side.

Usage:
    python main.py
"""

import os

from dotenv import load_dotenv

from utils import query_openai, query_ollama, display_results, log_results


def main():
    # Load environment variables from .env (no crash if file is missing)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    # Startup banner
    print("\n=== LLM Playground ===")
    print("Compare OpenAI and Ollama responses side by side.\n")

    if not api_key:
        print("[INFO] No OPENAI_API_KEY found in .env -- OpenAI calls will be skipped.")
        print("       To enable OpenAI, copy .env.example to .env and add your key.\n")

    print('Type your prompt and press Enter. Type "quit" or "exit" to stop.\n')

    while True:
        try:
            prompt = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # Query Ollama (always -- local and free)
        print("\n[Querying Ollama...] ", end="", flush=True)
        ollama_result = query_ollama(prompt)
        if ollama_result["error"]:
            print(f"error: {ollama_result['error']}")
        else:
            print("done.")

        # Query OpenAI (only if API key is available)
        if api_key:
            print("[Querying OpenAI...] ", end="", flush=True)
            openai_result = query_openai(prompt, api_key)
            if openai_result["error"]:
                print(f"error: {openai_result['error']}")
            else:
                print("done.")
        else:
            openai_result = {
                "response": None,
                "model": "N/A",
                "input_tokens": 0,
                "output_tokens": 0,
                "response_time": 0.0,
                "cost_usd": 0.0,
                "error": "No API key configured.",
            }

        # Display and log
        display_results(prompt, openai_result, ollama_result)
        log_results(prompt, openai_result, ollama_result)


if __name__ == "__main__":
    main()
