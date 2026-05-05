"""
ContextRevive — Ollama Client
The ONLY module that communicates with Ollama.
All other modules import from here.
"""

import json
import os
import re
import sys

# Ensure the project root is on sys.path so `config` is importable
# regardless of whether this file is run directly or imported as a module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests

from config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_EMBED_MODEL


def ask_ollama(prompt: str, system: str = "") -> str:
    """
    Send a generation request to Ollama and return the response text.

    Args:
        prompt:  The user/task prompt.
        system:  Optional system-level instruction.

    Returns:
        The generated text, or "" on any failure.
    """
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        print(f"[ERROR] ask_ollama failed: {e}")
        return ""


def get_embedding(text: str) -> list[float]:
    """
    Get a vector embedding for the given text via Ollama.

    Args:
        text: The text to embed.

    Returns:
        A list of floats (the embedding vector), or [] on failure.
    """
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": OLLAMA_EMBED_MODEL,
                "prompt": text,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"[WARN] get_embedding failed: {e}")
        return []


def test_ollama_connection() -> bool:
    """
    Check whether Ollama is reachable.

    Returns:
        True if the /api/tags endpoint responds with 200, False otherwise.
    """
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def parse_llm_json(raw: str) -> dict:
    """
    Robustly extract a JSON object from LLM output.

    llama3.1 sometimes wraps valid JSON in extra prose or markdown
    fences. This function tries two strategies before giving up.

    Args:
        raw: The raw string returned by the LLM.

    Returns:
        A parsed dict, or {} if extraction fails.
    """
    # Try 1: direct parse
    try:
        return json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        pass

    # Try 2: regex extraction of the first { ... } block
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, TypeError):
        pass

    return {}


# ---------------------------------------------------------------------------
# Quick smoke test — run this file directly to verify Ollama connectivity.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("  Ollama Client — Smoke Test")
    print("=" * 50)

    # 1. Connection check
    connected = test_ollama_connection()
    print(f"\n[Connection] Ollama reachable: {connected}")

    if not connected:
        print("Ollama is not running. Start it with: ollama serve")
    else:
        # 2. Embedding test
        embedding = get_embedding("test sentence")
        print(f"[Embedding]  Vector length: {len(embedding)}")

        # 3. Generation test
        answer = ask_ollama("What is 2 plus 2?")
        print(f"[Generation] Response: {answer}")