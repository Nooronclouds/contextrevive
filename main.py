"""
ContextRevive — AI Context Recovery Engine
Entry point: verifies Ollama connectivity and available models.
"""

import sys
import requests


def main():
    print("=" * 50)
    print("  ContextRevive — AI Context Recovery Engine")
    print("=" * 50)
    print()
    print("Checking Ollama connection...")

    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            # Check for required models
            llama_ready = any("llama3.1:8b" in name for name in model_names)
            embed_ready = any("nomic-embed-text" in name for name in model_names)

            if llama_ready:
                print("[OK] llama3.1:8b ready")
            else:
                print("[WARN] llama3.1:8b not found. Run: ollama pull llama3.1:8b")

            if embed_ready:
                print("[OK] nomic-embed-text ready")
            else:
                print("[WARN] nomic-embed-text not found. Run: ollama pull nomic-embed-text")

            print()
            print("ContextRevive engine ready.")
            print("Run: uvicorn api.server:app --reload --port 8000")
        else:
            print(f"ERROR: Ollama returned status {response.status_code}")
            sys.exit(1)

    except requests.ConnectionError:
        print("ERROR: Ollama is not running. Run: ollama serve")
        sys.exit(1)


if __name__ == "__main__":
    main()
