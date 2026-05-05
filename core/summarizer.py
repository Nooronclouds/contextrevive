"""
ContextRevive — Context Summarizer
Compresses long conversation histories into a summary + recent turns
to fit within a token budget for the agent.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ollama_client import ask_groq, ask_ollama, parse_llm_json  # noqa: F401
from config import USE_GROQ

llm = ask_groq if USE_GROQ else ask_ollama


class ContextSummarizer:
    """Compresses conversation context to stay within a token budget."""

    def estimate_tokens(self, turns: list) -> int:
        """Rough token estimate: total_words / 0.75."""
        total_words = 0
        for t in turns:
            total_words += len(t.content.split())
        return int(total_words / 0.75)

    def compress_context(self, turns: list, max_tokens: int = 512) -> dict:
        """
        Compress a list of ConversationTurn objects.

        If under budget, returns turns verbatim.
        Otherwise, summarizes older turns and keeps last 4 verbatim.
        """
        if not turns:
            return {"compressed": False, "turns": [], "summary": None}

        token_estimate = self.estimate_tokens(turns)

        if token_estimate <= max_tokens:
            return {
                "compressed": False,
                "turns": turns,
                "summary": None,
                "original_token_estimate": token_estimate,
                "compressed_token_estimate": token_estimate,
                "turns_compressed": 0,
            }

        recent_turns = turns[-4:]
        older_turns = turns[:-4]

        formatted = "\n".join(
            [f"Turn {t.turn_index} ({t.role}): {t.content}" for t in older_turns]
        )

        system = "You are an expert conversation summarizer. Be concise and factual."

        prompt = f"""Summarize this conversation history in 2-3 sentences.
Keep all key facts: names, numbers, decisions, problems mentioned.
Do not add opinions. Only state what was discussed.

CONVERSATION:
{formatted}

Summary:"""

        summary_text = llm(prompt, system).strip()

        summary_words = len(summary_text.split())
        recent_words = sum(len(t.content.split()) for t in recent_turns)
        compressed_estimate = int((summary_words + recent_words) / 0.75)

        return {
            "compressed": True,
            "summary": summary_text,
            "recent_turns": recent_turns,
            "older_turns_count": len(older_turns),
            "turns_compressed": len(older_turns),
            "original_token_estimate": token_estimate,
            "compressed_token_estimate": compressed_estimate,
            "compression_ratio": round(compressed_estimate / token_estimate, 2),
        }

    def format_for_agent(self, compression_result: dict, tracker) -> str:
        """Format the compression result as a single text block for the agent."""
        if not compression_result["compressed"]:
            return tracker.format_turns_as_text(compression_result["turns"])

        recent_text = tracker.format_turns_as_text(
            compression_result["recent_turns"]
        )
        return (
            f"[CONVERSATION SUMMARY — {compression_result['turns_compressed']} "
            f"turns compressed]:\n{compression_result['summary']}\n\n"
            f"[RECENT EXCHANGES]:\n{recent_text}"
        )

    def get_compression_stats(self, compression_result: dict) -> dict:
        """Return clean stats dict for UI/API."""
        was_compressed = compression_result.get("compressed", False)
        summary = compression_result.get("summary") or ""
        return {
            "was_compressed": was_compressed,
            "original_tokens": compression_result.get("original_token_estimate", 0),
            "compressed_tokens": compression_result.get(
                "compressed_token_estimate", 0
            ),
            "compression_ratio": compression_result.get("compression_ratio", 1.0),
            "turns_summarized": compression_result.get("turns_compressed", 0),
            "summary_preview": summary[:100],
        }


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from core.tracker import ConversationTracker, ConversationTurn

    summarizer = ContextSummarizer()

    print("=" * 60)
    print("  ContextSummarizer — Smoke Test")
    print("=" * 60)

    # ---- Test 1: Short conversation (no compression) ----
    print("\n--- Test 1: Short conversation (should NOT compress) ---")
    short_turns = [
        ConversationTurn(
            session_id="t1", turn_index=i + 1,
            role="user" if i % 2 == 0 else "ai",
            content=f"Short message {i + 1}.",
        )
        for i in range(5)
    ]
    result1 = summarizer.compress_context(short_turns, max_tokens=512)
    print(f"  compressed: {result1['compressed']}")
    print(f"  token estimate: {result1['original_token_estimate']}")

    # ---- Test 2: Long conversation (should compress) ----
    print("\n--- Test 2: Long conversation (SHOULD compress) ---")
    cs_dialogue = [
        ("user", "Hi, my name is Sarah Mitchell and my order number is ORD-48291."),
        ("ai", "Hello Sarah, I can help you with order ORD-48291. What seems to be the problem?"),
        ("user", "The package arrived yesterday but the laptop screen is cracked badly across the middle."),
        ("ai", "I'm sorry to hear that. I can process a replacement under our 30-day warranty policy."),
        ("user", "I paid $1,299 for it and I need it for work next Monday for a presentation."),
        ("ai", "Understood. I can ship a replacement with overnight shipping at no extra cost."),
        ("user", "That would be great. Do I need to send the broken one back first?"),
        ("ai", "No, we'll send the replacement first and include a prepaid return label."),
        ("user", "Perfect. Will it arrive by Friday for sure?"),
        ("ai", "Yes, with overnight shipping it will arrive Thursday or Friday at the latest."),
        ("user", "Can you also confirm my shipping address is 482 Elm Street, Boston MA 02134?"),
        ("ai", "Confirmed. The replacement will ship to 482 Elm Street, Boston MA 02134."),
        ("user", "Great, thank you so much for the quick help."),
        ("ai", "You're welcome. You'll get a tracking number by email within the hour."),
        ("user", "One more thing — should I expect any charge on my card?"),
    ]
    long_turns = [
        ConversationTurn(
            session_id="t2", turn_index=i + 1, role=role, content=content
        )
        for i, (role, content) in enumerate(cs_dialogue)
    ]
    result2 = summarizer.compress_context(long_turns, max_tokens=200)
    print(f"  compressed: {result2['compressed']}")
    print(f"  summary: {result2.get('summary')}")
    print(f"  compression ratio: {result2.get('compression_ratio')}")
    print("\n  format_for_agent output:")

    class _MiniTracker:
        format_turns_as_text = staticmethod(ConversationTracker.format_turns_as_text)

    print(summarizer.format_for_agent(result2, _MiniTracker()))

    # ---- Test 3: Real scenario ----
    print("\n--- Test 3: Real scenario (customer_support) ---")
    tracker = ConversationTracker()
    session_id = tracker.load_from_script("customer_support")
    turns = tracker.sessions[session_id]
    result3 = summarizer.compress_context(turns, max_tokens=150)
    stats = summarizer.get_compression_stats(result3)
    print("  Compression stats:")
    for k, v in stats.items():
        print(f"    {k}: {v}")
