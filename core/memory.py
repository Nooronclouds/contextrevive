"""
ContextRevive — Memory Manager
Priority-weighted memory system that stores conversation turns as vectors
and retrieves the most relevant memories for context reconstruction.
"""

import os
import sys
import uuid

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage.vector_store import VectorStore
from config import TOP_K_MEMORIES

# Keywords that indicate an "important" turn
_IMPORTANCE_KEYWORDS = {
    "?", "refund", "confirm", "order", "decision",
    "agree", "approved", "issue", "problem", "help",
}


class MemoryManager:
    """
    Manages conversation memories with a weighted priority system.

    Priority formula:
        priority = 0.35 * recency  +  0.30 * semantic_similarity
                 + 0.20 * importance  +  0.15 * access

    recency      — turn_index / total_turns  (later turns score higher)
    similarity   — 1 - cosine_distance       (closer = higher)
    importance   — 1.0 if content has keywords/numbers, else 0.4
    access       — starts at 0.5, +0.1 per retrieval (capped at 1.0)
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.priority_cache: dict[str, float] = {}   # memory_id → access_score
        self._session_turn_counts: dict[str, int] = {}  # session_id → total turns

    # ------------------------------------------------------------------
    # Priority helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _recency_score(turn_index: int, total_turns: int) -> float:
        """Higher for turns near the end of the conversation."""
        if total_turns == 0:
            return 0.0
        return turn_index / total_turns

    @staticmethod
    def _importance_score(content: str) -> float:
        """1.0 if content contains important keywords or numbers, else 0.4."""
        lower = content.lower()
        # Check for numbers
        if any(ch.isdigit() for ch in content):
            return 1.0
        # Check for keywords
        for kw in _IMPORTANCE_KEYWORDS:
            if kw in lower:
                return 1.0
        return 0.4

    def _access_score(self, memory_id: str) -> float:
        """Returns the current access score (0.5 base, +0.1 per access, max 1.0)."""
        return self.priority_cache.get(memory_id, 0.5)

    def _bump_access(self, memory_id: str) -> float:
        """Increment and return the updated access score."""
        current = self.priority_cache.get(memory_id, 0.5)
        updated = min(current + 0.1, 1.0)
        self.priority_cache[memory_id] = updated
        return updated

    def _compute_priority(
        self,
        turn_index: int,
        total_turns: int,
        semantic_similarity: float,
        content: str,
        memory_id: str,
    ) -> float:
        """Compute the full weighted priority score."""
        recency = self._recency_score(turn_index, total_turns)
        importance = self._importance_score(content)
        access = self._access_score(memory_id)

        return (
            0.35 * recency
            + 0.30 * semantic_similarity
            + 0.20 * importance
            + 0.15 * access
        )

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store_turn(self, session_id: str, turn) -> None:
        """
        Embed and store a single conversation turn.

        Args:
            session_id: The owning session.
            turn:       A ConversationTurn object (from tracker.py).
        """
        memory_id = str(uuid.uuid4())
        metadata = {
            "session_id": session_id,
            "turn_index": turn.turn_index,
            "role": turn.role,
            "word_count": turn.word_count,
        }
        self.vector_store.add_memory(memory_id, turn.content, metadata)

        # Track session size for recency calculations
        count = self._session_turn_counts.get(session_id, 0)
        self._session_turn_counts[session_id] = max(count, turn.turn_index)

        # Initialise access score
        self.priority_cache[memory_id] = 0.5

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve_relevant(
        self, session_id: str, query: str, top_k: int = TOP_K_MEMORIES
    ) -> list[dict]:
        """
        Find the most relevant memories for a query, ranked by priority.

        Args:
            session_id: Scope to this session.
            query:      Natural-language search string.
            top_k:      Number of top results to return.

        Returns:
            List of dicts with keys: memory_id, content, distance,
            metadata, priority_score.
        """
        results = self.vector_store.search_similar(
            query, session_id, n_results=10
        )
        total_turns = self._session_turn_counts.get(session_id, 1)

        scored: list[dict] = []
        for r in results:
            memory_id = r["memory_id"]
            turn_index = r["metadata"].get("turn_index", 1)
            similarity = 1.0 - r["distance"]  # cosine: 0=identical → sim=1

            priority = self._compute_priority(
                turn_index=turn_index,
                total_turns=total_turns,
                semantic_similarity=similarity,
                content=r["content"],
                memory_id=memory_id,
            )

            # Bump access score for this retrieval
            self._bump_access(memory_id)

            scored.append({
                **r,
                "priority_score": round(priority, 4),
            })

        # Sort by priority descending
        scored.sort(key=lambda x: x["priority_score"], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def get_memory_snapshot(self, session_id: str) -> dict:
        """
        Summary view of all memories in a session.

        Returns:
            {total, session_id, top_priorities}
        """
        all_memories = self.vector_store.get_all_for_session(session_id)

        # Find top 3 by cached priority (access score as proxy when
        # no query-time similarity is available)
        cached = [
            (mid, score)
            for mid, score in self.priority_cache.items()
        ]
        cached.sort(key=lambda x: x[1], reverse=True)

        return {
            "total": len(all_memories),
            "session_id": session_id,
            "top_priorities": cached[:3],
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dataclasses import dataclass

    # Minimal stand-in for ConversationTurn (avoid circular import)
    @dataclass
    class _FakeTurn:
        turn_index: int
        role: str
        content: str
        word_count: int = 0

        def __post_init__(self):
            self.word_count = len(self.content.split())

    print("=" * 55)
    print("  MemoryManager — Smoke Test")
    print("=" * 55)

    mm = MemoryManager()
    session = "smoke-test-session"

    # Store 5 fake turns about a damaged order
    turns = [
        _FakeTurn(1, "user", "Hi, I need help with my order"),
        _FakeTurn(2, "ai",   "Sure! What is your order number?"),
        _FakeTurn(3, "user", "Order 4421. The item arrived damaged."),
        _FakeTurn(4, "ai",   "I am sorry to hear that. Can you describe the damage?"),
        _FakeTurn(5, "user", "The screen is cracked and the box was torn open."),
    ]

    print("\n[Storing 5 turns...]")
    for t in turns:
        mm.store_turn(session, t)
        print(f"  Stored turn {t.turn_index} ({t.role}): {t.content[:50]}")

    # Retrieve relevant
    query = "refund for damaged item"
    print(f"\n[Query] '{query}'")
    results = mm.retrieve_relevant(session, query, top_k=3)
    for i, r in enumerate(results, 1):
        print(
            f"  #{i}  priority={r['priority_score']:.4f}  "
            f"dist={r['distance']:.4f}  "
            f"{r['content'][:55]}"
        )

    # Snapshot
    snap = mm.get_memory_snapshot(session)
    print(f"\n[Snapshot] Total memories: {snap['total']}")
    print(f"  Top priorities: {snap['top_priorities']}")

    # Cleanup
    mm.vector_store.delete_session_memories(session)
    print("\n[Cleanup] Test memories deleted.")