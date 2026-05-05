"""
ContextRevive — Vector Store
ChromaDB wrapper using manually provided embeddings from nomic-embed-text.
No built-in ChromaDB embedding functions are used.
"""

import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Suppress ChromaDB telemetry warnings (known bug in 0.4.x)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb

from core.ollama_client import get_embedding
from config import CHROMA_PATH


class VectorStore:
    """
    Persistent vector store backed by ChromaDB with cosine similarity.
    All embeddings are generated externally via Ollama's nomic-embed-text.
    """

    def __init__(self):
        chroma_path = os.path.abspath(CHROMA_PATH)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name="context_memories",
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_memory(self, memory_id: str, text: str, metadata: dict) -> None:
        """
        Embed text via Ollama and store it in ChromaDB.

        Args:
            memory_id: Unique identifier for this memory.
            text:      The content to embed and store.
            metadata:  Arbitrary metadata dict (must include session_id).
        """
        embedding = get_embedding(text)
        if not embedding:
            print(f"[WARN] VectorStore.add_memory: empty embedding for '{text[:40]}...'")
            return

        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search_similar(
        self, query: str, session_id: str, n_results: int = 5
    ) -> list[dict]:
        """
        Find the most similar memories to a query within a session.

        Args:
            query:      Natural-language search string.
            session_id: Scope results to this session.
            n_results:  Max number of results to return.

        Returns:
            List of dicts: {memory_id, content, distance, metadata}.
            Filtered to only include results with cosine distance < 0.5.
        """
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"session_id": session_id},
        )

        # Unpack ChromaDB's nested list format
        parsed: list[dict] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i in range(len(ids)):
            distance = distances[i]
            # Cosine distance filter: 0 = identical, 2 = opposite
            if distance < 0.5:
                parsed.append({
                    "memory_id": ids[i],
                    "content": documents[i],
                    "distance": round(distance, 4),
                    "metadata": metadatas[i],
                })

        return parsed

    def get_all_for_session(self, session_id: str) -> list[dict]:
        """
        Retrieve every stored memory for a given session.

        Returns:
            List of dicts: {memory_id, content, metadata}.
        """
        results = self.collection.get(where={"session_id": session_id})

        parsed: list[dict] = []
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        for i in range(len(ids)):
            parsed.append({
                "memory_id": ids[i],
                "content": documents[i],
                "metadata": metadatas[i],
            })

        return parsed

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    def delete_session_memories(self, session_id: str) -> None:
        """Remove all memories associated with a session."""
        results = self.collection.get(where={"session_id": session_id})
        ids = results.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
            print(f"[INFO] Deleted {len(ids)} memories for session {session_id[:8]}...")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  VectorStore — Smoke Test")
    print("=" * 55)

    store = VectorStore()

    test_session = "test-session-001"

    # Add a couple of test memories
    store.add_memory(
        "mem-1",
        "The customer reported a cracked screen on order 4421.",
        {"session_id": test_session, "turn_index": 3, "role": "user"},
    )
    store.add_memory(
        "mem-2",
        "A full refund has been initiated for the damaged item.",
        {"session_id": test_session, "turn_index": 8, "role": "ai"},
    )

    print("\n[Search] query='damaged order refund'")
    results = store.search_similar("damaged order refund", test_session, n_results=5)
    for r in results:
        print(f"  dist={r['distance']:.4f}  {r['content'][:60]}")

    print(f"\n[All] {len(store.get_all_for_session(test_session))} memories stored")

    # Cleanup
    store.delete_session_memories(test_session)
    print("[Cleanup] Test memories deleted.")