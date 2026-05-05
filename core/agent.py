"""
ContextRevive — Agent Orchestrator
Ties all modules together: tracker, memory, reconstructor, and Ollama.
This is the single entry point for the chat pipeline.
"""

import os
import sys
from dataclasses import dataclass

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.tracker import ConversationTracker
from core.memory import MemoryManager
from core.reconstructor import ContextReconstructor, ReconstructionResult
from core.ollama_client import ask_ollama


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    """Structured response from the ContextRevive agent."""
    response_text: str
    used_reconstruction: bool
    confidence: float          # 1.0 if no reconstruction needed
    strategy: str              # "none" | "silent" | "soft" | "ask_user"
    integrity_score: float
    inferred_summary: str      # "" if no reconstruction


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ContextReviveAgent:
    """
    Top-level orchestrator for ContextRevive.

    Workflow per chat():
        1. Record user message (tracker + memory)
        2. Detect gaps
        3. Reconstruct if gaps exist
        4. Build merged context
        5. Build strategy-aware system prompt
        6. Call llama3.1:8b for final response
        7. Record AI response
        8. Return AgentResponse
    """

    def __init__(self):
        self.tracker = ConversationTracker()
        self.memory = MemoryManager()
        self.reconstructor = ContextReconstructor()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def new_session(self, scenario_name: str | None = None) -> str:
        """
        Create a new session, optionally pre-loaded from a scenario script.

        Args:
            scenario_name: Key in conversations.json (e.g. "customer_support").
                           If None, an empty session is created.

        Returns:
            The new session_id.
        """
        if scenario_name:
            session_id = self.tracker.load_from_script(scenario_name)
            # Store every loaded turn in vector memory
            for turn in self.tracker.get_available_turns(session_id):
                self.memory.store_turn(session_id, turn)
        else:
            session_id = self.tracker.new_session()

        return session_id

    def simulate_gap(self, session_id: str, turn_indices: list[int]) -> dict:
        """
        Mark turns as missing and return the resulting gap report.

        Args:
            session_id:   Target session.
            turn_indices: 1-based indices to mark as unavailable.

        Returns:
            Dict with gap_report details and integrity_score.
        """
        self.tracker.simulate_gap(session_id, turn_indices)
        gap_report = self.tracker.detect_gaps(session_id)

        return {
            "gap_report": {
                "has_gaps": gap_report.has_gaps,
                "gap_ranges": gap_report.gap_ranges,
                "total_missing": gap_report.total_missing,
                "severity": gap_report.severity,
            },
            "integrity_score": gap_report.integrity_score,
        }

    # ------------------------------------------------------------------
    # Main chat pipeline
    # ------------------------------------------------------------------

    def chat(self, session_id: str, user_message: str) -> AgentResponse:
        """
        Process a user message through the full ContextRevive pipeline.

        Steps:
            1. Record user turn
            2. Detect gaps
            3. Reconstruct missing context (if any)
            4. Build merged context string
            5. Build strategy-aware prompts
            6. Generate response via Ollama
            7. Record AI turn
            8. Return structured AgentResponse
        """
        # Step 1 — Record user message
        new_turn = self.tracker.add_turn(session_id, "user", user_message)
        self.memory.store_turn(session_id, new_turn)

        # Step 2 — Detect gaps
        gap_report = self.tracker.detect_gaps(session_id)

        # Step 3 — Reconstruct if needed
        reconstruction: ReconstructionResult | None = None
        used_reconstruction = False

        if gap_report.has_gaps:
            reconstruction = self.reconstructor.reconstruct(
                session_id, gap_report, self.tracker, self.memory
            )
            used_reconstruction = True

        # Step 4 — Build context
        if used_reconstruction and reconstruction is not None:
            context = self.reconstructor.format_context_for_agent(
                session_id, reconstruction, self.tracker
            )
        else:
            available = self.tracker.get_available_turns(session_id, last_n=6)
            context = self.tracker.format_turns_as_text(available)

        # Step 5 — Build strategy-aware prompts
        strategy = reconstruction.strategy if reconstruction else "none"

        if strategy == "soft":
            style_instruction = (
                "Start your response with: Based on our earlier discussion,"
            )
        elif strategy == "ask_user":
            topic = (
                reconstruction.topics_inferred[0]
                if reconstruction and reconstruction.topics_inferred
                else "what we discussed"
            )
            style_instruction = (
                f"Ask the user: I want to make sure I have the right "
                f"context — could you briefly remind me what we discussed "
                f"about {topic}?"
            )
        else:
            style_instruction = "Respond naturally and helpfully."

        system_prompt = (
            "You are a helpful assistant continuing a conversation.\n"
            "Context marked [INFERRED] is reconstructed — treat it as "
            "background knowledge.\n"
            f"{style_instruction}\n"
            "Keep your response concise and relevant."
        )

        user_prompt = (
            f"CONVERSATION CONTEXT:\n{context}\n\n"
            f"USER MESSAGE: {user_message}\n\n"
            f"Your response:"
        )

        # Step 6 — Call Ollama
        response_text = ask_ollama(user_prompt, system_prompt)

        # Step 7 — Record AI response
        ai_turn = self.tracker.add_turn(session_id, "ai", response_text)
        self.memory.store_turn(session_id, ai_turn)

        # Step 8 — Return structured response
        return AgentResponse(
            response_text=response_text,
            used_reconstruction=used_reconstruction,
            confidence=reconstruction.confidence if reconstruction else 1.0,
            strategy=strategy,
            integrity_score=gap_report.integrity_score,
            inferred_summary=(
                reconstruction.inferred_summary if reconstruction else ""
            ),
        )

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def get_session_status(self, session_id: str) -> dict:
        """
        Full status snapshot of a session for the API/UI.

        Returns:
            Dict with integrity score, turn counts, gap report,
            memory snapshot, and full timeline.
        """
        gap_report = self.tracker.detect_gaps(session_id)
        memory_snapshot = self.memory.get_memory_snapshot(session_id)
        timeline = self.tracker.get_full_timeline(session_id)
        total_turns = len(self.tracker.sessions.get(session_id, []))
        available_count = int(gap_report.integrity_score * total_turns)

        return {
            "session_id": session_id,
            "integrity_score": gap_report.integrity_score,
            "total_turns": total_turns,
            "available_turns": available_count,
            "gap_report": {
                "has_gaps": gap_report.has_gaps,
                "gap_ranges": gap_report.gap_ranges,
                "total_missing": gap_report.total_missing,
                "gap_types": gap_report.gap_types,
                "severity": gap_report.severity,
            },
            "memory_snapshot": memory_snapshot,
            "timeline": timeline,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("  ContextReviveAgent — Smoke Test")
    print("=" * 60)

    # ------ TEST 1: Full conversation, no gap ------
    print("\n=== TEST 1: Full conversation, no gap ===")
    agent = ContextReviveAgent()
    sid = agent.new_session("customer_support")
    print(f"[Session] {sid[:12]}...")

    r = agent.chat(sid, "So did you process my refund?")
    print(f"Response: {r.response_text}")
    print(f"Reconstruction used: {r.used_reconstruction}")
    print(f"Strategy: {r.strategy}")
    print(f"Integrity: {r.integrity_score}")

    # Cleanup test 1
    agent.memory.vector_store.delete_session_memories(sid)

    # ------ TEST 2: Same question after gap ------
    print("\n=== TEST 2: Same question after gap ===")
    agent2 = ContextReviveAgent()
    sid2 = agent2.new_session("customer_support")
    gap_info = agent2.simulate_gap(sid2, [5, 6, 7, 8])
    print(f"[Gap] {gap_info}")

    r2 = agent2.chat(sid2, "So did you process my refund?")
    print(f"Response: {r2.response_text}")
    print(f"Reconstruction used: {r2.used_reconstruction}")
    print(f"Confidence: {r2.confidence:.2f}")
    print(f"Strategy: {r2.strategy}")
    print(f"Inferred: {r2.inferred_summary}")

    # Cleanup test 2
    agent2.memory.vector_store.delete_session_memories(sid2)

    # ------ Verdict ------
    print("\n=== ENGINE PROOF ===")
    print("Both responses reference refund/order context?")
    print("If yes — engine works.")