"""
ContextRevive — Context Reconstructor
The heart of the engine: uses llama3.1:8b to infer what happened
in missing conversation gaps based on surrounding evidence.
"""

import os
import sys
from dataclasses import dataclass

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ollama_client import ask_ollama, parse_llm_json
from core.tracker import ConversationTracker, GapReport
from core.memory import MemoryManager


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ReconstructionResult:
    """Output of a gap reconstruction attempt."""
    session_id: str
    gap_ranges: list
    inferred_summary: str
    confidence: float           # 0.0 – 1.0
    strategy: str               # "silent" | "soft" | "ask_user"
    topics_inferred: list       # e.g. ["damage report", "refund policy"]
    evidence_turns_used: list   # turn indices that informed the inference


# ---------------------------------------------------------------------------
# Context Reconstructor
# ---------------------------------------------------------------------------

class ContextReconstructor:
    """
    Analyses gaps in a conversation and uses llama3.1:8b to infer
    what was most likely discussed in the missing turns.

    Strategies:
        silent   — high confidence (≥ 0.75), merge automatically
        soft     — medium confidence (≥ 0.55), show to user with flag
        ask_user — low confidence (< 0.55), request human clarification
    """

    # ------------------------------------------------------------------
    # Main reconstruction pipeline
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        session_id: str,
        gap_report: GapReport,
        tracker: ConversationTracker,
        memory_manager: MemoryManager,
    ) -> ReconstructionResult:
        """
        Reconstruct missing context for the first detected gap.

        Args:
            session_id:     The session to reconstruct.
            gap_report:     Output of tracker.detect_gaps().
            tracker:        ConversationTracker with loaded turns.
            memory_manager: MemoryManager with stored turn vectors.

        Returns:
            A ReconstructionResult with the inferred summary,
            confidence, and recommended strategy.
        """
        # No gaps — nothing to do
        if not gap_report.has_gaps:
            return ReconstructionResult(
                session_id=session_id,
                gap_ranges=[],
                inferred_summary="No gaps detected.",
                confidence=1.0,
                strategy="silent",
                topics_inferred=[],
                evidence_turns_used=[],
            )

        # ----- STEP 1: Gather evidence -----
        gap_range = gap_report.gap_ranges[0]  # first gap for demo
        evidence = tracker.get_turns_around_gap(
            session_id, gap_range, window=3
        )
        before_turns = evidence["before"]
        after_turns = evidence["after"]

        # Retrieve relevant memories using the last available turn before gap
        query = before_turns[-1].content if before_turns else ""
        relevant_memories = memory_manager.retrieve_relevant(
            session_id, query=query, top_k=3
        ) if query else []

        # Track which turn indices were used as evidence
        evidence_indices = (
            [t.turn_index for t in before_turns]
            + [t.turn_index for t in after_turns]
        )

        # ----- STEP 2: Format evidence as text -----
        before_text = tracker.format_turns_as_text(before_turns)
        after_text = tracker.format_turns_as_text(after_turns)
        memory_text = "\n".join(
            [f"- {m['content']}" for m in relevant_memories]
        )
        gap_size = gap_range[1] - gap_range[0] + 1

        # ----- STEP 3: Build prompt -----
        system_prompt = (
            "You are an expert at reconstructing missing conversation "
            "context. Analyze the fragments given and infer what was likely "
            "discussed in the missing portion. Be conservative — only infer "
            "what the evidence strongly supports. Always respond with valid "
            "JSON only, no other text."
        )

        user_prompt = f"""A conversation has {gap_size} missing turns.

CONVERSATION BEFORE THE GAP:
{before_text}

CONVERSATION AFTER THE GAP:
{after_text}

RELEVANT STORED MEMORIES:
{memory_text if memory_text else "None available"}

The gap contains turns {gap_range[0]} to {gap_range[1]}.

What was most likely discussed in those missing turns?

Respond ONLY with this exact JSON format:
{{
  "inferred_summary": "2-3 sentence description of what was likely discussed",
  "topics_inferred": ["topic1", "topic2"],
  "confidence": 0.75,
  "reasoning": "brief explanation of why you are confident or not"
}}"""

        # ----- STEP 4: Call Ollama -----
        print(f"[Reconstructor] Calling llama3.1:8b for gap {gap_range}...")
        raw_response = ask_ollama(user_prompt, system_prompt)
        parsed = parse_llm_json(raw_response)

        if not parsed:
            confidence = 0.3
            inferred_summary = "Could not determine missing context reliably."
            topics_inferred = []
        else:
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # clamp to 0–1
            inferred_summary = parsed.get("inferred_summary", "")
            topics_inferred = parsed.get("topics_inferred", [])

        # ----- STEP 5: Determine strategy -----
        if confidence >= 0.75:
            strategy = "silent"
        elif confidence >= 0.55:
            strategy = "soft"
        else:
            strategy = "ask_user"

        return ReconstructionResult(
            session_id=session_id,
            gap_ranges=gap_report.gap_ranges,
            inferred_summary=inferred_summary,
            confidence=confidence,
            strategy=strategy,
            topics_inferred=topics_inferred,
            evidence_turns_used=evidence_indices,
        )

    # ------------------------------------------------------------------
    # Context formatter for the agent
    # ------------------------------------------------------------------

    def format_context_for_agent(
        self,
        session_id: str,
        reconstruction: ReconstructionResult,
        tracker: ConversationTracker,
    ) -> str:
        """
        Build a merged context string for the downstream agent.

        Combines available turns with the inferred summary (if confidence
        is high enough) inserted at the gap position.

        Args:
            session_id:     The session to format.
            reconstruction: The ReconstructionResult from reconstruct().
            tracker:        ConversationTracker with loaded turns.

        Returns:
            A formatted text block ready to be used as prompt context.
        """
        available_turns = tracker.get_available_turns(session_id, last_n=6)

        if not reconstruction.gap_ranges or reconstruction.strategy == "ask_user":
            # No reconstruction to merge, or confidence too low
            return tracker.format_turns_as_text(available_turns)

        # Build text with inferred context inserted at the gap position
        gap_start = reconstruction.gap_ranges[0][0]
        lines = []

        for t in available_turns:
            # Insert inferred context just before the first turn after the gap
            if t.turn_index >= gap_start and reconstruction.inferred_summary:
                lines.append(
                    f"\n[INFERRED CONTEXT — confidence "
                    f"{reconstruction.confidence:.0%}]: "
                    f"{reconstruction.inferred_summary}\n"
                )
                # Only insert once
                reconstruction.inferred_summary = ""

            lines.append(f"Turn {t.turn_index} ({t.role}): {t.content}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  ContextReconstructor — Smoke Test")
    print("=" * 60)

    tracker = ConversationTracker()
    memory_manager = MemoryManager()
    reconstructor = ContextReconstructor()

    # Load scenario and store all turns in memory
    session_id = tracker.load_from_script("customer_support")
    print(f"\n[Session] {session_id[:12]}...")

    print("[Storing all turns in memory...]")
    for turn in tracker.sessions[session_id]:
        memory_manager.store_turn(session_id, turn)

    # Simulate gap at turns 5-8
    tracker.simulate_gap(session_id, [5, 6, 7, 8])
    gap_report = tracker.detect_gaps(session_id)
    print(f"[Gap Report] ranges={gap_report.gap_ranges}, "
          f"severity={gap_report.severity}, "
          f"integrity={gap_report.integrity_score}")

    # Reconstruct
    print()
    result = reconstructor.reconstruct(
        session_id, gap_report, tracker, memory_manager
    )

    print(f"\n{'=' * 60}")
    print(f"  RECONSTRUCTION RESULT")
    print(f"{'=' * 60}")
    print(f"  Confidence:    {result.confidence:.0%}")
    print(f"  Strategy:      {result.strategy}")
    print(f"  Summary:       {result.inferred_summary}")
    print(f"  Topics:        {result.topics_inferred}")
    print(f"  Evidence used: turns {result.evidence_turns_used}")

    # Format merged context
    print(f"\n{'=' * 60}")
    print(f"  MERGED CONTEXT FOR AGENT")
    print(f"{'=' * 60}")
    merged = reconstructor.format_context_for_agent(
        session_id, result, tracker
    )
    print(merged)

    # Cleanup
    memory_manager.vector_store.delete_session_memories(session_id)
    print(f"\n[Cleanup] Done.")