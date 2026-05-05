"""
ContextRevive — Conversation Tracker
Pure Python logic for tracking conversation turns, detecting gaps,
and providing evidence windows for reconstruction.
NO Ollama calls in this file.
"""

import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SCENARIOS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scenarios", "conversations.json"
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    session_id: str
    turn_index: int
    role: str
    content: str
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    is_available: bool = True
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.content.split())


@dataclass
class GapReport:
    """Analysis of missing turns in a conversation session."""
    has_gaps: bool
    gap_ranges: list          # e.g. [[5, 8]]
    total_missing: int
    gap_types: list           # e.g. ["medium"]
    severity: str             # "none" | "low" | "medium" | "high"
    integrity_score: float    # available / total


# ---------------------------------------------------------------------------
# Conversation Tracker
# ---------------------------------------------------------------------------

class ConversationTracker:
    """
    Manages conversation sessions — loading scripts, tracking turns,
    simulating gaps, and producing gap reports.
    """

    def __init__(self):
        self.sessions: dict[str, list[ConversationTurn]] = {}
        self.scenario_data: dict | None = None

    # ------------------------------------------------------------------
    # Scenario loading
    # ------------------------------------------------------------------

    def load_scenarios(self) -> None:
        """Load conversation scripts from scenarios/conversations.json."""
        path = os.path.abspath(SCENARIOS_PATH)
        with open(path, "r", encoding="utf-8") as f:
            self.scenario_data = json.load(f)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def new_session(self) -> str:
        """Create a fresh empty session and return its ID."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        return session_id

    def load_from_script(self, scenario_name: str) -> str:
        """
        Load a named scenario script into a new session.

        Args:
            scenario_name: Key in conversations.json (e.g. "customer_support").

        Returns:
            The session_id of the newly created session.
        """
        if self.scenario_data is None:
            self.load_scenarios()

        script = self.scenario_data[scenario_name]
        session_id = self.new_session()

        for turn in script["turns"]:
            self.add_turn(session_id, turn["role"], turn["content"])

        return session_id

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def add_turn(self, session_id: str, role: str, content: str) -> ConversationTurn:
        """
        Append a new turn to an existing session.

        Args:
            session_id: Target session.
            role:       "user" or "ai".
            content:    The message text.

        Returns:
            The newly created ConversationTurn.
        """
        turns = self.sessions[session_id]
        turn = ConversationTurn(
            session_id=session_id,
            turn_index=len(turns) + 1,
            role=role,
            content=content,
        )
        turns.append(turn)
        return turn

    # ------------------------------------------------------------------
    # Gap simulation
    # ------------------------------------------------------------------

    def simulate_gap(self, session_id: str, turn_indices: list[int]) -> None:
        """
        Mark specific turns as unavailable to simulate missing/corrupted data.

        Args:
            session_id:   The session to modify.
            turn_indices: 1-based turn indices to mark as missing.
        """
        for turn in self.sessions[session_id]:
            if turn.turn_index in turn_indices:
                turn.is_available = False

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    def detect_gaps(self, session_id: str) -> GapReport:
        """
        Analyse a session for missing turns and return a structured report.

        Returns:
            A GapReport with gap ranges, severity, and integrity score.
        """
        turns = self.sessions[session_id]
        total = len(turns)

        if total == 0:
            return GapReport(
                has_gaps=False,
                gap_ranges=[],
                total_missing=0,
                gap_types=[],
                severity="none",
                integrity_score=1.0,
            )

        missing_indices = sorted(
            t.turn_index for t in turns if not t.is_available
        )

        # Group consecutive missing indices into ranges
        gap_ranges: list[list[int]] = []
        if missing_indices:
            start = missing_indices[0]
            end = missing_indices[0]
            for idx in missing_indices[1:]:
                if idx == end + 1:
                    end = idx
                else:
                    gap_ranges.append([start, end])
                    start = idx
                    end = idx
            gap_ranges.append([start, end])

        # Classify each gap range
        gap_types = []
        for r in gap_ranges:
            size = r[1] - r[0] + 1
            if size <= 2:
                gap_types.append("shallow")
            elif size <= 5:
                gap_types.append("medium")
            else:
                gap_types.append("deep")

        total_missing = len(missing_indices)
        available = total - total_missing
        integrity_score = available / total if total > 0 else 1.0

        # Severity from integrity score
        if integrity_score >= 0.9:
            severity = "none"
        elif integrity_score >= 0.7:
            severity = "low"
        elif integrity_score >= 0.5:
            severity = "medium"
        else:
            severity = "high"

        return GapReport(
            has_gaps=len(missing_indices) > 0,
            gap_ranges=gap_ranges,
            total_missing=total_missing,
            gap_types=gap_types,
            severity=severity,
            integrity_score=round(integrity_score, 4),
        )

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_available_turns(
        self, session_id: str, last_n: int | None = None
    ) -> list[ConversationTurn]:
        """
        Return only the turns that are still available (not missing).

        Args:
            session_id: Target session.
            last_n:     If given, return only the last N available turns.
        """
        available = [t for t in self.sessions[session_id] if t.is_available]
        if last_n is not None:
            available = available[-last_n:]
        return available

    def get_turns_around_gap(
        self, session_id: str, gap_range: list[int], window: int = 3
    ) -> dict[str, list[ConversationTurn]]:
        """
        Get available turns surrounding a gap for use as evidence.

        Args:
            session_id: Target session.
            gap_range:  [start_index, end_index] of the gap.
            window:     Max turns to retrieve on each side.

        Returns:
            {"before": [...], "after": [...]}
        """
        turns = self.sessions[session_id]
        gap_start, gap_end = gap_range

        before = [
            t for t in turns
            if t.is_available and t.turn_index < gap_start
        ][-window:]

        after = [
            t for t in turns
            if t.is_available and t.turn_index > gap_end
        ][:window]

        return {"before": before, "after": after}

    # ------------------------------------------------------------------
    # Display / serialization
    # ------------------------------------------------------------------

    def get_full_timeline(self, session_id: str) -> list[dict]:
        """
        Return every turn as a dict for the UI transparency panel.

        Each dict contains: turn_index, role, content_preview, status.
        """
        timeline = []
        for t in self.sessions[session_id]:
            timeline.append({
                "turn_index": t.turn_index,
                "role": t.role,
                "content_preview": t.content[:60],
                "status": "available" if t.is_available else "missing",
            })
        return timeline

    @staticmethod
    def format_turns_as_text(turns: list[ConversationTurn]) -> str:
        """
        Format a list of turns into a plain-text prompt string.

        Output:
            Turn 1 (user): content
            Turn 2 (ai): content
            ...
        """
        lines = []
        for t in turns:
            lines.append(f"Turn {t.turn_index} ({t.role}): {t.content}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tracker = ConversationTracker()

    # Load the customer_support scenario
    print("=" * 55)
    print("  ConversationTracker — Smoke Test")
    print("=" * 55)

    session_id = tracker.load_from_script("customer_support")
    print(f"\n[Session] {session_id}")

    # Full timeline before gap
    print("\n--- Full Timeline (before gap) ---")
    for entry in tracker.get_full_timeline(session_id):
        print(
            f"  Turn {entry['turn_index']:>2} [{entry['role']:>4}] "
            f"({entry['status']}) {entry['content_preview']}"
        )

    # Simulate gap at turns 5-8
    tracker.simulate_gap(session_id, [5, 6, 7, 8])
    print("\n--- Simulated gap at turns 5, 6, 7, 8 ---")

    # Full timeline after gap
    print("\n--- Full Timeline (after gap) ---")
    for entry in tracker.get_full_timeline(session_id):
        marker = "✗" if entry["status"] == "missing" else "✓"
        print(
            f"  {marker} Turn {entry['turn_index']:>2} [{entry['role']:>4}] "
            f"{entry['content_preview']}"
        )

    # Gap report
    report = tracker.detect_gaps(session_id)
    print(f"\n--- Gap Report ---")
    print(f"  Has gaps:        {report.has_gaps}")
    print(f"  Gap ranges:      {report.gap_ranges}")
    print(f"  Total missing:   {report.total_missing}")
    print(f"  Gap types:       {report.gap_types}")
    print(f"  Severity:        {report.severity}")
    print(f"  Integrity score: {report.integrity_score}")

    # Turns around the gap
    evidence = tracker.get_turns_around_gap(session_id, [5, 8], window=3)
    print(f"\n--- Evidence Window (3 turns each side) ---")
    print("  BEFORE gap:")
    for t in evidence["before"]:
        print(f"    Turn {t.turn_index} ({t.role}): {t.content}")
    print("  AFTER gap:")
    for t in evidence["after"]:
        print(f"    Turn {t.turn_index} ({t.role}): {t.content}")