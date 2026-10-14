"""
Tests for the note matching module.

Covers: candidate filtering, deterministic matcher logic, and prompt building.
"""

import pandas as pd
import pytest
from note_matcher import (
    find_candidate_notes, NoteMatcherDeterministic, build_llm_prompt, load_notes,
)


@pytest.fixture
def sample_notes():
    return pd.DataFrame([
        {"note_id": "N001", "date": pd.Timestamp("2025-02-24"),
         "applies_to": "Chennai-Bangalore",
         "note": "Heavy flooding disrupted truck movement, forcing detours and higher trip costs."},
        {"note_id": "N002", "date": pd.Timestamp("2025-01-20"),
         "applies_to": "Ahmedabad-Mumbai",
         "note": "A regional festival week saw a temporary surcharge."},
        {"note_id": "N003", "date": pd.Timestamp("2025-05-05"),
         "applies_to": "All Routes",
         "note": "Diesel prices rose nationwide, pushing up costs by 5-7%."},
        {"note_id": "N005", "date": pd.Timestamp("2024-07-29"),
         "applies_to": "Mumbai-Delhi",
         "note": "Scheduled highway maintenance caused minor delays; costs were not significantly affected."},
        {"note_id": "N006", "date": pd.Timestamp("2025-09-22"),
         "applies_to": "All Routes",
         "note": "Demand remained stable with no major disruptions reported."},
    ])


class TestFindCandidateNotes:
    def test_matches_route_and_date(self, sample_notes):
        candidates = find_candidate_notes(
            sample_notes, "Chennai-Bangalore", pd.Timestamp("2025-02-24")
        )
        assert "N001" in candidates["note_id"].values

    def test_matches_all_routes(self, sample_notes):
        candidates = find_candidate_notes(
            sample_notes, "Mumbai-Pune", pd.Timestamp("2025-05-05")
        )
        assert "N003" in candidates["note_id"].values

    def test_excludes_wrong_route(self, sample_notes):
        candidates = find_candidate_notes(
            sample_notes, "Delhi-Jaipur", pd.Timestamp("2025-02-24")
        )
        assert "N001" not in candidates["note_id"].values

    def test_excludes_distant_dates(self, sample_notes):
        candidates = find_candidate_notes(
            sample_notes, "Chennai-Bangalore", pd.Timestamp("2024-01-01")
        )
        assert "N001" not in candidates["note_id"].values


class TestDeterministicMatcher:
    def test_justified_with_cost_increasing_note(self):
        matcher = NoteMatcherDeterministic()
        result = matcher.evaluate(
            route="Chennai-Bangalore",
            week_of="2025-02-24",
            cost_per_tonne_km=5.0,
            pct_vs_history="+30% vs this route's past average",
            pct_vs_peers="+20% vs similar-length routes this week",
            candidate_notes=[{
                "note_id": "N001",
                "date": "2025-02-24",
                "applies_to": "Chennai-Bangalore",
                "note": "Heavy flooding disrupted truck movement, forcing detours and higher trip costs.",
            }],
        )
        assert result["justified"] is True
        assert result["matched_note_id"] == "N001"

    def test_not_justified_when_no_notes(self):
        matcher = NoteMatcherDeterministic()
        result = matcher.evaluate(
            route="Delhi-Jaipur",
            week_of="2024-11-11",
            cost_per_tonne_km=4.17,
            pct_vs_history="+35.5% vs this route's past average",
            pct_vs_peers="+21.0% vs similar-length routes this week",
            candidate_notes=[],
        )
        assert result["justified"] is False
        assert result["matched_note_id"] == ""

    def test_not_justified_when_note_says_no_impact(self):
        matcher = NoteMatcherDeterministic()
        result = matcher.evaluate(
            route="Mumbai-Delhi",
            week_of="2024-07-29",
            cost_per_tonne_km=3.5,
            pct_vs_history="+20% vs this route's past average",
            pct_vs_peers="+15% vs similar-length routes this week",
            candidate_notes=[{
                "note_id": "N005",
                "date": "2024-07-29",
                "applies_to": "Mumbai-Delhi",
                "note": "Scheduled highway maintenance caused minor delays; costs were not significantly affected.",
            }],
        )
        assert result["justified"] is False

    def test_not_justified_when_note_says_stable(self):
        matcher = NoteMatcherDeterministic()
        result = matcher.evaluate(
            route="Mumbai-Pune",
            week_of="2025-09-22",
            cost_per_tonne_km=3.98,
            pct_vs_history="+9.2% vs this route's past average",
            pct_vs_peers="+23.6% vs similar-length routes this week",
            candidate_notes=[{
                "note_id": "N006",
                "date": "2025-09-22",
                "applies_to": "All Routes",
                "note": "Demand remained stable with no major disruptions reported.",
            }],
        )
        assert result["justified"] is False


class TestBuildLLMPrompt:
    def test_prompt_includes_route_and_notes(self):
        prompt = build_llm_prompt(
            route="Chennai-Bangalore",
            week_of="2025-02-24",
            cost_per_tonne_km=5.0,
            pct_vs_history="+30% vs this route's past average",
            pct_vs_peers="+20% vs similar-length routes",
            candidate_notes=[{
                "note_id": "N001",
                "date": "2025-02-24",
                "applies_to": "Chennai-Bangalore",
                "note": "Flooding disrupted trucks.",
            }],
        )
        assert "Chennai-Bangalore" in prompt
        assert "N001" in prompt
        assert "Flooding" in prompt

    def test_prompt_handles_empty_notes(self):
        prompt = build_llm_prompt(
            route="Delhi-Jaipur",
            week_of="2024-11-11",
            cost_per_tonne_km=4.17,
            pct_vs_history="+35% vs history",
            pct_vs_peers="+21% vs peers",
            candidate_notes=[],
        )
        assert "No candidate notes found" in prompt
