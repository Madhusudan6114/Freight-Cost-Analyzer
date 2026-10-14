"""
Note matching and AI explanation engine.

Uses a combination of deterministic filtering (route + date window) and
LLM-based reasoning to match context notes to cost anomalies and generate
plain-English explanations.
"""

import json
import pandas as pd
from datetime import timedelta
from dataclasses import dataclass

from groq import Groq


@dataclass
class TokenTracker:
    """Tracks LLM token usage across all calls."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    call_count: int = 0
    model_name: str = ""

    def record(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_calls": self.call_count,
            "estimated_cost_usd": self._estimate_cost(),
        }

    def _estimate_cost(self) -> str:
        # Groq free tier: generous rate limits, no charge
        # Paid: Llama 3.3 70B ~ $0.59/1M input, $0.79/1M output
        input_cost = (self.total_input_tokens / 1_000_000) * 0.59
        output_cost = (self.total_output_tokens / 1_000_000) * 0.79
        total = input_cost + output_cost
        return f"${total:.4f} (at Groq paid rates; free under Groq free tier)"


def load_notes(path: str) -> pd.DataFrame:
    """Load context notes CSV."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def find_candidate_notes(notes: pd.DataFrame, route: str, week_of: pd.Timestamp,
                         window_days: int = 21) -> pd.DataFrame:
    """
    Find notes that could plausibly explain a cost anomaly on a given route and week.
    
    Matching criteria:
    - Note applies to this route OR to "All Routes"
    - Note date is within window_days before or after the week_of date
    
    This deterministic pre-filter narrows down candidates before LLM evaluation.
    """
    route_match = (notes["applies_to"] == route) | (notes["applies_to"] == "All Routes")

    week_start = week_of
    week_end = week_of + timedelta(days=6)

    date_match = (
        (notes["date"] >= week_start - timedelta(days=window_days)) &
        (notes["date"] <= week_end + timedelta(days=window_days))
    )

    return notes[route_match & date_match].copy()


def build_llm_prompt(route: str, week_of: str, cost_per_tonne_km: float,
                     pct_vs_history: str, pct_vs_peers: str,
                     candidate_notes: list[dict]) -> str:
    """Build the prompt for the LLM to evaluate whether notes justify a cost anomaly."""

    notes_text = ""
    if candidate_notes:
        for n in candidate_notes:
            notes_text += f"  - Note {n['note_id']} (date: {n['date']}, applies to: {n['applies_to']}): {n['note']}\n"
    else:
        notes_text = "  (No candidate notes found for this route and time period.)\n"

    prompt = f"""You are a freight cost analyst. A shipping route has an unusual cost increase. Your job is to determine if any of the provided context notes genuinely explain this cost rise.

ANOMALY:
- Route: {route}
- Week of: {week_of}
- Cost per tonne-km: {cost_per_tonne_km:.4f} INR
- {pct_vs_history}
- {pct_vs_peers}

CANDIDATE NOTES:
{notes_text}
RULES (follow exactly):
1. A note JUSTIFIES the cost rise ONLY if ALL of these are true:
   - The note describes something that would INCREASE costs (not just delays, not "no disruptions", not "costs were not affected")
   - The note applies to THIS route or "All Routes"
   - The note's date/time window overlaps with or closely precedes this anomaly week
2. If a note says costs were NOT affected, or mentions stable/normal conditions, it does NOT justify the rise.
3. If no note genuinely justifies the rise, say so honestly. Do NOT fabricate a reason.

Respond with EXACTLY this JSON (no markdown fences, no extra text):
{{
  "justified": true or false,
  "matched_note_id": "N00X" or "",
  "reason": "One or two sentences explaining the verdict. If justified, cite the note ID and reflect its content. If not justified, say no matching note was found."
}}"""
    return prompt


class NoteMatcherLLM:
    """Uses Groq (Llama) to evaluate whether context notes justify cost anomalies."""

    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile",
                 temperature: float = 0):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.tracker = TokenTracker(model_name=model_name)

    def evaluate(self, route: str, week_of: str, cost_per_tonne_km: float,
                 pct_vs_history: str, pct_vs_peers: str,
                 candidate_notes: list[dict]) -> dict:
        """
        Ask the LLM whether the candidate notes justify this anomaly.
        Returns dict with keys: justified, matched_note_id, reason.
        """
        prompt = build_llm_prompt(route, week_of, cost_per_tonne_km,
                                  pct_vs_history, pct_vs_peers, candidate_notes)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=300,
            response_format={"type": "json_object"},
        )

        # Track tokens
        if response.usage:
            self.tracker.record(
                response.usage.prompt_tokens or 0,
                response.usage.completion_tokens or 0,
            )

        return self._parse_response(response.choices[0].message.content)

    def _parse_response(self, text: str) -> dict:
        """Parse the LLM JSON response, with fallback for malformed output."""
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        try:
            result = json.loads(text)
            return {
                "justified": bool(result.get("justified", False)),
                "matched_note_id": str(result.get("matched_note_id", "")),
                "reason": str(result.get("reason", "LLM response missing reason.")),
            }
        except (json.JSONDecodeError, KeyError):
            return {
                "justified": False,
                "matched_note_id": "",
                "reason": f"LLM response could not be parsed. Raw: {text[:200]}",
            }


class NoteMatcherDeterministic:
    """
    Fallback note matcher that works without an LLM API key.
    Uses rule-based matching: checks if candidate notes describe cost-increasing events.
    """

    def __init__(self):
        self.tracker = TokenTracker(model_name="deterministic (no LLM)")
        # Keywords that indicate a cost-increasing event
        self._cost_increase_keywords = [
            "surcharge", "higher", "rose", "pushing up", "disrupted",
            "flooding", "flood", "detour", "higher trip costs",
            "diesel prices rose", "cost", "pricier"
        ]
        # Keywords that indicate NO cost impact
        self._no_impact_keywords = [
            "not significantly affected", "costs were not affected",
            "remained normal", "no significant disruptions", "stable",
            "no major disruptions", "absorbed by transporters without a rate change",
            "not part of this dataset"
        ]

    def evaluate(self, route: str, week_of: str, cost_per_tonne_km: float,
                 pct_vs_history: str, pct_vs_peers: str,
                 candidate_notes: list[dict]) -> dict:
        """Rule-based note evaluation."""
        if not candidate_notes:
            return {
                "justified": False,
                "matched_note_id": "",
                "reason": "No matching note found for this route or date range. "
                          "Cost rise looks unexplained and worth a human review.",
            }

        for note in candidate_notes:
            note_text = note["note"].lower()

            # Skip notes that explicitly say no cost impact
            if any(kw in note_text for kw in self._no_impact_keywords):
                continue

            # Check if note describes a cost-increasing event
            if any(kw in note_text for kw in self._cost_increase_keywords):
                return {
                    "justified": True,
                    "matched_note_id": note["note_id"],
                    "reason": f"Matches note {note['note_id']} dated {note['date']}: "
                              f"{note['note']}",
                }

        # Notes exist but none justify a cost increase
        closest = candidate_notes[0]
        return {
            "justified": False,
            "matched_note_id": "",
            "reason": f"The closest note ({closest['note_id']}, {closest['date']}) "
                      f"does not describe a reason for a cost rise on this route. "
                      f"No genuine justification found; flagged for review.",
        }
