# FreightTiger Shipping Cost Anomaly Detector

A smart assistant that monitors freight shipping costs across routes, detects abnormal cost increases, and uses AI to determine whether those increases are justified by real-world events (flooding, fuel hikes, festival surcharges) or unexplained and worth human review.

## Features

- **Weekly cost tracking** — Groups shipments by route and week (Mon–Sun), computes cost per tonne-km
- **Historical comparison** — Trailing 8-week rolling average per route (no look-ahead)
- **Peer comparison** — Same-week average across routes of the same type (Short/Medium/Long), excluding self
- **Anomaly flagging** — Flags route-weeks where cost exceeds thresholds vs. history or peers
- **AI-powered note matching** — Uses Groq API with open-weight models (Llama 3.3 70B) or deterministic fallback to check if context notes (flooding, diesel, festivals) justify the cost rise
- **Hallucination-safe** — Notes that say "no impact" or "stable" are never used to justify a rise; no note = unexplained
- **Reproducible** — Deterministic analytics; LLM runs at temperature 0; 3-run verification passes
- **Interactive Q&A** — Stretch goal: ask the assistant questions about findings in plain English

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Data processing | pandas |
| LLM | Groq API (`llama-3.3-70b-versatile`) via `groq` SDK |
| Configuration | python-dotenv |
| Testing | pytest |

## Project Structure

```
├── main.py                  # Pipeline entrypoint: analytics → note matching → CSV output
├── analytics.py             # Deterministic: weekly grouping, cost calc, rolling history, peer comparison, flagging
├── note_matcher.py          # LLM-based and deterministic note matching + token tracking
├── verify_reproducibility.py # Runs 3 times, diffs output to prove determinism
├── test_analytics.py        # 17 tests for core analytics logic
├── test_note_matcher.py     # 10 tests for note matching and prompt building
├── requirements.txt         # Python dependencies
├── .env.example             # Required environment variables
├── data/
│   ├── shipment_records.csv # Input: 2940 shipment records
│   ├── context_notes.csv    # Input: 10 real-world context notes
│   └── sample_output_format_v2.csv  # Reference output format
└── output/                  # Generated: anomaly reports, token logs
```

## Setup & Installation

```bash
# 1. Clone / unzip the project
# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set up Groq API key for LLM-powered matching
cp .env.example .env
# Edit .env and set GROQ_API_KEY (get a free key at https://console.groq.com/keys)
```

## Running

```bash
# Full run with LLM (requires GROQ_API_KEY in .env)
python main.py

# Run without LLM (deterministic matcher, no API key needed)
python main.py --no-llm

# Interactive Q&A mode (requires LLM)
python main.py --chat

# Custom thresholds / paths
python main.py --shipments data/shipment_records.csv --notes data/context_notes.csv
```

Output is written to `output/anomaly_report.csv` and `output/token_log.json`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | (required for LLM mode) | Groq API key from console.groq.com |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq-hosted open-source model |
| `LLM_TEMPERATURE` | `0` | Temperature for reproducibility |
| `HISTORY_THRESHOLD` | `15.0` | % above trailing average to flag |
| `PEER_THRESHOLD` | `15.0` | % above peer average to flag |

## Running Tests

```bash
python -m pytest test_analytics.py test_note_matcher.py -v
```

27 tests covering weekly grouping, cost-per-tonne-km calculation, rolling history, peer comparison, anomaly flagging, note candidate filtering, deterministic matching rules, and LLM prompt construction.

## Reproducibility Verification

```bash
python verify_reproducibility.py --no-llm    # deterministic mode
python verify_reproducibility.py              # LLM mode (with API key)
```

Runs the pipeline 3 times and diffs all graded columns (route, week_of, cost_per_tonne_km, vs_own_history, vs_similar_routes, flagged, matched_note_id). All 3 runs must match.

**Result:** All 3 runs produce identical output on all graded columns.

## How It Works

### Pipeline Steps

1. **Load & group** — Load `shipment_records.csv`, assign each shipment to its Monday–Sunday week, compute route as `origin-destination`
2. **Compute weekly cost** — For each route+week: `cost_per_tonne_km = sum(freight_cost) / sum(quantity × distance)` (weighted average across all shipments)
3. **vs. own history** — Trailing 8-week rolling average of prior weeks only (no look-ahead). First week has no history.
4. **vs. similar routes** — Same-week average across peer routes (same `route_type`), excluding itself
5. **Flag anomalies** — Flag if cost > 15% above history OR > 15% above peers
6. **Match notes** — For each flagged week, find context notes matching the route + date window, then evaluate whether they justify the rise
7. **Output CSV** — Write results matching the required format

### Note Matching Strategy

The note matcher uses a two-stage approach:

1. **Deterministic pre-filter**: Find notes where `applies_to` matches the route (or "All Routes") and `date` falls within ±21 days of the anomaly week
2. **Evaluation** (LLM or rule-based):
   - LLM mode: Groq evaluates whether the note genuinely describes a cost-increasing event using `llama-3.3-70b-versatile`
   - Deterministic mode: Keyword matching — notes with "surcharge", "flooding", "rose", etc. justify; notes with "not affected", "stable", "no disruptions" do not

### How I Convinced Myself It Was Right

1. **Sample output validation** — All 3 example rows from `sample_output_format_v2.csv` match exactly (cost, percentages, flag, note ID)
2. **Unit tests** — 27 tests covering every computation step, including edge cases (first week with no history, NaN handling, peer exclusion, route type isolation)
3. **Reproducibility** — 3 consecutive runs produce byte-identical output on all graded columns
4. **Manual note spot-checks** — Verified that:
   - N002 (festival surcharge on Ahmedabad-Mumbai) correctly justifies the Jan 20 2025 anomaly
   - N001 (Chennai-Bangalore flooding, Feb 24 – Mar 8) correctly justifies the 3 weeks during the disruption
   - N005 (Mumbai-Delhi maintenance, "costs not significantly affected") is correctly NOT used to justify
   - N006 ("stable demand, no disruptions") is correctly NOT used to justify Mumbai-Pune Sep 2025 anomaly

## Token / Cost Log (Deterministic Mode)

```json
{
  "model": "deterministic (no LLM)",
  "total_input_tokens": 0,
  "total_output_tokens": 0,
  "total_calls": 0,
  "estimated_cost_usd": "$0.0000"
}
```

When using Groq LLM (`llama-3.3-70b-versatile`): approximately 27 API calls (one per anomaly), ~300 tokens per call, completely free under Groq's free tier rate limits (or ~$0.006 at published rates).

## Known Limitations

- **Deterministic matcher edge case**: For Chennai-Bangalore week of Mar 17, both N001 (flooding) and N009 (return to normal) are candidates. The deterministic matcher picks N001 first. The LLM matcher handles this nuance correctly by recognizing the disruption had ended.
- **Threshold choice**: The 15% threshold is configurable but somewhat arbitrary. A lower threshold catches more subtle anomalies but increases noise.
- **No seasonal adjustment**: The trailing 8-week average doesn't account for seasonal cost patterns (e.g., monsoon season).
