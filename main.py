"""
Main pipeline: runs analytics, matches notes, generates output CSV.

Usage:
    python main.py                          # Full run with LLM (needs GROQ_API_KEY)
    python main.py --no-llm                 # Run with deterministic matcher (no API key needed)
    python main.py --run-id 1               # Tag output files with a run ID (for reproducibility checks)
    python main.py --chat                   # Interactive Q&A after analysis (stretch goal)
"""

import os
import sys
import json
import argparse
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from analytics import run_analytics
from note_matcher import (
    load_notes, find_candidate_notes,
    NoteMatcherLLM, NoteMatcherDeterministic, TokenTracker,
)


def process_anomalies(weekly: pd.DataFrame, notes: pd.DataFrame,
                      matcher) -> pd.DataFrame:
    """
    For each anomalous week, find candidate notes and ask the matcher
    to evaluate whether they justify the cost rise.
    """
    anomalies = weekly[weekly["is_anomaly"]].copy()

    results = []
    for _, row in anomalies.iterrows():
        candidates = find_candidate_notes(notes, row["route"], row["week_of"])
        candidate_dicts = candidates.to_dict("records") if not candidates.empty else []
        # Format dates as strings for the matcher
        for c in candidate_dicts:
            if isinstance(c.get("date"), pd.Timestamp):
                c["date"] = c["date"].strftime("%Y-%m-%d")

        verdict = matcher.evaluate(
            route=row["route"],
            week_of=row["week_of"].strftime("%Y-%m-%d"),
            cost_per_tonne_km=row["cost_per_tonne_km"],
            pct_vs_history=row["vs_own_history"],
            pct_vs_peers=row["vs_similar_routes"],
            candidate_notes=candidate_dicts,
        )
        results.append(verdict)

    anomalies["justified"] = [r["justified"] for r in results]
    anomalies["matched_note_id"] = [r["matched_note_id"] for r in results]
    anomalies["reason"] = [r["reason"] for r in results]

    return anomalies


def build_output(weekly: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final output CSV matching sample_output_format_v2.csv.
    Only includes anomaly rows (flagged weeks).
    
    Columns: route, week_of, cost_per_tonne_km, vs_own_history,
             vs_similar_routes, flagged, matched_note_id, reason
    """
    output = anomalies.copy()

    # Build the "flagged" column
    output["flagged"] = output.apply(
        lambda r: f"No (justified)" if r["justified"] else "Yes", axis=1
    )

    # Format week_of as date string
    output["week_of"] = output["week_of"].dt.strftime("%Y-%m-%d")

    # Round cost to 2 decimal places
    output["cost_per_tonne_km"] = output["cost_per_tonne_km"].round(2)

    # Select and order columns to match the required format
    output = output[["route", "week_of", "cost_per_tonne_km", "vs_own_history",
                      "vs_similar_routes", "flagged", "matched_note_id", "reason"]]

    return output.sort_values(["route", "week_of"]).reset_index(drop=True)


def run_chat(output_df: pd.DataFrame, notes: pd.DataFrame, matcher):
    """Stretch goal: interactive Q&A over findings using the LLM."""
    if not isinstance(matcher, NoteMatcherLLM):
        print("Chat mode requires LLM. Run without --no-llm to enable.")
        return

    # Build context from the analysis results
    context = "Here are the freight cost anomaly findings:\n\n"
    for _, row in output_df.iterrows():
        context += (f"Route: {row['route']}, Week: {row['week_of']}, "
                    f"Cost/tonne-km: {row['cost_per_tonne_km']}, "
                    f"{row['vs_own_history']}, {row['vs_similar_routes']}, "
                    f"Flagged: {row['flagged']}, Note: {row['matched_note_id']}, "
                    f"Reason: {row['reason']}\n")

    context += "\nContext notes:\n"
    for _, note in notes.iterrows():
        context += f"  {note['note_id']} ({note['date'].strftime('%Y-%m-%d')}, {note['applies_to']}): {note['note']}\n"

    system_msg = (f"{context}\n\nBased ONLY on the above findings and notes, "
                  f"answer questions concisely and accurately. "
                  f"Do not make up information that is not in the data above.")

    print("\n=== Freight Cost Assistant — Chat Mode ===")
    print("Ask questions about the analysis (type 'quit' to exit).\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break

        response = matcher.client.chat.completions.create(
            model=matcher.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=500,
        )

        if response.usage:
            matcher.tracker.record(
                response.usage.prompt_tokens or 0,
                response.usage.completion_tokens or 0,
            )

        print(f"\nAssistant: {response.choices[0].message.content}\n")


def main():
    parser = argparse.ArgumentParser(description="FreightTiger Shipping Cost Anomaly Detector")
    parser.add_argument("--no-llm", action="store_true",
                        help="Use deterministic note matcher (no API key needed)")
    parser.add_argument("--run-id", type=str, default="",
                        help="Tag output files with a run ID for reproducibility checks")
    parser.add_argument("--chat", action="store_true",
                        help="Enter interactive Q&A mode after analysis")
    parser.add_argument("--shipments", type=str, default="data/shipment_records.csv",
                        help="Path to shipment records CSV")
    parser.add_argument("--notes", type=str, default="data/context_notes.csv",
                        help="Path to context notes CSV")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory for output files")
    args = parser.parse_args()

    load_dotenv()

    # Config from env
    history_threshold = float(os.getenv("HISTORY_THRESHOLD", "15.0"))
    peer_threshold = float(os.getenv("PEER_THRESHOLD", "15.0"))

    # Validate input file paths
    if not os.path.exists(args.shipments):
        print(f"Error: Shipment records file not found at '{args.shipments}'.", file=sys.stderr)
        print("Please check the path or run from the project root.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.notes):
        print(f"Error: Context notes file not found at '{args.notes}'.", file=sys.stderr)
        print("Please check the path or run from the project root.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("FreightTiger Shipping Cost Anomaly Detector")
    print("=" * 60)

    # Step 1: Run deterministic analytics
    print("\n[1/4] Running analytics on shipment data...")
    weekly = run_analytics(args.shipments, history_threshold, peer_threshold)
    total_weeks = len(weekly)
    anomaly_count = weekly["is_anomaly"].sum()
    print(f"      Computed {total_weeks} route-weeks across {weekly['route'].nunique()} routes")
    print(f"      Found {anomaly_count} anomalous route-weeks")

    # Step 2: Set up note matcher
    print("\n[2/4] Setting up note matcher...")
    notes = load_notes(args.notes)
    print(f"      Loaded {len(notes)} context notes")

    if args.no_llm:
        matcher = NoteMatcherDeterministic()
        print("      Using deterministic matcher (no LLM)")
    else:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key or api_key == "your-api-key-here":
            print("      WARNING: No GROQ_API_KEY found. Falling back to deterministic matcher.")
            print("      Set GROQ_API_KEY in .env to use LLM-powered matching.")
            matcher = NoteMatcherDeterministic()
        else:
            model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
            temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
            matcher = NoteMatcherLLM(api_key, model_name, temperature)
            print(f"      Using Groq LLM ({model_name}, temperature={temperature})")

    # Step 3: Process anomalies with note matching
    print("\n[3/4] Matching anomalies against context notes...")
    anomalies = process_anomalies(weekly, notes, matcher)
    justified = anomalies["justified"].sum()
    unexplained = len(anomalies) - justified
    print(f"      {justified} justified (with matching note)")
    print(f"      {unexplained} unexplained (flagged for review)")

    # Step 4: Generate output
    print("\n[4/4] Generating output...")
    os.makedirs(args.output_dir, exist_ok=True)

    output = build_output(weekly, anomalies)
    suffix = f"_run{args.run_id}" if args.run_id else ""
    output_path = os.path.join(args.output_dir, f"anomaly_report{suffix}.csv")
    output.to_csv(output_path, index=False)
    print(f"      Output written to {output_path}")
    print(f"      Total rows: {len(output)}")

    # Save token/cost log
    tracker = matcher.tracker
    token_log = tracker.summary()
    token_log["timestamp"] = datetime.now().isoformat()
    token_log["run_id"] = args.run_id
    token_log["anomalies_processed"] = len(anomalies)

    token_log_path = os.path.join(args.output_dir, f"token_log{suffix}.json")
    with open(token_log_path, "w") as f:
        json.dump(token_log, f, indent=2)
    print(f"      Token log written to {token_log_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TOKEN / COST SUMMARY")
    print("=" * 60)
    print(f"  Model:          {token_log['model']}")
    print(f"  LLM calls:      {token_log['total_calls']}")
    print(f"  Input tokens:   {token_log['total_input_tokens']}")
    print(f"  Output tokens:  {token_log['total_output_tokens']}")
    print(f"  Estimated cost: {token_log['estimated_cost_usd']}")
    print("=" * 60)

    # Stretch: chat mode
    if args.chat:
        run_chat(output, notes, matcher)


if __name__ == "__main__":
    main()
