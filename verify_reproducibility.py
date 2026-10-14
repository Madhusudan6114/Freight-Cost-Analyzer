"""
Reproducibility verification: runs the pipeline 3 times and diffs the outputs.

Verifies that flags, numbers, and matched_note_ids are identical across runs.
Explanation wording may vary if using LLM, but verdicts and note IDs must match.
"""

import subprocess
import sys
import os
import pandas as pd


def run_pipeline(run_id: str, extra_args: list[str] = None) -> str:
    """Run main.py with a given run ID and return the output CSV path."""
    cmd = [sys.executable, "main.py", "--run-id", run_id]
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'=' * 40}")
    print(f"Run {run_id}")
    print(f"{'=' * 40}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".")
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Run {run_id} failed with code {result.returncode}")
    
    return os.path.join("output", f"anomaly_report_run{run_id}.csv")


def compare_runs(path1: str, path2: str, run1_id: str, run2_id: str) -> bool:
    """Compare two output CSVs. Returns True if they match on graded columns."""
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    # Columns that must be identical
    graded_cols = ["route", "week_of", "cost_per_tonne_km", "vs_own_history",
                   "vs_similar_routes", "flagged", "matched_note_id"]
    
    if len(df1) != len(df2):
        print(f"  MISMATCH: Run {run1_id} has {len(df1)} rows, Run {run2_id} has {len(df2)} rows")
        return False

    all_match = True
    for col in graded_cols:
        # Handle NaN comparison
        match = (df1[col].fillna("") == df2[col].fillna("")).all()
        if not match:
            print(f"  MISMATCH in column '{col}' between Run {run1_id} and Run {run2_id}:")
            diff_mask = df1[col].fillna("") != df2[col].fillna("")
            for idx in df1[diff_mask].index:
                print(f"    Row {idx}: '{df1.loc[idx, col]}' vs '{df2.loc[idx, col]}'")
            all_match = False

    return all_match


def main():
    extra_args = sys.argv[1:] if len(sys.argv) > 1 else ["--no-llm"]

    # Run 3 times
    paths = []
    for i in range(1, 4):
        path = run_pipeline(str(i), extra_args)
        paths.append(path)

    # Compare all pairs
    print("\n" + "=" * 60)
    print("REPRODUCIBILITY CHECK")
    print("=" * 60)

    all_match = True
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            match = compare_runs(paths[i], paths[j], str(i + 1), str(j + 1))
            status = "PASS" if match else "FAIL"
            print(f"  Run {i + 1} vs Run {j + 1}: {status}")
            all_match = all_match and match

    if all_match:
        print("\nAll 3 runs produced identical results on graded columns.")
    else:
        print("\nWARNING: Some runs produced different results!")

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
