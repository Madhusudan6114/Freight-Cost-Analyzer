"""
Core analytics for shipping cost anomaly detection.

All computations here are deterministic — no LLM calls.
Handles: weekly grouping, cost-per-tonne-km, rolling history, peer comparison, flagging.
"""

import pandas as pd
import numpy as np


def load_shipments(path: str) -> pd.DataFrame:
    """Load and validate shipment records."""
    df = pd.read_csv(path, parse_dates=["shipment_date"])
    required = ["shipment_id", "origin", "destination", "route_type",
                 "quantity_tonnes", "distance_km", "freight_cost_inr", "shipment_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in shipment data: {missing}")
    df["route"] = df["origin"] + "-" + df["destination"]
    return df


def assign_week(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each shipment to its ISO week (Monday–Sunday). week_of = that Monday's date."""
    df = df.copy()
    # Offset to Monday: weekday 0=Mon, so subtract weekday days to get to Monday
    df["week_of"] = df["shipment_date"] - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")
    return df


def compute_weekly_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekly cost_per_tonne_km for each route.
    
    Aggregates all shipments in a route+week, then computes:
        cost_per_tonne_km = sum(freight_cost) / sum(quantity * distance)
    
    This is the weighted average across all shipments in that week,
    which correctly accounts for varying shipment sizes.
    """
    df = df.copy()
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]

    weekly = (
        df.groupby(["route", "route_type", "week_of"])
        .agg(
            total_cost=("freight_cost_inr", "sum"),
            total_tonne_km=("tonne_km", "sum"),
            shipment_count=("shipment_id", "count"),
        )
        .reset_index()
    )
    weekly["cost_per_tonne_km"] = weekly["total_cost"] / weekly["total_tonne_km"]
    weekly = weekly.sort_values(["route", "week_of"]).reset_index(drop=True)
    return weekly


def add_vs_own_history(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Add trailing 8-week rolling average (strictly prior weeks, no look-ahead).
    
    For each route+week, the baseline is the average cost_per_tonne_km
    over the preceding 8 weeks that have data. If fewer than 8 prior weeks
    exist, we use all available prior weeks.
    """
    weekly = weekly.sort_values(["route", "week_of"]).copy()
    
    history_avg = []
    history_weeks_used = []

    for route in weekly["route"].unique():
        mask = weekly["route"] == route
        route_data = weekly.loc[mask, "cost_per_tonne_km"].values
        n = len(route_data)
        
        for i in range(n):
            if i == 0:
                history_avg.append(np.nan)
                history_weeks_used.append(0)
            else:
                lookback = route_data[max(0, i - 8):i]
                history_avg.append(lookback.mean())
                history_weeks_used.append(len(lookback))

    weekly["history_avg"] = history_avg
    weekly["history_weeks_used"] = history_weeks_used

    # Percentage change vs own history
    weekly["pct_vs_history"] = np.where(
        weekly["history_avg"].notna() & (weekly["history_avg"] > 0),
        ((weekly["cost_per_tonne_km"] - weekly["history_avg"]) / weekly["history_avg"]) * 100,
        np.nan,
    )
    return weekly


def add_vs_similar_routes(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Add peer comparison: average cost_per_tonne_km across other routes
    of the same route_type in the same week, excluding the route itself.
    """
    weekly = weekly.copy()

    # For each route_type + week, compute the mean across all routes
    type_week = (
        weekly.groupby(["route_type", "week_of"])
        .agg(
            sum_cost_tonne_km=("cost_per_tonne_km", "sum"),
            count_routes=("cost_per_tonne_km", "count"),
        )
        .reset_index()
    )

    weekly = weekly.merge(type_week, on=["route_type", "week_of"], how="left")

    # Peer average = (total - this route's value) / (count - 1)
    weekly["peer_avg"] = np.where(
        weekly["count_routes"] > 1,
        (weekly["sum_cost_tonne_km"] - weekly["cost_per_tonne_km"]) / (weekly["count_routes"] - 1),
        np.nan,
    )

    weekly["pct_vs_peers"] = np.where(
        weekly["peer_avg"].notna() & (weekly["peer_avg"] > 0),
        ((weekly["cost_per_tonne_km"] - weekly["peer_avg"]) / weekly["peer_avg"]) * 100,
        np.nan,
    )

    weekly.drop(columns=["sum_cost_tonne_km", "count_routes"], inplace=True)
    return weekly


def flag_anomalies(weekly: pd.DataFrame, history_threshold: float = 15.0,
                   peer_threshold: float = 15.0) -> pd.DataFrame:
    """
    Flag rows where cost is rising abnormally:
    - Cost is above history_threshold% vs its own trailing average, OR
    - Cost is above peer_threshold% vs similar routes in the same week.
    
    Only flag when the comparison data exists (not NaN).
    """
    weekly = weekly.copy()

    above_history = weekly["pct_vs_history"].fillna(-999) > history_threshold
    above_peers = weekly["pct_vs_peers"].fillna(-999) > peer_threshold

    weekly["is_anomaly"] = above_history | above_peers
    return weekly


def build_vs_history_text(row: pd.Series) -> str:
    """Format the vs_own_history column text like the sample output."""
    if pd.isna(row["pct_vs_history"]):
        return "insufficient history"
    sign = "+" if row["pct_vs_history"] >= 0 else ""
    return f"{sign}{row['pct_vs_history']:.1f}% vs this route's past average"


def build_vs_peers_text(row: pd.Series) -> str:
    """Format the vs_similar_routes column text like the sample output."""
    if pd.isna(row["pct_vs_peers"]):
        return "no peer routes this week"
    sign = "+" if row["pct_vs_peers"] >= 0 else ""
    return f"{sign}{row['pct_vs_peers']:.1f}% vs similar-length routes this week"


def run_analytics(shipments_path: str, history_threshold: float = 15.0,
                  peer_threshold: float = 15.0) -> pd.DataFrame:
    """Full deterministic analytics pipeline. Returns the weekly DataFrame with flags."""
    df = load_shipments(shipments_path)
    df = assign_week(df)
    weekly = compute_weekly_cost(df)
    weekly = add_vs_own_history(weekly)
    weekly = add_vs_similar_routes(weekly)
    weekly = flag_anomalies(weekly, history_threshold, peer_threshold)

    # Build display text columns
    weekly["vs_own_history"] = weekly.apply(build_vs_history_text, axis=1)
    weekly["vs_similar_routes"] = weekly.apply(build_vs_peers_text, axis=1)

    return weekly
