"""
Tests for the analytics module.

Covers: weekly grouping, cost-per-tonne-km calculation, rolling history,
peer comparison, and anomaly flagging.
"""

import pandas as pd
import numpy as np
import pytest
from analytics import (
    assign_week, compute_weekly_cost, add_vs_own_history,
    add_vs_similar_routes, flag_anomalies, load_shipments,
)


def make_shipment(shipment_id, origin, destination, route_type,
                  quantity, distance, cost, date):
    """Helper to create a single shipment row."""
    return {
        "shipment_id": shipment_id,
        "origin": origin,
        "destination": destination,
        "route_type": route_type,
        "material": "Steel Coils",
        "quantity_tonnes": quantity,
        "distance_km": distance,
        "freight_cost_inr": cost,
        "shipment_date": pd.Timestamp(date),
        "transporter": "Test Transport",
    }


class TestAssignWeek:
    def test_monday_stays_monday(self):
        df = pd.DataFrame([make_shipment("S1", "A", "B", "Short", 10, 100, 1000, "2024-01-01")])
        df["route"] = df["origin"] + "-" + df["destination"]
        result = assign_week(df)
        # 2024-01-01 is a Monday
        assert result.iloc[0]["week_of"] == pd.Timestamp("2024-01-01")

    def test_sunday_maps_to_prior_monday(self):
        df = pd.DataFrame([make_shipment("S1", "A", "B", "Short", 10, 100, 1000, "2024-01-07")])
        df["route"] = df["origin"] + "-" + df["destination"]
        result = assign_week(df)
        # 2024-01-07 is a Sunday -> week_of should be 2024-01-01 (Monday)
        assert result.iloc[0]["week_of"] == pd.Timestamp("2024-01-01")

    def test_wednesday_maps_to_monday(self):
        df = pd.DataFrame([make_shipment("S1", "A", "B", "Short", 10, 100, 1000, "2024-01-03")])
        df["route"] = df["origin"] + "-" + df["destination"]
        result = assign_week(df)
        # 2024-01-03 is Wednesday -> week_of should be 2024-01-01
        assert result.iloc[0]["week_of"] == pd.Timestamp("2024-01-01")


class TestComputeWeeklyCost:
    def test_single_shipment(self):
        df = pd.DataFrame([make_shipment("S1", "A", "B", "Short", 10, 100, 5000, "2024-01-01")])
        df["route"] = df["origin"] + "-" + df["destination"]
        df = assign_week(df)
        result = compute_weekly_cost(df)
        # cost_per_tonne_km = 5000 / (10 * 100) = 5.0
        assert result.iloc[0]["cost_per_tonne_km"] == pytest.approx(5.0)

    def test_two_shipments_same_week(self):
        """Two shipments in the same week should be aggregated as a weighted average."""
        df = pd.DataFrame([
            make_shipment("S1", "A", "B", "Short", 10, 100, 5000, "2024-01-01"),
            make_shipment("S2", "A", "B", "Short", 20, 100, 8000, "2024-01-03"),
        ])
        df["route"] = df["origin"] + "-" + df["destination"]
        df = assign_week(df)
        result = compute_weekly_cost(df)
        # total_cost = 13000, total_tonne_km = 10*100 + 20*100 = 3000
        # cost_per_tonne_km = 13000 / 3000 = 4.333...
        assert len(result) == 1
        assert result.iloc[0]["cost_per_tonne_km"] == pytest.approx(13000 / 3000)

    def test_two_different_weeks(self):
        df = pd.DataFrame([
            make_shipment("S1", "A", "B", "Short", 10, 100, 5000, "2024-01-01"),
            make_shipment("S2", "A", "B", "Short", 10, 100, 5000, "2024-01-08"),
        ])
        df["route"] = df["origin"] + "-" + df["destination"]
        df = assign_week(df)
        result = compute_weekly_cost(df)
        assert len(result) == 2


class TestVsOwnHistory:
    def test_first_week_has_no_history(self):
        df = pd.DataFrame([
            make_shipment("S1", "A", "B", "Short", 10, 100, 5000, "2024-01-01"),
        ])
        df["route"] = df["origin"] + "-" + df["destination"]
        df = assign_week(df)
        weekly = compute_weekly_cost(df)
        result = add_vs_own_history(weekly)
        assert pd.isna(result.iloc[0]["pct_vs_history"])

    def test_trailing_average_uses_prior_weeks(self):
        """With 3 weeks of data, week 3's history should average weeks 1 and 2."""
        rows = []
        for i, (date, cost) in enumerate([
            ("2024-01-01", 1000),  # cost/tkm = 1000/(10*100) = 1.0
            ("2024-01-08", 1500),  # cost/tkm = 1.5
            ("2024-01-15", 2000),  # cost/tkm = 2.0
        ]):
            rows.append(make_shipment(f"S{i}", "A", "B", "Short", 10, 100, cost, date))
        
        df = pd.DataFrame(rows)
        df["route"] = df["origin"] + "-" + df["destination"]
        df = assign_week(df)
        weekly = compute_weekly_cost(df)
        result = add_vs_own_history(weekly)
        
        # Week 3 history = avg(1.0, 1.5) = 1.25
        assert result.iloc[2]["history_avg"] == pytest.approx(1.25)
        # pct change = (2.0 - 1.25) / 1.25 * 100 = 60%
        assert result.iloc[2]["pct_vs_history"] == pytest.approx(60.0)

    def test_rolling_window_limited_to_8_weeks(self):
        """With 10 weeks of data, week 10's history should only use weeks 2-9 (8 weeks)."""
        rows = []
        for i in range(10):
            date = f"2024-{1 + (i * 7) // 28:02d}-{1 + (i * 7) % 28:02d}"
            cost = 1000 + i * 100  # Increasing cost
            rows.append(make_shipment(f"S{i}", "A", "B", "Short", 10, 100, cost, date))
        
        df = pd.DataFrame(rows)
        df["route"] = df["origin"] + "-" + df["destination"]
        df = assign_week(df)
        weekly = compute_weekly_cost(df)
        result = add_vs_own_history(weekly)
        
        # Week 10 (index 9) should use 8 prior weeks (indices 1-8)
        assert result.iloc[9]["history_weeks_used"] == 8

    def test_no_look_ahead(self):
        """Week 2's history should only use week 1, not week 3."""
        rows = []
        for i, (date, cost) in enumerate([
            ("2024-01-01", 1000),
            ("2024-01-08", 1500),
            ("2024-01-15", 9999),  # Future data, should not affect week 2
        ]):
            rows.append(make_shipment(f"S{i}", "A", "B", "Short", 10, 100, cost, date))
        
        df = pd.DataFrame(rows)
        df["route"] = df["origin"] + "-" + df["destination"]
        df = assign_week(df)
        weekly = compute_weekly_cost(df)
        result = add_vs_own_history(weekly)
        
        # Week 2 history = just week 1 = 1.0
        assert result.iloc[1]["history_avg"] == pytest.approx(1.0)


class TestVsSimilarRoutes:
    def test_peer_average_excludes_self(self):
        """Two Short routes in same week: each peer avg should be the other's cost."""
        df = pd.DataFrame([
            make_shipment("S1", "A", "B", "Short", 10, 100, 2000, "2024-01-01"),
            make_shipment("S2", "C", "D", "Short", 10, 100, 3000, "2024-01-02"),
        ])
        df["route"] = df["origin"] + "-" + df["destination"]
        df = assign_week(df)
        weekly = compute_weekly_cost(df)
        result = add_vs_similar_routes(weekly)
        
        # A-B peer avg should be C-D's cost = 3.0
        ab_row = result[result["route"] == "A-B"].iloc[0]
        assert ab_row["peer_avg"] == pytest.approx(3.0)
        
        # C-D peer avg should be A-B's cost = 2.0
        cd_row = result[result["route"] == "C-D"].iloc[0]
        assert cd_row["peer_avg"] == pytest.approx(2.0)

    def test_different_route_types_not_compared(self):
        """A Short and a Long route should not be peers."""
        df = pd.DataFrame([
            make_shipment("S1", "A", "B", "Short", 10, 100, 2000, "2024-01-01"),
            make_shipment("S2", "C", "D", "Long", 10, 100, 3000, "2024-01-02"),
        ])
        df["route"] = df["origin"] + "-" + df["destination"]
        df = assign_week(df)
        weekly = compute_weekly_cost(df)
        result = add_vs_similar_routes(weekly)
        
        # Each is the only route of its type, so peer_avg should be NaN
        assert pd.isna(result.iloc[0]["peer_avg"])
        assert pd.isna(result.iloc[1]["peer_avg"])


class TestFlagAnomalies:
    def test_flags_high_vs_history(self):
        weekly = pd.DataFrame([{
            "route": "A-B", "route_type": "Short",
            "pct_vs_history": 20.0, "pct_vs_peers": 5.0,
        }])
        result = flag_anomalies(weekly, history_threshold=15.0, peer_threshold=15.0)
        assert result.iloc[0]["is_anomaly"] == True

    def test_flags_high_vs_peers(self):
        weekly = pd.DataFrame([{
            "route": "A-B", "route_type": "Short",
            "pct_vs_history": 5.0, "pct_vs_peers": 20.0,
        }])
        result = flag_anomalies(weekly, history_threshold=15.0, peer_threshold=15.0)
        assert result.iloc[0]["is_anomaly"] == True

    def test_no_flag_when_normal(self):
        weekly = pd.DataFrame([{
            "route": "A-B", "route_type": "Short",
            "pct_vs_history": 5.0, "pct_vs_peers": 5.0,
        }])
        result = flag_anomalies(weekly, history_threshold=15.0, peer_threshold=15.0)
        assert result.iloc[0]["is_anomaly"] == False

    def test_no_flag_on_nan(self):
        """NaN comparisons should not trigger a flag."""
        weekly = pd.DataFrame([{
            "route": "A-B", "route_type": "Short",
            "pct_vs_history": np.nan, "pct_vs_peers": np.nan,
        }])
        result = flag_anomalies(weekly, history_threshold=15.0, peer_threshold=15.0)
        assert result.iloc[0]["is_anomaly"] == False


class TestLoadShipments:
    def test_loads_real_data(self):
        df = load_shipments("data/shipment_records.csv")
        assert len(df) == 2940
        assert "route" in df.columns
        assert set(df["route_type"].unique()) == {"Short", "Medium", "Long"}


class TestSampleOutputCompliance:
    """Verifies that the analytics pipeline strictly reproduces the assignment's official sample rows."""

    @pytest.fixture(scope="module")
    def pipeline_output(self):
        from analytics import run_analytics
        from note_matcher import load_notes, NoteMatcherDeterministic
        from main import process_anomalies, build_output

        weekly = run_analytics("data/shipment_records.csv")
        notes = load_notes("data/context_notes.csv")
        matcher = NoteMatcherDeterministic()
        anomalies = process_anomalies(weekly, notes, matcher)
        output = build_output(weekly, anomalies)
        return output

    def test_delhi_jaipur_ground_truth(self, pipeline_output):
        row = pipeline_output[
            (pipeline_output["route"] == "Delhi-Jaipur") &
            (pipeline_output["week_of"] == "2024-11-11")
        ]
        assert len(row) == 1, "Delhi-Jaipur 2024-11-11 must be flagged"
        r = row.iloc[0]
        assert r["cost_per_tonne_km"] == 4.17
        assert r["vs_own_history"] == "+35.5% vs this route's past average"
        assert r["vs_similar_routes"] == "+21.0% vs similar-length routes this week"
        assert r["flagged"] == "Yes"

    def test_ahmedabad_mumbai_ground_truth(self, pipeline_output):
        row = pipeline_output[
            (pipeline_output["route"] == "Ahmedabad-Mumbai") &
            (pipeline_output["week_of"] == "2025-01-20")
        ]
        assert len(row) == 1, "Ahmedabad-Mumbai 2025-01-20 must be flagged"
        r = row.iloc[0]
        assert r["cost_per_tonne_km"] == 3.29
        assert r["vs_own_history"] == "+29.5% vs this route's past average"
        assert r["vs_similar_routes"] == "+22.5% vs similar-length routes this week"
        assert r["flagged"] == "No (justified)"
        assert r["matched_note_id"] == "N002"

    def test_mumbai_pune_ground_truth(self, pipeline_output):
        row = pipeline_output[
            (pipeline_output["route"] == "Mumbai-Pune") &
            (pipeline_output["week_of"] == "2025-09-15")
        ]
        assert len(row) == 1, "Mumbai-Pune 2025-09-15 must be flagged"
        r = row.iloc[0]
        assert r["cost_per_tonne_km"] == 3.98
        assert r["vs_own_history"] == "+9.2% vs this route's past average"
        assert r["vs_similar_routes"] == "+23.6% vs similar-length routes this week"
        assert r["flagged"] == "Yes"

