import pandas as pd

from portfolio_positioning import (
    build_btc_positioning_history,
    build_spx_positioning_history,
    gold_positioning,
)


def test_spx_positioning_replays_risk_and_bottom_events_with_bottom_priority():
    dates = pd.date_range("2026-01-01", periods=7, freq="D")
    risk = pd.DataFrame({
        "Date": dates,
        "CurrentRiskEventMarker": [True, False, True, False, False, True, True],
        "CurrentRiskEventMarkerClass": [
            "RED FLAG", "INACTIVE", "RED FLAG + CREDIT CONFIRMATION",
            "INACTIVE", "INACTIVE", "RED FLAG + CREDIT CONFIRMATION", "RED FLAG",
        ],
    })
    correction = pd.DataFrame({
        "Date": dates,
        "ExtremeEpisodeDisplayMarker": [False, False, False, True, False, False, False],
        "ExtremeCapitulation2EpisodeMarker": [False, False, False, False, True, False, False],
        "DurableBottomConfirmed": [False, False, False, False, False, True, False],
    })
    history = build_spx_positioning_history(risk, correction)
    assert history["position_pct"].tolist() == [50, 50, 0, 33, 66, 100, 50]
    assert history.loc[5, "trigger"] == "DURABLE BOTTOM CONFIRMED"


def test_spx_level_two_is_a_66_percent_minimum_from_zero():
    date = pd.Timestamp("2026-01-01")
    risk = pd.DataFrame({
        "Date": [date, date + pd.Timedelta(days=1)],
        "CurrentRiskEventMarker": [True, False],
        "CurrentRiskEventMarkerClass": ["RED FLAG + CREDIT CONFIRMATION", "INACTIVE"],
    })
    correction = pd.DataFrame({
        "Date": [date, date + pd.Timedelta(days=1)],
        "ExtremeCapitulation2EpisodeMarker": [False, True],
    })
    history = build_spx_positioning_history(risk, correction)
    assert history["position_pct"].tolist() == [0, 66]


def test_gold_positioning_displays_selected_metal():
    assert gold_positioning("CASH")["position_pct"] == 0
    assert gold_positioning("GOLD") == {
        "position_pct": 100, "instrument": "Gold", "regime": "GOLD", "status": "READY",
    }
    assert gold_positioning("SILVER")["instrument"] == "Silver"
    assert gold_positioning("SILVER")["position_pct"] == 100


def test_btc_position_is_reduced_in_defensive_phases_and_zero_only_in_gold():
    history = pd.DataFrame({
        "Date": pd.date_range("2026-01-02", periods=10, freq="W-FRI"),
        "position": [None, "Gold", "BTC", "BTC", "BTC", "BTC", "BTC", "BTC", "BTC", "Gold"],
        "halving_phase": [
            "EARLY_EXPANSION",
            "EARLY_EXPANSION",
            "BOTTOMING_TRANSITION",
            "BOTTOMING_TRANSITION",
            "ACCUMULATION_PRE_HALVING",
            "ACCUMULATION_PRE_HALVING",
            "BEAR_DELEVERAGING",
            "BEAR_DELEVERAGING",
            "LATE_EXPANSION",
            "LATE_EXPANSION",
        ],
    })
    result = build_btc_positioning_history(history)
    assert result["position_pct"].tolist() == [0, 0, 75, 75, 100, 100, 75, 75, 100, 0]
    assert result.loc[2, "trigger"] == "BTC / GOLD: GOLD → BTC"
    assert result.loc[3, "trigger"] == ""
    assert result.loc[4, "trigger"] == "HALVING PHASE: ACCUMULATION_PRE_HALVING"


def test_btc_entry_in_allowed_phase_starts_at_100():
    history = pd.DataFrame({
        "Date": pd.date_range("2026-01-02", periods=3, freq="W-FRI"),
        "position": ["Gold", "BTC", "BTC"],
        "halving_phase": ["LATE_EXPANSION", "LATE_EXPANSION", "CYCLE_TOP_RISK"],
    })
    result = build_btc_positioning_history(history)
    assert result["position_pct"].tolist() == [0, 100, 100]

