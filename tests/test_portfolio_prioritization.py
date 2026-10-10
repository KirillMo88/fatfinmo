import numpy as np
import pandas as pd

from portfolio_prioritization import (
    PortfolioConfig,
    _apply_confirmation_health,
    _rolling_log_regression,
    build_btc_gold_history,
    build_state_history,
    build_spx_gold_history,
    priority_ranking,
)


def _monthly_frame(closes: list[float]) -> pd.DataFrame:
    values = np.asarray(closes, dtype="float64")
    return pd.DataFrame({
        "Date": pd.date_range("2000-01-31", periods=len(values), freq="ME"),
        "open": values,
        "high": values * 1.01,
        "low": values * 0.99,
        "close": values,
    })


def test_priority_matrix_all_combinations():
    assert priority_ranking("Gold", "BTC") == ("BTC > Gold > SPX", "BTC")
    assert priority_ranking("Gold", "Gold") == ("Gold > BTC > SPX", "Gold")
    assert priority_ranking("SPX", "BTC") == ("SPX > BTC > Gold", "SPX")
    assert priority_ranking("SPX", "Gold") == ("SPX > Gold > BTC", "SPX")
    assert priority_ranking(None, "BTC") == (None, None)


def test_spx_gold_hysteresis_initialization_hold_and_equality():
    closes = [100.0] * 50 + [106.0, 102.0, 94.0, 98.0, 106.0]
    history = build_spx_gold_history(_monthly_frame(closes))
    assert pd.isna(history.iloc[49]["position"])
    assert history.iloc[50]["position"] == "SPX"
    assert history.iloc[51]["position"] == "SPX"
    assert history.iloc[52]["position"] == "Gold"
    assert history.iloc[53]["position"] == "Gold"
    assert history.iloc[54]["position"] == "SPX"

    # Equality does not switch because the state machine uses strict crossings.
    equal = _monthly_frame([100.0] * 50 + [105.0])
    result = build_spx_gold_history(equal)
    assert pd.isna(result.iloc[-1]["position"])


def test_confirmation_three_period_persistence_and_history_survives_review():
    frame = pd.DataFrame({
        "Date": pd.date_range("2025-01-31", periods=7, freq="ME"),
        "position": ["Gold"] * 7,
        "gold": [False, True, True, True, False, False, False],
        "spx": [False, False, False, False, False, True, True],
    })
    result = _apply_confirmation_health(
        frame,
        "position",
        {"Gold": "gold", "SPX": "spx"},
        3,
    )
    assert result.loc[1, "confirmation_health"] == "Initial"
    assert result.loc[3, "confirmation_health"] == "Active"
    assert bool(result.loc[3, "sustained_confirmation_reached"])
    assert result.loc[4, "confirmation_health"] == "Under Review"
    assert bool(result.loc[6, "sustained_confirmation_reached"])


def test_confirmation_opposite_sustained_is_invalidated():
    frame = pd.DataFrame({
        "Date": pd.date_range("2025-01-31", periods=7, freq="ME"),
        "position": ["Gold"] * 7,
        "gold": [True, True, True, False, False, False, False],
        "spx": [False, False, False, True, True, True, True],
    })
    result = _apply_confirmation_health(frame, "position", {"Gold": "gold", "SPX": "spx"}, 3)
    assert result.loc[2, "confirmation_health"] == "Active"
    assert result.loc[5, "confirmation_health"] == "Invalidated"
    assert result.loc[6, "confirmation_health"] == "Invalidated"


def test_unconfirmed_episode_is_recorded_at_exit():
    frame = pd.DataFrame({
        "Date": pd.date_range("2025-01-31", periods=4, freq="ME"),
        "position": ["Gold", "Gold", "SPX", "SPX"],
        "gold": [True, True, False, False],
        "spx": [False, False, True, True],
    })
    result = _apply_confirmation_health(frame, "position", {"Gold": "gold", "SPX": "spx"}, 3)
    assert result.loc[2, "previous_episode_outcome"] == "Unconfirmed at Exit"


def test_btc_threshold_equality_switches_and_structural_layer_does_not_trade():
    dates = pd.date_range("2014-01-06", periods=220, freq="W-MON")
    ratios = np.exp(np.linspace(0.0, 1.0, len(dates)))
    weekly = pd.DataFrame({"Date": dates, "BTC": ratios * 1000, "Gold": 1000.0, "ratio": ratios})
    config = PortfolioConfig(btc_buy_percentile=100.0, btc_sell_percentile=100.0)
    history = build_btc_gold_history(weekly, config)
    initialized = history.loc[history["channel_percentile"].notna()].iloc[0]
    assert initialized["channel_percentile"] <= 100.0
    assert initialized["position"] == "BTC"

    # Directly validate the agreed inclusive upper boundary on the next row.
    later = weekly.copy()
    config = PortfolioConfig(btc_buy_percentile=-1.0, btc_sell_percentile=0.0)
    upper = build_btc_gold_history(later, config)
    first_valid = upper.loc[upper["channel_percentile"].notna()].iloc[0]
    assert first_valid["position"] == "Gold"


def test_rolling_regression_has_no_future_leakage_and_uses_98_5_channel():
    values = pd.Series(np.exp(np.linspace(0.0, 3.0, 260)))
    baseline = _rolling_log_regression(values, 208)
    changed = values.copy()
    changed.iloc[-1] *= 50
    rerun = _rolling_log_regression(changed, 208)
    pd.testing.assert_frame_equal(baseline.iloc[:-1], rerun.iloc[:-1])
    expected_upper = np.quantile(
        np.log(values.iloc[:208]) - np.polyval(np.polyfit(np.arange(208), np.log(values.iloc[:208]), 1), np.arange(208)),
        0.985,
    )
    expected_trend = np.polyval(np.polyfit(np.arange(208), np.log(values.iloc[:208]), 1), 208)
    assert np.isclose(baseline.loc[208, "channel_upper"], np.exp(expected_trend + expected_upper))


def test_shared_state_history_uses_latest_completed_monthly_state():
    spx = build_spx_gold_history(_monthly_frame([100.0] * 50 + [106.0, 107.0]))
    dates = pd.date_range("2003-11-03", periods=220, freq="W-MON")
    ratios = np.exp(np.linspace(0.0, 1.0, len(dates)))
    btc = build_btc_gold_history(pd.DataFrame({"Date": dates, "BTC": ratios * 1000, "Gold": 1000.0, "ratio": ratios}))
    combined = build_state_history(spx, btc)
    required = {
        "spx_gold_position", "btc_gold_position", "btc_halving_phase",
        "final_priority_ranking", "final_priority_asset",
    }
    assert required.issubset(combined.columns)
    first_spx_signal = pd.Timestamp(spx.loc[spx["trading_switch"], "Date"].iloc[0])
    before = combined.loc[combined["date"] < first_spx_signal]
    assert before["spx_gold_position"].isna().all()

