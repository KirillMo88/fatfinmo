import numpy as np
import pandas as pd

from broker_margin_leverage import (
    apply_broker_margin_percentiles,
    broker_leverage_interpretation,
    build_broker_margin_leverage_history,
    historical_percentile,
    weekly_log_spx,
)


def _fred_frame(index: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Close": values}, index=pd.to_datetime(index))


def test_uses_trailing_12_quarter_market_cap_average_and_converts_units() -> None:
    quarters = pd.period_range("2023Q3", "2026Q2", freq="Q-DEC")
    dates = [str(quarter.start_time.date()) for quarter in quarters]
    market_caps = [(71_377_019.23 * 12 - 93_607_725) / 11] * 11 + [93_607_725]
    loans = _fred_frame(dates, [600_000 + idx * 12_938.273 for idx in range(12)])
    loans.iloc[-1, 0] = 742_321
    market_cap = _fred_frame(dates, market_caps)

    result = build_broker_margin_leverage_history(loans, market_cap)
    latest = result.iloc[-1]

    assert latest["Date"] == pd.Timestamp("2026-06-30")
    assert latest["MarginLoansBillions"] == 742.321
    assert latest["EquityMarketCapTrillions"] == 93.607725
    assert np.isclose(latest["EquityMarketCap3YAvgTrillions"], 71.37701923)
    assert np.isclose(latest["MarginToMarketCap3YAvgPct"], 1.04)


def test_quarterly_ratio_does_not_match_different_quarters_by_row_position() -> None:
    dates = [str(quarter.start_time.date()) for quarter in pd.period_range("2023Q1", "2028Q1", freq="Q-DEC")]
    loans = _fred_frame(dates, [500_000 + idx * 10_000 for idx in range(len(dates))])
    market_values = [50_000_000 + idx * 1_000_000 for idx in range(len(dates))]
    market_values[8] = np.nan
    market_cap = _fred_frame(dates, market_values)

    result = build_broker_margin_leverage_history(loans, market_cap)

    assert len(result) == len(dates)
    # A missing observation invalidates the 12-quarter SMA until it leaves the window.
    assert result.loc[result["Date"].between("2025-03-31", "2025-12-31"), "MarginToMarketCap3YAvgPct"].isna().all()
    assert np.isfinite(result.loc[result["Date"].eq(pd.Timestamp("2028-03-31")), "MarginToMarketCap3YAvgPct"].iloc[0])


def test_pre_1990_history_initializes_10_year_percentile_at_1990() -> None:
    quarters = pd.period_range("1977Q3", "1992Q4", freq="Q-DEC")
    dates = [str(quarter.start_time.date()) for quarter in quarters]
    market_cap = _fred_frame(dates, np.linspace(4_000_000, 12_000_000, len(dates)).tolist())
    loans = _fred_frame(dates, np.linspace(5_000, 35_000, len(dates)).tolist())

    history = apply_broker_margin_percentiles(build_broker_margin_leverage_history(loans, market_cap), "10Y")
    q1_1990 = history.loc[history["Date"].eq(pd.Timestamp("1990-03-31"))].iloc[0]

    assert np.isfinite(q1_1990["MarginToMarketCap3YAvgPctile"])


def test_rolling_percentile_requires_full_window_and_excludes_future() -> None:
    series = pd.Series([*range(39), 39, 999, 998], dtype="float64")
    percentile = historical_percentile(series, "10Y")

    assert percentile.iloc[:39].isna().all()
    assert percentile.iloc[39] == 100.0
    assert percentile.iloc[40] == 100.0


def test_full_history_percentile_expands_only_through_current_observation() -> None:
    series = pd.Series([1.0, 3.0, 2.0, 99.0])

    percentile = historical_percentile(series, "Full History")

    assert percentile.tolist() == [100.0, 100.0, 66.66666666666666, 100.0]


def test_percentile_selector_recomputes_all_three_indicators() -> None:
    loans = list(np.linspace(100_000, 200_000, 45))
    frame = pd.DataFrame(
        {
            "MarginLoansBillions": np.asarray(loans) / 1_000,
            "MarginLoansYoY": np.linspace(-10, 30, 45),
            "MarginToMarketCap3YAvgPct": np.linspace(0.5, 1.0, 45),
        }
    )

    result = apply_broker_margin_percentiles(frame, "10Y")

    assert result["MarginLoansLevelPct"].iloc[-1] == 100.0
    assert result["MarginLoansGrowthPct"].iloc[-1] == 100.0
    assert result["MarginToMarketCap3YAvgPctile"].iloc[-1] == 100.0


def test_interpretation_matches_prompt_example_and_missing_values() -> None:
    title, detail = broker_leverage_interpretation(92.5, 80.0)
    missing_title, _ = broker_leverage_interpretation(np.nan, 47.5)

    assert title == "Rapid Credit Expansion / Elevated Trend-Adjusted Leverage"
    assert "trailing 3-year average" in detail
    assert missing_title == "Insufficient Data"


def test_weekly_spx_uses_weekly_close_and_natural_log() -> None:
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    daily = pd.DataFrame({"SPX_Close": np.arange(100, 110, dtype="float64")}, index=dates)

    weekly = weekly_log_spx(daily)

    assert len(weekly) == 2
    assert np.isclose(weekly.iloc[0]["LogSPX"], np.log(101.0))
    assert np.isclose(weekly.iloc[1]["LogSPX"], np.log(108.0))
