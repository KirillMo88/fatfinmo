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


def test_aligns_fred_series_by_quarter_and_converts_units() -> None:
    loans = _fred_frame(["2026-04-01", "2026-01-01"], [742_321, 622_199])
    market_cap = _fred_frame(["2026-01-01", "2026-04-01"], [80_266_112, 93_607_725])

    result = build_broker_margin_leverage_history(loans, market_cap)
    latest = result.iloc[-1]

    assert latest["Date"] == pd.Timestamp("2026-06-30")
    assert latest["MarginLoansBillions"] == 742.321
    assert latest["EquityMarketCapTrillions"] == 93.607725
    assert np.isclose(latest["MarginToMarketCapPct"], 100 * 742_321 / 93_607_725)


def test_quarterly_ratio_does_not_match_different_quarters_by_row_position() -> None:
    loans = _fred_frame(["2026-01-01", "2026-04-01"], [500_000, 700_000])
    market_cap = _fred_frame(["2026-01-01", "2026-07-01"], [50_000_000, 70_000_000])

    result = build_broker_margin_leverage_history(loans, market_cap)

    assert len(result) == 3
    assert result.loc[result["Date"].eq(pd.Timestamp("2026-06-30")), "MarginToMarketCapPct"].isna().all()
    assert np.isfinite(result.loc[result["Date"].eq(pd.Timestamp("2026-03-31")), "MarginToMarketCapPct"].iloc[0])


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
            "MarginToMarketCapPct": np.linspace(0.5, 1.0, 45),
        }
    )

    result = apply_broker_margin_percentiles(frame, "10Y")

    assert result["MarginLoansLevelPct"].iloc[-1] == 100.0
    assert result["MarginLoansGrowthPct"].iloc[-1] == 100.0
    assert result["MarginToMarketCapPctile"].iloc[-1] == 100.0


def test_interpretation_matches_prompt_example_and_missing_values() -> None:
    title, detail = broker_leverage_interpretation(92.5, 47.5)
    missing_title, _ = broker_leverage_interpretation(np.nan, 47.5)

    assert title == "Rapid Credit Expansion / Normal Leverage"
    assert "historically elevated rate" in detail
    assert missing_title == "Insufficient Data"


def test_weekly_spx_uses_weekly_close_and_natural_log() -> None:
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    daily = pd.DataFrame({"SPX_Close": np.arange(100, 110, dtype="float64")}, index=dates)

    weekly = weekly_log_spx(daily)

    assert len(weekly) == 2
    assert np.isclose(weekly.iloc[0]["LogSPX"], np.log(101.0))
    assert np.isclose(weekly.iloc[1]["LogSPX"], np.log(108.0))
