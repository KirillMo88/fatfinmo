from __future__ import annotations

import pandas as pd

from hy_oas import combine_hy_oas_sources, weekly_archive_available_frame


def test_weekly_archive_is_available_only_after_period_completion() -> None:
    raw = pd.DataFrame({"time": ["2021-12-20"], "close": [3.12]})

    result = weekly_archive_available_frame(raw)

    assert result.loc[0, "Date"] == pd.Timestamp("2021-12-27")
    assert result.loc[0, "HY_OAS"] == 3.12
    assert result.loc[0, "HYOASSourceFrequency"] == "WEEKLY_ARCHIVE"


def test_tradingview_daily_has_priority_over_fred_and_weekly() -> None:
    weekly = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2021-12-20", "2021-12-27"]),
            "HY_OAS": [3.12, 3.10],
            "HYOASSourceFrequency": "WEEKLY_ARCHIVE",
        }
    )
    tradingview = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2021-12-20", "2021-12-27"]),
            "HY_OAS": [3.41, 3.22],
            "HYOASSourceFrequency": "DAILY_TRADINGVIEW",
        }
    )
    fred = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2021-12-20", "2021-12-27"]),
            "HY_OAS": [3.40, 3.21],
            "HYOASSourceFrequency": "DAILY_FRED",
        }
    )

    result = combine_hy_oas_sources(weekly, tradingview, fred)

    assert result["HY_OAS"].tolist() == [3.41, 3.22]
    assert result["HYOASSourceFrequency"].tolist() == ["DAILY_TRADINGVIEW", "DAILY_TRADINGVIEW"]


def test_weekly_archive_is_used_only_when_daily_observation_is_missing() -> None:
    weekly = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2021-12-27"]),
            "HY_OAS": [3.12],
            "HYOASSourceFrequency": "WEEKLY_ARCHIVE",
        }
    )
    tradingview = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2021-12-20"]),
            "HY_OAS": [3.41],
            "HYOASSourceFrequency": "DAILY_TRADINGVIEW",
        }
    )

    result = combine_hy_oas_sources(weekly, tradingview, pd.DataFrame())

    assert result.loc[result["Date"].eq(pd.Timestamp("2021-12-20")), "HY_OAS"].iloc[0] == 3.41
    fallback = result.loc[result["Date"].eq(pd.Timestamp("2021-12-27"))].iloc[0]
    assert fallback["HY_OAS"] == 3.12
    assert fallback["HYOASSourceFrequency"] == "WEEKLY_ARCHIVE"


def test_future_daily_observations_do_not_change_prior_source_selection() -> None:
    weekly = weekly_archive_available_frame(
        pd.DataFrame({"time": ["2021-12-13", "2021-12-20"], "close": [3.36, 3.12]})
    )
    tradingview = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2021-12-20", "2021-12-21"]),
            "HY_OAS": [3.41, 3.25],
            "HYOASSourceFrequency": "DAILY_TRADINGVIEW",
        }
    )

    short = combine_hy_oas_sources(weekly, tradingview.iloc[:1], pd.DataFrame(), end_date="2021-12-20")
    full = combine_hy_oas_sources(weekly, tradingview, pd.DataFrame(), end_date="2021-12-20")

    pd.testing.assert_frame_equal(short, full)
