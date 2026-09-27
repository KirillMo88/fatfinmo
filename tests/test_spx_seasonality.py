from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from market_cycle_tab import build_spx_annual_seasonality_fig, build_spx_monthly_seasonality_fig
from spx_seasonality import calculate_spx_seasonality, monthly_closes_from_daily, weekly_closes_from_daily
from spy_macro_outlook import build_spy_macro_workbook


def seasonality_inputs(end: str = "2026-09-25") -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly_dates = pd.date_range("2010-01-01", end, freq="W-FRI")
    final_date = pd.Timestamp(end)
    if final_date.month == 12 and final_date not in weekly_dates:
        weekly_dates = weekly_dates.append(pd.DatetimeIndex([final_date])).sort_values()
    index = np.arange(len(weekly_dates), dtype=float)
    weekly_close = 100.0 * np.cumprod(1.001 + 0.006 * np.sin(index * 2.0 * np.pi / 52.0))
    weekly = pd.DataFrame({"Date": weekly_dates, "SPX_Close": weekly_close})

    monthly_end = pd.Timestamp(end).to_period("M").end_time.normalize()
    monthly_dates = pd.date_range("2009-12-31", monthly_end, freq="ME")
    monthly_index = np.arange(len(monthly_dates), dtype=float)
    monthly_close = 100.0 * np.cumprod(1.01 + 0.04 * np.sin(monthly_index * 2.0 * np.pi / 12.0))
    monthly = pd.DataFrame({"Date": monthly_dates, "SPX_Close": monthly_close})
    return weekly, monthly


def test_current_partial_year_is_excluded_from_all_historical_models() -> None:
    weekly, monthly = seasonality_inputs()
    result = calculate_spx_seasonality(weekly, monthly, "2026-09-25")

    assert result.metadata["Historical Start Year"] == 2010
    assert result.metadata["Historical End Year"] == 2025
    assert result.metadata["Number of Historical Years"] == 16
    assert result.metadata["Current Year"] == 2026
    assert result.current_year_actual["Date"].dt.year.eq(2026).all()
    assert not result.current_year_actual.empty
    assert set(result.monthly_statistics["Observation Count"]) == {16}


def test_current_year_price_changes_do_not_change_historical_model() -> None:
    weekly, monthly = seasonality_inputs()
    baseline = calculate_spx_seasonality(weekly, monthly, "2026-09-25")
    changed = weekly.copy()
    changed.loc[changed["Date"].dt.year.eq(2026), "SPX_Close"] *= 4.0
    revised = calculate_spx_seasonality(changed, monthly, "2026-09-25")

    pd.testing.assert_frame_equal(baseline.weekly_model, revised.weekly_model)
    pd.testing.assert_frame_equal(baseline.monthly_statistics, revised.monthly_statistics)


def test_model_uses_compounded_median_weekly_returns_not_median_levels() -> None:
    weekly, monthly = seasonality_inputs()
    result = calculate_spx_seasonality(weekly, monthly, "2026-09-25")
    model = result.weekly_model

    assert model.loc[0, "Median Cycle"] == 100.0
    expected_second_value = 100.0 * (1.0 + model.loc[1, "Median Weekly Return"])
    assert np.isclose(model.loc[1, "Median Cycle"], expected_second_value)
    expected_final_value = 100.0 * np.prod(1.0 + model.loc[1:, "Median Weekly Return"].to_numpy())
    assert np.isclose(model["Median Cycle"].iloc[-1], expected_final_value)


def test_monthly_returns_are_month_end_to_month_end_and_include_january_2010() -> None:
    weekly, monthly = seasonality_inputs()
    for year in range(2010, 2026):
        dec_index = monthly.index[monthly["Date"].eq(pd.Timestamp(year=year - 1, month=12, day=31))][0]
        jan_index = monthly.index[monthly["Date"].eq(pd.Timestamp(year=year, month=1, day=31))][0]
        monthly.loc[dec_index, "SPX_Close"] = 100.0
        monthly.loc[jan_index, "SPX_Close"] = 110.0

    result = calculate_spx_seasonality(weekly, monthly, "2026-09-25")

    jan = result.monthly_statistics.set_index("Month").loc["Jan"]
    assert np.isclose(jan["Average Monthly Return"], 0.10)
    assert jan["Observation Count"] == 16


def test_missing_month_is_not_filled_and_december_rolls_into_completed_sample_only_when_final_data_exists() -> None:
    weekly, monthly = seasonality_inputs("2026-12-31")
    missing_month = monthly["Date"].eq(pd.Timestamp("2012-05-31"))
    incomplete = calculate_spx_seasonality(weekly, monthly.loc[~missing_month], "2026-12-30")
    completed = calculate_spx_seasonality(weekly, monthly, "2026-12-31")

    assert incomplete.metadata["Historical End Year"] == 2025
    assert incomplete.monthly_statistics.set_index("Month").loc["May", "Observation Count"] < 16
    assert completed.metadata["Historical End Year"] == 2026
    assert completed.metadata["Number of Historical Years"] == 17


def test_completed_december_calendar_year_is_detected_on_last_market_day() -> None:
    weekly, monthly = seasonality_inputs("2021-12-30")

    result = calculate_spx_seasonality(weekly, monthly, "2021-12-30")

    assert result.metadata["Historical End Year"] == 2021
    assert result.metadata["Number of Historical Years"] == 12


def test_daily_weekly_aggregation_keeps_final_close_in_its_calendar_year() -> None:
    daily = pd.DataFrame(
        {"SPX_Close": [100.0, 101.0, 102.0, 103.0]},
        index=pd.to_datetime(["2026-12-28", "2026-12-29", "2026-12-30", "2026-12-31"]),
    )

    weekly = weekly_closes_from_daily(daily)

    assert weekly.iloc[-1]["Date"] == pd.Timestamp("2026-12-31")
    assert weekly.iloc[-1]["SPX_Close"] == 103.0


def test_daily_weekly_aggregation_omits_an_incomplete_current_week() -> None:
    daily = pd.DataFrame(
        {"SPX_Close": [100.0, 101.0, 102.0, 103.0]},
        index=pd.to_datetime(["2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23"]),
    )

    weekly = weekly_closes_from_daily(daily)

    assert weekly.iloc[-1]["Date"] == pd.Timestamp("2026-09-18")


def test_year_end_close_is_kept_when_new_year_holiday_closes_december_31() -> None:
    daily = pd.DataFrame(
        {"SPX_Close": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2021-12-28", "2021-12-29", "2021-12-30"]),
    )

    weekly = weekly_closes_from_daily(daily)

    assert weekly.iloc[-1]["Date"] == pd.Timestamp("2021-12-30")


def test_monthly_closes_use_last_daily_close_and_calendar_month_labels() -> None:
    daily = pd.DataFrame(
        {"SPX_Close": [100.0, 101.0]},
        index=pd.to_datetime(["2021-12-29", "2021-12-30"]),
    )

    monthly = monthly_closes_from_daily(daily)

    assert monthly.iloc[0]["Date"] == pd.Timestamp("2021-12-31")
    assert monthly.iloc[0]["SPX_Close"] == 101.0


def test_next_year_observation_rolls_forward_automatically() -> None:
    weekly, monthly = seasonality_inputs("2027-01-08")
    result = calculate_spx_seasonality(weekly, monthly, "2027-01-08")

    assert result.metadata["Historical End Year"] == 2026
    assert result.metadata["Current Year"] == 2027
    assert result.monthly_statistics["Observation Count"].eq(17).all()


def test_market_cycle_workbook_contains_seasonality_tables() -> None:
    weekly, monthly = seasonality_inputs()
    result = calculate_spx_seasonality(weekly, monthly, "2026-09-25")
    payload = build_spy_macro_workbook(
        pd.DataFrame({"Date": [pd.Timestamp("2026-09-25")], "SPX_Close": [1.0]}),
        None,
        seasonality_weekly=result.weekly_model,
        seasonality_monthly=result.monthly_statistics,
    )
    workbook = load_workbook(BytesIO(payload), read_only=True)

    assert workbook.sheetnames == ["Market Cycle", "SPX Seasonality Weekly", "SPX Seasonality Monthly"]
    assert "Current Year Actual" in {cell.value for cell in workbook["SPX Seasonality Weekly"][1]}
    assert "Observation Count" in {cell.value for cell in workbook["SPX Seasonality Monthly"][1]}


def test_seasonality_charts_show_model_calendar_and_all_three_views() -> None:
    weekly, monthly = seasonality_inputs()
    result = calculate_spx_seasonality(weekly, monthly, "2026-09-25")

    annual_fig = build_spx_annual_seasonality_fig(result)
    assert [trace.name for trace in annual_fig.data] == [
        "P25",
        "25–75% Historical Range",
        "Mean Seasonal Cycle",
        "Median Seasonal Cycle",
        "2026 Actual",
    ]
    assert list(annual_fig.layout.xaxis.ticktext) == ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    assert len(build_spx_monthly_seasonality_fig(result.monthly_statistics, "Median Monthly Performance", "Median Monthly Return").data) == 1
    assert len(build_spx_monthly_seasonality_fig(result.monthly_statistics, "Average Monthly Performance", "Average Monthly Return").data) == 1
