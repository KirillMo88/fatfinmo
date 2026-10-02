from __future__ import annotations

import numpy as np
import pandas as pd

from us_debt_m2 import build_us_debt_m2_figure, build_us_debt_m2_history


def test_debt_m2_ratio_aligns_quarter_end_debt_to_monthly_m2() -> None:
    debt_dates = pd.to_datetime(["2020-01-01", "2020-04-01"])
    m2_dates = pd.date_range("2020-01-01", periods=7, freq="MS")
    fred_data = pd.concat(
        [
            pd.DataFrame({"Series_ID": "GFDEBTN", "Date": debt_dates, "Value": [3_000_000, 3_600_000]}),
            pd.DataFrame({"Series_ID": "M2SL", "Date": m2_dates, "Value": [1500, 1500, 1500, 1600, 1600, 1600, 1700]}),
        ],
        ignore_index=True,
    )

    result = build_us_debt_m2_history(fred_data)

    march = result.loc[result["Date"].eq(pd.Timestamp("2020-03-31"))].iloc[0]
    april = result.loc[result["Date"].eq(pd.Timestamp("2020-04-30"))].iloc[0]
    june = result.loc[result["Date"].eq(pd.Timestamp("2020-06-30"))].iloc[0]
    assert march["FederalDebtUSD_Bn"] == 3000.0
    assert march["DebtM2Ratio"] == 2.0
    assert april["FederalDebtUSD_Bn"] == 3000.0
    assert june["FederalDebtUSD_Bn"] == 3600.0
    assert june["DebtM2Ratio"] == 2.25


def debt_m2_history_fixture() -> pd.DataFrame:
    dates = pd.date_range("1990-01-01", periods=240, freq="MS")
    ratio = np.linspace(0.8, 1.6, len(dates)) + 0.08 * np.sin(np.arange(len(dates)) / 8)
    debt = pd.DataFrame({"Series_ID": "GFDEBTN", "Date": dates, "Value": ratio * 1000 * 1000})
    m2 = pd.DataFrame({"Series_ID": "M2SL", "Date": dates, "Value": 1000.0})
    return build_us_debt_m2_history(pd.concat([debt, m2], ignore_index=True))


def test_debt_m2_history_computes_moving_averages_and_channel() -> None:
    history = debt_m2_history_fixture()

    assert history["SMA20"].notna().any()
    assert history["SMA200"].notna().any()
    assert history["Support"].notna().all()
    assert history["Resistance"].notna().all()
    assert (history["Support"] < history["Resistance"]).all()


def test_debt_m2_figure_includes_channel_and_event_comments() -> None:
    history = debt_m2_history_fixture()
    figure = build_us_debt_m2_figure(history)

    assert "Long-term support" in [trace.name for trace in figure.data]
    assert "Long-term resistance" in [trace.name for trace in figure.data]
    assert {"ASIAN CRISIS", "DOT-COM BUBBLE", "GFC + EURO CRISIS", "EVERYTHING BUBBLE"}.issubset(
        {annotation.text for annotation in figure.layout.annotations}
    )
