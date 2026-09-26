from io import BytesIO

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from btc_cycle import (
    BTC_PROJECTED_HALVING,
    build_btc_cycle_history,
    btc_cycle_export_xlsx,
    btc_cycle_validation,
    halving_cycle_position,
    halving_phase,
    liquidity_cycle_position,
    point_in_time_percentile,
)
from btc_cycle_tab import _build_btc_price_halving_figure, _build_macro_score_figure, _build_structural_cycles_figure


def test_halving_phases_follow_specified_progress_bands():
    assert halving_phase(0.0) == "EARLY_EXPANSION"
    assert halving_phase(20.0) == "LATE_EXPANSION"
    assert halving_phase(35.0) == "CYCLE_TOP_RISK"
    assert halving_phase(42.0) == "BEAR_DELEVERAGING"
    assert halving_phase(55.0) == "BOTTOMING_TRANSITION"
    assert halving_phase(70.0) == "ACCUMULATION_PRE_HALVING"


def test_halving_calendar_uses_actual_events_and_projected_april_2028():
    prior = halving_cycle_position("2024-04-19")
    current = halving_cycle_position("2026-09-25")

    assert prior["previous_halving"] == pd.Timestamp("2020-05-11")
    assert prior["next_halving"] == pd.Timestamp("2024-04-20")
    assert current["previous_halving"] == pd.Timestamp("2024-04-20")
    assert current["next_halving"] == BTC_PROJECTED_HALVING
    assert current["next_halving_projected"] is True
    assert current["phase"] == "BOTTOMING_TRANSITION"


def test_liquidity_phase_uses_only_troughs_known_at_signal_date():
    before_2019_anchor = liquidity_cycle_position("2018-10-01", "2018-10-01")
    after_2019_anchor = liquidity_cycle_position("2019-04-01", "2019-04-01")
    forecast_before_2019_anchor = liquidity_cycle_position("2020-01-01", "2018-10-01")

    assert before_2019_anchor["anchor"] == pd.Timestamp("2015-03-06")
    assert after_2019_anchor["anchor"] == pd.Timestamp("2019-03-08")
    assert forecast_before_2019_anchor["anchor"] == pd.Timestamp("2019-04-06")
    assert forecast_before_2019_anchor["anchor"] != pd.Timestamp("2019-03-08")


def test_secondary_percentile_is_trailing_and_point_in_time():
    dates = pd.date_range("2020-01-03", periods=180, freq="W-FRI")
    values = pd.Series(np.sin(np.arange(len(dates)) / 8.0), index=dates)
    original = point_in_time_percentile(values)
    changed = values.copy()
    changed.iloc[-1] = 1000.0
    changed_rank = point_in_time_percentile(changed)

    assert np.isnan(original.iloc[102])
    assert np.isfinite(original.iloc[103])
    assert np.isclose(original.iloc[-2], changed_rank.iloc[-2])
    assert changed_rank.iloc[-1] > original.iloc[-1]


def _cycle_inputs():
    btc_dates = pd.date_range("2014-01-03", "2026-09-25", freq="W-FRI")
    index = np.arange(len(btc_dates), dtype=float)
    btc = pd.DataFrame({"date": btc_dates, "Close": 1000.0 * np.exp(index / 230.0)})
    macro = pd.DataFrame(
        {
            "Date": btc_dates,
            "BTC_DXY_Close": 100.0 + np.sin(index / 12.0) * 4 + index / 500.0,
            "BTC_US2Y": 3.0 + np.sin(index / 17.0),
            "BTC_RealYield": 1.0 + np.cos(index / 23.0) * 0.6,
        }
    )
    m2_dates = pd.date_range("1990-01-01", "2026-08-01", freq="MS")
    m2_index = np.arange(len(m2_dates), dtype=float)
    cycle = pd.DataFrame({"Date": m2_dates, "PrimaryMarketCycle": np.sin(m2_index / 16.0)})
    return btc, cycle, macro


def test_btc_cycle_reconciles_scores_and_has_phase_only_projection():
    btc, canonical_cycle, macro = _cycle_inputs()
    history = build_btc_cycle_history(btc, canonical_cycle, macro)
    current = history.loc[~history["Projected"]].iloc[-1]
    projected = history.loc[history["Projected"]]

    assert not btc_cycle_validation(history)
    assert current["Liquidity_Trough_Anchor"] == pd.Timestamp("2022-10-28")
    available_m2 = canonical_cycle.loc[canonical_cycle["Date"].le(current["Date"])].iloc[-1]
    assert np.isclose(current["GlobalM2PrimaryCycle"], available_m2["PrimaryMarketCycle"])
    for horizon in ("3M", "6M", "9M", "12M"):
        inputs = (
            current[f"HalvingBase_{horizon}"]
            + current[f"LiquidityCycleModifier_{horizon}"]
            + current[f"SecondaryMacroModifier_{horizon}"]
        )
        assert np.isclose(current[f"BTC_MACRO_{horizon}"], np.clip(inputs, 0.0, 95.0))
        assert projected[f"SecondaryMacroModifier_{horizon}"].eq(0.0).all()
    assert projected["BTC_Price"].isna().all()
    assert projected["Historical_Projected_Flag"].eq("PROJECTED").all()
    assert projected["Date"].max() <= pd.Timestamp("2028-04-30")


def test_btc_cycle_xlsx_contains_full_history_and_projected_rows():
    btc, canonical_cycle, macro = _cycle_inputs()
    history = build_btc_cycle_history(btc, canonical_cycle, macro)

    workbook = load_workbook(BytesIO(btc_cycle_export_xlsx(history)), read_only=True)

    assert workbook.sheetnames == ["BTC Cycle"]
    sheet = workbook["BTC Cycle"]
    headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    assert "BTC_MACRO_3M" in headers
    assert "Historical_Projected_Flag" in headers
    assert sheet.max_row == len(history) + 1


def test_btc_cycle_charts_separate_observed_price_from_projected_scores():
    btc, canonical_cycle, macro = _cycle_inputs()
    history = build_btc_cycle_history(btc, canonical_cycle, macro)
    latest_observation = history.loc[~history["Projected"], "Date"].max()

    price_fig = _build_btc_price_halving_figure(history)
    liquidity_fig = _build_structural_cycles_figure(history)
    score_fig = _build_macro_score_figure(history, "3M")

    assert pd.to_datetime(price_fig.data[0].x).max() == latest_observation
    assert price_fig.layout.xaxis.range[1] == pd.Timestamp("2028-04-30")
    assert [trace.name for trace in liquidity_fig.data] == ["Global M2 Primary Liquidity Cycle"]
    assert len(score_fig.data) == 2
    assert score_fig.data[1].line.dash == "dash"
    assert pd.to_datetime(score_fig.data[1].x).max() <= pd.Timestamp("2028-04-30")
