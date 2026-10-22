from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import qqq_current_risk
from qqq_current_risk_data import _validate_ndfi, _values_on_dates
from qqq_current_risk import (
    QQQ_CURRENT_RISK_VIX_CONFIG,
    QQQ_CURRENT_RISK_VXN_CONFIG,
    calculate_qqq_current_risk_event_engine,
    calculate_qqq_current_risk_variant,
    calculate_qqq_current_risk_variants,
    calculate_qqq_shared_indicators,
    build_qqq_weekly_context,
    completed_weekly_ohlc,
    qqq_current_risk_new_event,
)


def qqq_risk_fixture() -> pd.DataFrame:
    dates = pd.bdate_range("2014-01-02", periods=420)
    count = len(dates)
    qqq = np.full(count, 100.0)
    iwm = np.full(count, 100.0)
    vix = np.full(count, 14.0)
    vxn = np.full(count, 17.0)
    ndfi = np.full(count, 85.0)
    hy = np.full(count, 3.0)
    iwm[-20:] = 100.0
    iwm[-1] = 95.0
    vix[-4:] = [14.0, 14.0, 14.0, 16.0]
    vxn[-4:] = [17.0, 17.0, 17.0, 19.0]
    ndfi[-6:] = [85.0, 83.0, 80.0, 75.0, 70.0, 65.0]
    hy[-6:] = [3.00, 3.10, 3.20, 3.30, 3.40, 3.50]
    return pd.DataFrame(
        {
            "Date": dates,
            "QQQ_RawClose": qqq,
            "IWM_RawClose": iwm,
            "VIX": vix,
            "VXN": vxn,
            "NDFI": ndfi,
            "HY_OAS": hy,
            "QQQ_SMA200W": 60.0,
            "QQQ_WeeklyRSI14": 70.0,
            "QQQ_26W_High": 100.0,
            "QQQ_WeeklyRSI26WMax": 90.0,
            "QQQ_1M_DD20_Probability": 12.0,
        }
    )


def test_engine_is_independent_from_spy_current_risk() -> None:
    source = Path(qqq_current_risk.__file__).read_text(encoding="utf-8")
    assert "from current_risk" not in source
    assert "import current_risk" not in source
    assert "calculate_current_risk_v1" not in source


def test_ndfi_session_timestamps_align_to_qqq_calendar_dates() -> None:
    ndfi = pd.DataFrame(
        {"Close": [48.0, 56.0]},
        index=pd.to_datetime(["2026-10-20 13:30:00", "2026-10-21 13:30:00"]),
    )
    validated = _validate_ndfi(ndfi, end=pd.Timestamp("2026-10-21"))

    assert validated.index.equals(pd.DatetimeIndex(["2026-10-20", "2026-10-21"]))
    aligned = _values_on_dates(validated, pd.Series(pd.to_datetime(["2026-10-20", "2026-10-21"])))
    np.testing.assert_array_equal(aligned, np.array([48.0, 56.0]))


def test_weekly_archive_registry_contains_qqq_risk_market_inputs() -> None:
    from historical_archive_job import model_market_ticker_registry

    tickers = set(model_market_ticker_registry())
    assert {"QQQ", "IWM", "^VIX", "^VXN"}.issubset(tickers)


def test_pvc_weights_and_vix_vxn_shared_values_are_identical() -> None:
    variants = calculate_qqq_current_risk_variants(qqq_risk_fixture())
    vix = variants["VIX"]
    vxn = variants["VXN"]
    columns = [
        "PVC_ExtensionContribution",
        "PVC_NearHighContribution",
        "PVC_ROCContribution",
        "PVC_BreadthContribution",
        "PVC_DailyRSIDivergenceContribution",
        "PVC_WeeklyRSIDivergenceContribution",
        "PVC_RSIBlowoffContribution",
        "PVC",
        "PVC_MAX10D",
    ]
    pd.testing.assert_frame_equal(vix[columns], vxn[columns])
    last = vix.iloc[-1]
    assert last["PVC"] == 70
    assert last["PVC_ExtensionContribution"] == 20
    assert last["PVC_NearHighContribution"] == 10
    assert last["PVC_ROCContribution"] == 0
    assert last["PVC_BreadthContribution"] == 15
    assert last["PVC_DailyRSIDivergenceContribution"] == 0
    assert last["PVC_WeeklyRSIDivergenceContribution"] == 10
    assert last["PVC_RSIBlowoffContribution"] == 15
    expected = vix["PVC"].rolling(10, min_periods=1).max()
    pd.testing.assert_series_equal(vix["PVC_MAX10D"], expected, check_names=False)


def test_activation_exact_four_conditions_and_six_escalation_channels() -> None:
    result = calculate_qqq_current_risk_variants(qqq_risk_fixture())["VIX"]
    last = result.iloc[-1]
    assert bool(last["CurrentRiskActivation"])
    assert bool(last["VolatilityBreakout"])
    assert last["Volatility3DChange"] >= 0.10
    assert bool(last["QQQActivationNearHigh"])
    assert last["CurrentRiskEscalationScoreRaw"] == 5
    assert last["CurrentRiskEscalationScore"] == 5
    assert last["CurrentMarketRiskState"] == "RED FLAG"
    assert last["CurrentRiskDisplayClass"] == "RED FLAG + CREDIT CONFIRMATION"


def test_inactive_gate_forces_published_escalation_to_zero() -> None:
    frame = qqq_risk_fixture()
    frame.loc[frame.index[-4:], ["VIX", "VXN"]] = [14.0, 17.0]
    result = calculate_qqq_current_risk_variants(frame)["VIX"].iloc[-1]
    assert not bool(result["CurrentRiskActivation"])
    assert result["CurrentRiskEscalationScoreRaw"] >= 1
    assert result["CurrentRiskEscalationScore"] == 0
    assert result["CurrentMarketRiskState"] == "NORMAL"


def test_only_calm_threshold_differs_in_escalation_configuration() -> None:
    frame = qqq_risk_fixture()
    frame.loc[frame.index[-11], "VIX"] = 17.0
    frame.loc[frame.index[-11], "VXN"] = 17.0
    shared = calculate_qqq_shared_indicators(frame)
    vix = calculate_qqq_current_risk_variant(shared, QQQ_CURRENT_RISK_VIX_CONFIG).iloc[-1]
    vxn = calculate_qqq_current_risk_variant(shared, QQQ_CURRENT_RISK_VXN_CONFIG).iloc[-1]
    assert not bool(vix["CurrentRiskCalmVolatility"])
    assert bool(vxn["CurrentRiskCalmVolatility"])
    assert vxn["CurrentRiskEscalationScoreRaw"] == vix["CurrentRiskEscalationScoreRaw"] + 1


def test_missing_required_input_is_data_incomplete_not_zero_risk() -> None:
    frame = qqq_risk_fixture()
    frame.loc[frame.index[-1], "NDFI"] = np.nan
    last = calculate_qqq_current_risk_variants(frame)["VIX"].iloc[-1]
    assert not bool(last["PVC_Complete"])
    assert pd.isna(last["PVC"])
    assert last["CurrentMarketRiskState"] == "DATA INCOMPLETE"
    assert pd.isna(last["CurrentRiskEscalationScore"])


def test_new_event_uses_only_preceding_five_sessions() -> None:
    activation = pd.Series([True, True, False, False, False, True, False, False, False, False, False, True])
    expected = pd.Series([True, False, False, False, False, False, False, False, False, False, False, True])
    pd.testing.assert_series_equal(qqq_current_risk_new_event(activation), expected)


def test_event_engine_preserves_credit_precedence_and_closes_after_five_sessions() -> None:
    activation = [True, True, False, False, False, False, False, True]
    state = ["HIGH RISK", "RED FLAG", "NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL", "LOW CONFIRMATION"]
    frame = pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-09-01", periods=len(activation)),
            "CurrentRiskActivation": activation,
            "CurrentRiskActivationAvailable": True,
            "CurrentMarketRiskState": state,
            "CurrentRiskDisplayClass": [
                "HIGH RISK", "RED FLAG + CREDIT CONFIRMATION", "INACTIVE", "INACTIVE",
                "INACTIVE", "INACTIVE", "INACTIVE", "LOW CONFIRMATION",
            ],
            "CurrentRiskCreditConfirmation": [False, True, False, False, False, False, False, False],
        }
    )
    result = calculate_qqq_current_risk_event_engine(frame)
    assert result.loc[1, "CurrentRiskEventMarkerClass"] == "RED FLAG + CREDIT CONFIRMATION"
    assert result.loc[1, "CurrentRiskEventCreditConfirmed"]
    assert result.loc[6, "CurrentRiskEventState"] == "CLOSED"
    assert result.loc[7, "CurrentRiskEventMarkerReason"] == "EVENT_START"


def test_holiday_week_is_mapped_by_actual_completion_without_lookahead() -> None:
    dates = pd.to_datetime(["2026-12-21", "2026-12-22", "2026-12-23", "2026-12-24"])
    frame = pd.DataFrame({"Close": [100, 101, 102, 103], "Low": [99, 100, 101, 102]}, index=dates)
    during_week = completed_weekly_ohlc(frame, now=pd.Timestamp("2026-12-24 23:00"))
    after_week = completed_weekly_ohlc(frame, now=pd.Timestamp("2026-12-26"))
    assert during_week.empty
    assert len(after_week) == 1
    assert after_week.iloc[0]["Date"] == pd.Timestamp("2026-12-24")


def test_future_rows_do_not_change_prior_daily_model_history() -> None:
    frame = qqq_risk_fixture()
    short = calculate_qqq_current_risk_variants(frame.iloc[:390].copy())["VIX"]
    full = calculate_qqq_current_risk_variants(frame)["VIX"].iloc[:390]
    columns = [
        "PVC", "PVC_MAX10D", "CurrentRiskActivation", "CurrentRiskEscalationScore",
        "CurrentMarketRiskState", "CurrentRiskNewEvent", "CurrentRiskEventMarker",
    ]
    pd.testing.assert_frame_equal(short[columns].reset_index(drop=True), full[columns].reset_index(drop=True))


def test_weekly_context_builds_qqq_specific_drawdown_inputs() -> None:
    dates = pd.bdate_range("2005-01-03", "2026-10-16")
    close = np.linspace(40.0, 600.0, len(dates)) * (1.0 + 0.02 * np.sin(np.arange(len(dates)) / 30.0))
    ohlcv = pd.DataFrame({"Close": close, "Low": close * 0.97}, index=dates)
    weekly = build_qqq_weekly_context(ohlcv, now=pd.Timestamp("2026-10-17"))
    required = {
        "QQQ_SMA200W", "QQQ_WeeklyRSI14", "QQQ_26W_High", "QQQ_WeeklyRSI26WMax",
        "MomentumCyclePhase", "QQQ_1M_DD20_Probability", "QQQ_1M_DD20_AnalogN",
        "QQQ_1M_DD20_SampleMode", "QQQ_1M_DD20_Confidence",
    }
    assert required.issubset(weekly.columns)
    assert weekly["QQQ_SMA200W"].notna().any()


def test_ui_tables_and_selected_variant_chart_consume_engine_outputs() -> None:
    from qqq_current_risk_tab import (
        activation_table,
        build_qqq_current_risk_signals_fig,
        diagnostic_table,
        escalation_table,
        pvc_table,
    )

    history = calculate_qqq_current_risk_variants(qqq_risk_fixture())["VIX"]
    current = history.iloc[-1].to_dict()
    assert len(pvc_table(current)) == 7
    assert len(activation_table(current, "VIX")) == 4
    assert len(escalation_table(current, "VIX")) == 6
    assert len(diagnostic_table(current, "VIX")) == 7
    figure = build_qqq_current_risk_signals_fig(history, "VIX")
    assert "VIX" in str(figure.layout.title.text)
    assert figure.data[0].name == "QQQ"
