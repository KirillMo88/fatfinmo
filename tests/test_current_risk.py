from __future__ import annotations

import numpy as np
import pandas as pd

from current_risk import calculate_current_risk_v1, classify_component_state, current_risk_new_event


def current_risk_fixture() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=420)
    spy = np.linspace(100.0, 145.0, len(dates))
    qqq = spy * 0.80
    vix = np.full(len(dates), 14.0)
    breadth = np.full(len(dates), 80.0)
    hy = np.full(len(dates), 3.10)

    spy[-25:] = np.linspace(132.0, 150.0, 25)
    qqq[-25:] = spy[-25:] * np.linspace(0.80, 0.74, 25)
    vix[-4:] = [14.0, 15.0, 16.0, 18.0]
    breadth[-6:] = [80.0, 72.0, 64.0, 57.0, 50.0, 44.0]
    hy[-6:] = [3.20, 3.28, 3.34, 3.40, 3.48, 3.58]

    return pd.DataFrame(
        {
            "Date": dates,
            "SPY_Close": spy,
            "QQQ": qqq,
            "VIX": vix,
            "SPXAboveSMA50D": breadth,
            "HY_OAS": hy,
            "SPY_SMA200W": spy / 1.30,
            "SPY_WeeklyRSI14": 70.0,
            "SPY_26W_High": spy,
            "SPY_WeeklyRSI26WMax": 70.0,
            "SPX_1M_DD20_Probability": 8.0,
        }
    )


def test_full_activation_and_credit_priority() -> None:
    result = calculate_current_risk_v1(current_risk_fixture())
    last = result.iloc[-1]
    assert bool(last["CurrentRiskActivation"])
    assert last["CurrentRiskEscalationScoreV2"] == 6
    assert last["CurrentMarketRiskState"] == "RED FLAG"
    assert last["CurrentRiskSignalClass"] == "RED FLAG + CREDIT CONFIRMATION"


def test_inactive_activation_forces_escalation_to_zero() -> None:
    frame = current_risk_fixture()
    frame.loc[frame.index[-4:], "VIX"] = 14.0
    result = calculate_current_risk_v1(frame)
    last = result.iloc[-1]
    assert not bool(last["CurrentRiskActivation"])
    assert last["CurrentRiskEscalationScoreV2"] == 0
    assert last["CurrentMarketRiskState"] == "NORMAL"
    assert last["CurrentRiskSignalClass"] == "INACTIVE"


def test_btc_has_no_effect_on_current_risk_v1() -> None:
    frame = current_risk_fixture()
    low_btc = frame.assign(BTC=1.0)
    high_btc = frame.assign(BTC=np.linspace(1.0, 1_000_000.0, len(frame)))
    first = calculate_current_risk_v1(low_btc)
    second = calculate_current_risk_v1(high_btc)
    columns = [
        "CurrentRiskHighBetaRisk",
        "CurrentRiskEscalationScoreV2",
        "CurrentRiskSignalClass",
    ]
    pd.testing.assert_frame_equal(first[columns], second[columns])


def test_pvc_max10d_is_trailing_memory() -> None:
    result = calculate_current_risk_v1(current_risk_fixture())
    expected = result["PVC_V2"].rolling(10, min_periods=1).max()
    pd.testing.assert_series_equal(result["PVC_MAX10D"], expected, check_names=False)


def test_future_rows_do_not_change_prior_history() -> None:
    frame = current_risk_fixture()
    short = calculate_current_risk_v1(frame.iloc[:380].copy())
    full = calculate_current_risk_v1(frame.copy()).iloc[:380]
    columns = [
        "CurrentRiskROC12", "CurrentRiskROC12Max20D", "CurrentRiskROCDivergenceGap",
        "PVC_ROCCondition", "PVC_V2", "PVC_MAX10D", "CurrentRiskActivation",
        "CurrentRiskNewEvent", "CurrentRiskHighBetaRisk",
    ]
    pd.testing.assert_frame_equal(short[columns].reset_index(drop=True), full[columns].reset_index(drop=True))


def test_roc12_deterioration_uses_raw_close_and_adds_15_pvc_points() -> None:
    frame = current_risk_fixture()
    frame["Date"] = pd.bdate_range(end="2025-02-03", periods=len(frame))
    raw = np.full(len(frame), 100.0)
    raw[-11] = 104.1
    raw[-1] = 100.8
    frame["SPY_RawClose"] = raw
    frame["SPY_Close"] = 85.0

    result = calculate_current_risk_v1(frame)
    row = result.iloc[-1]

    assert np.isclose(row["CurrentRiskROC12"], 0.8)
    assert np.isclose(row["CurrentRiskROC12Max20D"], 4.1)
    assert np.isclose(row["CurrentRiskROCDivergenceGap"], 3.3)
    assert bool(row["PVC_ROCCondition"])
    weighted_conditions = [
        (20, "PVC_ExtensionCondition"), (10, "PVC_NearHighCondition"),
        (15, "PVC_ROCCondition"), (15, "PVC_BreadthCondition"),
        (15, "PVC_DailyRSIDivergence"), (10, "PVC_WeeklyRSIDivergence"),
        (15, "PVC_BlowoffRSI"),
    ]
    assert row["PVC_V2"] == sum(weight for weight, column in weighted_conditions if row[column])


def test_roc12_blowoff_route_uses_current_12_day_return() -> None:
    frame = current_risk_fixture()
    raw = np.full(len(frame), 100.0)
    raw[-1] = 103.2
    frame["SPY_RawClose"] = raw

    row = calculate_current_risk_v1(frame).iloc[-1]

    assert np.isclose(row["CurrentRiskROC12"], 3.2)
    assert np.isclose(row["CurrentRiskROCDivergenceGap"], 0.0)
    assert bool(row["PVC_ROCCondition"])


def test_raw_prices_override_adjusted_prices_for_validated_extension() -> None:
    frame = current_risk_fixture()
    frame["Date"] = pd.bdate_range(end="2017-10-24", periods=len(frame))
    raw_spy = np.linspace(180.0, 256.56, len(frame))
    frame["SPY_Close"] = raw_spy * (223.53 / 256.56)
    frame["SPY_RawClose"] = raw_spy
    frame["QQQ"] = frame["SPY_Close"] * 0.80
    frame["QQQ_RawClose"] = raw_spy * 0.80
    frame["SPY_SMA200W"] = 211.13

    result = calculate_current_risk_v1(frame)
    last = result.iloc[-1]

    assert last["CurrentRiskPriceBasis"] == "RAW_CLOSE"
    assert last["CurrentRiskSPYRawClose"] == 256.56
    assert np.isclose(last["CurrentRiskSPYExtension200W"], 256.56 / 211.13 - 1.0)
    assert not bool(last["PVC_ExtensionCondition"])


def test_current_risk_new_event_clusters_previous_five_sessions() -> None:
    activation = pd.Series(
        [True, True, False, False, False, True, False, False, False, False, False, True, True],
        dtype=bool,
    )
    expected = pd.Series(
        [True, False, False, False, False, False, False, False, False, False, False, True, False],
        dtype=bool,
    )

    pd.testing.assert_series_equal(current_risk_new_event(activation), expected)


def test_2021_12_20_credit_confirmation_regression() -> None:
    frame = current_risk_fixture()
    frame["Date"] = pd.bdate_range(end="2021-12-20", periods=len(frame))
    frame["QQQ"] = frame["SPY_Close"] * 0.80
    frame.loc[frame.index[-6:], "HY_OAS"] = [3.20, 3.25, 3.30, 3.34, 3.38, 3.41]

    result = calculate_current_risk_v1(frame)
    row = result.loc[result["Date"].eq(pd.Timestamp("2021-12-20"))].iloc[0]

    assert row["HY_OAS"] == 3.41
    assert bool(row["CurrentRiskHYLevelConfirmation"])
    assert bool(row["CurrentRiskHYWidening"])
    assert bool(row["CurrentRiskActivation"])
    assert row["CurrentRiskEscalationScoreV2"] == 5
    assert bool(row["CurrentRiskCreditConfirmation"])
    assert row["CurrentRiskSignalClass"] == "RED FLAG + CREDIT CONFIRMATION"


def test_component_state_boundaries() -> None:
    assert classify_component_state(25) == "LOW"
    assert classify_component_state(26) == "MODERATE"
    assert classify_component_state(50) == "MODERATE"
    assert classify_component_state(51) == "ELEVATED"
    assert classify_component_state(75) == "ELEVATED"
    assert classify_component_state(76) == "HIGH"


def test_current_risk_output_remains_daily() -> None:
    frame = current_risk_fixture()
    result = calculate_current_risk_v1(frame)

    pd.testing.assert_series_equal(result["Date"], frame["Date"], check_names=False)
    assert result["CurrentRiskFrequency"].eq("DAILY").all()
