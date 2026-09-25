from __future__ import annotations

import numpy as np
import pandas as pd

from correction_bottom import add_extreme_capitulation, calculate_correction_bottom_indicator


def synthetic_2022_history() -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=520)
    close = np.linspace(100.0, 120.0, len(dates))
    vix = np.full(len(dates), 15.0)
    breadth50 = np.full(len(dates), 65.0)
    breadth200 = np.full(len(dates), 70.0)
    hy = np.full(len(dates), 3.0)

    close[300:321] = np.linspace(120.0, 107.0, 21)
    vix[300:321] = np.linspace(16.0, 38.0, 21)
    breadth50[300:321] = np.linspace(60.0, 8.0, 21)
    breadth200[300:321] = np.linspace(65.0, 18.0, 21)
    hy[300:321] = np.linspace(3.0, 5.5, 21)

    close[321:341] = np.linspace(108.0, 115.5, 20)
    vix[321:341] = np.linspace(36.0, 19.0, 20)
    breadth50[321:341] = np.linspace(10.0, 55.0, 20)
    breadth200[321:341] = np.linspace(20.0, 60.0, 20)
    hy[321:341] = np.linspace(5.4, 4.0, 20)

    close[341:441] = np.linspace(114.0, 115.0, 100)
    vix[341:441] = 18.0
    breadth50[341:441] = 58.0
    breadth200[341:441] = 62.0
    hy[341:441] = 3.8

    close[441:456] = np.linspace(115.0, 108.0, 15)
    vix[441:456] = np.linspace(18.0, 34.0, 15)
    breadth50[441:456] = np.linspace(58.0, 12.0, 15)
    breadth200[441:456] = np.linspace(62.0, 24.0, 15)
    hy[441:456] = np.linspace(3.8, 5.2, 15)

    close[456:481] = np.linspace(108.5, 116.0, 25)
    vix[456:481] = np.linspace(33.0, 17.0, 25)
    breadth50[456:481] = np.linspace(14.0, 62.0, 25)
    breadth200[456:481] = np.linspace(26.0, 65.0, 25)
    hy[456:481] = np.linspace(5.1, 3.5, 25)
    close[481:] = 116.0
    vix[481:] = 17.0
    breadth50[481:] = 62.0
    breadth200[481:] = 65.0
    hy[481:] = 3.5

    return pd.DataFrame(
        {
            "Date": dates,
            "SPY_Close": close,
            "SPY_Low": close * 0.997,
            "SPY_Volume": np.linspace(70_000_000, 95_000_000, len(dates)),
            "VIX": vix,
            "SPXAboveSMA50D": breadth50,
            "SPXAboveSMA200D": breadth200,
            "SPXAboveSMA50D_Observed": True,
            "SPXAboveSMA200D_Observed": True,
            "HY_OAS": hy,
        }
    )


def test_tactical_and_durable_require_active_correction() -> None:
    result = calculate_correction_bottom_indicator(synthetic_2022_history())
    assert not result.loc[~result["CorrectionActive"], "TacticalInternalEvent"].any()
    durable = result["DurableBottomConfirmed"]
    assert (result.loc[durable, "TacticalInternalEvent"] == True).all()  # noqa: E712


def test_correction_bottom_output_remains_daily() -> None:
    frame = synthetic_2022_history()
    result = calculate_correction_bottom_indicator(frame)

    pd.testing.assert_series_equal(result["Date"], frame["Date"], check_names=False)
    assert result["CorrectionFrequency"].eq("DAILY").all()


def test_2022_multiple_waves_allow_late_durable() -> None:
    result = calculate_correction_bottom_indicator(synthetic_2022_history())
    events = result.loc[result["TacticalInternalEvent"]]
    assert len(events) >= 2
    assert events["CorrectionCycleID"].nunique() == 1
    assert events["CorrectionWaveID"].nunique() >= 2
    assert result.loc[result["DurableBottomConfirmed"], "CorrectionWaveID"].nunique() >= 2


def test_new_wave_rearms_tactical_after_five_percent_decline() -> None:
    result = calculate_correction_bottom_indicator(synthetic_2022_history())
    tactical = result.index[result["TacticalInternalEvent"]].tolist()
    assert len(tactical) >= 2
    between = result.loc[tactical[0] + 1 : tactical[1] - 1]
    assert between["TacticalEligible"].any()
    first_wave = result.loc[tactical[0], "CorrectionWaveID"]
    second_wave = result.loc[tactical[1], "CorrectionWaveID"]
    assert first_wave != second_wave


def test_missing_breadth_and_hy_are_not_zeroes() -> None:
    frame = synthetic_2022_history()
    frame[["SPXAboveSMA50D", "SPXAboveSMA200D", "HY_OAS"]] = np.nan
    frame[["SPXAboveSMA50D_Observed", "SPXAboveSMA200D_Observed"]] = False
    result = calculate_correction_bottom_indicator(frame)
    assert result["Breadth50_Stress"].isna().all()
    assert result["Breadth200_Stress"].isna().all()
    assert result["BearRiskScore"].isna().all()
    assert result["ExtremeCapRaw"].isna().all()
    assert result["ExtremeCapitulation2Raw"].isna().all()
    assert result["ExtremeConditionsMet"].isna().all()


def test_future_data_does_not_change_prior_signals() -> None:
    frame = synthetic_2022_history()
    short = calculate_correction_bottom_indicator(frame.iloc[:400].copy())
    full = calculate_correction_bottom_indicator(frame.copy()).iloc[:400]
    columns = [
        "TacticalInternalEvent",
        "DurableBottomConfirmed",
        "CorrectionCycleID",
        "CorrectionWaveID",
        "VIX_Stress",
        "ExtremeCapRaw",
        "ExtremeCapitulation2Raw",
    ]
    pd.testing.assert_frame_equal(short[columns].reset_index(drop=True), full[columns].reset_index(drop=True))


def test_extreme_episode_clustering_preserves_raw_dates() -> None:
    dates = pd.bdate_range("2020-01-01", periods=30)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "CorrectionCurrentDrawdown": -0.20,
            "VIX_Stress": 99.0,
            "RSI_Stress": 99.0,
            "SPY5D_DownsideStress": 99.0,
            "Breadth50_Stress": 99.0,
            "Breadth200_Stress": 99.0,
            "ExtremeStressMean5": np.linspace(98.0, 100.0, 30),
            "SPYVolumeMaxPct5": 99.0,
            "SPYVolumeMaxRatio5": 2.0,
        }
    )
    keep = {1, 5, 12, 25}
    for idx in frame.index:
        if idx not in keep:
            frame.loc[idx, "VIX_Stress"] = 50.0
    add_extreme_capitulation(frame)
    assert frame["ExtremeCapRaw"].fillna(False).sum() == 4
    assert frame["ExtremeEpisodeMarker"].sum() == 2
    assert frame.loc[frame["ExtremeCapRaw"].fillna(False), "ExtremeEpisodeID"].nunique() == 2
    assert frame["ExtremeCapitulation2Raw"].fillna(False).sum() == 4
    assert frame["ExtremeCapitulation2EpisodeMarker"].sum() == 2
    assert frame["ExtremeEpisodeDisplayMarker"].sum() == 0


def test_extreme_capitulation_2_is_severity_gated_classic_subset() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-03-16", periods=4),
            "CorrectionCurrentDrawdown": [-0.16, -0.20, -0.16, -0.25],
            "VIX_Stress": [99.0, 99.0, 99.8, 99.8],
            "RSI_Stress": [99.0, 99.0, 99.0, 50.0],
            "SPY5D_DownsideStress": 99.0,
            "Breadth50_Stress": 99.0,
            "Breadth200_Stress": 99.0,
            "ExtremeStressMean5": [99.0, 99.1, 99.2, 90.0],
            "SPYVolumeMaxPct5": np.nan,
            "SPYVolumeMaxRatio5": np.nan,
        }
    )

    add_extreme_capitulation(frame)

    classic = frame["ExtremeCapRaw"].fillna(False).tolist()
    severe = frame["ExtremeCapitulation2Raw"].fillna(False).tolist()
    assert classic == [True, True, True, False]
    assert severe == [False, True, True, False]
    assert not (frame["ExtremeCapitulation2Raw"].fillna(False) & ~frame["ExtremeCapRaw"].fillna(False)).any()
    assert frame["ExtremeCapitulation2EpisodeMarker"].sum() == 1
    assert frame.loc[frame["ExtremeCapitulation2Raw"].fillna(False), "ExtremeCapitulation2EpisodeID"].nunique() == 1


def test_extreme_capitulation_2_clustering_splits_after_ten_sessions() -> None:
    dates = pd.bdate_range("2020-01-01", periods=25)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "CorrectionCurrentDrawdown": -0.21,
            "VIX_Stress": 99.0,
            "RSI_Stress": 99.0,
            "SPY5D_DownsideStress": 99.0,
            "Breadth50_Stress": 99.0,
            "Breadth200_Stress": 99.0,
            "ExtremeStressMean5": np.linspace(98.0, 100.0, len(dates)),
            "SPYVolumeMaxPct5": np.nan,
            "SPYVolumeMaxRatio5": np.nan,
        }
    )
    qualifying = {1, 11, 22}
    for idx in frame.index:
        if idx not in qualifying:
            frame.loc[idx, "VIX_Stress"] = 50.0

    add_extreme_capitulation(frame)

    assert frame["ExtremeCapitulation2Raw"].fillna(False).sum() == 3
    assert frame["ExtremeCapitulation2EpisodeMarker"].sum() == 2
    assert frame.loc[frame["ExtremeCapitulation2Raw"].fillna(False), "ExtremeCapitulation2EpisodeID"].nunique() == 2


def test_new_master_correction_clears_transient_context() -> None:
    frame = synthetic_2022_history()
    last_date = frame["Date"].iloc[-1]
    recovery_dates = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=12)
    second_wave_dates = pd.bdate_range(recovery_dates[-1] + pd.offsets.BDay(1), periods=12)
    recovery_close = np.linspace(frame["SPY_Close"].iloc[-1], 122.0, len(recovery_dates))
    second_close = np.linspace(122.0, 114.0, len(second_wave_dates))
    extension = pd.DataFrame(
        {
            "Date": recovery_dates.append(second_wave_dates),
            "SPY_Close": np.concatenate([recovery_close, second_close]),
            "VIX": 18.0,
            "SPXAboveSMA50D": 60.0,
            "SPXAboveSMA200D": 65.0,
            "SPXAboveSMA50D_Observed": True,
            "SPXAboveSMA200D_Observed": True,
            "HY_OAS": 3.5,
            "SPY_Volume": 80_000_000.0,
        }
    )
    extension["SPY_Low"] = extension["SPY_Close"] * 0.997
    result = calculate_correction_bottom_indicator(pd.concat([frame, extension], ignore_index=True))
    cycles = result["CorrectionCycleID"].dropna().unique().tolist()
    assert len(cycles) >= 2
    second_start = result[result["CorrectionCycleID"].eq(cycles[1])].iloc[0]
    assert str(second_start["CorrectionWaveID"]).endswith("W01")
    assert pd.isna(second_start["PostTacticalRecoveryHigh"])
    assert pd.isna(second_start["LatestDurableBottomDate"])


def test_every_durable_has_bear_risk_score_in_range() -> None:
    result = calculate_correction_bottom_indicator(synthetic_2022_history())
    durable = result.loc[result["DurableBottomConfirmed"]]
    assert not durable.empty
    assert durable["BearRiskScore"].between(0, 6).all()
    assert durable["BearRiskCategory"].isin(
        ["LOW BEAR RISK", "MODERATE BEAR RISK", "HIGH BEAR RISK"]
    ).all()


def test_volume_is_diagnostic_only_for_extreme_capitulation() -> None:
    frame = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-03-16")],
            "CorrectionCurrentDrawdown": [-0.25],
            "VIX_Stress": [99.0],
            "RSI_Stress": [99.0],
            "SPY5D_DownsideStress": [99.0],
            "Breadth50_Stress": [99.0],
            "Breadth200_Stress": [99.0],
            "ExtremeStressMean5": [99.0],
            "SPYVolumeMaxPct5": [np.nan],
            "SPYVolumeMaxRatio5": [np.nan],
        }
    )
    add_extreme_capitulation(frame)
    assert bool(frame.loc[0, "ExtremeCapRaw"])


def test_previous_ten_day_high_excludes_current_observation() -> None:
    frame = synthetic_2022_history().iloc[:260].copy()
    frame.loc[frame.index[-1], "SPY_Close"] = frame["SPY_Close"].iloc[:-1].max() * 1.25
    frame.loc[frame.index[-1], "SPY_Low"] = frame.loc[frame.index[-1], "SPY_Close"]
    result = calculate_correction_bottom_indicator(frame)
    expected = frame["SPY_Close"].iloc[-11:-1].max()
    assert result["Previous10DHigh"].iloc[-1] == expected
