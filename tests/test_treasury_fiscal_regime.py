from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import treasury_fiscal_regime as model
from treasury_fiscal_regime_tab import (
    build_buffers_chart, build_contributions_chart, build_financing_chart,
    build_financing_map, build_fiscal_chart, build_fiscal_decomposition_chart,
    build_net_liquidity_chart, build_policy_mix_chart,
)


def _released(dates: pd.DatetimeIndex, values: np.ndarray, lag_days: int) -> pd.DataFrame:
    return pd.DataFrame({
        "ObservationDate": dates,
        "AvailableDate": dates + pd.Timedelta(days=lag_days),
        "Value": values,
    })


def _sources() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    weeks = pd.date_range("2010-01-06", "2026-09-16", freq="W-WED")
    days = pd.bdate_range("2010-01-04", "2026-09-16")
    months = pd.date_range("2010-01-01", "2026-08-01", freq="MS")
    quarters = pd.date_range("2010-01-01", "2026-04-01", freq="QS")
    w = np.arange(len(weeks))
    d = np.arange(len(days))
    m = np.arange(len(months))
    q = np.arange(len(quarters))
    sources = {
        "WALCL": _released(weeks, 4_000_000 + 2500 * w + 100_000 * np.sin(w / 31), 1),
        "WTREGEN": _released(weeks, 300_000 + 180_000 * np.sin(w / 19), 1),
        "WDTGAL": _released(weeks, 320_000 + 190_000 * np.sin(w / 19), 1),
        "RRPONTSYD": _released(days, 200 + 100 * np.sin(d / 37), 1),
        "WRESBAL": _released(weeks, 2_000_000 + 1500 * w + 60_000 * np.sin(w / 30), 1),
        "GDP": _released(quarters, 15_000 + 245 * q, 60),
        "MTSR133FMS": _released(months, 270_000 + 800 * m + 22_000 * np.sin(m / 5), 45),
        "MTSO133FMS": _released(months, 410_000 + 1100 * m + 24_000 * np.cos(m / 6), 45),
        "MTSDS133FMS": _released(months, -140_000 - 300 * m, 45),
        "FGTSL": _released(quarters, 9_000_000 + 280_000 * q + 75_000 * np.sin(q / 4), 75),
        "BOGZ1FL313161110Q": _released(quarters, 2_000_000 + 65_000 * q + 30_000 * np.cos(q / 3), 75),
    }
    funding_dates = pd.date_range("2018-04-06", "2026-09-18", freq="W-FRI")
    f = np.arange(len(funding_dates))
    funding = pd.DataFrame({
        "Date": funding_dates,
        "ReservePressure": 0.7 + 0.3 * np.sin(f / 23),
        "MoneyMarketStress": 0.6 + 0.2 * np.cos(f / 17),
        "PersistentFundingFlag": False,
        "FundingState": "NORMAL",
        "CollateralStress": 0.3,
    })
    return sources, funding


def test_pit_normalization_excludes_future_observations() -> None:
    values = pd.Series(np.linspace(-2, 2, 100) + np.sin(np.arange(100) / 6))
    z = model.pit_z(values, 12)
    rank = model.pit_percentile(values, 12)
    changed = values.copy()
    changed.iloc[70:] = 1000
    pd.testing.assert_series_equal(z.iloc[:70], model.pit_z(changed, 12).iloc[:70])
    pd.testing.assert_series_equal(rank.iloc[:70], model.pit_percentile(changed, 12).iloc[:70])
    assert z.iloc[:12].isna().all()
    assert rank.iloc[:12].isna().all()


def test_mixed_reserve_units_and_component_identity() -> None:
    sources, funding = _sources()
    original = model.build_snapshot(sources, funding, pd.Timestamp("2026-09-18"))
    mixed = {**sources, "WRESBAL": sources["WRESBAL"].copy(), "WTREGEN": sources["WTREGEN"].copy()}
    mixed["WRESBAL"].loc[:750, "Value"] /= 1000
    mixed["WTREGEN"].loc[:750, "Value"] /= 1000
    actual = model.build_snapshot(mixed, funding, pd.Timestamp("2026-09-18"))
    pd.testing.assert_series_equal(actual.weekly["ReserveBufferRaw"], original.weekly["ReserveBufferRaw"])
    pd.testing.assert_series_equal(actual.weekly["WTREGEN"], original.weekly["WTREGEN"])
    week = actual.weekly.dropna(subset=["NetLiquidity", "MediumLiquidityImpulse"]).iloc[-1]
    assert week["NetLiquidity"] == pytest.approx(week["WALCL"] - week["WTREGEN"] - week["RRP"])
    assert week["MediumLiquidityImpulse"] == pytest.approx(
        week["FedImpulse_13W"] + week["TGAImpulse_13W"] + week["RRPImpulse_13W"]
    )
    assert week["TreasuryLiquidityImpulse"] == pytest.approx(
        0.30 * week["FastLiquidityZ"] + 0.50 * week["MediumLiquidityZ"] + 0.20 * week["SlowLiquidityZ"]
    )
    assert week["ReserveBufferRaw"] > 0.05


def test_vintage_scale_transition_does_not_rescale_later_small_values() -> None:
    source = pd.DataFrame({
        "ObservationDate": pd.to_datetime(["2025-11-05", "2025-11-12", "2025-11-19"]),
        "AvailableDate": pd.to_datetime(["2025-11-06", "2025-11-13", "2025-11-20"]),
        "Value": [940.979, 953_816.0, 50_000.0],
    })
    normalized = model._millions_from_initial_vintage(source)
    assert normalized["Value"].tolist() == pytest.approx([940_979, 953_816, 50_000])


def test_fred_fallback_uses_conservative_release_lags(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(model, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(model, "download_fred_series", lambda series_id, **_kwargs: pd.DataFrame({
        "Date": [pd.Timestamp("2026-04-01") if series_id == "FGTSL" else pd.Timestamp("2026-08-01")],
        "Value": [1_000_000.0],
    }))
    quarterly, status = model._load_delayed_current_vintage("FGTSL", None, False)
    assert status == "FRED_REVISED_HISTORY_CONSERVATIVE_LAG"
    assert quarterly.iloc[0]["AvailableDate"] == pd.Timestamp("2026-09-18")
    monthly, _ = model._load_delayed_current_vintage("MTSR133FMS", None, False)
    assert monthly.iloc[0]["AvailableDate"] == pd.Timestamp("2026-09-20")


def test_monthly_quarterly_release_dates_and_units() -> None:
    sources, funding = _sources()
    snapshot = model.build_snapshot(sources, funding, pd.Timestamp("2026-09-18"))
    weekly = snapshot.weekly.set_index("Date")
    assert pd.isna(weekly.loc["2010-01-29", "FiscalObservationDate"])
    assert weekly.loc["2026-05-29", "FinancingObservationDate"] == pd.Timestamp("2026-01-01")
    assert weekly.loc["2026-06-19", "FinancingObservationDate"] == pd.Timestamp("2026-04-01")
    fiscal = snapshot.fiscal.dropna(subset=["FiscalStanceRaw"]).iloc[-1]
    assert fiscal["Deficit12M"] == pytest.approx(fiscal["Outlays12M"] - fiscal["Receipts12M"])
    assert fiscal["FiscalStanceRaw"] == pytest.approx(fiscal["Deficit12M"] / fiscal["GDP"])
    assert fiscal["SpendingImpulse"] + fiscal["RevenueImpulse"] == pytest.approx(
        fiscal["FastFiscalImpulseRaw"]
    )
    financing = snapshot.financing.dropna(subset=["TotalNetIssuance4Q"]).iloc[-1]
    assert financing["FGTSL"] > 10_000
    assert financing["DurationNetIssuance4Q"] == pytest.approx(
        financing["TotalNetIssuance4Q"] - financing["BillNetIssuance4Q"]
    )


def test_absorption_and_financing_pressure_are_rule_based() -> None:
    row = pd.Series({
        "ReservePressurePercentile": 0.9, "MoneyMarketStressPercentile": 0.1,
        "PersistentFundingFlag": False, "FundingState": "NORMAL",
        "RRPBuffer": "DEPLETED", "TotalSupplyPercentile": 0.8,
        "DurationSupplyPercentile": 0.95,
    })
    assert model._absorption(row) == "TIGHT"
    row["AbsorptionCapacity"] = "TIGHT"
    assert model._financing_pressure(row) == "HIGH"
    row["FundingState"] = "PERSISTENT FUNDING PRESSURE"
    assert model._financing_pressure(row) == "SEVERE"
    row["ReservePressurePercentile"] = 0.3
    row["FundingState"] = "NORMAL"
    row["DurationSupplyPercentile"] = 0.1
    row["TotalSupplyPercentile"] = 0.1
    assert model._absorption(row) == "NORMAL"
    row["AbsorptionCapacity"] = "NORMAL"
    assert model._financing_pressure(row) == "LOW"


def test_missing_required_source_does_not_become_zero() -> None:
    sources, funding = _sources()
    sources["WTREGEN"] = model._empty_source()
    snapshot = model.build_snapshot(sources, funding, pd.Timestamp("2026-09-18"))
    assert snapshot.weekly["NetLiquidity"].isna().all()
    assert snapshot.weekly["TreasuryLiquidityState"].eq("DATA UNAVAILABLE").all()
    assert snapshot.weekly["PolicyMix"].eq("DATA UNAVAILABLE").all()


def test_all_treasury_charts_render() -> None:
    sources, funding = _sources()
    snapshot = model.build_snapshot(sources, funding, pd.Timestamp("2026-09-18"))
    weekly = snapshot.weekly.tail(60)
    fiscal = snapshot.fiscal.tail(24).assign(Date=lambda frame: frame["AvailableDate"])
    financing = snapshot.financing.tail(12).assign(Date=lambda frame: frame["AvailableDate"])
    figures = [
        build_net_liquidity_chart(weekly), build_contributions_chart(weekly, 13),
        build_buffers_chart(weekly, False), build_buffers_chart(weekly, True),
        build_fiscal_chart(fiscal), build_fiscal_decomposition_chart(fiscal, 3),
        build_financing_chart(financing, True), build_financing_map(weekly),
        build_policy_mix_chart(weekly),
    ]
    assert all(fig.layout.hovermode == "closest" and fig.data for fig in figures)


def test_policy_mix_history_combines_liquidity_fiscal_stance_and_regime_strip() -> None:
    sources, funding = _sources()
    frame = model.build_snapshot(sources, funding, pd.Timestamp("2026-09-18")).weekly.tail(60)
    figure = build_net_liquidity_chart(frame)
    assert [trace.type for trace in figure.data] == ["scatter", "scatter", "heatmap"]
    assert [trace.name for trace in figure.data[:2]] == ["Net Liquidity", "12M Deficit / GDP"]
    assert not any("LiquidityZ" in str(trace) for trace in figure.data)
    assert figure.data[-1].y == ("Policy mix",)
    assert figure.layout.yaxis.title.text == "Net Liquidity, USD bn"
    assert figure.layout.yaxis2.title.text == "12M Deficit / GDP, %"
