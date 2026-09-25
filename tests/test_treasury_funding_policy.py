from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import treasury_funding_policy as model
from treasury_fiscal_regime_tab import (
    build_near_term_refinancing_chart,
    build_policy_components_chart,
    build_policy_score_chart,
    build_rollover_chart,
    build_rollover_yoy_chart,
)


def _market_rows(date: str = "2026-08-31") -> pd.DataFrame:
    record = pd.Timestamp(date)
    definitions = [
        ("Bills Maturity Value", "912AAA001", 100_000, 3.5, None, record + pd.Timedelta(days=90)),
        ("Notes", "912AAA002", 200_000, 3.8, 2.0, record + pd.Timedelta(days=300)),
        ("Bonds", "912AAA003", 300_000, 4.2, 3.0, record + pd.Timedelta(days=1500)),
        ("Inflation-Protected Securities", "912AAA004", 150_000, 1.8, 1.5, record + pd.Timedelta(days=250)),
        ("Floating Rate Notes", "912AAA005", 50_000, 3.9, 0.2, record + pd.Timedelta(days=500)),
    ]
    rows = []
    for security_type, cusip, amount, yield_pct, coupon, maturity in definitions:
        rows.append({
            "record_date": record, "security_type_desc": "Marketable",
            "security_class1_desc": security_type, "security_class2_desc": cusip,
            "issue_date": record - pd.Timedelta(days=120), "maturity_date": maturity,
            "outstanding_amt": amount, "issued_amt": amount,
            "interest_rate_pct": coupon, "yield_pct": yield_pct,
        })
    return pd.DataFrame(rows)


def _totals(date: str = "2026-08-31", amount: float = 800_000) -> pd.DataFrame:
    return pd.DataFrame([{
        "record_date": date, "security_type_desc": "Total Marketable",
        "security_class_desc": "_", "total_mil_amt": amount,
    }])


def test_mspd_security_filter_deduplicates_cusip_and_reconciles() -> None:
    raw = _market_rows()
    duplicate = raw.iloc[[0]].copy()
    duplicate["outstanding_amt"] = 90_000
    raw = pd.concat([raw, duplicate, pd.DataFrame([{
        "record_date": "2026-08-31", "security_type_desc": "Total Marketable",
        "security_class1_desc": "Total Marketable", "security_class2_desc": "_",
        "issue_date": None, "maturity_date": None, "outstanding_amt": 800_000,
        "issued_amt": None, "interest_rate_pct": None, "yield_pct": None,
    }])], ignore_index=True)
    history, diagnostics = model.build_treasury_funding_history(raw, _totals())
    latest = history.iloc[-1]
    assert diagnostics["DuplicateCUSIPRows"] == 2
    assert latest["TotalMarketableDebt"] == pytest.approx(.8)
    assert latest["Rollover12M"] == pytest.approx(.45)
    assert latest["RolloverIntensity"] == pytest.approx(56.25)
    assert latest["ReconciliationFlag"] == "PASS"
    assert latest["Rollover12M"] <= latest["TotalMarketableDebt"]


def test_default_scenario_weights_and_cohort_accounting() -> None:
    config = model.default_config()
    model.validate_config(config)
    assert all(sum(shares.values()) == pytest.approx(1.0) for shares in config["issuance_shares"].values())
    securities, _ = model.normalize_mspd_market(_market_rows())
    forecast = model.run_cohort_forecast(securities, config)
    assert set(forecast["Scenario"]) == {"LONG", "BASE", "SHORT"}
    assert set(forecast["Year"]) == set(model.FORECAST_YEARS)
    assert forecast["AccountingGap"].abs().max() < 1e-10
    assert (forecast["GrossFinancingRequirement"] ==
            forecast["PrincipalRollover"] + forecast["NewNetFinancing"]).all()
    base = forecast.loc[forecast["Scenario"].eq("BASE")].set_index("Year")
    assert base.loc[2028, "PrincipalRollover"] > 0


def test_policy_response_is_point_in_time_and_preserves_missing_values() -> None:
    weeks = pd.date_range("2015-01-02", periods=230, freq="W-FRI")
    liquidity = pd.DataFrame({
        "Date": weeks,
        "NetLiquidity": 4_000 + np.arange(len(weeks)) * 2 + np.sin(np.arange(len(weeks)) / 8) * 20,
        "WRESBAL": 2_200 + np.arange(len(weeks)) * 1.5 + np.cos(np.arange(len(weeks)) / 9) * 15,
    })
    months = pd.date_range("2014-12-01", periods=60, freq="MS")
    monthly = pd.DataFrame({
        "date": months,
        "global_cb_assets_usd_bn": 15_000 + np.arange(len(months)) * 35 + np.sin(np.arange(len(months)) / 4) * 100,
    })
    first = model.build_policy_response(monthly, liquidity)
    changed = monthly.copy()
    changed.loc[changed.index[-8:], "global_cb_assets_usd_bn"] *= 5
    second = model.build_policy_response(changed, liquidity)
    cutoff = weeks[-45]
    pd.testing.assert_series_equal(
        first.loc[first["Date"].lt(cutoff), "CBImpulsePercentile"].reset_index(drop=True),
        second.loc[second["Date"].lt(cutoff), "CBImpulsePercentile"].reset_index(drop=True),
    )
    score = first["PolicyResponseScore"].dropna()
    assert score.between(0, 100).all()
    assert first.loc[first[list(model.POLICY_WEIGHTS)].isna().any(axis=1), "PolicyResponseScore"].isna().all()
    assert sum(model.POLICY_WEIGHTS.values()) == pytest.approx(1.0)


def test_new_charts_render_with_closest_hover() -> None:
    annual = pd.DataFrame({
        "Year": [2024, 2025, 2026], "Rollover12M": [7.0, 7.5, 8.0],
        "RolloverIntensity": [25, 26, 27], "PortfolioAvgRate": [3, 3.2, 3.4],
        "WAMMonths": [70, 69, 68], "BillShare": [20, 21, 22],
    })
    forecast = pd.DataFrame([
        {"Year": year, "Scenario": scenario, "PrincipalRollover": 8 + (year - 2027),
         "RolloverIntensity": 28, "PortfolioAvgRate": 3.6, "WAMMonths": 67, "BillShare": 23}
        for year in (2027, 2028) for scenario in ("LONG", "BASE", "SHORT")
    ])
    policy = pd.DataFrame({
        "Date": pd.date_range("2025-01-03", periods=8, freq="W-FRI"),
        "PolicyResponseScore": np.linspace(30, 70, 8),
        "CBImpulsePercentile": np.linspace(20, 60, 8),
        "USNLImpulsePercentile": np.linspace(30, 70, 8),
        "BankReservesImpulsePercentile": np.linspace(40, 80, 8),
    })
    figures = [
        build_rollover_chart(annual, forecast), build_rollover_yoy_chart(annual, forecast),
        build_policy_score_chart(policy), build_policy_components_chart(policy),
    ]
    assert all(figure.data and figure.layout.hovermode == "closest" for figure in figures)


def _near_term_securities(end: str = "2022-12-31") -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2019-01-31", end, freq="ME")
    security_types = ("Bills", "Notes", "Bonds", "TIPS", "FRN")
    for sequence, date in enumerate(dates):
        for offset, months in enumerate((1, 2, 4, 5, 8)):
            rows.append({
                "RecordDate": date, "MaturityDate": date + pd.DateOffset(months=months),
                "CUSIP": f"TEST{sequence:04d}{offset}", "SecurityType": security_types[offset],
                "OutstandingMil": 50_000 + sequence * 100 + offset * 1_000,
                "EffectiveRatePct": 3.0,
            })
    return pd.DataFrame(rows)


def test_near_term_refinancing_reconciles_and_is_pit_safe() -> None:
    securities = _near_term_securities()
    first, diagnostics = model.build_near_term_refinancing(securities)
    assert diagnostics["Next3MWithinNext6M"]
    assert diagnostics["Trailing12MPositiveInRegime"]
    assert diagnostics["MaturityLedgerDuplicateRows"] == 0
    assert diagnostics["Next3MComponentMaxError"] < 1e-12
    assert diagnostics["Next6MComponentMaxError"] < 1e-12
    assert diagnostics["RefinancingWeightSum"] == pytest.approx(1.0)
    regime = first.loc[first["record_date"].ge(model.REFINANCING_REGIME_START)]
    assert regime["next_3m_rollover"].le(regime["next_6m_rollover"]).all()
    assert regime[["pressure_ratio_3m", "pressure_ratio_6m"]].gt(0).all().all()
    assert regime[["pressure_3m_percentile", "pressure_6m_percentile"]].apply(
        lambda series: series.between(0, 100).all()
    ).all()
    expected = .6 * regime["pressure_3m_percentile"] + .4 * regime["pressure_6m_percentile"]
    np.testing.assert_allclose(regime["near_term_refinancing_pressure"], expected)

    extended, _ = model.build_near_term_refinancing(_near_term_securities("2023-12-31"))
    original = first.loc[first["record_date"].le(pd.Timestamp("2022-12-31"))].set_index("record_date")
    comparison = extended.loc[extended["record_date"].le(pd.Timestamp("2022-12-31"))].set_index("record_date")
    pd.testing.assert_frame_equal(
        original[["pressure_3m_percentile", "pressure_6m_percentile"]],
        comparison[["pressure_3m_percentile", "pressure_6m_percentile"]],
    )


def test_near_term_charts_render() -> None:
    frame, _ = model.build_near_term_refinancing(_near_term_securities())
    frame = frame.rename(columns={"record_date": "Date"})
    pressure = build_near_term_refinancing_chart(frame)
    assert len(pressure.data) == 3
    assert pressure.layout.hovermode == "closest"
    assert pressure.layout.height == 345
