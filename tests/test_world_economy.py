from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from world_economy import (
    _cagr,
    _weo_gdp_to_usd_bn,
    build_world_economy_frames,
    parse_bis_bulk_csv,
    parse_bis_dsr_bulk_csv,
    parse_oecd_long_term_rates_csv,
    parse_weo_countries_frame,
)


def _bis_frame() -> pd.DataFrame:
    observations = [
        ("2016-Q1", 100.0), ("2021-Q1", 120.0), ("2025-Q4", 130.0), ("2026-Q1", 135.0),
    ]
    return pd.DataFrame([
        {"iso3": "USA", "bis_iso2": "US", "country_name": "United States", "year": int(period[:4]),
         "quarter": int(period[-1]), "period": period, "debt_gdp_pct": value,
         "bis_series_id": "Q.US.C.A.M.770.A", "observation_status": ""}
        for period, value in observations
    ])


def _imf_frame() -> pd.DataFrame:
    rows = []
    for year in range(2016, 2027):
        rows.extend([
            {"iso3": "USA", "country_name": "United States", "indicator_id": "NGDPD", "year": year,
             "value": 18_000 + (year - 2016) * 1_000, "latest_actual_year": 2025, "scale": "Billions"},
            {"iso3": "USA", "country_name": "United States", "indicator_id": "GGXCNL_NGDP", "year": year,
             "value": -5 + (year - 2016) * 0.1, "latest_actual_year": 2025, "scale": "Percent of GDP"},
            {"iso3": "USA", "country_name": "United States", "indicator_id": "GGX_NGDP", "year": year,
             "value": 35 + (year - 2016) * 0.1, "latest_actual_year": 2025, "scale": "Percent of GDP"},
            {"iso3": "USA", "country_name": "United States", "indicator_id": "BCA_NGDPD", "year": year,
             "value": -2 + (year - 2016) * 0.2, "latest_actual_year": 2025, "scale": "Percent of GDP"},
            {"iso3": "USA", "country_name": "United States", "indicator_id": "NGDP_RPCH", "year": year,
             "value": 2.0 + (year - 2023) * 0.1, "latest_actual_year": 2025, "scale": "Percent change"},
        ])
    return pd.DataFrame(rows)


def _dsr_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"iso3": "USA", "bis_iso2": "US", "country_name": "United States", "year": year,
         "quarter": 1, "period": f"{year}-Q1", "dsr_pct": value}
        for year, value in ((2016, 20.0), (2021, 18.0), (2026, 16.6))
    ])


def _long_term_rates_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"iso3": "USA", "period": f"{year}-{month:02d}", "year": year, "month": month,
         "long_term_interest_rate_pct": value}
        for year, month, value in ((2016, 8, 1.5), (2021, 8, 1.3), (2026, 8, 4.2))
    ])


def test_formula_helpers_preserve_sign_and_require_positive_gdp() -> None:
    assert np.isclose(_cagr(200, 100, 5), (2 ** (1 / 5) - 1) * 100)
    assert np.isnan(_cagr(0, 100, 5))
    assert _weo_gdp_to_usd_bn(1250, "Millions") == 1.25
    assert _weo_gdp_to_usd_bn(1250, "Billions") == 1250


def test_snapshot_calculations_use_matching_bis_quarters_and_three_real_growth_years() -> None:
    table, history, diagnostics, reference_year, warning = build_world_economy_frames(
        _bis_frame(), _imf_frame(), dsr=_dsr_frame(), long_term_rates=_long_term_rates_frame(), reference_year=2026,
    )
    row = table.iloc[0]
    assert reference_year == 2026
    assert warning is None
    assert row["bis_observation_quarter"] == "2026Q1"
    assert np.isclose(row["government_expenditure_pct"], 36.0)
    assert np.isclose(row["current_account_pct"], 0.0)
    assert np.isclose(row["current_account_delta_5y_pp"], 1.0)
    assert np.isclose(row["current_account_delta_10y_pp"], 2.0)
    assert row["dsr_observation_quarter"] == "2026-Q1"
    assert np.isclose(row["dsr_delta_5y_pp"], -1.4)
    assert np.isclose(row["dsr_delta_10y_pp"], -3.4)
    assert np.isclose(row["long_term_interest_rate_pct"], 4.2)
    assert row["long_term_interest_rate_observation_month"] == "2026-08"
    assert np.isclose(row["long_term_interest_rate_delta_5y_pp"], 2.9)
    assert np.isclose(row["long_term_interest_rate_delta_10y_pp"], 2.7)
    assert row["debt_delta_5y_pp"] == 15.0
    assert row["debt_delta_10y_pp"] == 35.0
    assert np.isclose(row["fiscal_balance_usd_bn"], row["nominal_gdp_usd_bn"] * row["fiscal_balance_pct"] / 100)
    assert np.isclose(row["avg_real_gdp_growth_3y_pct"], np.mean([2.0, 2.1, 2.2]))
    assert history.loc[history["is_provisional"], "bis_observation_quarter"].iloc[0] == "2026-Q1"
    annual_2026 = history.loc[history["iso3"].eq("USA") & history["year"].eq(2026) & ~history["is_provisional"]].iloc[0]
    assert np.isclose(annual_2026["total_debt_gdp_annual_avg_pct"], 135.0)
    assert np.isclose(annual_2026["dsr_annual_avg_pct"], 16.6)
    assert np.isclose(annual_2026["long_term_interest_rate_annual_avg_pct"], 4.2)
    assert np.isclose(annual_2026["interest_on_total_debt_pct_gdp"], 5.67)
    assert np.isclose(annual_2026["current_account_pct"], 0.0)
    assert np.isclose(annual_2026["government_expenditure_pct"], 36.0)
    assert diagnostics["bis_countries_mapped"] == 1


def test_bis_parser_selects_only_quarterly_total_nfc_percent_gdp_series() -> None:
    raw = pd.DataFrame([
        {"FREQ": "Q:Quarterly", "BORROWERS_CTY": "US:United States", "TC_BORROWERS": "C:Non-financial sector",
         "TC_LENDERS": "A:All sectors", "VALUATION": "M:Market value", "UNIT_TYPE": "770:Percent of GDP",
         "TC_ADJUST": "A:Adjusted", "TIME_PERIOD": "2026-Q1", "OBS_VALUE": "135.0", "OBS_STATUS": "A"},
        {"FREQ": "Q:Quarterly", "BORROWERS_CTY": "US:United States", "TC_BORROWERS": "P:Private non-financial sector",
         "TC_LENDERS": "A:All sectors", "VALUATION": "M:Market value", "UNIT_TYPE": "770:Percent of GDP",
         "TC_ADJUST": "A:Adjusted", "TIME_PERIOD": "2026-Q1", "OBS_VALUE": "90.0", "OBS_STATUS": "A"},
    ])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("WS_TC_csv_flat.csv", raw.to_csv(index=False))
    parsed = parse_bis_bulk_csv(buffer.getvalue())
    assert len(parsed) == 1
    assert parsed.iloc[0]["bis_series_id"] == "Q.US.C.A.M.770.A"
    assert parsed.iloc[0]["debt_gdp_pct"] == 135.0


def test_bis_dsr_parser_selects_private_nonfinancial_sector_quarterly_percent_series() -> None:
    raw = pd.DataFrame([
        {"FREQ": "Q: Quarterly", "BORROWERS_CTY": "US: United States", "DSR_BORROWERS": "P: Private non-financial sector",
         "TIME_PERIOD": "2026-Q1", "OBS_VALUE": "16.6", "UNIT_MEASURE": "367: Per cent", "UNIT_MULT": "0: Units"},
        {"FREQ": "Q: Quarterly", "BORROWERS_CTY": "US: United States", "DSR_BORROWERS": "H: Households",
         "TIME_PERIOD": "2026-Q1", "OBS_VALUE": "7.0", "UNIT_MEASURE": "367: Per cent", "UNIT_MULT": "0: Units"},
    ])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("WS_DSR_csv_flat.csv", raw.to_csv(index=False))
    parsed = parse_bis_dsr_bulk_csv(buffer.getvalue())
    assert len(parsed) == 1
    assert parsed.iloc[0]["iso3"] == "USA"
    assert parsed.iloc[0]["dsr_pct"] == 16.6


def test_oecd_parser_selects_monthly_long_term_rates_in_percent_per_annum() -> None:
    raw = pd.DataFrame([
        {"REF_AREA": "USA", "FREQ": "M", "MEASURE": "IRLT", "UNIT_MEASURE": "PA",
         "TIME_PERIOD": "2026-08", "OBS_VALUE": "4.2"},
        {"REF_AREA": "USA", "FREQ": "A", "MEASURE": "IRLT", "UNIT_MEASURE": "PA",
         "TIME_PERIOD": "2026", "OBS_VALUE": "4.0"},
        {"REF_AREA": "USA", "FREQ": "M", "MEASURE": "IR3TIB", "UNIT_MEASURE": "PA",
         "TIME_PERIOD": "2026-08", "OBS_VALUE": "3.5"},
        {"REF_AREA": "USA", "FREQ": "M", "MEASURE": "IRLT", "UNIT_MEASURE": "IX",
         "TIME_PERIOD": "2026-08", "OBS_VALUE": "4.2"},
    ])
    parsed = parse_oecd_long_term_rates_csv(raw.to_csv(index=False).encode("utf-8"))
    assert len(parsed) == 1
    assert parsed.iloc[0]["iso3"] == "USA"
    assert parsed.iloc[0]["period"] == "2026-08"
    assert parsed.iloc[0]["long_term_interest_rate_pct"] == 4.2


def test_weo_parser_converts_gdp_usd_billions_but_leaves_percent_series_unchanged() -> None:
    raw = pd.DataFrame([
        {"COUNTRY.ID": "USA", "COUNTRY": "United States", "INDICATOR.ID": "NGDPD", "LATEST_ACTUAL_ANNUAL_DATA": 2025,
         "SCALE": "Millions", "2026": 28_000},
        {"COUNTRY.ID": "USA", "COUNTRY": "United States", "INDICATOR.ID": "GGXCNL_NGDP", "LATEST_ACTUAL_ANNUAL_DATA": 2025,
         "SCALE": "Percent of GDP", "2026": -6.1},
        {"COUNTRY.ID": "USA", "COUNTRY": "United States", "INDICATOR.ID": "GGX_NGDP", "LATEST_ACTUAL_ANNUAL_DATA": 2025,
         "SCALE": "Percent of GDP", "2026": 37.4},
        {"COUNTRY.ID": "USA", "COUNTRY": "United States", "INDICATOR.ID": "BCA_NGDPD", "LATEST_ACTUAL_ANNUAL_DATA": 2025,
         "SCALE": "Percent of GDP", "2026": 1.2},
        {"COUNTRY.ID": "USA", "COUNTRY": "United States", "INDICATOR.ID": "NGDP_RPCH", "LATEST_ACTUAL_ANNUAL_DATA": 2025,
         "SCALE": "Percent change", "2026": 1.8},
    ])
    parsed = parse_weo_countries_frame(raw)
    gdp = parsed.loc[parsed["indicator_id"].eq("NGDPD"), "value"].iloc[0]
    fiscal = parsed.loc[parsed["indicator_id"].eq("GGXCNL_NGDP"), "value"].iloc[0]
    expenditure = parsed.loc[parsed["indicator_id"].eq("GGX_NGDP"), "value"].iloc[0]
    current_account = parsed.loc[parsed["indicator_id"].eq("BCA_NGDPD"), "value"].iloc[0]
    assert gdp == 28.0
    assert fiscal == -6.1
    assert expenditure == 37.4
    assert current_account == 1.2


def test_reference_year_advances_dynamic_growth_years_without_inventing_bis_quarter() -> None:
    imf = _imf_frame()
    next_year = imf.loc[imf["year"].eq(2026)].copy()
    next_year["year"] = 2027
    next_year["value"] = next_year.apply(
        lambda row: row["value"] + (1_000 if row["indicator_id"] == "NGDPD" else 0.1), axis=1
    )
    imf = pd.concat([imf, next_year], ignore_index=True)
    table, history, _, year, _ = build_world_economy_frames(_bis_frame(), imf, reference_year=2027)
    row = table.iloc[0]
    assert year == 2027
    assert row["bis_observation_quarter"] == "2026Q1"
    assert "real_gdp_growth_2024_pct" in table.columns
    assert "real_gdp_growth_2025_pct" in table.columns
    assert "real_gdp_growth_2026_pct" in table.columns
    assert history.loc[history["is_provisional"], "year"].empty


def test_missing_imf_latest_actual_year_is_treated_as_forecast_for_provisional_bis_country() -> None:
    bis = _bis_frame()
    bis.loc[len(bis)] = {
        "iso3": "CAN", "bis_iso2": "CA", "country_name": "Canada", "year": 2026,
        "quarter": 1, "period": "2026-Q1", "debt_gdp_pct": 90.0,
        "bis_series_id": "Q.CA.C.A.M.770.A", "observation_status": "",
    }
    imf = _imf_frame()
    imf["latest_actual_year"] = imf["latest_actual_year"].astype("Int64")

    _, history, _, _, _ = build_world_economy_frames(bis, imf, reference_year=2026)

    canada = history.loc[history["iso3"].eq("CAN") & history["is_provisional"]].iloc[0]
    assert canada["fiscal_status"] == "IMF forecast"
    assert canada["gdp_status"] == "IMF forecast"
    assert canada["real_growth_status"] == "IMF forecast"
