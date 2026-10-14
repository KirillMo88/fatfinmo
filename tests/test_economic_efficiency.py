from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from economic_efficiency import (
    build_country_history,
    build_efficiency_table,
    calculate_efficiency_history,
    eligible_countries,
    resolve_reference_year,
)


CODES = ["USA", "CHN", "DEU", "JPN", "IND", "GBR", "FRA", "ITA", "BRA", "CAN", "MEX"]


def _pwt() -> pd.DataFrame:
    names = {
        "USA": "United States", "CHN": "China", "DEU": "Germany", "JPN": "Japan",
        "IND": "India", "GBR": "United Kingdom", "FRA": "France", "ITA": "Italy",
        "BRA": "Brazil", "CAN": "Canada", "MEX": "Mexico",
    }
    rows = []
    for rank, code in enumerate(CODES, start=1):
        for year in (2013, 2023):
            elapsed = 0 if year == 2013 else 1
            rows.append({
                "countrycode": code,
                "country": names[code],
                "year": year,
                "ctfp": 1.05 - rank * 0.03 + elapsed * 0.01,
                "rtfpna": (1.0 + rank * 0.01) * (1.10 + rank * 0.005) ** elapsed,
                "cgdpo": (20_000 - rank * 500) * (1.4 ** elapsed),
                "cn": (55_000 - rank * 900) * (1.25 ** elapsed),
                "rgdpna": (18_000 - rank * 450) * (1.30 ** elapsed),
                "rnna": (50_000 - rank * 800) * (1.15 ** elapsed),
                "emp": (100 - rank) * (1.05 ** elapsed),
                "avh": 1_800 - rank * 5,
            })
    return pd.DataFrame(rows)


def _gdp() -> pd.DataFrame:
    rows = []
    for year in (2013, 2023):
        for rank, code in enumerate(CODES, start=1):
            rows.append({
                "iso3": code,
                "country_name": code,
                "year": year,
                "nominal_gdp_usd": (30_000 - rank * 1_000) * 1e9,
                "is_country": True,
            })
    return pd.DataFrame(rows)


def _services() -> pd.DataFrame:
    return pd.DataFrame([
        {"iso3": code, "year": 2022 if code == "USA" else 2023, "services_gdp_pct": 60 + rank}
        for rank, code in enumerate(CODES)
    ])


def test_efficiency_formulas_use_correct_pwt_stock_concepts() -> None:
    history = calculate_efficiency_history(_pwt())
    current = history.loc[history["iso3"].eq("USA") & history["year"].eq(2023)].iloc[0]
    base = history.loc[history["iso3"].eq("USA") & history["year"].eq(2013)].iloc[0]

    assert current["tfp_level"] == current["ctfp"] * 100
    assert np.isclose(current["tfp_growth_10y_pct"], (current["rtfpna"] / base["rtfpna"] - 1) * 100)
    assert np.isclose(current["capital_productivity_current_pct"], current["cgdpo"] / current["cn"] * 100)
    assert np.isclose(current["capital_productivity_real_pct"], current["rgdpna"] / current["rnna"] * 100)
    assert np.isclose(
        current["capital_productivity_change_10y_pct"],
        ((current["rgdpna"] / current["rnna"]) / (base["rgdpna"] / base["rnna"]) - 1) * 100,
    )
    assert np.isclose(current["gdp_per_hour_ppp"], current["cgdpo"] / (current["emp"] * current["avh"]))
    assert np.isclose(
        current["gdp_per_hour_change_10y_pct"],
        ((current["rgdpna"] / (current["emp"] * current["avh"])) /
         (base["rgdpna"] / (base["emp"] * base["avh"])) - 1) * 100,
    )


def test_ten_year_metrics_require_exact_endpoint() -> None:
    pwt = _pwt()
    pwt = pwt.loc[~(pwt["countrycode"].eq("CHN") & pwt["year"].eq(2013))]
    history = calculate_efficiency_history(pwt)
    china = history.loc[history["iso3"].eq("CHN") & history["year"].eq(2023)].iloc[0]
    assert pd.isna(china["tfp_growth_10y_pct"])
    assert pd.isna(china["capital_productivity_change_10y_pct"])
    assert pd.isna(china["gdp_per_hour_change_10y_pct"])


def test_reference_year_and_selector_require_complete_pwt_data() -> None:
    assert resolve_reference_year(_pwt(), _gdp()) == 2023
    countries = eligible_countries(_pwt(), 2023)
    assert {"USA", "DEU", "CHN"}.issubset(set(countries["iso3"]))
    incomplete = _pwt()
    incomplete.loc[incomplete["countrycode"].eq("MEX") & incomplete["year"].eq(2023), "avh"] = np.nan
    assert "MEX" not in set(eligible_countries(incomplete, 2023)["iso3"])


def test_selected_country_is_appended_and_services_year_is_explicit() -> None:
    table = build_efficiency_table(
        _pwt(), _services(), _gdp(), reference_year=2023, selected_iso3="MEX"
    )
    assert len(table) == 11
    assert table.iloc[-1]["iso3"] == "MEX"
    assert table.iloc[-1]["position"] == "Selected"
    assert bool(table.iloc[-1]["is_selected"])

    usa = table.loc[table["iso3"].eq("USA")].iloc[0]
    assert usa["services_year"] == 2022
    assert bool(usa["services_is_older"])


def test_top10_selection_is_not_duplicated() -> None:
    table = build_efficiency_table(
        _pwt(), _services(), _gdp(), reference_year=2023, selected_iso3="DEU"
    )
    assert len(table) == 10
    assert int(table["iso3"].eq("DEU").sum()) == 1
    assert bool(table.loc[table["iso3"].eq("DEU"), "is_selected"].iloc[0])


def test_country_history_preserves_real_series_and_original_services_dates() -> None:
    history = build_country_history(
        _pwt(), _services(), _gdp(), iso3="USA", reference_year=2023, observations=30
    )
    assert history["year"].tolist() == [2013, 2023]
    assert history.loc[history["year"].eq(2023), "services_gdp_pct"].isna().all()
    assert history.loc[history["year"].eq(2013), "capital_productivity_real_pct"].notna().all()


def test_module_is_rendered_immediately_before_world_trade() -> None:
    source = Path("world_economy_tab.py").read_text(encoding="utf-8")
    efficiency_call = source.index("render_economic_efficiency_module(efficiency_snapshot, gdp_snapshot)")
    trade_call = source.index("render_world_trade_module(trade_snapshot, macro_snapshot=snapshot)")
    assert efficiency_call < trade_call
