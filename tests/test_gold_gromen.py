from pathlib import Path

import numpy as np
import pandas as pd

from gold_regime.gromen import (
    HISTORICAL_US_GOLD_OZ,
    TROY_OZ_PER_TONNE,
    US_GOLD_COMPONENTS,
    build_convergence_panel,
    build_model2_history,
    build_us_gold_coverage_history,
    build_wgc_ytd_display,
    calculate_adaptive_matrix,
    calculate_imf_positive_ca,
    calculate_static_scenarios,
    calculate_world_bank_positive_ca,
    load_wgc_data,
    monetary_gold_flow_tonnes,
    prepare_wgc_annual,
    required_gold_price,
    select_calibration_year,
)


def _wgc_row(year: int = 2025) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period": str(year),
                "year": year,
                "is_complete": True,
                "total_supply_tonnes": 5_000.0,
                "total_mine_supply_tonnes": 3_600.0,
                "recycled_gold_tonnes": 1_400.0,
                "jewellery_fabrication_tonnes": 1_700.0,
                "technology_tonnes": 300.0,
                "investment_tonnes": 1_500.0,
                "central_banks_tonnes": 900.0,
                "otc_and_other_tonnes": 600.0,
                "lbma_gold_price_usd_oz": 3_000.0,
            }
        ]
    )


def test_us_gold_coverage_uses_hard_debt_switch_and_all_eight_gold_components():
    dates = pd.to_datetime(["2003-01-15", "2003-02-15", "2011-12-15", "2012-01-15", "2012-02-15"])
    gold = pd.Series([350.0, 360.0, 1_600.0, 1_700.0, 1_750.0], index=dates)
    rows = [
        ["FDHBFIN", "2002-10-01", 2.0],
        ["FORTREASPOS69995", "2003-02-01", 3_000.0],
        ["FORTREASPOS69995", "2012-01-01", 5_000.0],
        ["FORTREASPOS69995", "2012-02-01", 5_100.0],
    ]
    rows.extend([[series_id, "2012-01-01", float(index)] for index, series_id in enumerate(US_GOLD_COMPONENTS, start=1)])
    rows.extend([[series_id, "2012-02-01", float(index)] for index, series_id in enumerate(US_GOLD_COMPONENTS[:-1], start=1)])
    fred = pd.DataFrame(rows, columns=["Series_ID", "Date", "Value"])

    history = build_us_gold_coverage_history(gold, fred).set_index("date")

    assert history.loc[pd.Timestamp("2003-01-01"), "foreign_debt_source"] == "FDHBFIN"
    assert history.loc[pd.Timestamp("2003-01-01"), "foreign_debt_usd"] == 2e9
    assert history.loc[pd.Timestamp("2003-02-01"), "foreign_debt_source"] == "FORTREASPOS69995"
    assert history.loc[pd.Timestamp("2003-02-01"), "foreign_debt_usd"] == 3e9
    assert history.loc[pd.Timestamp("2011-12-01"), "us_gold_oz"] == HISTORICAL_US_GOLD_OZ
    assert history.loc[pd.Timestamp("2012-01-01"), "us_gold_oz"] == sum(range(1, 9))
    assert np.isnan(history.loc[pd.Timestamp("2012-02-01"), "us_gold_oz"])
    assert np.isnan(history.loc[pd.Timestamp("2012-02-01"), "coverage_ratio"])


def test_required_gold_price_formula():
    assert required_gold_price(10_000.0, 2.0, 0.40) == 2_000.0


def test_global_positive_ca_excludes_aggregates_and_does_not_turn_missing_into_zero():
    actual = {"AAA", "BBB", "CCC", "DDD"}
    ca = pd.DataFrame(
        {
            "country_code": ["AAA", "BBB", "CCC", "WLD"],
            "year": [2024, 2024, 2024, 2024],
            "current_account_usd": [100.0, -50.0, 0.0, 9_999.0],
        }
    )
    gdp = pd.DataFrame(
        {
            "country_code": ["AAA", "BBB", "CCC", "DDD", "WLD"],
            "year": [2024] * 5,
            "gdp_usd": [40.0, 30.0, 20.0, 10.0, 1000.0],
        }
    )

    result = calculate_world_bank_positive_ca(ca, gdp, actual, [2024]).iloc[0]

    assert result["global_positive_ca_usd"] == 100.0
    assert result["covered_countries"] == 3
    assert np.isclose(result["country_coverage"], 0.75)
    assert np.isclose(result["gdp_coverage"], 0.90)


def test_imf_gate_requires_gdp_and_prior_year_surplus_coverage_without_count_gate():
    eligible = {"AAA", "BBB", "CCC", "DDD"}
    ca = pd.DataFrame(
        {
            "country_code": ["AAA", "BBB", "CCC", "DDD", "AAA", "BBB", "CCC"],
            "year": [2023, 2023, 2023, 2023, 2024, 2024, 2024],
            "current_account_usd": [400.0, 300.0, 250.0, 50.0, 100.0, -50.0, 0.0],
            "latest_actual_year": [2023, 2023, 2023, 2023, 2023, 2024, 2024],
        }
    )
    gdp = pd.DataFrame(
        {
            "country_code": ["AAA", "BBB", "CCC", "DDD"] * 2,
            "year": [2023] * 4 + [2024] * 4,
            "gdp_usd": [400.0, 300.0, 250.0, 50.0] * 2,
        }
    )
    result = calculate_imf_positive_ca(ca, gdp, eligible, [2023, 2024], publication_year=2026)
    row = result.loc[result["year"] == 2024].iloc[0]

    assert row["global_positive_ca_usd"] == 100.0
    assert row["available_economy_count"] == 3
    assert row["total_eligible_economy_count"] == 4
    assert np.isclose(row["gdp_coverage"], 0.95)
    assert np.isclose(row["prior_year_surplus_coverage"], 0.95)
    assert row["data_status"] == "ESTIMATE"
    assert row["is_valid"]


def test_imf_prior_surplus_gate_can_fail_even_when_gdp_coverage_passes():
    ca = pd.DataFrame(
        {
            "country_code": ["AAA", "BBB", "AAA"],
            "year": [2023, 2023, 2024],
            "current_account_usd": [50.0, 950.0, 1.0],
            "latest_actual_year": [2023, 2023, 2023],
        }
    )
    gdp = pd.DataFrame(
        {
            "country_code": ["AAA", "BBB", "AAA", "BBB"],
            "year": [2023, 2023, 2024, 2024],
            "gdp_usd": [99.0, 1.0, 99.0, 1.0],
        }
    )
    row = calculate_imf_positive_ca(ca, gdp, {"AAA", "BBB"}, [2023, 2024], publication_year=2026).iloc[-1]
    assert np.isclose(row["gdp_coverage"], 0.99)
    assert np.isclose(row["prior_year_surplus_coverage"], 0.05)
    assert not row["is_valid"]


def test_calibration_year_is_latest_exact_intersection_without_year_mixing():
    wgc = pd.DataFrame({"year": [2023, 2024, 2025, 2026], "is_complete": [True, True, True, True]})
    ca = pd.DataFrame(
        {
            "year": [2023, 2024, 2025, 2026],
            "is_valid": [True, True, True, True],
            "data_status": ["ACTUAL", "ACTUAL", "ESTIMATE", "FORECAST"],
            "is_completed_calendar_year": [True, True, True, True],
        }
    )

    assert select_calibration_year(wgc, ca) == 2025

    prepared = prepare_wgc_annual(pd.concat([_wgc_row(2024), _wgc_row(2025)], ignore_index=True))
    ca_values = pd.DataFrame(
        {
            "year": [2024, 2025],
            "global_positive_ca_usd": [1e12, 9e12],
            "gdp_coverage": [0.95, 0.99],
            "prior_year_surplus_coverage": [0.96, 0.99],
            "is_valid": [True, False],
            "data_status": ["ACTUAL", "FORECAST"],
            "is_completed_calendar_year": [True, True],
        }
    )
    history = build_model2_history(prepared, ca_values)
    assert history["year"].tolist() == [2024]
    assert history.iloc[0]["global_positive_ca_usd"] == 1e12


def test_wgc_reconciliation_tolerance_and_jewellery_fabrication():
    valid = prepare_wgc_annual(_wgc_row())
    assert valid.iloc[0]["balance_gap_tonnes"] == 0.0
    assert valid.iloc[0]["balance_tolerance_tonnes"] == 1.0
    assert valid.iloc[0]["balance_valid"]

    invalid = _wgc_row()
    invalid.loc[0, "jewellery_fabrication_tonnes"] -= 1.01
    result = prepare_wgc_annual(invalid).iloc[0]
    assert np.isclose(result["balance_gap_tonnes"], 1.01)
    assert not result["balance_valid"]


def test_wgc_ytd_is_annualized_for_display_but_cannot_be_calibration_year():
    quarterly = pd.DataFrame(
        [
            {"period": "2026Q1", "year": 2026, "quarter": 1, "is_published": True},
            {"period": "2026Q2", "year": 2026, "quarter": 2, "is_published": True},
        ]
    )
    for column in (
        "total_supply_tonnes",
        "total_mine_supply_tonnes",
        "recycled_gold_tonnes",
        "jewellery_fabrication_tonnes",
        "technology_tonnes",
        "investment_tonnes",
        "central_banks_tonnes",
        "otc_and_other_tonnes",
    ):
        quarterly[column] = [100.0, 150.0]
    quarterly["lbma_gold_price_usd_oz"] = [4_000.0, 5_000.0]

    ytd = build_wgc_ytd_display(quarterly).iloc[0]
    assert ytd["total_supply_tonnes"] == 500.0
    assert ytd["published_quarters"] == 2
    assert ytd["display_only"]
    assert ytd["lbma_gold_price_usd_oz"] == 4_500.0

    wgc = pd.DataFrame({"year": [2025], "is_complete": [True]})
    ca = pd.DataFrame(
        {
            "year": [2025, 2026],
            "is_valid": [True, True],
            "data_status": ["ESTIMATE", "FORECAST"],
            "is_completed_calendar_year": [True, False],
        }
    )
    assert select_calibration_year(wgc, ca) == 2025


def test_static_and_adaptive_valuations_reconcile_to_equations():
    base = prepare_wgc_annual(_wgc_row()).iloc[0]
    positive_ca = 2e12
    static = calculate_static_scenarios(positive_ca, base["broad_flow_tonnes"])
    expected = 0.5 * positive_ca / (base["broad_flow_tonnes"] * TROY_OZ_PER_TONNE)
    assert np.isclose(static.loc[static["target_share"] == 0.5, "implied_gold_price"].iloc[0], expected)

    matrix = calculate_adaptive_matrix(positive_ca, base)
    price = float(matrix.loc[np.isclose(matrix["theta"], 0.5), 0.5].iloc[0])
    lhs = price * TROY_OZ_PER_TONNE * 0.5 * monetary_gold_flow_tonnes(price, base)
    assert np.isfinite(price)
    assert np.isclose(lhs, 0.5 * positive_ca, rtol=1e-8)

    convergence = build_convergence_panel(3_000.0, static, matrix)
    assert convergence.iloc[0]["valuation"] == "Current TradingView"
    assert len(convergence) == 9


def test_repository_wgc_dataset_contains_completed_2025_and_display_only_2026_ytd():
    annual, quarterly = load_wgc_data(Path("data") / "gold_regime")
    assert int(annual.loc[annual["is_complete"], "year"].max()) == 2025
    assert quarterly.loc[quarterly["is_published"], "period"].iloc[-1] == "2026Q2"


def test_luke_gromen_block_is_immediately_after_demand_structure():
    source = Path("gold_regime/macro2_view.py").read_text(encoding="utf-8")
    demand = source.index("    render_demand_structure(selected_range, range_end)")
    gromen = source.index("    render_luke_gromen_gold_models(luke_gromen_snapshot)")
    diagnostics = source.index("    render_macro2_diagnostics(current)")
    assert demand < gromen < diagnostics
