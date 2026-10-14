from __future__ import annotations

import pandas as pd
import pytest

from world_economy_tab import (
    _add_trajectory_average_separator,
    _build_sovereign_cds_figures,
    _country_trajectory_table,
    _top_gdp_cds_countries,
    _trajectory_chart_data,
)


def test_trajectory_chart_data_includes_recent_five_and_ten_year_averages() -> None:
    reference_year = 2026
    years = list(range(reference_year - 10, reference_year + 1))
    history = pd.DataFrame({
        "year": years,
        "is_provisional": False,
        "real_gdp_growth_pct": years,
        "fiscal_balance_pct": years,
        "current_account_pct": years,
        "total_debt_gdp_annual_avg_pct": years,
        "long_term_interest_rate_annual_avg_pct": years,
        "interest_on_total_debt_pct_gdp": years,
    })

    chart = _trajectory_chart_data(history, reference_year)

    assert chart["Period"].tolist() == [str(year) for year in years] + ["5Y Avg", "10Y Avg"]
    assert chart.loc[chart["Period"].eq("5Y Avg"), "real_gdp_growth_pct"].iloc[0] == 2024
    assert chart.loc[chart["Period"].eq("10Y Avg"), "real_gdp_growth_pct"].iloc[0] == 2021.5


def test_trajectory_average_separator_sits_between_years_and_averages() -> None:
    class FigureStub:
        def __init__(self) -> None:
            self.shapes: list[dict] = []

        def add_shape(self, **shape: object) -> None:
            self.shapes.append(shape)

    figure = FigureStub()
    periods = [str(year) for year in range(2016, 2027)] + ["5Y Avg", "10Y Avg"]

    _add_trajectory_average_separator(figure, pd.DataFrame({"Period": periods}))

    assert len(figure.shapes) == 1
    assert figure.shapes[0]["x0"] == 10.5
    assert figure.shapes[0]["x1"] == 10.5
    assert figure.shapes[0]["line"]["dash"] == "dot"


def test_top_gdp_cds_excludes_russia_and_replaces_it_with_next_country() -> None:
    countries = pd.DataFrame({
        "gdp_rank": list(range(1, 13)),
        "iso3": ["USA", "CHN", "DEU", "JPN", "GBR", "IND", "FRA", "ITA", "RUS", "BRA", "CAN", "MEX"],
        "country_name": [
            "United States", "China", "Germany", "Japan", "United Kingdom", "India",
            "France", "Italy", "Russian Federation", "Brazil", "Canada", "Mexico",
        ],
        "cds_5y_bps": [20, 50, 15, 30, 25, 80, 12, 10, 13800, 100, 18, 140],
    })

    selected = _top_gdp_cds_countries(countries)

    assert len(selected) == 10
    assert "RUS" not in selected["iso3"].tolist()
    assert "CAN" in selected["iso3"].tolist()
    assert selected.iloc[0]["gdp_rank"] == 11


def test_sovereign_cds_figures_share_country_order_and_show_six_month_variation() -> None:
    countries = pd.DataFrame({
        "country_name": ["Canada", "China", "United States"],
        "cds_5y_bps": [39.6, 36.41, 37.36],
        "cds_var_6m_pct": [0.0, -14.55, 0.05],
        "cds_pd_pct": [0.66, 0.61, 0.62],
    })

    level, variation = _build_sovereign_cds_figures(countries)

    expected_order = ["Canada", "China", "United States"]
    assert list(level.layout.yaxis.categoryarray) == expected_order
    assert list(variation.layout.yaxis.categoryarray) == expected_order
    assert list(variation.data[0].x) == [0.0, -14.55, 0.05]
    assert list(variation.data[0].marker.color) == ["#8b96a6", "#70927d", "#b77a62"]
    assert variation.layout.yaxis.showticklabels is False
    assert variation.layout.xaxis.range[0] == pytest.approx(-variation.layout.xaxis.range[1])


def test_country_trajectory_table_places_government_debt_rows_after_total_debt() -> None:
    history = pd.DataFrame({
        "year": [2025, 2026],
        "is_provisional": [False, False],
        "nominal_gdp_usd_bn": [27_000.0, 28_000.0],
        "nominal_gdp_growth_pct": [5.0, 3.7],
        "real_gdp_growth_pct": [2.0, 2.2],
        "total_debt_gdp_annual_avg_pct": [250.0, 251.0],
        "government_debt_gdp_pct": [108.5, 110.0],
        "government_debt_coverage_pct": [3.41, float("nan")],
        "long_term_interest_rate_annual_avg_pct": [4.2, 4.5],
        "interest_on_total_debt_pct_gdp": [10.5, 11.3],
        "dsr_annual_avg_pct": [13.8, 13.9],
        "fiscal_balance_pct": [-6.8, -7.5],
        "government_expenditure_pct": [37.7, 37.9],
        "current_account_pct": [-3.6, -3.7],
    })

    trajectory = _country_trajectory_table(history, 2026)

    labels = trajectory["Indicator"].tolist()
    total_debt_index = labels.index("Total Debt / GDP")
    assert labels[total_debt_index + 1:total_debt_index + 3] == [
        "Government Debt / GDP", "Government Debt Coverage",
    ]
    assert trajectory.loc[trajectory["Indicator"].eq("Government Debt Coverage"), "2025"].iloc[0] == "3.41"
    assert trajectory.loc[trajectory["Indicator"].eq("Government Debt Coverage"), "2026"].iloc[0] == "—"
