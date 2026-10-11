from __future__ import annotations

import pandas as pd

from world_economy_tab import (
    _add_trajectory_average_separator,
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
