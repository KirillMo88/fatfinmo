from __future__ import annotations

import pandas as pd

from world_economy_tab import _trajectory_chart_data


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
