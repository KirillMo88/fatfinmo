from __future__ import annotations

import pandas as pd

from market_cycle_multiples import (
    MULTPL_METRICS,
    MULTPL_METRIC_GROUPS,
    multiples_range_bounds,
    parse_multpl_table,
)
from market_cycle_tab import build_multiple_metric_fig


def test_multpl_catalog_contains_requested_nine_indicators_in_three_rows() -> None:
    keys = {metric.key for metric in MULTPL_METRICS}

    assert len(keys) == 9
    assert len(MULTPL_METRIC_GROUPS) == 3
    assert all(len(group) == 3 for group in MULTPL_METRIC_GROUPS)
    assert {key for group in MULTPL_METRIC_GROUPS for key in group} == keys


def test_parse_multpl_table_keeps_estimate_flag_and_parses_percent_values() -> None:
    page = """
    <table>
      <thead><tr><th>Date</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Sep 25, 2026</td><td>† (Estimate) 3.82%</td></tr>
        <tr><td>Jun 30, 2026</td><td>3.50%</td></tr>
      </tbody>
    </table>
    """

    result = parse_multpl_table(page)

    assert result["Date"].is_monotonic_increasing
    assert result["Value"].tolist() == [3.5, 3.82]
    assert result["Estimate"].tolist() == [False, True]


def test_multiples_range_is_shared_and_max_has_no_forced_start() -> None:
    metrics = {
        "monthly": pd.DataFrame(
            {"Date": pd.to_datetime(["2014-06-01", "2026-06-01"]), "Value": [1.0, 2.0]}
        ),
        "annual": pd.DataFrame(
            {"Date": pd.to_datetime(["2010-01-01", "2025-01-01"]), "Value": [3.0, 4.0]}
        ),
    }

    start_10y, end = multiples_range_bounds(metrics, "10Y")
    start_20y, _ = multiples_range_bounds(metrics, "20Y")
    start_max, max_end = multiples_range_bounds(metrics, "MAX")

    assert start_10y == pd.Timestamp("2016-06-01")
    assert start_20y == pd.Timestamp("2006-06-01")
    assert end == max_end == pd.Timestamp("2026-06-01")
    assert start_max is None


def test_multiple_chart_shows_reported_and_estimated_series_with_range() -> None:
    metric = MULTPL_METRICS[0]
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2025-01-01", "2026-01-01"]),
            "Value": [1.5, 2.5, 3.0],
            "Estimate": [False, False, True],
        }
    )

    fig = build_multiple_metric_fig(metric, data, pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31"))

    assert list(fig.data[0].y) == [2.5]
    assert fig.data[0].customdata.tolist() == ["Reported"]
    assert fig.layout.yaxis.ticksuffix == "x"
    assert list(fig.layout.xaxis.range) == [pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31")]
