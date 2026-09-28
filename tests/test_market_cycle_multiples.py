from __future__ import annotations

import pandas as pd

from market_cycle_multiples import (
    DERIVED_MULTPL_METRICS,
    MULTPL_METRICS,
    MULTPL_METRIC_GROUPS,
    calculate_earnings_growth_12m,
    calculate_sp500_pe_15y_percentile,
    clean_sp500_pe_history,
    load_multpl_metrics,
    multiples_range_bounds,
    parse_multpl_table,
)
from market_cycle_tab import (
    build_multiple_metric_fig,
    read_persistent_snapshot_cache,
    spy_macro_source_fingerprint,
    write_persistent_snapshot_cache,
)


def test_multpl_catalog_contains_requested_source_indicators_and_three_display_rows() -> None:
    keys = {metric.key for metric in MULTPL_METRICS}
    display_keys = keys | {metric.key for metric in DERIVED_MULTPL_METRICS}

    assert len(keys) == 8
    assert "real_earnings_growth" not in keys
    assert "dividend_yield" not in keys
    assert "earnings" in keys
    assert len(MULTPL_METRIC_GROUPS) == 3
    assert [len(group) for group in MULTPL_METRIC_GROUPS] == [4, 3, 3]
    assert {key for group in MULTPL_METRIC_GROUPS for key in group} == display_keys
    assert MULTPL_METRIC_GROUPS[0] == ("sp500_ps", "sp500_pe", "sp500_pe_15y_percentile", "shiller_pe")
    assert MULTPL_METRIC_GROUPS[1] == ("earnings", "earnings_growth_12m", "earnings_yield")


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


def test_persistent_snapshot_cache_round_trips_and_checks_identity(tmp_path) -> None:
    cache_path = tmp_path / "market-cycle.pkl"
    expected = {"latest": pd.Timestamp("2026-09-28"), "values": [1, 2, 3]}

    write_persistent_snapshot_cache(cache_path, "schema-1", "source-a", expected)

    assert read_persistent_snapshot_cache(cache_path, "schema-1", "source-a") == expected
    assert read_persistent_snapshot_cache(cache_path, "schema-2", "source-a") is None
    assert read_persistent_snapshot_cache(cache_path, "schema-1", "source-b") is None


def test_spy_macro_cache_fingerprint_tracks_latest_spx_input() -> None:
    base = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-09-21", "2026-09-28"]),
            "SPX_Close": [6500.0, 6600.0],
        }
    )

    first = spy_macro_source_fingerprint(base)
    changed = base.assign(SPX_Close=[6500.0, 6601.0])

    assert first != spy_macro_source_fingerprint(changed)


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


def test_sp500_pe_excludes_jan_through_sep_2009_only() -> None:
    dates = pd.to_datetime(["2008-12-01", "2009-01-01", "2009-09-01", "2009-10-01"])
    frame = pd.DataFrame({"Date": dates, "Value": [15.0, 16.0, 17.0, 18.0], "Estimate": False})

    cleaned = clean_sp500_pe_history(frame)

    assert cleaned["Date"].tolist() == [pd.Timestamp("2008-12-01"), pd.Timestamp("2009-10-01")]


def test_sp500_pe_15y_percentile_uses_full_trailing_fifteen_year_window() -> None:
    dates = pd.date_range("2000-01-01", "2016-01-01", freq="MS")
    pe = pd.DataFrame({"Date": dates, "Value": range(1, len(dates) + 1), "Estimate": False})

    percentile = calculate_sp500_pe_15y_percentile(pe)

    assert percentile["Date"].min() == pd.Timestamp("2015-01-01")
    assert len(percentile) == 13
    assert percentile.iloc[0]["Value"] == 100.0
    assert percentile.iloc[-1]["Value"] == 100.0


def test_earnings_growth_is_12_month_roc_matched_by_calendar_month() -> None:
    dates = pd.date_range("2020-01-31", periods=27, freq="ME").delete(5)
    values = [100.0 + i for i in range(27) if i != 5]
    earnings = pd.DataFrame({"Date": dates, "Value": values, "Estimate": False})
    earnings.loc[earnings["Date"].eq(pd.Timestamp("2021-01-31")), "Value"] = 110.0
    earnings.loc[earnings["Date"].eq(pd.Timestamp("2022-01-31")), "Value"] = 121.0
    earnings.loc[earnings["Date"].eq(pd.Timestamp("2022-01-31")), "Estimate"] = True

    growth = calculate_earnings_growth_12m(earnings)

    jan_2022 = growth.loc[growth["Date"].eq(pd.Timestamp("2022-01-31"))].iloc[0]
    assert abs(jan_2022["Value"] - 10.0) < 1e-10
    assert bool(jan_2022["Estimate"])
    assert not growth["Date"].eq(pd.Timestamp("2021-06-30")).any()


def test_multpl_loader_includes_derived_earnings_growth(monkeypatch) -> None:
    earnings = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=13, freq="ME"),
            "Value": [100.0] * 12 + [110.0],
            "Estimate": [False] * 13,
        }
    )

    def fake_fetch(metric):
        if metric.key == "earnings":
            return earnings
        return pd.DataFrame({"Date": pd.to_datetime(["2020-01-31"]), "Value": [1.0], "Estimate": [False]})

    monkeypatch.setattr("market_cycle_multiples.fetch_multpl_metric", fake_fetch)
    data, errors = load_multpl_metrics()

    assert not errors
    assert "earnings_growth_12m" in data
    assert abs(data["earnings_growth_12m"].iloc[-1]["Value"] - 10.0) < 1e-10
