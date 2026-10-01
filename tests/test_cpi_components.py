from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import cpi_components as cpi


def sample_raw(months: int = 37) -> pd.DataFrame:
    dates = pd.date_range("2023-08-01", periods=months, freq="MS")
    rows = []
    for series_number, metadata in enumerate(cpi.CPI_SERIES.values()):
        for position, date in enumerate(dates):
            rows.append(
                {
                    "Series_ID": metadata["series_id"],
                    "Date": date,
                    "Value": 100.0 + (series_number + 1) * position * 0.35,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("horizon,months", list(cpi.CPI_HORIZONS.items()))
def test_all_horizons_use_annualized_raw_index_growth(horizon: str, months: int) -> None:
    raw = sample_raw()
    breakdown = cpi.calculate_cpi_breakdown(raw, horizon)
    row = breakdown.iloc[0]
    expected = ((row["CurrentIndex"] / row["ComparisonIndex"]) ** (12 / months) - 1) * 100
    assert row["InflationRate"] == pytest.approx(expected)
    assert row["ComparisonMonth"] == row["ReferenceMonth"] - pd.DateOffset(months=months)


def test_12m_is_standard_yoy() -> None:
    row = cpi.calculate_cpi_breakdown(sample_raw(), "12M").iloc[0]
    expected_yoy = (row["CurrentIndex"] / row["ComparisonIndex"] - 1) * 100
    assert row["InflationRate"] == pytest.approx(expected_yoy)


def test_parser_ignores_m13_and_keeps_only_monthly_rows() -> None:
    payload = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": metadata["series_id"],
                    "data": [
                        {"year": "2026", "period": "M08", "value": "123.456"},
                        {"year": "2026", "period": "M13", "value": "999"},
                    ],
                }
                for metadata in cpi.CPI_SERIES.values()
            ]
        },
    }
    raw = cpi.parse_bls_cpi_payload(payload)
    assert len(raw) == 6
    assert raw["Date"].eq(pd.Timestamp("2026-08-01")).all()
    assert raw["Value"].eq(123.456).all()


def test_latest_common_month_shifts_every_series_when_one_is_missing() -> None:
    raw = sample_raw()
    latest = raw["Date"].max()
    missing_series = next(iter(cpi.CPI_SERIES.values()))["series_id"]
    raw = raw.loc[~(raw["Series_ID"].eq(missing_series) & raw["Date"].eq(latest))]

    assert cpi.latest_common_cpi_month(raw) == latest - pd.DateOffset(months=1)
    breakdown = cpi.calculate_cpi_breakdown(raw, "12M")
    assert breakdown["ReferenceMonth"].eq(latest - pd.DateOffset(months=1)).all()
    assert breakdown["CurrentIndex"].nunique() == 6


def test_horizon_switching_uses_cached_raw_history_without_another_fetch(tmp_path) -> None:
    raw = sample_raw()
    calls = []

    def fetcher(**_kwargs):
        calls.append(1)
        return raw

    cache_path = tmp_path / "raw_monthly.csv"
    meta_path = tmp_path / "cache_meta.json"
    now = pd.Timestamp.now(tz="UTC")
    cached_raw, _ = cpi.load_cpi_raw_history(cache_path=cache_path, meta_path=meta_path, now=now, fetcher=fetcher)
    cached_raw, _ = cpi.load_cpi_raw_history(cache_path=cache_path, meta_path=meta_path, now=now + pd.Timedelta(hours=1), fetcher=fetcher)
    for horizon in cpi.CPI_HORIZONS:
        cpi.calculate_cpi_breakdown(cached_raw, horizon)
    assert len(calls) == 1


def test_negative_inflation_renders_below_zero_with_signed_bar_label() -> None:
    breakdown = cpi.calculate_cpi_breakdown(sample_raw(), "12M")
    breakdown.loc[breakdown["Category"].eq("Energy"), "InflationRate"] = -1.4
    fig = cpi.build_cpi_components_chart(breakdown, "12M")
    assert fig.data[0].text[1] == "-1.4% (-0.09 pp)"
    assert fig.layout.yaxis.range[0] < 0 < fig.layout.yaxis.range[1]


def test_component_bar_shows_weighted_headline_cpi_contribution() -> None:
    breakdown = cpi.calculate_cpi_breakdown(sample_raw(), "12M")
    breakdown.loc[breakdown["Category"].eq("Energy"), "InflationRate"] = 16.3
    fig = cpi.build_cpi_components_chart(breakdown, "12M")

    assert fig.data[0].text[1] == "16.3% (+1.04 pp)"
    assert fig.data[0].text[4:] == (f"{fig.data[0].y[4]:.1f}%", f"{fig.data[0].y[5]:.1f}%")
    assert fig.data[0].customdata[1][7] == "+1.04 pp"
    assert fig.data[0].customdata[4][7] == "Not applicable"


def test_component_order_and_fixed_2026_weights() -> None:
    breakdown = cpi.calculate_cpi_breakdown(sample_raw(), "12M")
    assert breakdown["Category"].tolist() == ["Food", "Energy", "Shelter", "All other items", "CPI", "Core CPI"]
    assert breakdown["Weight_2026"].tolist() == pytest.approx([13.698, 6.383, 35.625, 44.294, 100.0, 79.919])
    fig = cpi.build_cpi_components_chart(breakdown, "12M")
    assert list(fig.data[0].x) == [0, 1, 2, 3, 5, 6]
    assert list(fig.layout.xaxis.tickvals) == [0, 1, 2, 3, 5, 6]
    assert list(fig.layout.xaxis.ticktext) == [
        "Food<br>13.7%", "Energy<br>6.4%", "Shelter<br>35.6%", "All other items<br>44.3%", "CPI<br>100.0%", "Core CPI<br>79.9%"
    ]


def test_missing_comparison_month_is_reported_without_substituting_horizon() -> None:
    raw = sample_raw()
    reference = cpi.latest_common_cpi_month(raw)
    comparison = reference - pd.DateOffset(months=24)
    series_id = cpi.CPI_SERIES["Food"]["series_id"]
    raw = raw.loc[~(raw["Series_ID"].eq(series_id) & raw["Date"].eq(comparison))]
    with pytest.raises(cpi.CPIDataError, match="Insufficient CPI history for selected period"):
        cpi.calculate_cpi_breakdown(raw, "24M")


def test_cache_fallback_is_returned_when_bls_refresh_fails(tmp_path) -> None:
    raw = sample_raw()
    cache_path = tmp_path / "raw_monthly.csv"
    meta_path = tmp_path / "cache_meta.json"
    raw.to_csv(cache_path, index=False)
    meta_path.write_text(json.dumps({"fetched_at_utc": "2020-01-01T00:00:00+00:00"}), encoding="utf-8")

    def failing_fetcher(**_kwargs):
        raise cpi.CPIDataError("simulated BLS outage")

    cached, status = cpi.load_cpi_raw_history(
        cache_path=cache_path,
        meta_path=meta_path,
        now=pd.Timestamp("2026-10-01", tz="UTC"),
        fetcher=failing_fetcher,
    )
    assert not cached.empty
    assert status["used_cache_fallback"] is True
    assert status["stale"] is False
    assert "simulated BLS outage" in status["error"]
