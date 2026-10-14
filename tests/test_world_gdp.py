from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from world_gdp import WorldGDPService
from world_gdp_tab import (
    _build_top10_share_change_figure,
    _focused_additive_history,
    _format_change,
    _format_number,
    _gdp_display_scale,
    _render_per_capita_chart,
    _top10_display_table,
    _top10_world_share_change,
)


def _frame(years: list[int], country_count: int = 12) -> pd.DataFrame:
    rows = []
    for year in years:
        year_offset = year - min(years)
        country_values = []
        for index in range(country_count):
            nominal = float((country_count - index) * 1e12 * (1 + year_offset / 100))
            ppp = float((index + 2) * 8e11 * (1 + year_offset / 120))
            per_capita = float(10_000 + index * 1_000 + year_offset * 100)
            country_values.append((nominal, ppp, per_capita))
            rows.append({
                "iso3": f"C{index:02d}",
                "country_name": f"Country {index:02d}",
                "region_name": "Test",
                "is_country": True,
                "year": year,
                "nominal_gdp_usd": nominal,
                "gdp_ppp_intl_usd": ppp,
                "gdp_per_capita_usd": per_capita,
            })
        rows.append({
            "iso3": "WLD",
            "country_name": "World",
            "region_name": "",
            "is_country": False,
            "year": year,
            "nominal_gdp_usd": sum(value[0] for value in country_values) * 1.1,
            "gdp_ppp_intl_usd": sum(value[1] for value in country_values) * 1.1,
            "gdp_per_capita_usd": 12_500 + year_offset * 80,
        })
    return pd.DataFrame(rows)


def test_top10_uses_nominal_rank_and_exact_ten_year_change() -> None:
    frame = _frame([2014, 2024])
    frame.loc[
        frame["iso3"].eq("C03") & frame["year"].eq(2014), "gdp_per_capita_usd"
    ] = np.nan

    table = WorldGDPService.build_top10_table(frame, reference_year=2024)

    assert len(table) == 10
    assert table["iso3"].tolist() == [f"C{index:02d}" for index in range(10)]
    expected = (1.1 - 1) * 100
    assert table.iloc[0]["nominal_gdp_usd_change_10y_pct"] == pytest.approx(expected)
    assert pd.isna(table.loc[table["iso3"].eq("C03"), "gdp_per_capita_usd_change_10y_pct"]).all()


def test_dynamic_top10_and_selected_country_reconcile_without_double_counting() -> None:
    frame = _frame([2000, 2010])
    # Reverse the nominal ranking in 2010 so annual membership changes.
    mask = frame["year"].eq(2010) & frame["is_country"]
    frame.loc[mask, "nominal_gdp_usd"] = sorted(frame.loc[mask, "nominal_gdp_usd"])
    frame.loc[frame["year"].eq(2010) & frame["iso3"].eq("WLD"), "nominal_gdp_usd"] = (
        frame.loc[mask, "nominal_gdp_usd"].sum() * 1.1
    )

    history = WorldGDPService.build_additive_history(
        frame, metric="nominal", reference_year=2010, selected_iso3="C11"
    )

    year_2000 = history.loc[history["year"].eq(2000)]
    year_2010 = history.loc[history["year"].eq(2010)]
    assert year_2000.loc[year_2000["iso3"].eq("C11"), "selected_outside_top10"].item()
    assert len(year_2010.loc[year_2010["iso3"].eq("C11")]) == 1
    assert year_2000["share_pct"].sum() == pytest.approx(100.0)
    assert year_2010["value"].sum() == pytest.approx(year_2010["world_value"].iloc[0])


def test_country_focus_additive_view_contains_only_the_selected_country() -> None:
    history = WorldGDPService.build_additive_history(
        _frame([2000, 2010]), metric="nominal", reference_year=2010, selected_iso3="C11"
    )

    focused = _focused_additive_history(history, "c11")

    assert set(focused["iso3"]) == {"C11"}
    assert focused["year"].tolist() == [2000, 2010]
    assert focused["country_name"].eq("Country 11").all()


def test_country_gdp_scale_uses_billions_below_one_trillion() -> None:
    assert _gdp_display_scale(pd.Series([18e9, 22e9])) == (1e9, "B")
    assert _gdp_display_scale(pd.Series([900e9, 1.2e12])) == (1e12, "T")


def test_ppp_history_never_creates_pre_1990_observations() -> None:
    history = WorldGDPService.build_additive_history(
        _frame([1980, 1990, 2000]), metric="ppp", reference_year=2000
    )

    assert history["year"].min() == 1990


def test_material_negative_other_residual_is_rejected() -> None:
    frame = _frame([2024])
    frame.loc[frame["iso3"].eq("WLD"), "nominal_gdp_usd"] = 1.0

    with pytest.raises(ValueError, match="exceed the WLD aggregate materially"):
        WorldGDPService.build_additive_history(frame, metric="nominal", reference_year=2024)


def test_per_capita_selected_view_contains_only_country_and_world() -> None:
    history = WorldGDPService.build_per_capita_history(
        _frame([1960, 1970, 2024]), reference_year=2024, selected_iso3="C11"
    )

    assert set(history["iso3"]) == {"C11", "WLD"}
    assert (history.loc[history["iso3"].eq("WLD"), "relative_world_pct"] == 100.0).all()


def test_per_capita_hover_shows_only_the_closest_series(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr("world_gdp_tab.st.markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "world_gdp_tab.st.plotly_chart",
        lambda figure, **kwargs: captured.setdefault("figure", figure),
    )
    history = WorldGDPService.build_per_capita_history(
        _frame([1960, 1970, 2024]), reference_year=2024
    )

    _render_per_capita_chart(history, display_mode="Absolute", selected_iso3=None)

    assert captured["figure"].layout.hovermode == "closest"


def test_table_value_formatting_uses_units_signs_and_na() -> None:
    assert _format_number(30.7697, "${:,.2f}T") == "$30.77T"
    assert _format_number(41.259824, "${:,.2f}T") == "$41.26T"
    assert _format_number(np.nan, "${:,.2f}T") == "N/A"
    assert _format_change(72.44) == "+72.4%"
    assert _format_change(-13.16) == "−13.2%"
    assert _format_change(np.nan) == "N/A"


def test_top10_display_groups_all_change_columns_on_the_right() -> None:
    table = WorldGDPService.build_top10_table(_frame([2014, 2024]), reference_year=2024)

    display = _top10_display_table(table)

    assert display.columns.tolist() == [
        "Rank",
        "Country",
        "Nominal GDP, $T",
        "GDP PPP, Intl$T",
        "GDP per capita, $",
        "Nominal GDP 10Y Change",
        "GDP PPP 10Y Change",
        "GDP per capita 10Y Change",
    ]


def test_top10_world_share_change_uses_the_current_top10_and_exact_20_year_window() -> None:
    frame = _frame([2004, 2024])
    current_country = frame["year"].eq(2024) & frame["iso3"].eq("C00")
    frame.loc[current_country, "nominal_gdp_usd"] *= 1.5
    current_countries = frame["year"].eq(2024) & frame["is_country"]
    frame.loc[frame["year"].eq(2024) & frame["iso3"].eq("WLD"), "nominal_gdp_usd"] = (
        frame.loc[current_countries, "nominal_gdp_usd"].sum() * 1.1
    )
    table = WorldGDPService.build_top10_table(frame, reference_year=2024)

    change = _top10_world_share_change(frame, table, 2024)

    assert len(change) == 10
    assert set(change["iso3"]) == set(table["iso3"])
    c00 = change.loc[change["iso3"].eq("C00")].iloc[0]
    assert c00["current_share_pct"] > c00["base_share_pct"]
    assert c00["share_change_pp"] == pytest.approx(
        c00["current_share_pct"] - c00["base_share_pct"]
    )
    figure = _build_top10_share_change_figure(change, 2024)
    assert figure.layout.hovermode == "closest"
    assert list(figure.data[0].y) == change["country_name"].tolist()


def test_latest_common_year_applies_150_economy_coverage() -> None:
    complete_year = datetime.now(timezone.utc).year - 2
    newer_year = complete_year + 1
    frame = _frame([complete_year, newer_year], country_count=151)
    missing_codes = [f"C{index:02d}" for index in range(2)]
    frame.loc[
        frame["year"].eq(newer_year) & frame["iso3"].isin(missing_codes), "gdp_ppp_intl_usd"
    ] = np.nan

    year, coverage = WorldGDPService.latest_common_year(frame, return_coverage=True)

    assert year == complete_year
    assert coverage["gdp_ppp_intl_usd"] == 151


def test_failed_refresh_keeps_last_valid_version(tmp_path, monkeypatch) -> None:
    service = WorldGDPService(tmp_path)
    frame = _frame([1960, 1990, 2014, 2024], country_count=151)
    metadata = {"reference_year": 2024, "coverage": {}, "validation_status": "valid"}
    monkeypatch.setattr(service, "_download", lambda: (frame, metadata, {"countries": []}))
    first = service.load(force_refresh=True)
    assert first["status"] == "current"

    def fail():
        raise RuntimeError("source unavailable")

    monkeypatch.setattr(service, "_download", fail)
    fallback = service.load(force_refresh=True)

    assert fallback["status"] == "cached"
    assert "source unavailable" in fallback["warning"]
    columns = ["iso3", "year", "is_country", *WorldGDPService.INDICATORS]
    pd.testing.assert_frame_equal(
        fallback["data"][columns].sort_values(["year", "iso3"]).reset_index(drop=True),
        WorldGDPService._coerce_frame(frame)[columns].sort_values(["year", "iso3"]).reset_index(drop=True),
        check_dtype=False,
    )
