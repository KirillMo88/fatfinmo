from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import funding_conditions as funding
from funding_conditions import (
    EXPORT_FIELDS, MODEL_VERSION, FundingSnapshot, build_history, calendar_technical_flag,
    classify_funding_state, expanding_robust_z, persistent_funding_flag,
)
from funding_conditions_tab import (
    build_collateral_chart, build_core_chart, build_money_market_chart,
    build_reserve_chart, build_state_chart,
)


def _released(dates: pd.DatetimeIndex, values: np.ndarray, lag: int = 1) -> pd.DataFrame:
    return pd.DataFrame({
        "ObservationDate": dates,
        "AvailableDate": dates + pd.offsets.BDay(lag),
        "Value": values,
    })


def _sources() -> tuple[dict[str, pd.DataFrame], pd.Series]:
    dates = pd.bdate_range("2018-04-02", periods=600)
    t = np.arange(len(dates))
    dff = np.full(len(dates), 2.0)
    sofr = dff + 0.25 + 0.07 * np.sin(t / 19) + 0.025 * np.sin(t / 7)
    reserves_dates = pd.date_range("2010-01-06", periods=650, freq="W-WED")
    r = np.arange(len(reserves_dates))
    gdp_dates = pd.date_range("2010-01-01", periods=55, freq="QS")
    sources = {
        "SOFR99": _released(dates, sofr),
        "DFF": _released(dates, dff),
        "IORB": _released(pd.bdate_range("2021-07-29", periods=10), np.full(10, 2.0)),
        "WRESBAL": _released(reserves_dates, 2_000_000 + 1000 * r + 20_000 * np.sin(r / 17)),
        "GDP": pd.DataFrame({"ObservationDate": gdp_dates,
                             "AvailableDate": gdp_dates + pd.Timedelta(days=120),
                             "Value": 15_000 + 80 * np.arange(len(gdp_dates))}),
    }
    move_dates = pd.bdate_range("2010-01-04", periods=3000)
    move = pd.Series(95 + 8 * np.sin(np.arange(len(move_dates)) / 30), index=move_dates)
    return sources, move


def test_robust_normalization_is_point_in_time_and_one_sided() -> None:
    values = pd.Series(0.2 + 0.04 * np.sin(np.arange(150) / 9))
    original = expanding_robust_z(values)
    changed = values.copy()
    changed.iloc[120:] = 1000
    pd.testing.assert_series_equal(original.iloc[:120], expanding_robust_z(changed).iloc[:120])
    assert original.iloc[:51].isna().all()
    assert pd.notna(original.iloc[51])
    assert original.dropna().between(-4, 4).all()


def test_calendar_flag_and_persistence_require_off_calendar_evidence() -> None:
    dates = pd.bdate_range("2026-03-23", "2026-04-07")
    flag = calendar_technical_flag(dates)
    assert flag.loc[pd.Timestamp("2026-03-27")]
    assert flag.loc[pd.Timestamp("2026-04-03")]
    assert not flag.loc[pd.Timestamp("2026-03-26")]
    stress = pd.Series([3.0] * 5, index=range(5))
    assert not persistent_funding_flag(stress, pd.Series([True] * 5)).iloc[-1]
    assert persistent_funding_flag(stress, pd.Series([True, True, False, False, False])).iloc[-1]
    moderate = pd.Series([1.4] * 15)
    assert persistent_funding_flag(moderate, pd.Series([False] * 15)).iloc[-1]


def test_funding_state_keeps_move_and_reserve_watch_separate() -> None:
    frame = pd.DataFrame({
        "PersistentFundingFlag": [False, False, True, True, False],
        "TechnicalFundingFlag": [False, True, False, False, False],
        "MoneyMarketStress": [0.2, 1.4, 2.2, 2.2, 0.2],
        "ReservePressure": [2.5, 0.2, 0.2, 1.2, 0.2],
        "ReserveVulnerability": [2.0, 0.1, 0.1, 1.6, 0.1],
        "CollateralStress": [0.1, 0.1, 0.2, 0.2, 3.0],
    })
    assert classify_funding_state(frame).tolist() == [
        "NORMAL", "TECHNICAL FUNDING PRESSURE", "PERSISTENT FUNDING PRESSURE",
        "SYSTEMIC FUNDING STRESS", "TREASURY VOLATILITY",
    ]


def test_released_sources_and_core_formula() -> None:
    sources, move = _sources()
    snapshot = build_history(sources, move)
    daily, weekly = snapshot.daily, snapshot.weekly
    first = daily.iloc[0]
    assert first["Date"] > first["ObservationDate"]
    assert daily.loc[daily["Date"].lt(pd.Timestamp("2021-07-29")), "IORB"].isna().all()
    assert daily["ReserveGDP"].dropna().between(0.05, 0.25).all()
    assert daily["MoneyMarketStress"].dropna().between(0, 4).all()
    complete = daily.dropna(subset=["FundingCore"]).iloc[-1]
    assert complete["FundingSpreadRaw"] == pytest.approx(complete["SOFR99"] - complete["DFF"])
    assert complete["FundingCore"] == pytest.approx(0.60 * complete["MoneyMarketStress"] + 0.40 * complete["ReservePressure"])
    assert complete["FundingConditionsModelVersion"] == MODEL_VERSION
    assert weekly["Date"].dt.dayofweek.eq(4).all()
    assert set(EXPORT_FIELDS).issubset(weekly.columns)
    expected_unconfirmed = (daily["MoneyMarketStress"].gt(1) & ~daily["TechnicalFundingFlag"] &
                            ~daily["PersistentFundingFlag"])
    pd.testing.assert_series_equal(daily["UnconfirmedFundingPressureFlag"], expected_unconfirmed,
                                   check_names=False)

    changed_move = move * 3
    other = build_history(sources, changed_move).daily
    pd.testing.assert_series_equal(daily["FundingCore"], other["FundingCore"])
    assert not daily["CollateralStress"].equals(other["CollateralStress"])


def test_missing_move_is_partial_without_changing_core() -> None:
    sources, move = _sources()
    full = build_history(sources, move).daily
    partial = build_history(sources, None).daily
    pd.testing.assert_series_equal(full["FundingCore"], partial["FundingCore"])
    assert partial["DataCoverage"].eq("PARTIAL DATA").any()
    assert not partial["FundingState"].eq("TREASURY VOLATILITY").any()


def test_gdp_and_reserves_are_not_available_before_release() -> None:
    sources, move = _sources()
    reserves = funding._reserve_history(sources["WRESBAL"], sources["GDP"])
    first_release = sources["GDP"].iloc[0]["AvailableDate"]
    assert reserves.loc[reserves["AvailableDate"].lt(first_release), "ReserveGDP"].isna().all()
    assert reserves.loc[reserves["AvailableDate"].ge(first_release), "ReserveGDP"].notna().any()
    delayed = {**sources, "GDP": sources["GDP"].copy()}
    delayed["GDP"]["AvailableDate"] += pd.Timedelta(days=90)
    later = funding._reserve_history(delayed["WRESBAL"], delayed["GDP"])
    check_date = first_release + pd.Timedelta(days=90)
    assert reserves.loc[reserves["AvailableDate"].ge(first_release) & reserves["AvailableDate"].lt(check_date), "ReserveGDP"].notna().any()
    assert later.loc[later["AvailableDate"].lt(first_release + pd.Timedelta(days=90)), "ReserveGDP"].isna().all()


def test_mixed_wresbal_vintage_units_do_not_create_false_reserve_shock() -> None:
    sources, move = _sources()
    expected = build_history(sources, move)
    mixed = {**sources, "WRESBAL": sources["WRESBAL"].copy()}
    mixed["WRESBAL"].loc[:400, "Value"] /= 1000
    actual = build_history(mixed, move)
    for column in ("WRESBAL", "ReserveGDP", "ReserveGDP_Z", "ReserveChange13W",
                   "ReserveDrain", "ReservePressure", "FundingCore"):
        pd.testing.assert_series_equal(actual.daily[column], expected.daily[column],
                                       check_names=False, rtol=1e-10, atol=1e-10)
    pd.testing.assert_series_equal(actual.daily["FundingState"], expected.daily["FundingState"])


def test_asof_handles_multiple_observations_released_on_one_date() -> None:
    source = pd.DataFrame({"ObservationDate": [pd.Timestamp("2026-01-01")],
                           "AvailableDate": [pd.Timestamp("2026-01-30")], "Value": [100.0]})
    dates = pd.DatetimeIndex(["2026-01-29", "2026-01-30", "2026-01-30"])
    result = funding._asof(source, dates)
    assert pd.isna(result.iloc[0])
    assert result.iloc[1:].eq(100.0).all()


def test_initial_release_loader_uses_realtime_start(tmp_path, monkeypatch) -> None:
    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"observations": [{"date": "2026-01-01", "realtime_start": "2026-01-29", "value": "31000"}]}

    monkeypatch.setattr(funding, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(funding, "get_fred_api_key", lambda _key: "test")
    monkeypatch.setattr(funding.httpx, "get", lambda *_args, **_kwargs: Response())
    frame, status = funding._load_initial_release("GDP", None)
    assert status == "FRED_INITIAL_RELEASE"
    assert frame.iloc[0]["AvailableDate"] == pd.Timestamp("2026-01-29")


def test_iorb_fallback_starts_in_2021_and_lags_observation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(funding, "STORAGE_DIR", tmp_path)
    source = pd.DataFrame({"Series_ID": ["IORB", "IORB"],
                           "Date": pd.to_datetime(["2021-07-28", "2021-07-29"]), "Value": [0.1, 0.15]})
    monkeypatch.setattr(funding, "download_fred_series", lambda *_args, **_kwargs: source)
    frame, status = funding._load_iorb_fallback(None)
    assert status == "FRED_CURRENT_VINTAGE_1BD_LAG"
    assert frame["ObservationDate"].tolist() == [pd.Timestamp("2021-07-29")]
    assert frame.iloc[0]["AvailableDate"] == pd.Timestamp("2021-07-30")


def test_all_funding_charts_render() -> None:
    sources, move = _sources()
    snapshot = build_history(sources, move)
    daily, weekly = snapshot.daily.tail(40), snapshot.weekly.tail(8)
    assert [trace.type for trace in build_core_chart(weekly).data] == ["scatter", "scatter", "scatter", "heatmap"]
    assert len(build_money_market_chart(daily).data) == 4
    assert len(build_reserve_chart(weekly).data) == 5
    assert len(build_collateral_chart(weekly).data) == 3
    assert build_state_chart(weekly).data[0].type == "heatmap"


def test_funding_hover_shows_only_the_hovered_series() -> None:
    sources, move = _sources()
    snapshot = build_history(sources, move)
    charts = (
        build_core_chart(snapshot.weekly.tail(8)),
        build_money_market_chart(snapshot.daily.tail(40)),
        build_reserve_chart(snapshot.weekly.tail(8)),
        build_collateral_chart(snapshot.weekly.tail(8)),
        build_state_chart(snapshot.weekly.tail(8)),
    )
    for chart in charts:
        assert chart.layout.hovermode == "closest"
        for trace in chart.data:
            if trace.type == "scatter":
                assert "%{y:" in trace.hovertemplate
                assert trace.name in trace.hovertemplate
                assert trace.text is None


def test_export_maps_exact_week_without_future_fill(monkeypatch) -> None:
    from macro_research_export import add_funding_conditions

    history = pd.DataFrame({"Date": [pd.Timestamp("2026-09-18")], **{name: [1.0] for name in EXPORT_FIELDS}})
    history["FundingConditionsModelVersion"] = MODEL_VERSION
    snapshot = FundingSnapshot(pd.DataFrame(), history, {"TimingConvention": "test"})
    monkeypatch.setattr("macro_research_export.read_funding_snapshot", lambda: snapshot)
    dataset = pd.DataFrame({"Date": pd.to_datetime(["2026-09-18", "2026-09-25"])})
    metadata = []
    add_funding_conditions(dataset, metadata, None)
    assert dataset.loc[0, "FundingCore"] == 1.0
    assert pd.isna(dataset.loc[1, "FundingCore"])
    assert dataset.loc[0, "FundingConditionsModelVersion"] == MODEL_VERSION
    assert len(metadata) == len(EXPORT_FIELDS)
