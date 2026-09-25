from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import rates_financial_conditions as rates_fc
from rates_financial_conditions import (
    FRED_IDS,
    MARKET_TICKERS,
    MODEL_VERSION,
    REGIMES,
    _confirmation,
    _weekly_asof,
    build_history,
    expanding_z,
    forward_return_stats,
    validation_episodes,
    momentum,
)
from rates_financial_conditions_tab import (
    CURVE_26W_EXPLANATIONS,
    build_component_chart,
    build_confirmation_chart,
    build_curve_chart,
    build_regime_map,
    build_transmission_chart,
)


def _inputs() -> tuple[pd.DatetimeIndex, dict[str, pd.Series], dict[str, pd.Series]]:
    dates = pd.date_range("2010-01-01", periods=850, freq="W-FRI")
    t = np.arange(len(dates), dtype=float)
    fred = {
        "DGS2": 1.5 + 0.004 * t + 0.35 * np.sin(t / 23),
        "DGS10": 3.0 + 0.002 * t + 0.20 * np.sin(t / 31),
        "DFII10": 0.8 + 0.003 * t + 0.15 * np.sin(t / 27),
        "FEDFUNDS": 1.0 + 0.002 * t,
        "BAMLH0A0HYM2": 4.0 + 0.3 * np.sin(t / 21),
        "BAMLC0A0CM": 1.5 + 0.12 * np.sin(t / 17),
        "NFCI": 0.2 * np.sin(t / 18),
        "ANFCI": 0.2 * np.sin(t / 19),
    }
    market = {
        "DXY": 90 + 0.01 * t + 2 * np.sin(t / 24),
        "MOVE": 100 + 8 * np.sin(t / 14),
        "SPY": 100 * (1.002 ** t),
        "QQQ": 100 * (1.003 ** t),
        "GLD": 100 * (1.001 ** t),
        "BTC": 100 * (1.005 ** t),
    }
    return dates, {key: pd.Series(value, index=dates) for key, value in fred.items()}, {key: pd.Series(value, index=dates) for key, value in market.items()}


def test_expanding_z_uses_only_past_and_current_observations() -> None:
    values = pd.Series(np.arange(200, dtype=float))
    original = expanding_z(values)
    changed_future = values.copy()
    changed_future.iloc[180:] = 1_000_000
    pd.testing.assert_series_equal(original.iloc[:180], expanding_z(changed_future).iloc[:180])
    assert original.iloc[:155].isna().all()


def test_weekly_alignment_respects_release_lags() -> None:
    calendar = pd.date_range("2026-01-02", periods=4, freq="W-FRI")
    series = pd.Series([3.0], index=[calendar[0]])
    assert pd.isna(_weekly_asof(series, calendar, "chicago").iloc[0])
    assert _weekly_asof(series, calendar, "chicago").iloc[1] == 3.0
    assert pd.isna(_weekly_asof(series, calendar, "fred_daily").iloc[0])
    assert _weekly_asof(series, calendar, "fred_daily").iloc[1] == 3.0
    released = pd.Series([3.0], index=[calendar[0] + pd.Timedelta(days=5)])
    assert pd.isna(_weekly_asof(released, calendar, "released").iloc[0])
    assert _weekly_asof(released, calendar, "released").iloc[1] == 3.0


def test_initial_release_cache_uses_publication_dates(tmp_path, monkeypatch) -> None:
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            dates = pd.date_range("1990-01-05", periods=180, freq="W-FRI")
            return {"observations": [
                {"date": str(date.date()), "realtime_start": str((date + pd.Timedelta(days=5)).date()), "value": "1.2"}
                for date in dates
            ]}

    monkeypatch.setattr(rates_fc, "FRED_INITIAL_CACHE", tmp_path / "initial.parquet")
    requests = []

    def fake_get(_url, **kwargs):
        requests.append(kwargs["params"])
        return Response()

    monkeypatch.setattr(rates_fc.httpx, "get", fake_get)
    initial = rates_fc._load_initial_releases("test", refresh=True)
    assert set(initial["Series_ID"]) == set(rates_fc.INITIAL_RELEASE_IDS)
    assert initial["Date"].min() > initial["ObservationDate"].min()
    assert len(initial) == 3 * 180
    assert len(requests) == 12
    rates_fc._load_initial_releases("test", refresh=True)
    assert len(requests) == 15


def test_core_formulas_and_independent_confirmation() -> None:
    dates, fred, market = _inputs()
    assert set(fred) == set(FRED_IDS)
    assert set(market) == set(MARKET_TICKERS)
    history = build_history(dates, fred, market)
    row = history.iloc[-1]
    assert row["RatesFinancialConditionsModelVersion"] == MODEL_VERSION
    assert row["RatesFinancialConditionsRegime"] in REGIMES
    assert row["RatesPressureScore"] == pytest.approx(0.65 * row["US2YMomentum"] + 0.35 * row["RealYieldMomentum"])
    assert row["CreditLevel"] == pytest.approx(0.70 * row["HY_OAS_Z"] + 0.30 * row["IG_OAS_Z"])
    assert row["FinancialConditionsLevel"] == pytest.approx(0.45 * row["CreditLevel"] + 0.30 * row["DXY_Level"] + 0.25 * row["MOVE_Level"])
    assert row["FinancialConditionsDirectionScore"] == pytest.approx(0.45 * row["CreditDirectionScore"] + 0.30 * row["DXYMomentum"] + 0.25 * row["MOVEMomentum"])
    assert row["US2YMomentum"] == pytest.approx(momentum(history["DGS2"]).iloc[-1])
    assert row["NFCILevel"] == pytest.approx(row["NFCI"])
    assert row["FCConfirmationStatus"] in {"3_OF_3_CONFIRMED_EASING", "3_OF_3_CONFIRMED_TIGHTENING", "PARTIAL_CONFIRMATION", "DIVERGING"}

    changed = {**fred, "NFCI": fred["NFCI"] * -3, "ANFCI": fred["ANFCI"] * -4, "FEDFUNDS": fred["FEDFUNDS"] * -2}
    other = build_history(dates, changed, market).iloc[-1]
    assert other["RatesPressureScore"] == pytest.approx(row["RatesPressureScore"])
    assert other["FinancialConditionsLevel"] == pytest.approx(row["FinancialConditionsLevel"])
    assert other["FinancialConditionsDirectionScore"] == pytest.approx(row["FinancialConditionsDirectionScore"])
    assert other["RatesFinancialConditionsRegime"] == row["RatesFinancialConditionsRegime"]


def test_confirmation_and_returns_periods() -> None:
    assert _confirmation("EASING", "TIGHTENING", "TIGHTENING") == ("1_OF_3", "DIVERGING", "LOW")
    assert _confirmation("EASING", "EASING", "EASING") == ("3_OF_3", "3_OF_3_CONFIRMED_EASING", "HIGH")
    dates, fred, market = _inputs()
    history = build_history(dates, fred, market)
    returns = forward_return_stats(history)
    assert len(returns) == 4 * 3 * 4
    assert returns["N"].ge(0).all()
    assert returns.loc[returns["Asset"].eq("BTC"), "N"].sum() > 0
    assert returns.loc[returns["Asset"].eq("BTC"), "N"].sum() < returns.loc[returns["Asset"].eq("SPY"), "N"].sum()
    assert returns["Average Adverse Excursion"].dropna().le(0).all()
    assert len(validation_episodes(history)) >= 6


def test_credit_weekly_archive_is_available_only_after_week_end(tmp_path, monkeypatch) -> None:
    import tradingview_mcp

    monday = pd.date_range("1997-01-06", periods=510, freq="W-MON")
    bars = [{"t": int(date.timestamp()), "c": 3.0} for date in monday]
    monkeypatch.setattr(tradingview_mcp, "call_tool", lambda _tool, _args: {"bars": bars})
    monkeypatch.setattr(rates_fc, "CREDIT_ARCHIVE", tmp_path / "credit.parquet")
    archive, status = rates_fc._load_credit_archive(refresh=True)
    assert status == "TRADINGVIEW_WEEKLY"
    assert set(archive["Series_ID"]) == {"BAMLH0A0HYM2", "BAMLC0A0CM"}
    assert archive["Date"].min() == monday[0] + pd.Timedelta(days=7)


def test_all_rates_charts_render_from_model_history() -> None:
    dates, fred, market = _inputs()
    frame = build_history(dates, fred, market).tail(52)
    transmission = build_transmission_chart(frame)
    assert [trace.type for trace in transmission.data] == ["scatter", "scatter", "heatmap"]
    assert transmission.layout.xaxis.matches == "x2"
    assert len(build_confirmation_chart(frame).data) == 3
    assert len(build_component_chart(frame).data) == 3
    assert len(build_regime_map(frame, 26).data) == 2
    assert [trace.type for trace in build_curve_chart(frame, 13).data] == ["scatter", "scatter", "scatter", "heatmap"]


def test_curve_26w_explanations_cover_all_curve_regimes() -> None:
    assert set(CURVE_26W_EXPLANATIONS) == {
        "BULL STEEPENING", "BULL FLATTENING", "BEAR STEEPENING", "BEAR FLATTENING"
    }
    assert all("Rates are " in explanation for explanation in CURVE_26W_EXPLANATIONS.values())


def test_export_uses_exact_completed_week_without_forward_fill(monkeypatch) -> None:
    from macro_research_export import add_rates_financial_conditions

    history = pd.DataFrame({"Date": [pd.Timestamp("2026-09-18")], **{name: [1.0] for name in rates_fc.EXPORT_FIELDS}})
    history["RatesFinancialConditionsModelVersion"] = MODEL_VERSION
    snapshot = rates_fc.RatesSnapshot(history, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"TimingConvention": "test"})
    monkeypatch.setattr("macro_research_export.read_rates_fc_snapshot", lambda: snapshot)
    dataset = pd.DataFrame({"Date": pd.to_datetime(["2026-09-18", "2026-09-25"])})
    metadata = []
    add_rates_financial_conditions(dataset, metadata, None)
    assert dataset.loc[0, "RatesPressureScore"] == 1.0
    assert pd.isna(dataset.loc[1, "RatesPressureScore"])
    assert dataset.loc[0, "RatesFinancialConditionsModelVersion"] == MODEL_VERSION
    assert len(metadata) == len(rates_fc.EXPORT_FIELDS)
