from __future__ import annotations

import numpy as np
import pandas as pd

import market_cycle
from market_cycle import (
    _outside_frozen_breadth_window,
    build_weekly_frame,
    calculate_log_ppiaco_spx,
    load_frozen_breadth_archive,
    load_tradingview_breadth_series,
)
from market_cycle_tab import build_multi_layer_market_cycles_fig


def _close_frame(dates: pd.DatetimeIndex, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Close": values, "Low": values}, index=dates)


def test_log_ppiaco_spx_uses_natural_log_and_preserves_missing_values() -> None:
    result = calculate_log_ppiaco_spx(
        pd.Series([50.0, 100.0, 100.0, 0.0]),
        pd.Series([100.0, 200.0, np.nan, 400.0]),
    )

    assert np.isclose(result.iloc[0], np.log(0.5))
    assert np.isclose(result.iloc[1], np.log(0.5))
    assert result.iloc[2:].isna().all()


def test_weekly_market_cycle_aligns_fred_ppiaco_and_calculates_log_ratio(monkeypatch) -> None:
    monkeypatch.setattr(market_cycle, "load_spx_reference_weekly", lambda _end: pd.DataFrame())
    spx_dates = pd.date_range("2025-01-01", "2025-02-28", freq="B")
    ppi_dates = pd.DatetimeIndex(["2025-01-01", "2025-02-01"])
    raw = {
        "^GSPC": _close_frame(spx_dates, np.linspace(5_000.0, 5_200.0, len(spx_dates)).tolist()),
        "PPIACO": _close_frame(ppi_dates, [250.0, 260.0]),
    }

    weekly = build_weekly_frame(raw, pd.Timestamp("2025-02-28"))

    assert weekly["PPIACO"].notna().any()
    valid = weekly.dropna(subset=["PPIACO_SPX_Log"]).iloc[-1]
    assert np.isclose(valid["PPIACO_SPX_Log"], np.log(valid["PPIACO"] / valid["SPX_Close"]))


def test_frozen_breadth_window_filters_datetime_index_with_comparisons() -> None:
    dates = pd.DatetimeIndex(["2014-10-06", "2025-12-31", "2026-01-02"])
    frame = _close_frame(dates, [50.0, 60.0, 61.0])

    result = _outside_frozen_breadth_window(frame, "S5FI")

    assert result.index.tolist() == [pd.Timestamp("2026-01-02")]


def test_s5fi_archive_includes_supplied_2026_daily_rows(monkeypatch, tmp_path) -> None:
    archive = tmp_path / "s5fi.csv"
    archive.write_text(
        "time,open,high,low,close\n"
        "31/12/2025,60,61,59,60\n"
        "02/01/2026,61,62,60,61\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(market_cycle, "S5FI_ARCHIVE_PATH", archive)
    monkeypatch.setattr(market_cycle, "S5FI_LEGACY_ARCHIVE_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(market_cycle, "S5FI_RUNTIME_ARCHIVE_PATH", tmp_path / "runtime.csv")

    loaded = load_frozen_breadth_archive("S5FI", pd.Timestamp("2026-01-02"))

    assert loaded.index.tolist() == [pd.Timestamp("2025-12-31"), pd.Timestamp("2026-01-02")]
    assert loaded.loc[pd.Timestamp("2026-01-02"), "Close"] == 61.0


def test_s5fi_daily_loader_ignores_monthly_tradingview_and_keeps_archive(monkeypatch, tmp_path) -> None:
    archive = tmp_path / "s5fi.csv"
    archive.write_text(
        "time,open,high,low,close\n"
        "30/09/2026,28,29,27,28\n"
        "26/10/2026,59,60,58,59\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(market_cycle, "S5FI_ARCHIVE_PATH", archive)
    monkeypatch.setattr(market_cycle, "S5FI_LEGACY_ARCHIVE_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(market_cycle, "S5FI_RUNTIME_ARCHIVE_PATH", tmp_path / "runtime.csv")
    monkeypatch.setattr(market_cycle, "load_tradingview_ohlcv_frame", lambda *_args: pd.DataFrame())
    monkeypatch.setattr(
        market_cycle,
        "load_investing_breadth_series",
        lambda *_args: _close_frame(pd.DatetimeIndex(["2026-11-25"]), [59.24]),
    )

    loaded = load_tradingview_breadth_series("S5FI", pd.Timestamp("2026-11-25"))

    assert loaded.loc[pd.Timestamp("2026-09-30"), "Close"] == 28.0
    assert loaded.loc[pd.Timestamp("2026-11-25"), "Close"] == 59.24
    persisted = pd.read_csv(tmp_path / "runtime.csv")
    persisted["time"] = pd.to_datetime(persisted["time"], dayfirst=True)
    persisted_row = persisted.loc[persisted["time"] == pd.Timestamp("2026-11-25")].iloc[-1]
    assert persisted_row["close"] == 59.24


def test_multi_layer_figure_places_ppiaco_52w_change_after_structural_extension() -> None:
    figure = build_multi_layer_market_cycles_fig(
        pd.DataFrame(),
        pd.Timestamp("2000-01-01"),
        pd.Timestamp("2026-01-01"),
    )
    subplot_titles = [annotation.text for annotation in figure.layout.annotations]

    assert subplot_titles[-2:] == ["Structural Extension", "52W Change PPIACO"]
    assert figure.layout.height == 900


def test_multi_layer_figure_plots_only_ppiaco_52w_change_in_fifth_panel() -> None:
    dates = pd.date_range("2024-01-31", periods=24, freq="ME")
    spx = pd.Series(np.linspace(4_000.0, 6_000.0, len(dates)))
    ppiaco = pd.Series(np.linspace(245.0, 265.0, len(dates)))
    history = pd.DataFrame(
        {
            "Date": dates,
            "SPX_Close": spx,
            "PPIACO": ppiaco,
            "PPIACO_SPX_Ratio": ppiaco / spx,
            "PPIACO_SPX_Log": np.log(ppiaco / spx),
            "PrimaryMarketCycle": np.sin(np.linspace(0.0, 4.0, len(dates))),
            "LongMarketExtensionCycle": np.cos(np.linspace(0.0, 3.0, len(dates))),
            "StructuralExtensionSmooth": np.linspace(20.0, 80.0, len(dates)),
            "StructuralExtensionPct": np.linspace(0.2, 0.8, len(dates)),
        }
    )

    figure = build_multi_layer_market_cycles_fig(history, dates.min(), dates.max())
    trace = next(item for item in figure.data if item.name == "52W Change PPIACO")

    expected = ppiaco.pct_change(12, fill_method=None) * 100.0
    np.testing.assert_allclose(np.asarray(trace.y, dtype="float64"), expected, equal_nan=True)
    np.testing.assert_allclose(np.asarray(trace.customdata, dtype="float64"), ppiaco)
    assert trace.yaxis == "y6"
