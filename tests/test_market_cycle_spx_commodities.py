from __future__ import annotations

import numpy as np
import pandas as pd

import market_cycle
from market_cycle import build_weekly_frame, calculate_log_spx_ppiaco
from market_cycle_tab import build_multi_layer_market_cycles_fig


def _close_frame(dates: pd.DatetimeIndex, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Close": values, "Low": values}, index=dates)


def test_log_spx_ppiaco_uses_natural_log_and_preserves_missing_values() -> None:
    result = calculate_log_spx_ppiaco(
        pd.Series([100.0, 200.0, np.nan, 400.0]),
        pd.Series([50.0, 100.0, 100.0, 0.0]),
    )

    assert np.isclose(result.iloc[0], np.log(2.0))
    assert np.isclose(result.iloc[1], np.log(2.0))
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
    valid = weekly.dropna(subset=["SPX_PPIACO_Log"]).iloc[-1]
    assert np.isclose(valid["SPX_PPIACO_Log"], np.log(valid["SPX_Close"] / valid["PPIACO"]))


def test_multi_layer_figure_places_spx_commodities_cycle_after_structural_extension() -> None:
    figure = build_multi_layer_market_cycles_fig(
        pd.DataFrame(),
        pd.Timestamp("2000-01-01"),
        pd.Timestamp("2026-01-01"),
    )
    subplot_titles = [annotation.text for annotation in figure.layout.annotations]

    assert subplot_titles[-2:] == ["Structural Extension", "SPX/Commodities Cycle"]
    assert figure.layout.height == 900


def test_multi_layer_figure_plots_log_spx_ppiaco_in_fifth_panel() -> None:
    dates = pd.date_range("2024-01-31", periods=24, freq="ME")
    spx = pd.Series(np.linspace(4_000.0, 6_000.0, len(dates)))
    ppiaco = pd.Series(np.linspace(245.0, 265.0, len(dates)))
    ratio = spx / ppiaco
    history = pd.DataFrame(
        {
            "Date": dates,
            "SPX_Close": spx,
            "PPIACO": ppiaco,
            "SPX_PPIACO_Ratio": ratio,
            "SPX_PPIACO_Log": np.log(ratio),
            "PrimaryMarketCycle": np.sin(np.linspace(0.0, 4.0, len(dates))),
            "LongMarketExtensionCycle": np.cos(np.linspace(0.0, 3.0, len(dates))),
            "StructuralExtensionSmooth": np.linspace(20.0, 80.0, len(dates)),
            "StructuralExtensionPct": np.linspace(0.2, 0.8, len(dates)),
        }
    )

    figure = build_multi_layer_market_cycles_fig(history, dates.min(), dates.max())
    trace = next(item for item in figure.data if item.name == "SPX/Commodities Cycle")

    np.testing.assert_allclose(np.asarray(trace.y, dtype="float64"), np.log(ratio))
    assert trace.yaxis == "y6"
