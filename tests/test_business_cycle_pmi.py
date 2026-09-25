from __future__ import annotations

import pandas as pd

import business_cycle


def test_pmi_release_fallback_uses_release_dates() -> None:
    frame = business_cycle.load_pmi_release_fallback("2010-01-01")

    assert len(frame) >= 150
    assert frame["Series_ID"].eq("NAPM").all()
    assert frame["IsReleaseDated"].all()
    assert frame["Value"].between(25, 75).all()


def test_release_dated_pmi_is_available_in_release_week() -> None:
    source = pd.Series([48.0, 51.0], index=pd.to_datetime(["2026-08-03", "2026-09-01"]))
    calendar = pd.date_range("2026-07-31", "2026-09-11", freq="W-FRI")

    aligned = business_cycle.align_fred_series_to_weekly(
        source,
        calendar,
        "NAPM",
        release_dated=True,
    )

    assert aligned.loc[pd.Timestamp("2026-08-07")] == 48.0
    assert aligned.loc[pd.Timestamp("2026-09-04")] == 51.0


def test_business_cycle_history_builds_pmi_3mma_from_release_history() -> None:
    pmi = business_cycle.load_pmi_release_fallback("2024-01-01")
    calendar = pd.date_range("2024-01-05", "2026-09-18", freq="W-FRI")

    history = business_cycle.build_business_cycle_history(calendar, pmi)

    assert history["NAPM"].notna().sum() > 100
    assert history["ISM_3MMA"].notna().sum() > 100
    assert history["ISM_3MMA"].iloc[-1] > 0
