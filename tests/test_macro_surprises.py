from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from macro_surprises import (
    build_surprise_history,
    calculate_release_surprises,
    classify_inflation_signal,
    classify_turning_signal,
    infer_cpi_definition,
    load_seed_releases,
    merge_release_updates,
    normalize_release_block,
    select_point_in_time_releases,
)


def test_seed_file_loads_complete_valid_history() -> None:
    releases, metadata = load_seed_releases()

    counts = releases.groupby("Indicator").size().to_dict()
    assert counts == {
        "CPI": 172,
        "Initial Jobless Claims": 883,
        "PMI": 209,
        "Retail Sales": 210,
    }
    assert metadata["RejectedRows"] == 36
    assert infer_cpi_definition(releases) == "CPI YoY"
    assert releases.loc[releases["Indicator"].eq("PMI"), "SeriesDefinition"].eq("ISM Manufacturing PMI").all()


def test_expanding_zscore_is_point_in_time_and_respects_minimum_history() -> None:
    dates = pd.date_range("2020-01-01", periods=30, freq="MS")
    block = pd.DataFrame(
        {
            "ReleaseDate": dates,
            "ReleaseTime": "08:30:00",
            "Actual": np.arange(30, dtype=float) + np.sin(np.arange(30)),
            "Forecast": np.arange(30, dtype=float),
            "Previous": np.arange(30, dtype=float),
        }
    )
    releases, _ = normalize_release_block(
        block,
        indicator="PMI",
        definition="ISM Manufacturing PMI",
        source_name="test",
        source_url="",
    )
    before = calculate_release_surprises(releases)
    assert before["ZSurprise"].iloc[:24].isna().all()
    assert np.isfinite(before["ZSurprise"].iloc[24])

    future = block.tail(1).copy()
    future["ReleaseDate"] = pd.Timestamp("2030-01-01")
    future["Actual"] = 999.0
    future["Forecast"] = 0.0
    future_release, _ = normalize_release_block(
        future,
        indicator="PMI",
        definition="ISM Manufacturing PMI",
        source_name="test",
        source_url="",
    )
    after = calculate_release_surprises(pd.concat([releases, future_release], ignore_index=True))
    pd.testing.assert_series_equal(before["ZSurprise"], after["ZSurprise"].iloc[:-1], check_names=False)


def test_claims_surprise_sign_is_inverted() -> None:
    dates = pd.date_range("2020-01-01", periods=60, freq="W-THU")
    block = pd.DataFrame(
        {
            "ReleaseDate": dates,
            "ReleaseTime": "08:30:00",
            "Actual": np.arange(60, dtype=float) + 200.0,
            "Forecast": np.arange(60, dtype=float) + 190.0,
            "Previous": np.arange(60, dtype=float) + 195.0,
        }
    )
    releases, _ = normalize_release_block(
        block,
        indicator="Initial Jobless Claims",
        definition="Initial Jobless Claims",
        source_name="test",
        source_url="",
    )
    calculated = calculate_release_surprises(releases)
    assert calculated["RawSurprise"].eq(-10.0).all()


def test_decay_dynamic_weights_breadth_and_context_signals() -> None:
    release_date = pd.Timestamp("2026-01-02")
    releases = pd.DataFrame(
        {
            "Indicator": ["PMI", "Retail Sales", "Initial Jobless Claims", "CPI"],
            "ReleaseDate": [release_date] * 4,
            "ZSurprise": [1.0, -1.0, 1.0, -0.5],
        }
    )
    context = pd.DataFrame(
        {
            "date": [release_date],
            "BusinessCycleDirection": ["IMPROVING"],
            "InflationState": ["FALLING"],
        }
    )
    history = build_surprise_history(releases, pd.DatetimeIndex([release_date, release_date + timedelta(days=28)]), context)

    first = history.iloc[0]
    assert first["GrowthSurpriseScore"] == 0.4
    assert first["GrowthBreadth"] == 0.7
    assert first["TurningSignal"] == "CONFIRMED IMPROVEMENT"
    assert first["CPIContribution"] == -0.5
    assert first["InflationSignal"] == "DISINFLATION CONFIRMED"

    later = history.iloc[1]
    assert np.isclose(later["PMIContribution"], 0.5)
    assert np.isclose(later["ClaimsContribution"], 0.25)
    assert np.isclose(later["GrowthSurpriseScore"], 0.15)


def test_revision_is_preserved_but_does_not_rewrite_model_history() -> None:
    block = pd.DataFrame(
        {
            "ReleaseDate": ["2026-01-02"],
            "ReleaseTime": ["08:30"],
            "Actual": [50.0],
            "Forecast": [49.0],
            "Previous": [48.0],
        }
    )
    original, _ = normalize_release_block(
        block,
        indicator="PMI",
        definition="ISM Manufacturing PMI",
        source_name="seed",
        source_url="",
    )
    changed = block.copy()
    changed["Actual"] = 51.0
    revision, _ = normalize_release_block(
        changed,
        indicator="PMI",
        definition="ISM Manufacturing PMI",
        source_name="live",
        source_url="https://example.com",
    )
    merged = merge_release_updates(original, revision)

    assert len(merged) == 2
    assert merged["RevisionDetected"].sum() == 1
    selected = select_point_in_time_releases(merged)
    assert len(selected) == 1
    assert selected.iloc[0]["Actual"] == 50.0


def test_signal_classification_matrix() -> None:
    assert classify_turning_signal("IMPROVING", -0.21) == "SLOWDOWN WARNING"
    assert classify_turning_signal("DETERIORATING", 0.21) == "POTENTIAL BOTTOMING"
    assert classify_inflation_signal("RISING", -0.21) == "DISINFLATION CHALLENGE"
    assert classify_inflation_signal("FALLING", 0.21) == "REACCELERATION CHALLENGE"
