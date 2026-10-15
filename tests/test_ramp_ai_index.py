from __future__ import annotations

import pytest

from ramp_ai_index import (
    DATASET_KEYS,
    SEED_PATH,
    build_token_maker,
    build_token_price,
    snapshot_from_seed_workbook,
    validate_datasets,
)


def test_seed_workbook_covers_all_requested_charts() -> None:
    snapshot = snapshot_from_seed_workbook(SEED_PATH)
    assert tuple(snapshot.datasets) == DATASET_KEYS
    validate_datasets(snapshot.datasets)
    assert snapshot.latest_dates["adoption_models"] == "2026-09-01"
    assert snapshot.latest_dates["token_volume_maker"] == "2026-10-11"
    assert not snapshot.datasets["token_volume_maker"]["Date"].isna().any()


def test_token_volume_index_and_share_use_maker_aggregates() -> None:
    props = {
        "tokenVolumes": [
            {"period_end": "2026-01-04", "metric_key": "token_volume", "model_maker": "openai", "metric_value": 30},
            {"period_end": "2026-01-04", "metric_key": "token_volume", "model_maker": "openai", "metric_value": 20},
            {"period_end": "2026-01-04", "metric_key": "token_volume", "model_maker": "anthropic", "metric_value": 50},
            {"period_end": "2026-01-11", "metric_key": "token_volume", "model_maker": "anthropic", "metric_value": 100},
        ]
    }
    indexed = build_token_maker(props, "token_volume", "index").set_index("Date")
    shares = build_token_maker(props, "token_volume", "share").set_index("Date")
    assert indexed.loc["2026-01-11", "Anthropic"] == pytest.approx(100.0)
    assert indexed.loc["2026-01-04", "OpenAI"] == pytest.approx(50.0)
    assert shares.loc["2026-01-04", ["Anthropic", "OpenAI"]].sum() == pytest.approx(1.0)


def test_token_price_is_cost_per_million_tokens() -> None:
    props = {
        "tokenPrices": [
            {"usage_date": "2026-01-01", "token_type": "all_tokens", "model_maker": "openai", "token_cost_usd_7d": 2, "token_count_7d": 2_000_000},
            {"usage_date": "2026-01-01", "token_type": "all_tokens", "model_maker": "anthropic", "token_cost_usd_7d": 6, "token_count_7d": 3_000_000},
        ]
    }
    frame = build_token_price(props)
    assert frame.loc[0, "OpenAI"] == pytest.approx(1.0)
    assert frame.loc[0, "Anthropic"] == pytest.approx(2.0)
    assert frame.loc[0, "OpenAI & Anthropic"] == pytest.approx(1.6)
