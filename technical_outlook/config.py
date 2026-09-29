from __future__ import annotations

from typing import Any

from elliott_waves.config import AssetSpec


MODEL_VERSION = "TECHNICAL_OUTLOOK_V1"
CONFIG_VERSION = "TECHNICAL_OUTLOOK_CONFIG_V1"

CORE_ASSETS: dict[str, AssetSpec] = {
    "SPY": AssetSpec(
        "SPY", "SPY", "yahoo_finance", "SPY", "etf", "USD", "USD_per_share",
        "XNYS", "America/New_York", "raw_ohlc", "1D", "Yahoo Finance",
        "SPDR S&P 500 ETF Trust; OHLCV is not a total-return series.",
    ),
    "QQQ": AssetSpec(
        "QQQ", "QQQ", "yahoo_finance", "QQQ", "etf", "USD", "USD_per_share",
        "XNAS", "America/New_York", "raw_ohlc", "1D", "Yahoo Finance",
        "Invesco QQQ Trust; OHLCV is not a total-return series.",
    ),
    "GLD": AssetSpec(
        "GLD", "GLD", "yahoo_finance", "GLD", "etf", "USD", "USD_per_share",
        "XNYS", "America/New_York", "raw_ohlc", "1D", "Yahoo Finance",
        "SPDR Gold Shares ETF; Technical Outlook intentionally analyses GLD.",
    ),
    "BTC-USD": AssetSpec(
        "BTC-USD", "BTC-USD", "yahoo_finance", "BTC-USD", "crypto", "USD", "USD_per_BTC",
        "24X7_UTC", "Etc/UTC", "raw_ohlc", "1D", "Yahoo Finance",
        "Yahoo Finance BTC-USD spot history with reported volume.",
    ),
}

CONFIG: dict[str, Any] = {
    "model_version": MODEL_VERSION,
    "config_version": CONFIG_VERSION,
    "pivot": {
        "atr_period": 14,
        "degrees": {
            "MAJOR": {"timeframe": "MONTHLY", "atr_multiplier": 2.25, "minimum_reversal_pct": 6.0},
            "INTERMEDIATE": {"timeframe": "WEEKLY", "atr_multiplier": 1.75, "minimum_reversal_pct": 3.5},
            "MINOR": {"timeframe": "WEEKLY", "atr_multiplier": 0.90, "minimum_reversal_pct": 1.5},
        },
    },
    "ma": {"windows": [50, 100, 200], "slope_window": 6},
    "extension_thresholds": {"low": 5.0, "normal": 15.0, "high": 30.0},
    "divergence": {"active_bars": 12, "minimum_indicator_delta": 0.25},
    "volume_profile": {"bins": 30, "value_area": 0.70, "lookback_weekly_bars": 156},
    "support_resistance": {
        "cluster_atr_multiplier": 0.75,
        "cluster_price_fraction": 0.012,
        "weights": {
            "major_swing": 3.0,
            "minor_swing": 2.0,
            "breakout": 3.0,
            "sma200": 2.8,
            "sma100": 2.2,
            "sma50": 1.8,
            "poc": 3.0,
            "hvn": 2.5,
            "lvn": 1.4,
            "fibonacci": 1.6,
            "round_number": 0.7,
        },
    },
    "elliott_weights": {
        "fibonacci": 0.15,
        "momentum": 0.15,
        "wave3": 0.10,
        "divergence": 0.08,
        "volume": 0.07,
        "channel": 0.08,
        "time": 0.07,
        "alternation": 0.05,
        "internal_structure": 0.15,
        "parent_consistency": 0.10,
    },
    "elliott_complexity_penalties": {
        "IMPULSE": 0.0,
        "LEADING_DIAGONAL": 2.0,
        "ENDING_DIAGONAL": 2.0,
        "ZIGZAG": 0.0,
        "FLAT": 2.0,
        "EXPANDED_FLAT": 3.0,
        "TRIANGLE": 3.0,
        "WXY": 5.0,
        "WXYXZ": 10.0,
    },
    "elliott_parent_inconsistency_penalty": 15.0,
    "elliott_confidence": {"high_min_score": 75.0, "high_gap": 15.0, "medium_min_score": 55.0, "medium_gap": 7.0},
    "elliott_search": {"max_start_pivots": 14, "max_candidates_per_degree": 80, "recency_penalty_per_bar": 1.0, "recency_penalty_cap": 12.0},
    "scenario_weights": {
        "trend": 0.25,
        "momentum": 0.20,
        "elliott": 0.15,
        "volume": 0.08,
        "extension_risk": 0.08,
        "divergence": 0.08,
        "support_resistance": 0.08,
        "historical_analog": 0.08,
    },
    "analogs": {"neighbors": 20, "minimum_sample": 8, "minimum_spacing_weeks": 13},
}


def yahoo_asset_spec(ticker: str) -> AssetSpec:
    symbol = ticker.strip().upper()
    return AssetSpec(
        canonical_asset_id=symbol,
        display_name=symbol,
        source_id="yahoo_finance",
        provider_symbol=symbol,
        instrument_type="user_requested_asset",
        currency="USD",
        price_unit="provider_units",
        session_calendar="24X7_UTC" if symbol.endswith("-USD") else "XNYS",
        source_timezone="Etc/UTC" if symbol.endswith("-USD") else "America/New_York",
        adjustment_mode="raw_ohlc",
        base_timeframe="1D",
        provider_label="Yahoo Finance",
        provenance_note="User-requested Technical Outlook asset.",
    )
