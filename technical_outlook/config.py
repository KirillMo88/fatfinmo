from __future__ import annotations

from typing import Any

from market_data import AssetSpec


MODEL_VERSION = "TECHNICAL_OUTLOOK_V3"
CONFIG_VERSION = "TECHNICAL_OUTLOOK_CONFIG_V3"
SR_ENGINE_VERSION = "SR_ENGINE_V1"
SCENARIO_ENGINE_VERSION = "SCENARIO_ENGINE_V1"
LEGACY_SR_ENGINE_VERSION = "SR_ENGINE_V0"
LEGACY_SCENARIO_ENGINE_VERSION = "SCENARIO_ENGINE_V0"

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
            "MAJOR": {"timeframe": "WEEKLY", "atr_multiplier": 1.75, "minimum_reversal_pct": 3.5},
            "INTERMEDIATE": {"timeframe": "DAILY", "atr_multiplier": 1.35, "minimum_reversal_pct": 1.75},
            "MINOR": {"timeframe": "DAILY", "atr_multiplier": 0.80, "minimum_reversal_pct": 0.75},
        },
    },
    "ma": {"windows": [50, 100, 200], "slope_window": 6},
    "extension_thresholds": {"low": 5.0, "normal": 15.0, "high": 30.0},
    "divergence": {"active_bars": 12, "minimum_indicator_delta": 0.25},
    "volume_profile": {"bins": 30, "value_area": 0.70, "lookback_bars": 500},
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
    "support_resistance_v1": {
        "member_weights": {
            "structural_swing": 4.0,
            "weekly_swing": 3.0,
            "daily_swing": 2.0,
            "minor_swing": 1.0,
            "poc": 3.0,
            "hvn": 2.5,
            "sma200": 2.5,
            "sma100": 2.0,
            "sma50": 1.5,
            "fibonacci": 1.75,
            "round_number": 0.5,
            "channel_boundary": 2.0,
            "lvn": 0.0,
        },
        "family_breadth_increment": 0.25,
        "family_breadth_cap": 0.50,
        "confluence_thresholds": {"medium": 2.5, "high": 5.0, "very_high": 8.0},
        "daily": {
            "base_price_fraction": 0.004,
            "atr_multiplier": 0.35,
            "radius_cap_fraction": 0.006,
            "min_width_fraction": 0.002,
            "min_width_atr_multiplier": 0.15,
            "max_total_width_fraction": 0.012,
            "minimum_touch_separation": 5,
            "reaction_window": 10,
            "recency_bars": [63, 126, 252, 504],
        },
        "weekly": {
            "base_price_fraction": 0.009,
            "atr_multiplier": 0.55,
            "radius_cap_fraction": 0.0125,
            "min_width_fraction": 0.004,
            "min_width_atr_multiplier": 0.20,
            "max_total_width_fraction": 0.025,
            "minimum_touch_separation": 3,
            "reaction_window": 4,
            "recency_bars": [13, 26, 52, 104],
        },
        "mad_multiplier": 2.0,
        "episode_exit_atr": 0.5,
        "minimum_reaction_atr": 0.5,
        "minimum_hold_episodes": 3,
        "cross_timeframe_overlap": 0.30,
        "cross_timeframe_duplicate_overlap": 0.70,
        "cross_timeframe_bonus": 1.0,
        "cross_timeframe_bonus_cap": 1.0,
    },
    "scenario_v1": {
        "component_weights": {
            "confluence": {"LOW": 0.5, "MEDIUM": 1.25, "HIGH": 2.25, "VERY_HIGH": 3.0},
            "strength": {"WEAK": 0.0, "MODERATE": 0.75, "STRONG": 1.5, "VERY_STRONG": 2.0},
        },
        "timeframe_component": {
            "SHORT": {"DAILY": 1.0, "WEEKLY": 0.5, "CROSS_TIMEFRAME": 1.0},
            "MEDIUM": {"DAILY": 0.75, "WEEKLY": 1.0, "CROSS_TIMEFRAME": 1.0},
            "6M": {"DAILY": 0.25, "WEEKLY": 1.0, "CROSS_TIMEFRAME": 1.0},
        },
        "cross_timeframe_component": 0.5,
        "horizon_bars": {
            "market": {"SHORT": 20, "MEDIUM": 63, "6M": 126},
            "crypto": {"SHORT": 28, "MEDIUM": 91, "6M": 182},
        },
        "primary_min_relevance": 5.0,
        "primary_max_reachability": 1.25,
        "extended_min_relevance": 4.5,
        "extended_max_reachability": 2.0,
        "structural_min_relevance": 4.0,
        "structural_max_reachability": 3.0,
        "break_buffer_atr": 0.10,
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
