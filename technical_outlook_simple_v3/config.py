from __future__ import annotations

from copy import deepcopy
from typing import Any

from elliott_waves.config import AssetSpec


UI_NAME = "Technical Outlook v3"
MODEL_VERSION = "TECHNICAL_OUTLOOK_SIMPLE_V3"
CONFIG_VERSION = "TECHNICAL_OUTLOOK_SIMPLE_V3_CONFIG"
SR_ENGINE_VERSION = "SR_ENGINE_SIMPLE_V3"
SCENARIO_ENGINE_VERSION = "SCENARIO_ENGINE_SIMPLE_V3"

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
        "SPDR Gold Shares ETF; SIMPLE v3 intentionally analyses GLD.",
    ),
    "BTC-USD": AssetSpec(
        "BTC-USD", "BTC-USD", "yahoo_finance", "BTC-USD", "crypto", "USD", "USD_per_BTC",
        "24X7_UTC", "Etc/UTC", "raw_ohlc", "1D", "Yahoo Finance",
        "Yahoo Finance BTC-USD spot history with reported volume.",
    ),
}

SWING_CONFIGS: dict[str, dict[str, Any]] = {
    "SPY": {
        "source": "ASSET_SPECIFIC",
        "weekly": {"atr_multiplier": 2.75, "min_reversal_pct": 0.09},
        "daily": {"atr_multiplier": 2.25, "min_reversal_pct": 0.05},
    },
    "QQQ": {
        "source": "ASSET_SPECIFIC",
        "weekly": {"atr_multiplier": 2.50, "min_reversal_pct": 0.10},
        "daily": {"atr_multiplier": 2.25, "min_reversal_pct": 0.06},
    },
    "GLD": {
        "source": "ASSET_SPECIFIC",
        "weekly": {"atr_multiplier": 2.50, "min_reversal_pct": 0.09},
        "daily": {"atr_multiplier": 2.25, "min_reversal_pct": 0.055},
    },
    "BTC-USD": {
        "source": "ASSET_SPECIFIC",
        "calibration_start": "2020-01-01",
        "weekly": {"atr_multiplier": 2.25, "min_reversal_pct": 0.22},
        "daily": {"atr_multiplier": 2.25, "min_reversal_pct": 0.15},
    },
}

CONFIG: dict[str, Any] = {
    "model_version": MODEL_VERSION,
    "config_version": CONFIG_VERSION,
    "sr_engine_version": SR_ENGINE_VERSION,
    "scenario_engine_version": SCENARIO_ENGINE_VERSION,
    "chart": {"bars": 500, "interactive": False},
    "moving_averages": {"windows": [50, 100, 200], "slope_window": 6},
    "volume_profile": {
        "weekly_lookback_bars": 260,
        "daily_lookback_bars": 252,
        "profile_bins": 40,
        "smoothing_method": "gaussian",
        "gaussian_sigma": 1.0,
        "minimum_peak_separation_bins": 2,
        "hvn_min_prominence_poc_fraction": 0.08,
        "hvn_min_height_percentile": 0.60,
        "max_hvn": 4,
        "value_area": 0.70,
    },
    "fibonacci": {
        "retracements": [0.236, 0.382, 0.500, 0.618, 0.786],
        "extensions": [1.0, 1.272, 1.618],
        "developing_quality_multiplier": 0.75,
    },
    "family_quality": {
        "breadth_increment": 0.25,
        "breadth_cap": 0.50,
        "weights": {
            "structural_swing": 4.0,
            "weekly_swing": 3.0,
            "daily_swing": 2.0,
            "poc": 3.0,
            "hvn": 2.5,
            "fib_0.236": 1.0,
            "fib_0.382": 1.5,
            "fib_0.500": 1.25,
            "fib_0.618": 1.75,
            "fib_0.786": 1.25,
            "sma50": 1.5,
            "sma100": 2.0,
            "sma200": 2.5,
        },
    },
    "clustering": {
        "mad_multiplier": 2.0,
        "weekly": {
            "base_price_fraction": 0.009,
            "atr_multiplier": 0.55,
            "radius_cap_fraction": 0.0125,
            "min_width_fraction": 0.004,
            "min_width_atr_multiplier": 0.20,
            "max_total_width_fraction": 0.025,
        },
        "daily": {
            "base_price_fraction": 0.004,
            "atr_multiplier": 0.35,
            "radius_cap_fraction": 0.006,
            "min_width_fraction": 0.002,
            "min_width_atr_multiplier": 0.15,
            "max_total_width_fraction": 0.012,
        },
    },
    "strength": {
        "minimum_reaction_atr": 0.5,
        "minimum_hold_episodes": 3,
        "episode_exit_atr": 0.5,
        "weekly": {
            "minimum_touch_separation": 3,
            "reaction_window": 4,
            "recency_bars": [13, 26, 52, 104],
        },
        "daily": {
            "minimum_touch_separation": 5,
            "reaction_window": 10,
            "recency_bars": [63, 126, 252, 504],
        },
        "class_thresholds": {"moderate": 2.5, "strong": 5.0, "very_strong": 7.5},
    },
    "display": {"lower_cutoff_fraction": 0.40, "classes": ["HIGH", "VERY_HIGH"]},
    "scenario": {
        "weights": {
            "market_structure": 0.30,
            "momentum": 0.25,
            "ma_structure": 0.15,
            "support_resistance": 0.10,
            "volume_context": 0.10,
            "historical_analogs": 0.05,
            "extension_risk": 0.05,
        },
        "extension_risk": {
            "normal_threshold_pct": 15.0,
            "extreme_threshold_pct": 30.0,
            "max_neutral_shift_pp": 8,
            "max_opposite_shift_pp": 2,
        },
        "minimum_history_weeks": 200,
        "reachability": {"primary_atr": 8.0, "extended_atr": 16.0},
    },
    "analogs": {"neighbors": 20, "minimum_sample": 8, "minimum_spacing_weeks": 13},
    "llm": {"weekly_zone_limit": 5, "daily_zone_limit": 5, "max_chars": 60_000},
}


def swing_config(ticker: str) -> dict[str, Any]:
    symbol = str(ticker).strip().upper()
    if symbol in SWING_CONFIGS:
        return deepcopy(SWING_CONFIGS[symbol])
    value = deepcopy(SWING_CONFIGS["SPY"])
    value["source"] = "SPY_DEFAULT"
    return value


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
        provenance_note="User-requested Technical Outlook SIMPLE v3 asset; SPY swing defaults apply.",
    )
