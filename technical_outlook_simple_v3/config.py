from __future__ import annotations

from copy import deepcopy
from typing import Any

from market_data import AssetSpec


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
    "GOLD": AssetSpec(
        "GOLD", "GOLD", "tradingview_mcp", "OANDA:XAUUSD", "commodity", "USD", "USD_per_oz",
        "TRADINGVIEW_OANDA_XAUUSD", "Etc/UTC", "raw_ohlc", "1D", "TradingView MCP",
        "TradingView OANDA:XAUUSD; GOLD uses the OANDA feed with reported volume rather than the GLD ETF.",
    ),
    # Compatibility entry for existing callers. Canonical SIMPLE v3 refreshes
    # use GOLD via CORE_ASSET_KEYS and normalize GLD in the service layer.
    "GLD": AssetSpec(
        "GLD", "GLD", "yahoo_finance", "GLD", "etf", "USD", "USD_per_share",
        "XNYS", "America/New_York", "raw_ohlc", "1D", "Yahoo Finance",
        "Compatibility alias only; SIMPLE v3 no longer refreshes the GLD ETF.",
    ),
    "BTC-USD": AssetSpec(
        "BTC-USD", "BTC-USD", "yahoo_finance", "BTC-USD", "crypto", "USD", "USD_per_BTC",
        "24X7_UTC", "Etc/UTC", "raw_ohlc", "1D", "Yahoo Finance",
        "Yahoo Finance BTC-USD spot history with reported volume.",
    ),
}

CORE_ASSET_KEYS = ("SPY", "QQQ", "GOLD", "BTC-USD")

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
    "chart": {"bars": 300, "interactive": False},
    "moving_averages": {"windows": [50, 100, 200], "slope_window": 6},
    "volume_profile": {
        "weekly_lookback_bars": 300,
        "daily_lookback_bars": 300,
        "profile_bins": 24,
        "smoothing_method": "none",
        "gaussian_sigma": 0.0,
        "minimum_peak_separation_bins": 2,
        "local_peak_min_poc_fraction": 0.25,
        "max_local_peaks": 3,
        "value_area": 0.70,
    },
    "fibonacci": {
        "retracements": [0.382, 0.500, 0.618],
        "extensions": [1.272, 1.618],
    },
    "key_point_scores": {
        "swing_structure": {"one": 50.0, "two": 75.0, "three_plus": 100.0},
        "moving_average": {"sma100": 50.0, "sma200": 100.0},
        "volume_acceptance": {"poc": 100.0, "local_peak": 50.0},
        "fibonacci": {
            "strategic_0.382": 75.0,
            "strategic_0.500_0.618": 100.0,
            "tactical_0.382": 50.0,
            "tactical_0.500_0.618": 75.0,
        },
    },
    # Legacy quality weights are retained for snapshot compatibility only;
    # SIMPLE v3 zone classes use key_point_scores above.
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
            "radius_cap_fraction": 0.020,
            "min_width_fraction": 0.004,
            "min_width_atr_multiplier": 0.20,
            "max_total_width_fraction": 0.040,
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
    "display": {
        "lower_cutoff_fraction": 0.40,
        "key_point_filter_options": ["OFF", "HIGH", "HIGH + MID", "ALL"],
    },
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
    if symbol == "GOLD":
        symbol = "GLD"
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
