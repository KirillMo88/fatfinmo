from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ENGINE_VERSION = "ELLIOTT_WAVE_ENGINE_V2"
RULE_PROFILE = "CLASSIC_ARITHMETIC_V2"


@dataclass(frozen=True)
class AssetSpec:
    canonical_asset_id: str
    display_name: str
    source_id: str
    provider_symbol: str
    instrument_type: str
    currency: str
    price_unit: str
    session_calendar: str
    source_timezone: str
    adjustment_mode: str
    base_timeframe: str
    provider_label: str
    provenance_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ASSET_SPECS: dict[str, AssetSpec] = {
    "SPX": AssetSpec(
        canonical_asset_id="SPX",
        display_name="SPX",
        source_id="yahoo_finance",
        provider_symbol="^GSPC",
        instrument_type="price_index",
        currency="USD",
        price_unit="index_points",
        session_calendar="XNYS",
        source_timezone="America/New_York",
        adjustment_mode="raw_ohlc",
        base_timeframe="1D",
        provider_label="Yahoo Finance",
        provenance_note="S&P 500 price index; not SPY and not a total-return series.",
    ),
    "NDX": AssetSpec(
        canonical_asset_id="NDX",
        display_name="NDX",
        source_id="yahoo_finance",
        provider_symbol="^NDX",
        instrument_type="price_index",
        currency="USD",
        price_unit="index_points",
        session_calendar="XNAS",
        source_timezone="America/New_York",
        adjustment_mode="raw_ohlc",
        base_timeframe="1D",
        provider_label="Yahoo Finance",
        provenance_note="Nasdaq-100 price index; not QQQ and not Nasdaq Composite.",
    ),
    "GOLD": AssetSpec(
        canonical_asset_id="GOLD",
        display_name="GOLD",
        source_id="tradingview_mcp",
        provider_symbol="TVC:GOLD",
        instrument_type="cfd_index_proxy",
        currency="USD",
        price_unit="USD_per_troy_ounce",
        session_calendar="TRADINGVIEW_TVC_GOLD",
        source_timezone="Etc/UTC",
        adjustment_mode="provider_raw",
        base_timeframe="1W",
        provider_label="TradingView MCP",
        provenance_note="TradingView CFDs on Gold (US$ / OZ); explicitly not labelled as physical spot.",
    ),
    "BTCUSD": AssetSpec(
        canonical_asset_id="BTCUSD",
        display_name="BTCUSD",
        source_id="tradingview_mcp",
        provider_symbol="INDEX:BTCUSD",
        instrument_type="crypto_history_index",
        currency="USD",
        price_unit="USD_per_BTC",
        session_calendar="24X7_UTC",
        source_timezone="Etc/UTC",
        adjustment_mode="provider_raw",
        base_timeframe="1W",
        provider_label="TradingView MCP",
        provenance_note="Bitcoin all time history index; not BTCUSDT and not an exchange trading pair.",
    ),
}


ENGINE_PARAMETERS: dict[str, Any] = {
    "engine_version": ENGINE_VERSION,
    "rule_profile": RULE_PROFILE,
    "measurement_mode": "arithmetic",
    "recalculation": "nightly_closed_bars",
    "atr_period": 14,
    "pivot_atr_multipliers": [1.5, 3.0, 6.0],
    "max_pivots_per_stream": 500,
    "max_depth": 4,
    "beam_per_interval_pattern": 20,
    "max_active_scenarios": 128,
    "display_scenarios": 3,
    "flat_min_b": 0.90,
    "barrier_tolerance_fraction": 0.05,
    "retracement_ratio_tolerance": 0.03,
    "extension_ratio_tolerance": 0.10,
    "experimental_enabled": False,
    "core_patterns": [
        "IMPULSE",
        "ZIGZAG",
        "FLAT_REGULAR",
        "FLAT_EXPANDED",
        "TRIANGLE_CONTRACTING",
        "TRIANGLE_BARRIER",
        "DOUBLE_ZIGZAG",
        "DOUBLE_THREE",
        "ENDING_DIAGONAL_CONTRACTING_33333",
    ],
    "conditional_patterns": ["IMPULSE_TRUNCATED_5"],
    "experimental_patterns_disabled": [
        "FLAT_RUNNING",
        "LEADING_CONTRACTING_33333",
        "LEADING_CONTRACTING_53535",
    ],
    "unsupported_patterns": [
        "TRIANGLE_EXPANDING",
        "DIAGONAL_EXPANDING",
        "TRIANGLE_WITH_NESTED_TRIANGLE",
        "TRIPLE_ZIGZAG",
        "TRIPLE_THREE",
    ],
    "legacy_wave5_targets": True,
    "extra_extended_wave5_targets": False,
    "user_A_to_wave5_reference": True,
    "user_A_to_wave5_in_ranking": False,
    "momentum_in_hard_rules": False,
    "macro_inputs_in_engine": False,
    "probability_output": False,
    "price_target_date_output": False,
}


PATTERN_LABELS = {
    "IMPULSE": ["0", "1", "2", "3", "4", "5"],
    "IMPULSE_TRUNCATED_5": ["0", "1", "2", "3", "4", "5"],
    "ENDING_DIAGONAL_CONTRACTING_33333": ["0", "1", "2", "3", "4", "5"],
    "ZIGZAG": ["S", "A", "B", "C"],
    "FLAT_REGULAR": ["S", "A", "B", "C"],
    "FLAT_EXPANDED": ["S", "A", "B", "C"],
    "TRIANGLE_CONTRACTING": ["S", "A", "B", "C", "D", "E"],
    "TRIANGLE_BARRIER": ["S", "A", "B", "C", "D", "E"],
    "DOUBLE_ZIGZAG": ["S", "W", "X", "Y"],
    "DOUBLE_THREE": ["S", "W", "X", "Y"],
}
