from __future__ import annotations

from copy import deepcopy


GOLD_REGIME_CONFIG = {
    "percentile": {
        "window_weeks": 156,
        "minimum_weeks": 104,
    },
    "structural_macro": {
        "dxy_weight": 0.35,
        "real_yield_weight": 0.55,
        "us2y_weight": 0.10,
        "window_weeks": 13,
    },
    "forward_macro_risk": {
        "us2y_weight": 0.60,
        "wti_weight": 0.40,
        "us2y_window_weeks": 13,
        "wti_window_weeks": 26,
    },
    "tactical_flow": {
        "etf_weight": 0.85,
        "cot_weight": 0.15,
        "etf_flow_window_weeks": 4,
        "cot_change_window_weeks": 4,
    },
    "etf_tickers": ["GLD", "IAU", "GLDM", "IAUM", "SGOL", "OUNZ", "AAAU", "BAR"],
    "cot": {
        "url": "https://publicreporting.cftc.gov/api/v3/views/72hh-3qpy/export.csv",
        "report": "DISAGGREGATED_FUTURES_ONLY",
        "market": "COMEX_GOLD",
        "cache_ttl_seconds": 21600,
    },
    "freshness_days": {
        "market_prices": 7,
        "fred_yields": 10,
        "etf_flows": 10,
        "cot": 14,
        "structural_demand": 120,
    },
    "structural_demand": {
        "score": None,
        "monetary_demand_share": None,
        "central_bank_4q_purchases": None,
        "demand_rotation_yoy": None,
        "last_updated": None,
    },
}


def gold_regime_config() -> dict:
    return deepcopy(GOLD_REGIME_CONFIG)
