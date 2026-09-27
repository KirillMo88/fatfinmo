"""Scenario-based BTC price forecast using historical halving-cycle multiples."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BTC_BOTTOM_2026 = 57_749.0
DEFAULT_HALVING_TO_TOP_MULTIPLIERS = {
    "Conservative": 1.20,
    "Base": 1.35,
    "Strong liquidity": 1.50,
}
HISTORICAL_CYCLE_INPUTS = (
    {
        "cycle": "2015–2017",
        "bottom_to_halving": 3.20,
        "halving_to_top": 26.57,
        "bottom_to_top": 85.09,
        "bottom_to_halving_days": 540,
        "halving_to_top_days": 524,
        "bottom_to_top_days": 1064,
    },
    {
        "cycle": "2018–2021",
        "bottom_to_halving": 3.04,
        "halving_to_top": 6.52,
        "bottom_to_top": 19.79,
        "bottom_to_halving_days": 514,
        "halving_to_top_days": 550,
        "bottom_to_top_days": 1064,
    },
    {
        "cycle": "2022–2025",
        "bottom_to_halving": 3.86,
        "halving_to_top": 1.92,
        "bottom_to_top": 7.40,
        "bottom_to_halving_days": 512,
        "halving_to_top_days": 531,
        "bottom_to_top_days": 1043,
    },
)


def historical_return_table() -> pd.DataFrame:
    rows = []
    for cycle in HISTORICAL_CYCLE_INPUTS:
        rows.append(
            {
                "Cycle": cycle["cycle"],
                "Bottom → Halving": _format_multiple_return(cycle["bottom_to_halving"]),
                "Halving → Top": _format_multiple_return(cycle["halving_to_top"]),
                "Bottom → Top": _format_multiple_return(cycle["bottom_to_top"]),
            }
        )
    return pd.DataFrame(rows)


def historical_timing_table() -> pd.DataFrame:
    rows = []
    for cycle in HISTORICAL_CYCLE_INPUTS:
        rows.append(
            {
                "Cycle": cycle["cycle"],
                "Bottom → Top": _format_duration(cycle["bottom_to_top_days"]),
                "Bottom → Halving": _format_duration(cycle["bottom_to_halving_days"]),
                "Halving → Top": _format_duration(cycle["halving_to_top_days"]),
            }
        )
    return pd.DataFrame(rows)


def build_btc_halving_price_forecast(
    bottom_price: float = DEFAULT_BTC_BOTTOM_2026,
    halving_to_top_multipliers: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Calculate price-at-halving and cycle-top scenarios from editable assumptions."""
    multipliers = dict(DEFAULT_HALVING_TO_TOP_MULTIPLIERS)
    if halving_to_top_multipliers is not None:
        multipliers.update(halving_to_top_multipliers)
    if set(multipliers) != set(DEFAULT_HALVING_TO_TOP_MULTIPLIERS):
        raise ValueError("Provide Conservative, Base, and Strong liquidity multipliers")

    bottom_price = float(bottom_price)
    if not np.isfinite(bottom_price) or bottom_price <= 0:
        raise ValueError("BTC bottom price must be a positive finite number")
    if any(not np.isfinite(value) or value <= 0 for value in multipliers.values()):
        raise ValueError("Scenario multipliers must be positive finite numbers")

    historical_bottom_to_halving = [cycle["bottom_to_halving"] for cycle in HISTORICAL_CYCLE_INPUTS]
    average_bottom_to_halving = float(np.mean(historical_bottom_to_halving))
    price_at_halving = bottom_price * average_bottom_to_halving
    cycle_top_prices = {
        name: price_at_halving * float(multipliers[name])
        for name in DEFAULT_HALVING_TO_TOP_MULTIPLIERS
    }

    result: dict[str, Any] = {
        "BTC_Bottom_2026": bottom_price,
        "BTC_BottomToHalving_2015_2017": historical_bottom_to_halving[0],
        "BTC_BottomToHalving_2018_2021": historical_bottom_to_halving[1],
        "BTC_BottomToHalving_2022_2025": historical_bottom_to_halving[2],
        "BTC_Avg_BottomToHalving": average_bottom_to_halving,
        "BTC_HalvingToTop_Conservative": float(multipliers["Conservative"]),
        "BTC_HalvingToTop_Base": float(multipliers["Base"]),
        "BTC_HalvingToTop_StrongLiquidity": float(multipliers["Strong liquidity"]),
        "BTC_PriceAtHalving": price_at_halving,
        "BTC_CycleTop_Conservative": cycle_top_prices["Conservative"],
        "BTC_CycleTop_Base": cycle_top_prices["Base"],
        "BTC_CycleTop_StrongLiquidity": cycle_top_prices["Strong liquidity"],
        "BTC_CycleTop_Average": float(np.mean(list(cycle_top_prices.values()))),
    }
    errors = validate_btc_halving_price_forecast(result)
    if errors:
        raise ArithmeticError("BTC halving price forecast failed validation: " + "; ".join(errors))
    return result


def validate_btc_halving_price_forecast(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_average = float(
        np.mean(
            [
                result["BTC_BottomToHalving_2015_2017"],
                result["BTC_BottomToHalving_2018_2021"],
                result["BTC_BottomToHalving_2022_2025"],
            ]
        )
    )
    if not np.isclose(result["BTC_Avg_BottomToHalving"], expected_average, rtol=0, atol=1e-12):
        errors.append("Average Bottom-to-Halving multiple does not match historical mean")
    if not np.isclose(
        result["BTC_PriceAtHalving"], result["BTC_Bottom_2026"] * result["BTC_Avg_BottomToHalving"], rtol=0, atol=1e-8
    ):
        errors.append("Price at halving does not reconcile")

    scenario_pairs = (
        ("Conservative", "BTC_HalvingToTop_Conservative", "BTC_CycleTop_Conservative"),
        ("Base", "BTC_HalvingToTop_Base", "BTC_CycleTop_Base"),
        ("Strong liquidity", "BTC_HalvingToTop_StrongLiquidity", "BTC_CycleTop_StrongLiquidity"),
    )
    for label, multiplier_key, price_key in scenario_pairs:
        if not np.isclose(
            result[price_key], result["BTC_PriceAtHalving"] * result[multiplier_key], rtol=0, atol=1e-8
        ):
            errors.append(f"{label} cycle-top estimate does not reconcile")

    average_top = float(
        np.mean(
            [
                result["BTC_CycleTop_Conservative"],
                result["BTC_CycleTop_Base"],
                result["BTC_CycleTop_StrongLiquidity"],
            ]
        )
    )
    if not np.isclose(result["BTC_CycleTop_Average"], average_top, rtol=0, atol=1e-8):
        errors.append("Average cycle-top forecast does not reconcile")
    return errors


def _format_multiple_return(multiple: float) -> str:
    return f"{multiple:.2f}x / {(multiple - 1.0) * 100:+,.0f}%"


def _format_duration(days: int) -> str:
    weeks = days / 7.0
    months = days / (365.2425 / 12.0)
    week_label = f"{weeks:.1f}".rstrip("0").rstrip(".")
    return f"{days} days / {week_label} weeks / {months:.1f} months"
