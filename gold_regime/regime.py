from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GOLD_REGIME_CONFIG
from .utils import classify_score


TACTICAL_FLOW_LABELS = ("STRONG_BULLISH_FLOW", "BULLISH_FLOW", "NEUTRAL", "BEARISH_FLOW", "STRONG_BEARISH_FLOW")
ALPHA_LABELS = ("STRONG", "POSITIVE", "MODERATE", "NEUTRAL", "WEAK")
STATE_PRIORITY = [
    "HIGH_RISK",
    "FLOW_DIVERGENCE_WARNING",
    "HIGH_CONVICTION_LONG",
    "STRONG_TREND_WITH_FLOW_SUPPORT",
    "MACRO_WARNING",
    "BULLISH",
    "MACRO_TURNING_BULLISH",
    "BEARISH",
    "NEUTRAL",
]


def calculate_gold_tactical_flow(etf_score: float, cot_score: float, config: dict | None = None) -> tuple[float, str]:
    cfg = (config or GOLD_REGIME_CONFIG)["tactical_flow"]
    if not np.isfinite(etf_score) or not np.isfinite(cot_score):
        return np.nan, "PARTIAL_DATA"
    score = float(
        np.clip(
            float(cfg["etf_weight"]) * float(etf_score)
            + float(cfg["cot_weight"]) * float(cot_score),
            0.0,
            100.0,
        )
    )
    return score, classify_tactical_flow_state(score)


def classify_tactical_flow_state(score: float) -> str:
    return classify_score(float(score), TACTICAL_FLOW_LABELS) if np.isfinite(score) else "PARTIAL_DATA"


def classify_alpha(score: float) -> str:
    return classify_score(float(score), ALPHA_LABELS) if np.isfinite(score) else "DATA_INCOMPLETE"


def determine_flow_flags(
    alpha: float,
    tactical_flow: float,
    etf_score: float,
    cot_score: float,
    structural_macro: float = np.nan,
    forward_macro_risk: float = np.nan,
    long_liquidity_cycle: str | None = None,
    previous_tactical_flow: float | None = None,
) -> dict[str, bool | str]:
    flow_recovery = False
    flow_deterioration = False
    if previous_tactical_flow is not None and np.isfinite(previous_tactical_flow) and np.isfinite(tactical_flow):
        flow_recovery = previous_tactical_flow < 40.0 and tactical_flow >= 60.0
        flow_deterioration = previous_tactical_flow > 60.0 and tactical_flow <= 40.0

    cot_etf_divergence = ""
    etf_led_bullish_divergence = False
    speculative_cot_divergence = False
    if np.isfinite(etf_score) and np.isfinite(cot_score):
        etf_led_bullish_divergence = etf_score >= 70.0 and cot_score <= 35.0
        speculative_cot_divergence = cot_score >= 70.0 and etf_score <= 30.0
        if etf_led_bullish_divergence:
            cot_etf_divergence = "ETF_BULLISH_COT_BEARISH"
        elif speculative_cot_divergence:
            cot_etf_divergence = "ETF_BEARISH_COT_BULLISH"

    macro_flow_conflict = (
        np.isfinite(structural_macro)
        and np.isfinite(forward_macro_risk)
        and np.isfinite(tactical_flow)
        and structural_macro < 40.0
        and forward_macro_risk > 60.0
        and tactical_flow >= 60.0
    )
    long_cycle_headwind = str(long_liquidity_cycle or "").upper() in {"DECELERATING_EXPANSION", "CONTRACTION"}
    flags = {
        "FLOW_CONFIRMATION_BULLISH": bool(np.isfinite(alpha) and np.isfinite(tactical_flow) and alpha >= 65.0 and tactical_flow >= 60.0),
        "FLOW_DIVERGENCE_WARNING": bool(np.isfinite(alpha) and np.isfinite(tactical_flow) and alpha >= 70.0 and tactical_flow < 30.0),
        "FLOW_RECOVERY": bool(flow_recovery),
        "FLOW_DETERIORATION": bool(flow_deterioration),
        "COT_ETF_DIVERGENCE": cot_etf_divergence,
        "ETF_LED_BULLISH_DIVERGENCE": bool(etf_led_bullish_divergence),
        "SPECULATIVE_COT_DIVERGENCE": bool(speculative_cot_divergence),
        "MACRO_FLOW_CONFLICT": bool(macro_flow_conflict),
        "LONG_CYCLE_HEADWIND": bool(long_cycle_headwind),
    }
    active = [
        key
        for key in [
            "ETF_LED_BULLISH_DIVERGENCE",
            "SPECULATIVE_COT_DIVERGENCE",
            "MACRO_FLOW_CONFLICT",
            "LONG_CYCLE_HEADWIND",
            "FLOW_DIVERGENCE_WARNING",
        ]
        if flags.get(key)
    ]
    flags["ACTIVE_DIVERGENCE_FLAGS"] = ", ".join(active) if active else "NONE"
    return flags


def determine_gold_regime(
    alpha: float,
    structural_macro: float,
    forward_macro_risk: float,
    tactical_flow: float,
) -> str:
    values = [alpha, structural_macro, forward_macro_risk, tactical_flow]
    if any(not np.isfinite(value) for value in values):
        return "DATA_INCOMPLETE"

    conditions = {
        "HIGH_RISK": forward_macro_risk >= 80.0 and structural_macro < 40.0 and tactical_flow < 40.0 and alpha < 60.0,
        "FLOW_DIVERGENCE_WARNING": alpha >= 70.0 and tactical_flow < 30.0,
        "HIGH_CONVICTION_LONG": alpha >= 70.0 and structural_macro >= 60.0 and forward_macro_risk <= 40.0 and tactical_flow >= 60.0,
        "STRONG_TREND_WITH_FLOW_SUPPORT": alpha >= 70.0 and tactical_flow >= 70.0 and (structural_macro < 60.0 or forward_macro_risk > 40.0),
        "MACRO_WARNING": alpha >= 60.0 and (structural_macro < 40.0 or forward_macro_risk > 60.0) and tactical_flow >= 40.0,
        "BULLISH": alpha >= 65.0 and structural_macro >= 50.0 and forward_macro_risk <= 60.0 and tactical_flow >= 50.0,
        "MACRO_TURNING_BULLISH": structural_macro >= 60.0 and forward_macro_risk <= 40.0 and alpha < 60.0,
        "BEARISH": alpha < 50.0 and structural_macro < 40.0 and tactical_flow < 40.0,
        "NEUTRAL": True,
    }
    for state in STATE_PRIORITY:
        if conditions[state]:
            return state
    return "NEUTRAL"


def additional_structural_demand_context(alpha: float, forward_macro_risk: float, structural_demand_score: float | None) -> str:
    if structural_demand_score is None or not np.isfinite(structural_demand_score):
        return ""
    if np.isfinite(alpha) and alpha >= 60.0 and np.isfinite(forward_macro_risk) and forward_macro_risk > 60.0 and structural_demand_score >= 60.0:
        return "STRONG STRUCTURAL DEMAND"
    return ""


def generate_gold_regime_explanation(current: dict) -> str:
    regime = current.get("gold_regime", "DATA_INCOMPLETE")
    alpha = current.get("gold_alpha", np.nan)
    structural_macro = current.get("structural_macro_score", np.nan)
    forward_risk = current.get("forward_macro_risk", np.nan)
    tactical_flow = current.get("tactical_flow_score", np.nan)
    etf_score = current.get("etf_flow_score", np.nan)
    cot_score = current.get("cot_momentum_score", np.nan)
    flags = current.get("ACTIVE_DIVERGENCE_FLAGS", "NONE")

    if regime == "DATA_INCOMPLETE":
        return "Gold Regime is not fully available because at least one required automatic block is missing. The model does not substitute neutral placeholder values."

    trend = (
        "strong"
        if np.isfinite(alpha) and alpha >= 70.0
        else "constructive"
        if np.isfinite(alpha) and alpha >= 60.0
        else "weak"
        if np.isfinite(alpha) and alpha < 50.0
        else "mixed"
    )
    macro = (
        "supportive"
        if np.isfinite(structural_macro) and structural_macro >= 60.0
        else "unfavorable"
        if np.isfinite(structural_macro) and structural_macro < 40.0
        else "neutral"
    )
    risk = (
        "extreme"
        if np.isfinite(forward_risk) and forward_risk >= 80.0
        else "elevated"
        if np.isfinite(forward_risk) and forward_risk > 60.0
        else "contained"
        if np.isfinite(forward_risk) and forward_risk <= 40.0
        else "moderate"
    )
    etf = (
        "confirming"
        if np.isfinite(etf_score) and etf_score >= 60.0
        else "not confirming"
        if np.isfinite(etf_score) and etf_score < 40.0
        else "mixed"
    )
    cot = (
        "confirming"
        if np.isfinite(cot_score) and cot_score >= 60.0
        else "diverging/cooling"
        if np.isfinite(cot_score) and cot_score <= 35.0 and np.isfinite(etf_score) and etf_score >= 70.0
        else "weak"
        if np.isfinite(cot_score) and cot_score < 40.0
        else "neutral"
    )
    conflict = "present" if current.get("MACRO_FLOW_CONFLICT") else "not active"

    templates = {
        "HIGH_CONVICTION_LONG": "Risk/reward is strongly favorable because price trend, macro, forward risk and capital flows are aligned.",
        "BULLISH": "Risk/reward is favorable, but conviction is lower than a high-conviction long setup.",
        "STRONG_TREND_WITH_FLOW_SUPPORT": "Risk/reward remains constructive because strong price trend and ETF-led demand are offsetting imperfect macro conditions.",
        "MACRO_WARNING": "Risk/reward is more fragile: trend remains intact, but macro pressure is a meaningful headwind.",
        "FLOW_DIVERGENCE_WARNING": "Risk/reward is deteriorating because strong price trend is not confirmed by capital flows.",
        "MACRO_TURNING_BULLISH": "Risk/reward is improving, but price confirmation is still missing.",
        "BEARISH": "Risk/reward is unfavorable because price weakness, weak macro and weak demand confirmation are aligned.",
        "HIGH_RISK": "Risk/reward is poor because forward macro risk is extreme and neither trend nor flows are providing support.",
        "NEUTRAL": "Risk/reward is balanced because the stronger rule-based states are not currently active.",
    }

    return (
        f"Gold price trend is {trend}. Structural macro is {macro}. Forward macro risk is {risk}. "
        f"ETF flows are {etf}; COT is {cot}. Macro-vs-flow conflict is {conflict}. "
        f"Current setup is best described as {regime}. {templates.get(regime, templates['NEUTRAL'])} "
        f"Active divergence flags: {flags}."
    )


def apply_gold_regime_history(history: pd.DataFrame, gold_alpha: float | None) -> pd.DataFrame:
    if history.empty:
        return history
    out = history.copy()
    out["gold_alpha"] = np.nan if gold_alpha is None else gold_alpha
    out["gold_alpha_state"] = out["gold_alpha"].map(classify_alpha)
    out[["tactical_flow_score", "flow_state"]] = out.apply(
        lambda row: pd.Series(
            calculate_gold_tactical_flow(
                row.get("etf_flow_score", np.nan),
                row.get("cot_momentum_score", np.nan),
            )
        ),
        axis=1,
    )
    out["gold_regime"] = out.apply(
        lambda row: determine_gold_regime(
            row.get("gold_alpha", np.nan),
            row.get("structural_macro_score", np.nan),
            row.get("forward_macro_risk", np.nan),
            row.get("tactical_flow_score", np.nan),
        ),
        axis=1,
    )
    flag_rows = out.apply(
        lambda row: determine_flow_flags(
            row.get("gold_alpha", np.nan),
            row.get("tactical_flow_score", np.nan),
            row.get("etf_flow_score", np.nan),
            row.get("cot_momentum_score", np.nan),
            row.get("structural_macro_score", np.nan),
            row.get("forward_macro_risk", np.nan),
            row.get("long_liquidity_cycle"),
        ),
        axis=1,
    )
    if not flag_rows.empty:
        flag_frame = pd.DataFrame(list(flag_rows), index=out.index)
        out = pd.concat([out, flag_frame], axis=1)
    return out
