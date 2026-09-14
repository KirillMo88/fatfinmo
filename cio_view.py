from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from bybit_derivatives import BYBIT_STORAGE_PATH, read_bybit_storage
from finance_core import download_completed_ohlcv
from fund_flows import FundFlowCache, default_fund_flow_cache_path
from global_liquidity import GLOBAL_LIQUIDITY_STORAGE_DIR, read_global_liquidity
from global_macro_tab import load_global_macro_snapshot
from gold_regime import build_gold_regime_snapshot, gold_regime_config


CIO_MODEL = os.environ.get("CIO_MODEL", "gpt-5.6-sol")
CIO_REASONING_EFFORT = os.environ.get("CIO_REASONING_EFFORT", "medium")
CIO_CRITIC_ENABLED = os.environ.get("CIO_CRITIC_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
CIO_CRITIC_REASONING_EFFORT = os.environ.get("CIO_CRITIC_REASONING_EFFORT", "medium")
CIO_MONTHLY_COST_WARNING_USD = float(os.environ.get("CIO_MONTHLY_COST_WARNING_USD", "5") or 5)
CIO_REFRESH_SECONDS = 7 * 24 * 60 * 60
CIO_STORAGE_DIR = Path(os.environ.get("CIO_STORAGE_DIR", str(Path(__file__).with_name("persistent") / "cio_view")))
CIO_STORAGE_PATH = CIO_STORAGE_DIR / "cio_view.json"

ASSET_TICKERS = ["SPY", "QQQ", "GLD", "BTC-USD"]
BTC_SPOT_ETF_FLOW_TICKERS = ("IBIT", "FBTC", "GBTC", "ARKB", "BITB", "BTCO", "EZBC", "HODL", "BRRR", "BTCW")
GLOBAL_MACRO_INSTRUMENTS = {
    "U.S. Dollar Index",
    "EUR/USD",
    "USD/JPY",
    "USD/CNY",
    "U.S. 5-Year Breakeven Inflation Rate",
    "U.S. 10-Year Breakeven Inflation Rate",
    "U.S. 1-Year Inflation Expectations",
    "U.S. 10-Year Real Yield",
    "Federal Funds Effective Rate",
    "U.S. 2-Year Treasury Yield",
    "U.S. 10-Year Treasury Yield",
    "U.S. 2Y-10Y Treasury Curve",
    "U.S. 3M-10Y Treasury Curve",
    "Germany 10-Year Government Bond Yield",
    "China 10-Year Government Bond Yield",
    "Japan 10-Year Government Bond Yield",
    "U.S. ISM Manufacturing PMI",
    "U.S. ISM Services PMI",
    "U.S. Initial Jobless Claims",
    "Chicago Fed National Activity Index",
    "WTI Crude Oil",
    "Copper",
    "CBOE Volatility Index",
    "ICE BofA MOVE Index",
    "U.S. High Yield Option-Adjusted Spread",
    "U.S. Investment Grade Option-Adjusted Spread",
    "Chicago Fed Adjusted National Financial Conditions Index",
}

CIO_SYSTEM_PROMPT = """You are the CIO synthesis layer of a quantitative macro and market regime system.

You receive precomputed indicators and regime outputs.

Do not recalculate indicators.
Do not invent missing values.
Do not override the quantitative models.
Do not claim certainty.
Do not give deterministic investment advice.

Your job is to synthesize supplied signals, identify confirmations and conflicts,
assess the financial-system environment, produce a 3-month outlook for SPY, QQQ,
GLD and BTC-USD, and identify what would change the view.

Different assets have different drivers. Do not use one universal macro model.
Treat Global Liquidity differently for equities, Gold and BTC according to the
supplied model architecture.

Return only valid JSON matching the requested structure."""


@dataclass(frozen=True)
class CioRunResult:
    snapshot: dict[str, Any]
    analyst_result: dict[str, Any]
    critic_result: dict[str, Any] | None
    final_result: dict[str, Any]
    usage: dict[str, Any]
    status: str
    generated_at: str
    next_scheduled_at: str


def render_cio_view_tab(table_df: pd.DataFrame, market_snapshot: dict[str, Any], fred_api_key: str | None = None) -> None:
    st.subheader("CIO View")
    st.caption("Macro & Market Outlook - weekly LLM synthesis over deterministic regime outputs.")

    controls = st.columns([1.2, 4.8])
    with controls[0]:
        regenerate = st.button("Regenerate CIO View", use_container_width=True, key="cio_regenerate")
    with controls[1]:
        st.markdown(
            f"<div style='padding-top:1.55rem; color:#94a3b8; font-size:0.78rem;'>Storage: {CIO_STORAGE_PATH}</div>",
            unsafe_allow_html=True,
        )

    if regenerate:
        with st.spinner("Generating CIO View..."):
            result = generate_cio_view(table_df, market_snapshot, fred_api_key, force=True)
    else:
        result = load_or_build_cio_view(table_df, market_snapshot, fred_api_key)

    _render_cio_top_cards(result)
    final = result.final_result
    _render_dimensions(final.get("dimensions", {}))
    _render_cross_regime(final.get("cross_regime_signals", []))
    _render_asset_outlook(final.get("asset_outlook_3m", {}))
    _render_scenarios(final.get("scenarios", {}))
    _render_key_risks(final.get("key_risks", []))
    _render_change_view(final.get("what_would_change_view", {}))
    _render_diagnostics(result)


def load_or_build_cio_view(table_df: pd.DataFrame, market_snapshot: dict[str, Any], fred_api_key: str | None) -> CioRunResult:
    cached = read_cio_cache()
    if cached and cache_is_fresh(cached):
        return cio_result_from_cache(cached, "CACHED")
    if cached:
        return cio_result_from_cache(cached, "STALE")
    snapshot = build_cio_snapshot(table_df, market_snapshot, fred_api_key)
    final = deterministic_cio_view(snapshot, "CIO_VIEW_UNAVAILABLE")
    return CioRunResult(
        snapshot=snapshot,
        analyst_result=final,
        critic_result=None,
        final_result=final,
        usage={"model": CIO_MODEL, "reasoning_effort": CIO_REASONING_EFFORT, "status": "NO_PREVIOUS_ANALYSIS"},
        status="CIO_VIEW_UNAVAILABLE",
        generated_at=snapshot["as_of_date"],
        next_scheduled_at=next_weekly_timestamp(snapshot["as_of_date"]),
    )


def generate_cio_view(
    table_df: pd.DataFrame,
    market_snapshot: dict[str, Any],
    fred_api_key: str | None,
    force: bool = False,
) -> CioRunResult:
    cached = read_cio_cache()
    if cached and cache_is_fresh(cached) and not force:
        return cio_result_from_cache(cached, "CACHED")
    snapshot = build_cio_snapshot(table_df, market_snapshot, fred_api_key)
    started = time.time()
    usage: dict[str, Any] = {
        "model": CIO_MODEL,
        "reasoning_effort": CIO_REASONING_EFFORT,
        "critic_enabled": CIO_CRITIC_ENABLED,
        "estimated_cost": None,
    }
    try:
        analyst = call_cio_llm(snapshot, CIO_REASONING_EFFORT)
        usage["analyst_usage"] = analyst.pop("_response_usage", None)
        critic = call_cio_critic(snapshot, analyst) if CIO_CRITIC_ENABLED else None
        if critic:
            usage["critic_usage"] = critic.pop("_response_usage", None)
        final = critic or analyst
        status = "CURRENT"
    except Exception as exc:
        previous = cio_result_from_cache(cached, "STALE") if cached else None
        if previous:
            return previous
        final = deterministic_cio_view(snapshot, "LLM_UNAVAILABLE")
        analyst = final
        critic = None
        usage["error"] = str(exc)
        status = "LLM_UNAVAILABLE"
    usage["latency_seconds"] = round(time.time() - started, 2)
    generated_at = datetime.now(timezone.utc).isoformat()
    result = CioRunResult(
        snapshot=snapshot,
        analyst_result=analyst,
        critic_result=critic,
        final_result=final,
        usage=usage,
        status=status,
        generated_at=generated_at,
        next_scheduled_at=next_weekly_timestamp(generated_at),
    )
    write_cio_cache(result)
    return result


def build_cio_snapshot(table_df: pd.DataFrame, market_snapshot: dict[str, Any], fred_api_key: str | None = None) -> dict[str, Any]:
    as_of = datetime.now(timezone.utc).isoformat()
    raw_liquidity, monthly_liquidity, weekly_liquidity = read_global_liquidity()
    liquidity = compact_liquidity_snapshot(monthly_liquidity, weekly_liquidity, market_snapshot)
    macro = compact_global_macro_snapshot(fred_api_key)
    gold = compact_gold_snapshot(table_df, fred_api_key)
    btc = compact_btc_snapshot(table_df, market_snapshot, liquidity)
    assets = {ticker: asset_snapshot(ticker, table_df) for ticker in ASSET_TICKERS}
    data_quality = data_quality_snapshot(raw_liquidity, monthly_liquidity, weekly_liquidity, macro, gold, btc)
    return {
        "as_of_date": as_of,
        "market_regime": compact_market_snapshot(market_snapshot),
        "global_liquidity": liquidity,
        "global_macro": macro,
        "gold_regime": gold,
        "btc_regime": btc,
        "asset_market_data": assets,
        "data_quality": data_quality,
    }


def compact_market_snapshot(market: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "final_state": "Final_Market_State",
        "structural_regime": "Market_Regime",
        "spy_price": "SPY_Price",
        "spy_vs_sma40w": "SPY_vs_SMA40W_%",
        "spy_52w_drawdown": "SPY_Drawdown_52W_%",
        "realized_vol_13w": "SPY_Volatility_13W_%",
        "fast_transition_risk": "Fast_Transition_Risk",
        "fast_risk_state": "Fast_Transition_State",
        "fast_risk_direction_4w": "Fast_Risk_Direction_4W",
        "fast_risk_direction_4w_state": "Fast_Risk_Direction_4W_State",
        "macro_transition_risk": "Macro_Transition_Risk",
        "macro_risk_state": "Macro_Transition_State",
        "global_liquidity_backdrop": "Global_Liquidity_Backdrop",
        "negative_confirmations": "Negative_Confirmation_Count",
    }
    out = {target: clean_value(market.get(source)) for target, source in keys.items()}
    out["confirmations"] = {
        "wti": clean_value(market.get("WTI_Confirmation")),
        "real_yield_10y": clean_value(market.get("Real_Yield_10Y_Confirmation")),
        "iwm_spy": clean_value(market.get("IWM_SPY_Confirmation")),
        "xli_xlp": clean_value(market.get("XLI_XLP_Confirmation")),
        "rsi_divergence": clean_value(market.get("RSI_Divergence")),
    }
    return out


def compact_liquidity_snapshot(monthly: pd.DataFrame, weekly: pd.DataFrame, market: dict[str, Any]) -> dict[str, Any]:
    monthly_latest = latest_row(monthly)
    weekly_latest = latest_row(weekly)
    return {
        "global_m2_usd_bn": clean_value(monthly_latest.get("global_m2_usd_bn")),
        "global_m2_trend": clean_value(market.get("Global_Liquidity_Backdrop")),
        "m2_13w": clean_value(monthly_latest.get("global_m2_3m_pct")),
        "m2_26w": clean_value(monthly_latest.get("global_m2_6m_pct")),
        "m2_52w": clean_value(monthly_latest.get("global_m2_12m_pct")),
        "m2_impulse": clean_value(monthly_latest.get("global_m2_impulse")),
        "global_cb_assets_usd_bn": clean_value(monthly_latest.get("global_cb_assets_usd_bn")),
        "cb_impulse": clean_value(weekly_latest.get("global_cb_impulse")),
        "us_net_liquidity_usd_bn": clean_value(weekly_latest.get("us_net_liquidity_usd_bn")),
        "us_net_liquidity_impulse": clean_value(weekly_latest.get("us_net_liquidity_impulse")),
        "global_liquidity_score": clean_value(market.get("Global_Liquidity_Score")),
        "global_liquidity_direction_13w": clean_value(market.get("Global_Liquidity_Direction_13W")),
        "global_liquidity_direction": clean_value(market.get("Global_Liquidity_Direction_13W_State")),
        "global_liquidity_backdrop": clean_value(market.get("Global_Liquidity_Backdrop")),
        "long_liquidity_cycle_phase": clean_value(market.get("Long_Liquidity_Cycle")),
        "last_updated": clean_value(market.get("Global_Liquidity_Last_Updated")),
        "data_status": clean_value(market.get("Global_Liquidity_Data_Status")),
    }


def compact_global_macro_snapshot(fred_api_key: str | None) -> dict[str, Any]:
    try:
        frame = load_global_macro_snapshot(fred_api_key or "", 0)
    except Exception as exc:
        return {"status": "ERROR", "error": str(exc), "items": []}
    if frame.empty:
        return {"status": "MISSING", "items": []}
    keep = frame[frame["Instrument"].isin(GLOBAL_MACRO_INSTRUMENTS)].copy()
    fields = ["Block", "Instrument", "Current", "1M", "3M", "6M", "12M", "Unit", "Source", "Data Status"]
    items = []
    for row in keep[[column for column in fields if column in keep.columns]].to_dict("records"):
        items.append({snake_key(key): clean_value(value) for key, value in row.items()})
    return {"status": "OK", "items": items}


def compact_gold_snapshot(table_df: pd.DataFrame, fred_api_key: str | None) -> dict[str, Any]:
    try:
        gold_alpha = table_value(table_df, "GLD", "Alpha_Score")
        snapshot = build_gold_regime_snapshot(gold_alpha=gold_alpha, fred_api_key=fred_api_key, config=gold_regime_config())
        current = dict(snapshot.current or {})
        history = snapshot.history.copy() if hasattr(snapshot, "history") else pd.DataFrame()
        returns = price_returns_from_history(history, "gold_price")
        return {
            "final_state": clean_value(current.get("gold_regime")),
            "gold_alpha": clean_value(current.get("gold_alpha")),
            "structural_macro": clean_value(current.get("structural_macro_score")),
            "forward_macro_risk": clean_value(current.get("forward_macro_risk")),
            "tactical_flow": clean_value(current.get("tactical_flow_score")),
            "etf_flow_score": clean_value(current.get("etf_flow_score")),
            "cot_momentum_score": clean_value(current.get("cot_momentum_score")),
            "divergence_flags": clean_value(current.get("ACTIVE_DIVERGENCE_FLAGS")),
            "structural_demand_status": clean_value(current.get("structural_demand_status")),
            "long_liquidity_cycle_phase": clean_value(current.get("long_liquidity_cycle")),
            "price_changes": returns,
            "last_updated": clean_value(current.get("date")),
            "data_status": "OK",
        }
    except Exception as exc:
        return {"final_state": "DATA_INCOMPLETE", "data_status": "ERROR", "error": str(exc)}


def compact_btc_snapshot(table_df: pd.DataFrame, market: dict[str, Any], liquidity: dict[str, Any]) -> dict[str, Any]:
    btc = asset_snapshot("BTC-USD", table_df)
    bybit = latest_bybit_btc()
    etf = latest_btc_etf_flows()
    alpha = table_value(table_df, "BTC-USD", "Alpha_Score")
    liq_score = to_float(liquidity.get("global_liquidity_score"))
    dxy_risk = to_float(market.get("Macro_DXY_Risk", market.get("DXY_Risk")))
    us2y_risk = to_float(market.get("US2Y_Risk"))
    structural = weighted_mean([liq_score, 100.0 - dxy_risk, 100.0 - us2y_risk], [0.40, 0.40, 0.20])
    forward_risk = weighted_mean([dxy_risk, us2y_risk, 100.0 - liq_score], [0.55, 0.30, 0.15])
    phase = btc_halving_phase()
    return {
        "final_state": btc_final_state(alpha, liquidity, etf, bybit, forward_risk),
        "halving_phase": phase["phase"],
        "months_since_halving": phase["months_since_halving"],
        "months_since_cycle_top": phase["months_since_cycle_top"],
        "global_liquidity_score": clean_value(liq_score),
        "global_liquidity_direction": clean_value(liquidity.get("global_liquidity_direction")),
        "structural_macro": clean_value(structural),
        "forward_macro_risk": clean_value(forward_risk),
        "alpha": clean_value(alpha),
        "etf_flow_state": etf.get("state"),
        "etf_flow_4w": clean_value(etf.get("flow_4w")),
        "open_interest_usd": clean_value(bybit.get("open_interest_usd")),
        "oi_change_1w": clean_value(bybit.get("oi_change_1w_pct")),
        "oi_change_4w": clean_value(bybit.get("oi_change_4w_pct")),
        "funding_rate": clean_value(bybit.get("funding_1d")),
        "funding_7d": clean_value(bybit.get("funding_7d")),
        "funding_28d": clean_value(bybit.get("funding_28d")),
        "perpetual_premium_basis": clean_value(bybit.get("perp_premium_pct")),
        "cycle_bottom_status": btc_bottom_status(phase["phase"], alpha, liquidity, etf, bybit, forward_risk),
        "price_changes": {key: btc.get(key) for key in ["return_1m", "return_3m", "return_6m", "return_12m"]},
        "drawdown_from_cycle_top": clean_value(btc.get("drawdown_from_ath")),
        "data_status": "OK",
    }


def asset_snapshot(ticker: str, table_df: pd.DataFrame) -> dict[str, Any]:
    row = table_row(table_df, ticker)
    out = {
        "ticker": ticker,
        "current_price": latest_close(ticker),
        "return_1m": clean_value(row.get("Perf_1M_%")),
        "return_3m": clean_value(row.get("Perf_3M_%")),
        "return_6m": clean_value(row.get("Perf_6M_%")),
        "return_12m": clean_value(row.get("Perf_12M_%")),
        "drawdown_from_ath": clean_value(row.get("Price_vs_ATH_%")),
        "distance_from_52w_high": clean_value(row.get("Price_vs_52W_High_%")),
        "alpha_score": clean_value(row.get("Alpha_Score")),
        "alpha_state": clean_value(row.get("Alpha_State")),
    }
    return out


def call_cio_llm(snapshot: dict[str, Any], reasoning_effort: str) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=get_openai_api_key())
    prompt = (
        f"Analyze this CIO snapshot as of {snapshot['as_of_date']}.\n\n"
        "Produce financial system assessment, dimension states, cross-regime signals, "
        "SPY/QQQ/GLD/BTC-USD 3M outlook, base/bull/bear scenarios, key risks, and what would change the view.\n"
        "Use only supplied inputs.\n\n"
        f"CIO Snapshot JSON:\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}"
    )
    response = client.responses.create(
        model=CIO_MODEL,
        reasoning={"effort": reasoning_effort},
        instructions=CIO_SYSTEM_PROMPT,
        input=prompt,
    )
    text = getattr(response, "output_text", "") or ""
    result = parse_json_text(text)
    result.setdefault("as_of_date", snapshot["as_of_date"])
    result["_response_usage"] = extract_response_usage(response)
    return result


def call_cio_critic(snapshot: dict[str, Any], analyst: dict[str, Any]) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=get_openai_api_key())
    prompt = (
        "Audit the CIO analysis. Correct unsupported conclusions, missing-data inventions, "
        "driver mismatches for SPY/QQQ/GLD/BTC, overuse of short-term Fast Risk, misuse of Global Liquidity for Gold, "
        "and confidence values that are too high. Return corrected final JSON only.\n\n"
        f"Snapshot:\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}\n\n"
        f"Analyst JSON:\n{json.dumps(analyst, ensure_ascii=False, indent=2)}"
    )
    response = client.responses.create(
        model=CIO_MODEL,
        reasoning={"effort": CIO_CRITIC_REASONING_EFFORT},
        instructions=CIO_SYSTEM_PROMPT,
        input=prompt,
    )
    text = getattr(response, "output_text", "") or ""
    result = parse_json_text(text)
    result.setdefault("as_of_date", snapshot["as_of_date"])
    result["_response_usage"] = extract_response_usage(response)
    return result


def deterministic_cio_view(snapshot: dict[str, Any], status: str) -> dict[str, Any]:
    market = snapshot.get("market_regime", {})
    liquidity = snapshot.get("global_liquidity", {})
    gold = snapshot.get("gold_regime", {})
    btc = snapshot.get("btc_regime", {})
    label = deterministic_system_label(market, liquidity)
    return {
        "as_of_date": snapshot["as_of_date"],
        "overall_system_state": {
            "label": label,
            "summary": "Deterministic fallback view generated from existing regime outputs; LLM synthesis has not produced a current weekly analysis.",
        },
        "dimensions": fallback_dimensions(market, liquidity),
        "cross_regime_signals": fallback_cross_regime_signals(market, liquidity, gold, btc),
        "asset_outlook_3m": {
            "SPY": fallback_asset_outlook("SPY", snapshot),
            "QQQ": fallback_asset_outlook("QQQ", snapshot),
            "GLD": fallback_asset_outlook("GLD", snapshot),
            "BTC-USD": fallback_asset_outlook("BTC-USD", snapshot),
        },
        "scenarios": fallback_scenarios(),
        "key_risks": fallback_key_risks(snapshot),
        "what_would_change_view": fallback_change_view(),
        "data_quality_notes": [status, *snapshot.get("data_quality", {}).get("notes", [])],
    }


def deterministic_system_label(market: dict[str, Any], liquidity: dict[str, Any]) -> str:
    final = str(market.get("final_state", ""))
    liq_dir = str(liquidity.get("global_liquidity_direction", ""))
    if "STRESS" in final:
        return "SYSTEMIC_STRESS"
    if "WARNING" in final or "DETERIORATING" in liq_dir:
        return "RISK_ON_WITH_WARNING"
    if str(market.get("structural_regime", "")).upper() == "BULL":
        return "RISK_ON_EXPANSION"
    return "LIQUIDITY_DIVERGENCE"


def fallback_dimensions(market: dict[str, Any], liquidity: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        "growth": {"state": "NEUTRAL", "explanation": "Use Global Macro PMI/claims block for the current weekly LLM view."},
        "inflation": {"state": "NEUTRAL", "explanation": "Inflation block is available in the supplied snapshot."},
        "liquidity": {"state": state_from_score(liquidity.get("global_liquidity_score")), "explanation": str(liquidity.get("global_liquidity_backdrop", "n/a"))},
        "rates": {"state": "NEUTRAL", "explanation": "Rates and real-yield data are monitored in Global Macro."},
        "financial_conditions": {"state": "NEUTRAL", "explanation": "VIX/MOVE/spreads are included in Global Macro."},
        "credit": {"state": "NEUTRAL", "explanation": "Credit spread state requires LLM synthesis over supplied current/trend changes."},
        "risk_appetite": {"state": state_from_risk(market.get("fast_transition_risk")), "explanation": str(market.get("fast_risk_state", "n/a"))},
        "market_trend": {"state": str(market.get("structural_regime", "DATA_INCOMPLETE")), "explanation": str(market.get("final_state", "n/a"))},
        "systemic_stress": {"state": state_from_risk(market.get("macro_transition_risk")), "explanation": str(market.get("macro_risk_state", "n/a"))},
    }


def fallback_cross_regime_signals(market: dict[str, Any], liquidity: dict[str, Any], gold: dict[str, Any], btc: dict[str, Any]) -> list[dict[str, str]]:
    signals = []
    if str(market.get("structural_regime", "")).upper() == "BULL" and "DETERIORATING" in str(liquidity.get("global_liquidity_direction", "")):
        signals.append({"title": "PRICE_LIQUIDITY_DIVERGENCE", "description": "Market trend remains constructive while liquidity direction is deteriorating."})
    if to_float(gold.get("tactical_flow")) >= 60 and to_float(gold.get("forward_macro_risk")) >= 60:
        signals.append({"title": "GOLD_MACRO_FLOW_CONFLICT", "description": "Gold flow support is positive but forward macro risk is elevated."})
    if str(btc.get("cycle_bottom_status", "")).endswith("WATCH"):
        signals.append({"title": "BTC_BOTTOMING_NOT_CONFIRMED", "description": "BTC timing setup is active but confirmation set is incomplete."})
    if not signals:
        signals.append({"title": "NO_MAJOR_CONFLICT_DETECTED", "description": "Fallback synthesis did not identify a high-priority cross-regime conflict."})
    return signals[:5]


def fallback_asset_outlook(ticker: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    asset = snapshot.get("asset_market_data", {}).get(ticker, {})
    alpha = to_float(asset.get("alpha_score"))
    bias = "NEUTRAL_POSITIVE" if alpha >= 60 else "NEUTRAL_NEGATIVE" if alpha < 40 and np.isfinite(alpha) else "NEUTRAL"
    return {
        "ticker": ticker,
        "bias_3m": bias,
        "confidence": 0.55,
        "supporting_factors": ["Existing deterministic regime outputs are available."],
        "risk_factors": ["LLM synthesis is unavailable or stale."],
        "expected_environment": "Use the latest successful CIO generation for a fuller qualitative view.",
        "what_would_make_more_bullish": ["GlobalLiquidityDirection > 0", "Alpha Score > 60"],
        "what_would_make_more_bearish": ["GlobalLiquidityDirection deteriorates", "Alpha Score < 40"],
    }


def fallback_scenarios() -> dict[str, dict[str, Any]]:
    return {
        "base": {"scenario_conditions": ["Current major signals persist"], "system_implication": "Mixed regime; wait for LLM weekly synthesis.", "SPY": "NEUTRAL", "QQQ": "NEUTRAL", "GLD": "NEUTRAL", "BTC": "NEUTRAL"},
        "bull": {"scenario_conditions": ["Liquidity direction improves", "Rates/DXY ease", "Credit remains benign"], "system_implication": "Risk appetite broadens.", "SPY": "POSITIVE", "QQQ": "POSITIVE", "GLD": "MIXED", "BTC": "POSITIVE"},
        "bear": {"scenario_conditions": ["Liquidity deterioration continues", "DXY/rates rise", "Credit and volatility widen"], "system_implication": "Correction risk rises.", "SPY": "NEGATIVE", "QQQ": "NEGATIVE", "GLD": "MIXED", "BTC": "NEGATIVE"},
    }


def fallback_key_risks(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"rank": 1, "title": "Liquidity deterioration", "description": "Global liquidity direction remains a key cross-asset risk.", "affected_assets": ["SPY", "QQQ", "BTC-USD"], "trigger_to_watch": "GlobalLiquidityDirection13W < -5"},
        {"rank": 2, "title": "Rates / USD pressure", "description": "Higher yields and a stronger DXY can pressure duration and crypto assets.", "affected_assets": ["QQQ", "BTC-USD", "GLD"], "trigger_to_watch": "US2Y / DXY 3M change"},
        {"rank": 3, "title": "Credit stress", "description": "HY/IG spread widening would change the risk-on interpretation.", "affected_assets": ["SPY", "QQQ", "BTC-USD"], "trigger_to_watch": "HY OAS / IG OAS"},
    ]


def fallback_change_view() -> dict[str, dict[str, list[str]]]:
    common_bull = ["GlobalLiquidityDirection > 0", "US2Y falls", "DXY weakens"]
    common_bear = ["GlobalLiquidityDirection < -10", "HY/IG spreads widen", "VIX/MOVE accelerate"]
    return {
        "SPY": {"bullish_triggers": common_bull, "bearish_triggers": common_bear},
        "QQQ": {"bullish_triggers": common_bull + ["10Y real yield falls"], "bearish_triggers": common_bear + ["10Y real yield rises"]},
        "GLD": {"bullish_triggers": ["Gold Alpha > 60", "ETF Flow Score > 70", "real yields fall"], "bearish_triggers": ["Gold Alpha < 40", "ETF Flow Score < 30", "US2Y / DXY rise"]},
        "BTC-USD": {"bullish_triggers": ["BTC Alpha > 60", "BTC ETF flows positive", "funding normalized"], "bearish_triggers": ["BTC Alpha < 40", "ETF flows turn negative", "OI/funding become overheated"]},
    }


def _render_cio_top_cards(result: CioRunResult) -> None:
    final = result.final_result
    state = final.get("overall_system_state", {})
    snapshot = result.snapshot
    cards = [
        ("Overall System State", state.get("label", "n/a"), result.status),
        ("Market Regime", snapshot.get("market_regime", {}).get("final_state", "n/a"), snapshot.get("market_regime", {}).get("structural_regime", "n/a")),
        ("Global Liquidity", snapshot.get("global_liquidity", {}).get("global_liquidity_backdrop", "n/a"), snapshot.get("global_liquidity", {}).get("global_liquidity_direction", "n/a")),
        ("Gold Regime", snapshot.get("gold_regime", {}).get("final_state", "n/a"), f"Alpha {snapshot.get('gold_regime', {}).get('gold_alpha', 'n/a')}"),
        ("BTC Regime", snapshot.get("btc_regime", {}).get("final_state", "n/a"), snapshot.get("btc_regime", {}).get("halving_phase", "n/a")),
        ("Last Analysis", result.generated_at[:19], result.status),
        ("Next Scheduled", result.next_scheduled_at[:19], "weekly"),
        ("Data Quality", snapshot.get("data_quality", {}).get("status", "n/a"), f"{len(snapshot.get('data_quality', {}).get('notes', []))} notes"),
    ]
    for start in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, (label, value, detail) in zip(cols, cards[start : start + 4]):
            with col:
                _metric_card(label, value, detail)


def _render_dimensions(dimensions: dict[str, Any]) -> None:
    st.markdown("### Financial System State")
    rows = [{"Dimension": key.replace("_", " ").title(), "State": value.get("state"), "Explanation": value.get("explanation")} for key, value in dimensions.items() if isinstance(value, dict)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_cross_regime(signals: list[Any]) -> None:
    st.markdown("### Key Cross-Asset Signals")
    cols = st.columns(min(5, max(1, len(signals))))
    for col, item in zip(cols, signals[:5]):
        with col:
            if isinstance(item, dict):
                _metric_card(item.get("title", "Signal"), item.get("label", item.get("state", "")), item.get("description", ""))
            else:
                _metric_card(str(item), "", "")


def _render_asset_outlook(outlook: dict[str, Any]) -> None:
    st.markdown("### 3M Asset Outlook")
    cols = st.columns(4)
    for col, ticker in zip(cols, ASSET_TICKERS):
        item = outlook.get(ticker, {}) if isinstance(outlook, dict) else {}
        with col:
            st.markdown(f"#### {ticker} - 3M Outlook")
            _metric_card("Bias", item.get("bias_3m", "n/a"), f"Confidence {item.get('confidence', 'n/a')}")
            st.caption(str(item.get("expected_environment", "")))
            st.markdown("**Top Supports**")
            st.markdown(items_markdown(item.get("supporting_factors", [])))
            st.markdown("**Top Risks**")
            st.markdown(items_markdown(item.get("risk_factors", [])))
            st.markdown("**More Bullish If**")
            st.markdown(items_markdown(item.get("what_would_make_more_bullish", [])))
            st.markdown("**More Bearish If**")
            st.markdown(items_markdown(item.get("what_would_make_more_bearish", [])))


def _render_scenarios(scenarios: dict[str, Any]) -> None:
    st.markdown("### 3M Scenario Analysis")
    cols = st.columns(3)
    for col, key, title in zip(cols, ["base", "bull", "bear"], ["Base Case", "Bull Case", "Bear Case"]):
        scenario = scenarios.get(key, {}) if isinstance(scenarios, dict) else {}
        with col:
            st.markdown(f"#### {title}")
            st.markdown(items_markdown(scenario.get("scenario_conditions", [])))
            st.caption(str(scenario.get("system_implication", "")))
            rows = [{"Asset": asset, "Impact": scenario.get(asset, scenario.get(asset.replace("-USD", ""), "n/a"))} for asset in ASSET_TICKERS]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_key_risks(risks: list[Any]) -> None:
    st.markdown("### Key Risks")
    rows = []
    for idx, risk in enumerate(risks[:5], start=1):
        if isinstance(risk, dict):
            rows.append(
                {
                    "Rank": risk.get("rank", idx),
                    "Risk": risk.get("title", "n/a"),
                    "Description": risk.get("description", ""),
                    "Assets": ", ".join(map(str, risk.get("affected_assets", []))),
                    "Trigger": risk.get("trigger_to_watch", ""),
                }
            )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_change_view(change_view: dict[str, Any]) -> None:
    st.markdown("### What Would Change Our View?")
    tabs = st.tabs(ASSET_TICKERS)
    for tab, ticker in zip(tabs, ASSET_TICKERS):
        item = change_view.get(ticker, {}) if isinstance(change_view, dict) else {}
        with tab:
            left, right = st.columns(2)
            with left:
                st.markdown("**Bullish Triggers**")
                st.markdown(items_markdown(item.get("bullish_triggers", [])))
            with right:
                st.markdown("**Bearish Triggers**")
                st.markdown(items_markdown(item.get("bearish_triggers", [])))


def _render_diagnostics(result: CioRunResult) -> None:
    st.markdown("### First Run Diagnostics")
    st.caption(f"Model: {CIO_MODEL} | reasoning: {CIO_REASONING_EFFORT} | critic: {CIO_CRITIC_ENABLED}")
    st.json(
        {
            "status": result.status,
            "generated_at": result.generated_at,
            "next_scheduled_at": result.next_scheduled_at,
            "usage": result.usage,
            "monthly_cost_warning_usd": CIO_MONTHLY_COST_WARNING_USD,
        },
        expanded=False,
    )
    with st.expander("A. Snapshot JSON sent to GPT", expanded=True):
        st.json(result.snapshot)
    with st.expander("B. Analyst result", expanded=True):
        st.json(result.analyst_result)
    with st.expander("C. Critic result", expanded=True):
        st.json(result.critic_result or {})
    with st.expander("D. Final result", expanded=True):
        st.json(result.final_result)


def _metric_card(label: Any, value: Any, detail: Any = "") -> None:
    st.markdown(
        f"""
<div style="padding:0.65rem 0; line-height:1.15;">
  <div style="font-size:0.72rem; color:#94a3b8; font-weight:700;">{escape(label)}</div>
  <div style="font-size:1.0rem; color:#f8fafc; font-weight:800;">{escape(value)}</div>
  <div style="font-size:0.72rem; color:#cbd5e1;">{escape(detail)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def latest_btc_etf_flows() -> dict[str, Any]:
    try:
        cache = FundFlowCache(default_fund_flow_cache_path())
        frames = []
        for ticker in BTC_SPOT_ETF_FLOW_TICKERS:
            observations = cache.load_observations(ticker, pd.Timestamp("2024-01-01").date())
            if observations:
                frames.append(pd.DataFrame({"date": [obs.date for obs in observations], "flow": [obs.net_flow for obs in observations]}))
        if not frames:
            return {"state": "PARTIAL_DATA"}
        daily = pd.concat(frames, ignore_index=True)
        weekly = daily.assign(date=pd.to_datetime(daily["date"]), flow=pd.to_numeric(daily["flow"], errors="coerce")).set_index("date").resample("W-FRI")["flow"].sum()
        if weekly.empty:
            return {"state": "PARTIAL_DATA"}
        flow_4w = float(weekly.tail(4).sum())
        state = "POSITIVE" if flow_4w > 0 else "NEGATIVE" if flow_4w < 0 else "NEUTRAL"
        return {"state": state, "flow_1w": clean_value(weekly.iloc[-1]), "flow_4w": clean_value(flow_4w), "last_updated": pd.Timestamp(weekly.index[-1]).date().isoformat()}
    except Exception as exc:
        return {"state": "ERROR", "error": str(exc)}


def latest_bybit_btc() -> dict[str, Any]:
    try:
        frame = read_bybit_storage(BYBIT_STORAGE_PATH)
    except Exception as exc:
        return {"data_status": "ERROR", "error": str(exc)}
    if frame.empty or "asset" not in frame.columns:
        return {"data_status": "MISSING"}
    btc = frame[frame["asset"].astype(str).str.upper().eq("BTC-USD")].copy()
    if btc.empty:
        return {"data_status": "MISSING"}
    btc["date"] = pd.to_datetime(btc["date"], errors="coerce")
    row = btc.dropna(subset=["date"]).sort_values("date").tail(1).to_dict("records")
    return {key: clean_value(value) for key, value in (row[0] if row else {}).items()}


def btc_halving_phase() -> dict[str, Any]:
    current = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    halving = pd.Timestamp("2024-04-20")
    top = pd.Timestamp("2025-10-01")
    months_since_halving = (current - halving).days / 30.4375
    months_since_top = (current - top).days / 30.4375 if current >= top else math.nan
    if months_since_halving < 6:
        phase = "POST_HALVING_EARLY"
    elif months_since_halving < 12:
        phase = "BULL_EXPANSION"
    elif months_since_halving < 18:
        phase = "LATE_BULL_PEAK_WINDOW"
    elif months_since_halving < 30:
        phase = "POST_PEAK_BEAR"
    else:
        phase = "ACCUMULATION_PRE_HALVING"
    return {"phase": phase, "months_since_halving": clean_value(months_since_halving), "months_since_cycle_top": clean_value(months_since_top)}


def btc_final_state(alpha: Any, liquidity: dict[str, Any], etf: dict[str, Any], bybit: dict[str, Any], forward_risk: Any) -> str:
    alpha_value = to_float(alpha)
    risk = to_float(forward_risk)
    if alpha_value >= 60 and etf.get("state") == "POSITIVE" and risk < 50:
        return "BULLISH_WITH_FLOW_SUPPORT"
    if risk >= 60 or "DETERIORATING" in str(liquidity.get("global_liquidity_direction", "")):
        return "MACRO_RISK_WARNING"
    if alpha_value < 40 and np.isfinite(alpha_value):
        return "WEAK_TREND"
    return "NEUTRAL"


def btc_bottom_status(phase: str, alpha: Any, liquidity: dict[str, Any], etf: dict[str, Any], bybit: dict[str, Any], forward_risk: Any) -> str:
    if phase not in {"POST_PEAK_BEAR", "ACCUMULATION_PRE_HALVING"}:
        return "NO_BOTTOM_SIGNAL"
    signals = 0
    signals += int(to_float(alpha) >= 50)
    signals += int(etf.get("state") == "POSITIVE")
    signals += int(to_float(bybit.get("oi_change_4w_pct")) < 0)
    signals += int(abs(to_float(bybit.get("funding_28d"))) < 0.01)
    signals += int(to_float(forward_risk) < 45)
    signals += int(str(liquidity.get("global_liquidity_direction")) in {"ACCELERATING", "IMPROVING", "STABLE"})
    if signals >= 5:
        return "BOTTOM_CONFIRMED"
    if signals >= 4:
        return "CANDIDATE_BOTTOM"
    if signals >= 3:
        return "BOTTOMING_WATCH"
    return "BOTTOM_NOT_CONFIRMED"


def data_quality_snapshot(raw: pd.DataFrame, monthly: pd.DataFrame, weekly: pd.DataFrame, macro: dict[str, Any], gold: dict[str, Any], btc: dict[str, Any]) -> dict[str, Any]:
    notes = []
    missing = []
    stale = []
    partial = []
    for name, frame in [("raw_liquidity", raw), ("monthly_liquidity", monthly), ("weekly_liquidity", weekly)]:
        if frame.empty:
            missing.append(name)
    for item in macro.get("items", []):
        status = str(item.get("data_status", ""))
        if status == "MISSING":
            missing.append(str(item.get("instrument")))
        elif "STALE" in status:
            stale.append(str(item.get("instrument")))
        elif "PARTIAL" in status or "FALLBACK" in status:
            partial.append(str(item.get("instrument")))
    if gold.get("data_status") == "ERROR":
        notes.append(f"Gold regime error: {gold.get('error')}")
    if btc.get("data_status") == "ERROR":
        notes.append(f"BTC regime error: {btc.get('error')}")
    status = "OK" if not missing and not stale else "PARTIAL_DATA"
    return {"status": status, "missing_series": missing, "stale_series": stale, "partial_data": partial, "notes": notes}


def table_row(table_df: pd.DataFrame, ticker: str) -> dict[str, Any]:
    if table_df is None or table_df.empty or "Ticker" not in table_df.columns:
        return {}
    rows = table_df[table_df["Ticker"].astype(str).str.upper().eq(ticker.upper())]
    return rows.tail(1).to_dict("records")[0] if not rows.empty else {}


def table_value(table_df: pd.DataFrame, ticker: str, column: str) -> float:
    return to_float(table_row(table_df, ticker).get(column))


def latest_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    return frame.dropna(how="all").tail(1).to_dict("records")[0]


def latest_close(ticker: str) -> Any:
    try:
        frame = download_completed_ohlcv(ticker, period="2y")
        close = pd.to_numeric(frame.get("Close", pd.Series(dtype="float64")), errors="coerce").dropna()
        return clean_value(close.iloc[-1]) if not close.empty else None
    except Exception:
        return None


def price_returns_from_history(frame: pd.DataFrame, price_col: str) -> dict[str, Any]:
    if frame.empty or price_col not in frame.columns:
        return {"1m": None, "3m": None, "6m": None, "12m": None}
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    series = pd.Series(pd.to_numeric(data[price_col], errors="coerce").values, index=data["date"]).dropna().sort_index()
    if series.empty:
        return {"1m": None, "3m": None, "6m": None, "12m": None}
    end = series.index[-1]
    return {label: clean_value(performance_on_or_before(series, end, days)) for label, days in [("1m", 30), ("3m", 90), ("6m", 182), ("12m", 365)]}


def performance_on_or_before(series: pd.Series, end: pd.Timestamp, days: int) -> float:
    start = end - pd.DateOffset(days=int(days))
    before = series.loc[:start]
    if before.empty:
        return math.nan
    v0 = float(before.iloc[-1])
    v1 = float(series.loc[:end].iloc[-1])
    return math.nan if v0 == 0 or not np.isfinite(v0) or not np.isfinite(v1) else (v1 / v0 - 1.0) * 100.0


def weighted_mean(values: list[float], weights: list[float]) -> float:
    arr = np.array(values, dtype="float64")
    w = np.array(weights, dtype="float64")
    mask = np.isfinite(arr)
    return float(np.average(arr[mask], weights=w[mask])) if mask.any() else math.nan


def state_from_score(value: Any) -> str:
    score = to_float(value)
    if not np.isfinite(score):
        return "NEUTRAL"
    if score >= 70:
        return "VERY_POSITIVE"
    if score >= 55:
        return "POSITIVE"
    if score >= 40:
        return "NEUTRAL"
    if score >= 25:
        return "NEGATIVE"
    return "VERY_NEGATIVE"


def state_from_risk(value: Any) -> str:
    risk = to_float(value)
    if not np.isfinite(risk):
        return "NEUTRAL"
    if risk >= 70:
        return "VERY_NEGATIVE"
    if risk >= 50:
        return "NEGATIVE"
    if risk >= 30:
        return "NEUTRAL"
    return "POSITIVE"


def read_cio_cache() -> dict[str, Any] | None:
    if not CIO_STORAGE_PATH.exists():
        return None
    try:
        return json.loads(CIO_STORAGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_cio_cache(result: CioRunResult) -> None:
    CIO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    CIO_STORAGE_PATH.write_text(
        json.dumps(
            {
                "snapshot": result.snapshot,
                "analyst_result": result.analyst_result,
                "critic_result": result.critic_result,
                "final_result": result.final_result,
                "usage": result.usage,
                "status": result.status,
                "generated_at": result.generated_at,
                "next_scheduled_at": result.next_scheduled_at,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def cio_result_from_cache(cache: dict[str, Any], status: str) -> CioRunResult:
    return CioRunResult(
        snapshot=cache.get("snapshot", {}),
        analyst_result=cache.get("analyst_result", {}),
        critic_result=cache.get("critic_result"),
        final_result=cache.get("final_result", {}),
        usage=cache.get("usage", {}),
        status=status,
        generated_at=str(cache.get("generated_at", "")),
        next_scheduled_at=str(cache.get("next_scheduled_at", "")),
    )


def cache_is_fresh(cache: dict[str, Any]) -> bool:
    generated = pd.to_datetime(cache.get("generated_at"), errors="coerce", utc=True)
    if pd.isna(generated):
        return False
    return (pd.Timestamp.now(tz="UTC") - generated).total_seconds() < CIO_REFRESH_SECONDS


def next_weekly_timestamp(value: str) -> str:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        ts = pd.Timestamp.now(tz="UTC")
    return (ts + pd.Timedelta(seconds=CIO_REFRESH_SECONDS)).isoformat()


def get_openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        key = str(st.secrets["OPENAI_API_KEY"]).strip() if "OPENAI_API_KEY" in st.secrets else ""
    except StreamlitSecretNotFoundError:
        key = ""
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    return key


def parse_json_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    try:
        result = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        result = json.loads(stripped[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("CIO response JSON must be an object.")
    return result


def extract_response_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        key: clean_value(getattr(usage, key))
        for key in ("input_tokens", "output_tokens", "total_tokens")
        if hasattr(usage, key)
    } or None


def clean_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if not isinstance(value, (list, dict, tuple, set)) and pd.isna(value):
        return None
    return value


def to_float(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else math.nan
    except Exception:
        return math.nan


def snake_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("/", "_")


def items_markdown(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return "- n/a"
    return "\n".join(f"- {escape(item)}" for item in items[:5])


def escape(value: Any) -> str:
    import html

    return html.escape("n/a" if value is None else str(value))
