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

Return only valid JSON matching the requested structure.

Required JSON keys:
as_of_date, analysis_source, overall_system_state, dimensions, cross_regime_signals,
asset_outlook_3m, scenarios, key_risks, what_would_change_view, data_quality_notes,
critic_changes_summary.

Dimension vocabularies:
Growth = ACCELERATING/STABLE/SLOWING/CONTRACTING/UNKNOWN.
Inflation = RISING/STABLE/FALLING/REACCELERATING/UNKNOWN.
Liquidity state must include level and direction, not a single generic word.
Rates = EASING/NEUTRAL/RESTRICTIVE/TIGHTENING/UNKNOWN.
Financial Conditions = EASY/NEUTRAL/TIGHTENING/TIGHT/UNKNOWN.
Credit = BENIGN/NORMAL/DETERIORATING/STRESSED/UNKNOWN.
Risk Appetite = POSITIVE/NEUTRAL/NEGATIVE/UNKNOWN.
Market Trend = BULL/BULL_WITH_WARNING/DETERIORATING/CORRECTION/STRESS/UNKNOWN.
Systemic Stress = LOW/ELEVATED/HIGH/EXTREME/UNKNOWN.

Asset driver rules:
SPY prioritizes structural market regime, macro transition risk, liquidity score/direction,
credit, breadth, DXY/rates. QQQ is more sensitive than SPY to liquidity, real yields,
US2Y, DXY and financial conditions. GLD prioritizes Gold Alpha, Gold Structural Macro,
Gold Forward Macro Risk, Gold Tactical Flow, ETF flows, COT positioning and rates; do not
use Global Liquidity as a primary Gold trigger. BTC uses Halving Phase x Global Liquidity
x Trend/Alpha x ETF flows x OI/funding/basis; halving phase alone is not directional.

For each asset output bias_3m, confidence, expected_environment, top_supports, top_risks,
more_bullish_if, more_bearish_if. Confidence must be 0.50-0.85; cap at 0.60 for important
missing data and at 0.65 for major conflicts.

Key risks must include status ACTIVE/WATCH/NOT_ACTIVE and trigger_to_escalate. Do not list
an already-active condition as a future trigger."""


@dataclass(frozen=True)
class CioRunResult:
    snapshot: dict[str, Any]
    analyst_result: dict[str, Any]
    critic_result: dict[str, Any] | None
    final_result: dict[str, Any]
    usage: dict[str, Any]
    status: str
    analysis_source: str
    analyst_status: str
    critic_status: str
    generated_at: str
    next_scheduled_at: str
    analyst_raw: dict[str, Any] | None = None
    analyst_normalized: dict[str, Any] | None = None
    critic_raw: dict[str, Any] | None = None
    critic_normalized: dict[str, Any] | None = None


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
    final = safe_dict(result.final_result)
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
    return generate_cio_view(table_df, market_snapshot, fred_api_key, force=True)


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
        "openai_api": "FAILED",
        "analyst": "FAILED",
        "critic": "SKIPPED" if not CIO_CRITIC_ENABLED else "FAILED",
        "estimated_cost": None,
    }
    analyst_raw: dict[str, Any] | None = None
    analyst: dict[str, Any]
    critic_raw: dict[str, Any] | None = None
    critic: dict[str, Any] | None = None
    analysis_source = "DETERMINISTIC_FALLBACK"
    analyst_status = "FAILED"
    critic_status = "SKIPPED" if not CIO_CRITIC_ENABLED else "FAILED"
    try:
        analyst_raw = call_cio_llm(snapshot, CIO_REASONING_EFFORT)
        usage["analyst_usage"] = analyst_raw.pop("_response_usage", None)
        analyst = normalize_cio_result(analyst_raw, snapshot, "LLM_ANALYST")
        usage["openai_api"] = "CONNECTED"
        usage["analyst"] = "SUCCESS"
        usage["analyst_schema_valid"] = "YES" if not analyst.get("_validation_errors") else "NO"
        analyst_status = "SUCCESS"
        analysis_source = "LLM_ANALYST"
        status = "CURRENT"
    except Exception as exc:
        analyst_raw = None
        analyst = normalize_cio_result(deterministic_cio_view(snapshot, "LLM_UNAVAILABLE"), snapshot, "DETERMINISTIC_FALLBACK")
        critic = None
        final = analyst
        usage["analyst_error"] = brief_error(exc)
        usage["analyst_schema_valid"] = "NO"
        status = "LLM_UNAVAILABLE"
    else:
        if CIO_CRITIC_ENABLED:
            try:
                critic_raw = call_cio_critic(snapshot, analyst)
                usage["critic_usage"] = critic_raw.pop("_response_usage", None)
                critic = normalize_cio_result(critic_raw, snapshot, "LLM_CRITIC_CORRECTED")
                usage["critic"] = "SUCCESS"
                usage["critic_schema_valid"] = "YES" if not critic.get("_validation_errors") else "NO"
                critic_status = "SUCCESS"
                analysis_source = "LLM_CRITIC_CORRECTED"
            except Exception as exc:
                usage["critic"] = "FAILED"
                usage["critic_error"] = brief_error(exc)
                usage["critic_schema_valid"] = "NO"
                critic_status = "FAILED"
        final = critic or analyst
    final = normalize_cio_result(final, snapshot, analysis_source)
    usage["analysis_source"] = analysis_source
    usage["analyst_status"] = analyst_status
    usage["critic_status"] = critic_status
    usage["normalization_applied"] = "YES"
    usage["validation_errors_count"] = len(final.get("_validation_errors", []))
    usage["section_sources"] = final.get("section_sources", {})
    usage["sections_using_fallback"] = [name for name, source in final.get("section_sources", {}).items() if source == "FALLBACK"]
    usage["token_usage"] = summarize_token_usage(usage)
    usage["estimated_cost"] = estimate_usage_cost(usage["token_usage"])
    usage["latency_seconds"] = round(time.time() - started, 2)
    generated_at = datetime.now(timezone.utc).isoformat()
    result = CioRunResult(
        snapshot=snapshot,
        analyst_result=analyst,
        critic_result=critic,
        final_result=final,
        usage=usage,
        status=status,
        analysis_source=analysis_source,
        analyst_status=analyst_status,
        critic_status=critic_status,
        generated_at=generated_at,
        next_scheduled_at=next_weekly_timestamp(generated_at),
        analyst_raw=analyst_raw,
        analyst_normalized=analyst,
        critic_raw=critic_raw,
        critic_normalized=critic,
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
    snapshot = {
        "as_of_date": as_of,
        "market_regime": compact_market_snapshot(market_snapshot),
        "global_liquidity": liquidity,
        "global_macro": macro,
        "gold_regime": gold,
        "btc_regime": btc,
        "asset_market_data": assets,
        "data_quality": data_quality,
    }
    validation_errors = validate_cio_snapshot(snapshot)
    snapshot["snapshot_validation_errors"] = validation_errors
    if validation_errors:
        snapshot["data_quality"]["status"] = "PARTIAL_DATA"
        snapshot["data_quality"].setdefault("invalid_scores", []).extend(validation_errors)
        snapshot["data_quality"].setdefault("notes", []).append(f"{len(validation_errors)} score-like fields failed 0-100 validation.")
    return snapshot


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
        "positioning_risk": "PositioningRisk",
        "positioning_state": "PositioningState",
        "tail_risk_flag": "TailRiskFlag",
        "tail_risk_reason": "TailRiskReason",
        "liquidity_warning": "LiquidityWarning",
        "credit_warning": "CreditWarning",
        "fast_warning": "FastWarning",
        "macro_warning": "MacroWarning",
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
        current_alpha = clean_value(current.get("gold_alpha"))
        gold_alpha_valid = is_score_like(current_alpha)
        if not gold_alpha_valid and is_score_like(gold_alpha):
            current_alpha = clean_value(gold_alpha)
            gold_alpha_valid = True
        return {
            "final_state": clean_value(current.get("gold_regime")),
            "gold_alpha": current_alpha if gold_alpha_valid else None,
            "gold_alpha_valid": gold_alpha_valid,
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
    current_price = latest_close(ticker)
    if current_price is None:
        current_price = first_valid(row, ["Current_Price", "Price", "Last", "Close", "WeeklyClose_Last"])
    out = {
        "ticker": ticker,
        "current_price": clean_value(current_price),
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
        "Use only supplied inputs. Treat snapshot_validation_errors as invalid data, not normal valid scores. "
        "Make SPY, QQQ, GLD and BTC-USD asset-specific; avoid generic reused supports/risks/triggers.\n\n"
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
        "halving treated as deterministic BTC forecast, active risks incorrectly shown as future triggers, "
        "SPY/QQQ not differentiated enough, invalid/null values treated as valid, and confidence values that are too high. "
        "Return the corrected result using EXACTLY the same JSON schema and key names as the Analyst input. "
        "Do not rename fields. Do not change object/list/string types. Do not add alternate root structures. "
        "Return corrected final JSON only, with critic_changes_summary explaining material corrections.\n\n"
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
        "analysis_source": "DETERMINISTIC_FALLBACK",
        "overall_system_state": {
            "label": label,
            "summary": deterministic_system_summary(snapshot, label),
        },
        "dimensions": fallback_dimensions(snapshot),
        "cross_regime_signals": fallback_cross_regime_signals(market, liquidity, gold, btc),
        "asset_outlook_3m": {
            "SPY": fallback_asset_outlook("SPY", snapshot),
            "QQQ": fallback_asset_outlook("QQQ", snapshot),
            "GLD": fallback_asset_outlook("GLD", snapshot),
            "BTC-USD": fallback_asset_outlook("BTC-USD", snapshot),
        },
        "scenarios": fallback_scenarios(snapshot),
        "key_risks": fallback_key_risks(snapshot),
        "what_would_change_view": fallback_change_view(),
        "critic_changes_summary": [],
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


def deterministic_system_summary(snapshot: dict[str, Any], label: str) -> str:
    market = snapshot.get("market_regime", {})
    liquidity = snapshot.get("global_liquidity", {})
    return (
        f"{label}: market structure is {market.get('structural_regime', 'UNKNOWN')} / "
        f"{market.get('final_state', 'UNKNOWN')}; liquidity level is "
        f"{state_from_liquidity_score(liquidity.get('global_liquidity_score'))} and direction is "
        f"{liquidity.get('global_liquidity_direction', 'UNKNOWN')}; macro transition risk is "
        f"{market.get('macro_transition_risk', 'n/a')}; systemic stress is estimated as "
        f"{systemic_stress_state(market)}."
    )


def fallback_dimensions(snapshot: dict[str, Any]) -> dict[str, dict[str, str]]:
    market = snapshot.get("market_regime", {})
    liquidity = snapshot.get("global_liquidity", {})
    macro = snapshot.get("global_macro", {})
    growth_state = growth_dimension_state(macro)
    inflation_state = inflation_dimension_state(macro)
    rates_state = rates_dimension_state(macro)
    fc_state = financial_conditions_state(macro, market)
    credit_state = credit_dimension_state(macro)
    liquidity_level = state_from_liquidity_score(liquidity.get("global_liquidity_score"))
    liquidity_direction = str(liquidity.get("global_liquidity_direction", "UNKNOWN"))
    return {
        "growth": {"state": growth_state, "explanation": "Derived from PMI, CFNAI and jobless-claims items when present."},
        "inflation": {"state": inflation_state, "explanation": "Derived from breakevens, inflation expectations and WTI changes."},
        "liquidity": {"state": liquidity_level, "level": liquidity_level, "direction": liquidity_direction, "explanation": f"Score {liquidity.get('global_liquidity_score', 'n/a')}; 13W direction {liquidity_direction}."},
        "rates": {"state": rates_state, "explanation": "Derived from US2Y, 10Y real yield and curve pressure."},
        "financial_conditions": {"state": fc_state, "explanation": "Derived from VIX/MOVE, credit spreads and fast transition risk."},
        "credit": {"state": credit_state, "explanation": "Derived from HY/IG spread levels and changes."},
        "risk_appetite": {"state": risk_appetite_state(market), "explanation": str(market.get("fast_risk_state", "n/a"))},
        "market_trend": {"state": market_trend_state(market), "explanation": str(market.get("final_state", "n/a"))},
        "systemic_stress": {"state": systemic_stress_state(market), "explanation": f"Macro risk {market.get('macro_transition_risk', 'n/a')}; fast risk {market.get('fast_transition_risk', 'n/a')}."},
    }


def fallback_cross_regime_signals(market: dict[str, Any], liquidity: dict[str, Any], gold: dict[str, Any], btc: dict[str, Any]) -> list[dict[str, str]]:
    signals = []
    if str(market.get("structural_regime", "")).upper() == "BULL" and liquidity_direction_value(liquidity) < -10:
        signals.append({"name": "PRICE_LIQUIDITY_DIVERGENCE", "title": "PRICE_LIQUIDITY_DIVERGENCE", "description": "Structural equity trend is still bull while global liquidity momentum is deteriorating.", "affected_assets": ["SPY", "QQQ", "BTC-USD"]})
    if to_float(gold.get("gold_alpha")) >= 60 and to_float(gold.get("tactical_flow")) >= 60 and (to_float(gold.get("structural_macro")) < 40 or to_float(gold.get("forward_macro_risk")) > 60):
        signals.append({"name": "GOLD_MACRO_FLOW_CONFLICT", "title": "GOLD_MACRO_FLOW_CONFLICT", "description": "Gold alpha/flows are supportive while structural macro or forward macro risk is not.", "affected_assets": ["GLD"]})
    if str(btc.get("halving_phase")) in {"POST_PEAK_BEAR", "ACCUMULATION_PRE_HALVING"} and (to_float(btc.get("alpha")) >= 50 or str(btc.get("etf_flow_state")) == "POSITIVE") and "DETERIORATING" in str(btc.get("global_liquidity_direction")):
        signals.append({"name": "BTC_CYCLE_LIQUIDITY_CONFLICT", "title": "BTC_CYCLE_LIQUIDITY_CONFLICT", "description": "BTC cycle/flow setup is improving while global liquidity direction remains adverse.", "affected_assets": ["BTC-USD"]})
    if systemic_stress_state(market) == "LOW":
        signals.append({"name": "NO_SYSTEMIC_CREDIT_STRESS", "title": "NO_SYSTEMIC_CREDIT_STRESS", "description": "Risk warning is currently more liquidity/macro than systemic stress.", "affected_assets": ["SPY", "QQQ", "BTC-USD"]})
    if not signals:
        signals.append({"name": "NO_HIGH_PRIORITY_CONFLICT", "title": "NO_HIGH_PRIORITY_CONFLICT", "description": "Rule fallback did not identify a high-priority cross-regime conflict.", "affected_assets": ASSET_TICKERS})
    return signals[:5]


def fallback_asset_outlook(ticker: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    asset = snapshot.get("asset_market_data", {}).get(ticker, {})
    market = snapshot.get("market_regime", {})
    liquidity = snapshot.get("global_liquidity", {})
    gold = snapshot.get("gold_regime", {})
    btc = snapshot.get("btc_regime", {})
    alpha = to_float(asset.get("alpha_score"))
    bias = asset_fallback_bias(ticker, snapshot, alpha)
    supports, risks, bullish, bearish = asset_specific_lists(ticker, snapshot)
    return {
        "ticker": ticker,
        "bias_3m": bias,
        "confidence": fallback_confidence(snapshot),
        "expected_environment": asset_environment_text(ticker, market, liquidity, gold, btc),
        "top_supports": supports,
        "top_risks": risks,
        "supporting_factors": supports,
        "risk_factors": risks,
        "more_bullish_if": bullish,
        "more_bearish_if": bearish,
        "what_would_make_more_bullish": bullish,
        "what_would_make_more_bearish": bearish,
    }


def fallback_scenarios(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    market = snapshot.get("market_regime", {})
    liquidity = snapshot.get("global_liquidity", {})
    base_conditions = [
        f"Market state remains {market.get('final_state', 'UNKNOWN')}",
        f"Liquidity direction remains {liquidity.get('global_liquidity_direction', 'UNKNOWN')}",
        f"Macro transition risk remains {market.get('macro_transition_risk', 'n/a')}",
    ]
    bull_conditions = dynamic_bull_conditions(snapshot)
    bear_conditions = dynamic_bear_conditions(snapshot)
    return {
        "base": {"scenario_conditions": base_conditions, "system_implication": deterministic_system_summary(snapshot, deterministic_system_label(market, liquidity)), "SPY": "CONSTRUCTIVE_CONSTRAINED", "QQQ": "HIGHER_SENSITIVITY", "GLD": "DRIVER_DEPENDENT", "BTC-USD": "CYCLE_LIQUIDITY_DEPENDENT"},
        "bull": {"scenario_conditions": bull_conditions, "system_implication": "Risk appetite broadens if liquidity/rates improve without credit stress.", "SPY": "POSITIVE", "QQQ": "POSITIVE_HIGH_BETA", "GLD": "POSITIVE_IF_GOLD_DRIVERS_CONFIRM", "BTC-USD": "POSITIVE_IF_LIQUIDITY_AND_FLOWS_CONFIRM"},
        "bear": {"scenario_conditions": bear_conditions, "system_implication": "Correction risk rises if liquidity deterioration spreads into macro/credit stress.", "SPY": "NEGATIVE", "QQQ": "MORE_NEGATIVE_THAN_SPY", "GLD": "MIXED_UNLESS_GOLD_FLOWS_HOLD", "BTC-USD": "NEGATIVE_IF_LIQUIDITY_AND_LEVERAGE_WORSEN"},
    }


def fallback_key_risks(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    market = snapshot.get("market_regime", {})
    liquidity = snapshot.get("global_liquidity", {})
    liq_active = "DETERIORATING" in str(liquidity.get("global_liquidity_direction", "")) or liquidity_direction_value(liquidity) < -10
    macro_watch = to_float(market.get("macro_transition_risk")) >= 30
    return [
        {"rank": 1, "title": "Liquidity deterioration", "status": "ACTIVE" if liq_active else "WATCH", "description": "Global liquidity direction is the main cross-asset medium-term risk.", "affected_assets": ["SPY", "QQQ", "BTC-USD"], "trigger_to_escalate": "GlobalLiquidityScore < 40 or DETERIORATING_FAST persists another 4 weeks"},
        {"rank": 2, "title": "Macro transition risk", "status": "ACTIVE" if macro_watch else "WATCH", "description": "DXY, Fed liquidity and US2Y can move macro pressure from warning to correction risk.", "affected_assets": ["SPY", "QQQ", "BTC-USD"], "trigger_to_escalate": "MacroTransitionRisk > 40 with DXY/US2Y rising"},
        {"rank": 3, "title": "Credit stress", "status": "WATCH", "description": "Credit widening would turn a liquidity warning into broader risk-off stress.", "affected_assets": ["SPY", "QQQ", "BTC-USD"], "trigger_to_escalate": "HY/IG OAS widen while VIX/MOVE rise"},
        {"rank": 4, "title": "Gold macro-flow conflict", "status": "ACTIVE" if any(s.get("name") == "GOLD_MACRO_FLOW_CONFLICT" for s in fallback_cross_regime_signals(market, liquidity, snapshot.get("gold_regime", {}), snapshot.get("btc_regime", {}))) else "WATCH", "description": "Gold can stay supported by flows while macro scores are fragile.", "affected_assets": ["GLD"], "trigger_to_escalate": "Gold Alpha < 50 and ETF/COT support weakens"},
    ]


def fallback_change_view() -> dict[str, dict[str, list[str]]]:
    return {
        "SPY": {"bullish_triggers": ["GlobalLiquidityDirection > 0", "MacroTransitionRisk < 20", "HY/IG spreads stable or tightening", "Breadth confirmations improve"], "bearish_triggers": ["GlobalLiquidityScore < 40", "MacroTransitionRisk > 40", "HY/IG spreads widen", "Breadth deteriorates"]},
        "QQQ": {"bullish_triggers": ["GlobalLiquidityDirection > 0", "10Y real yield falls", "US2Y falls", "QQQ Alpha > 60"], "bearish_triggers": ["Liquidity deteriorates further", "10Y real yield rises", "DXY strengthens", "QQQ Alpha < 40"]},
        "GLD": {"bullish_triggers": ["Gold Alpha >= 70", "ETF Flow Score >= 70", "Real yield falling", "Structural Macro improving"], "bearish_triggers": ["Gold Alpha < 50", "ETF Flow < 40", "Structural Macro < 40", "Forward Macro Risk high and flows weaken"]},
        "BTC-USD": {"bullish_triggers": ["Halving setup aligns with improving liquidity", "BTC Alpha > 60", "BTC ETF flows improve", "OI/funding/basis remain non-extreme"], "bearish_triggers": ["Global liquidity deteriorates", "BTC Alpha < 40", "ETF flows weaken", "OI/funding leverage rises without price confirmation"]},
    }


def _render_cio_top_cards(result: CioRunResult) -> None:
    final = safe_dict(result.final_result)
    state = safe_dict(final.get("overall_system_state"))
    snapshot = safe_dict(result.snapshot)
    gold = safe_dict(snapshot.get("gold_regime"))
    btc = safe_dict(snapshot.get("btc_regime"))
    quality = safe_dict(snapshot.get("data_quality"))
    cards = [
        ("Overall System State", state.get("label") or "n/a", result.status),
        ("Analysis Source", display_analysis_source(result.analysis_source), f"Analyst {result.analyst_status} / Critic {result.critic_status}"),
        ("Market Regime", safe_dict(snapshot.get("market_regime")).get("final_state", "n/a"), safe_dict(snapshot.get("market_regime")).get("structural_regime", "n/a")),
        ("Global Liquidity", safe_dict(snapshot.get("global_liquidity")).get("global_liquidity_backdrop", "n/a"), safe_dict(snapshot.get("global_liquidity")).get("global_liquidity_direction", "n/a")),
        ("Gold Regime", gold.get("final_state", "n/a"), f"Alpha {display_value(gold.get('gold_alpha', 'n/a'))}"),
        ("BTC Tactical / Regime State", btc.get("final_state", "n/a"), ""),
        ("BTC Halving Cycle Phase", btc.get("halving_phase", "n/a"), ""),
        ("Last Analysis", result.generated_at[:19], result.status),
        ("Next Scheduled", result.next_scheduled_at[:19], "weekly"),
        ("Data Quality", quality.get("status", "n/a"), "; ".join(data_quality_reasons(snapshot)[:2]) or f"{len(safe_list(quality.get('notes')))} notes"),
    ]
    for start in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, (label, value, detail) in zip(cols, cards[start : start + 4]):
            with col:
                _metric_card(label, value, detail)


def _render_dimensions(dimensions: dict[str, Any]) -> None:
    st.markdown("### Financial System State")
    rows = []
    for key, value in safe_dict(dimensions).items():
        item = safe_dict(value)
        state = item.get("state") or item.get("level") or "UNKNOWN"
        if key == "liquidity":
            state = f"{item.get('state', 'UNKNOWN')} | Level {item.get('level', 'UNKNOWN')} | Direction {item.get('direction', 'UNKNOWN')}"
        rows.append({"Dimension": key.replace("_", " ").title(), "State": display_value(state), "Explanation": safe_str(item.get("explanation") or "")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_cross_regime(signals: list[Any]) -> None:
    st.markdown("### Key Cross-Asset Signals")
    items = safe_list(signals)
    cols = st.columns(min(5, max(1, len(items))))
    for col, item in zip(cols, items[:5]):
        with col:
            data = safe_dict(item)
            signal = data.get("signal") or data.get("name") or data.get("title") or data.get("label") or "UNNAMED_SIGNAL"
            assets = ", ".join(normalize_assets(data.get("affected_assets", data.get("assets", []))))
            _metric_card(signal, assets, data.get("description", safe_str(item) if not data else ""))


def _render_asset_outlook(outlook: dict[str, Any]) -> None:
    st.markdown("### 3M Asset Outlook")
    cols = st.columns(4)
    for col, ticker in zip(cols, ASSET_TICKERS):
        item = safe_dict(safe_dict(outlook).get(ticker))
        with col:
            st.markdown(f"#### {ticker} - 3M Outlook")
            _metric_card("Bias", item.get("bias", item.get("bias_3m", "n/a")), f"Confidence {display_value(item.get('confidence', 'n/a'))}")
            st.markdown(safe_str(item.get("expected_environment", "")))
            st.markdown("**Top Supports**")
            st.markdown(items_markdown(item.get("top_supports", item.get("supporting_factors", []))))
            st.markdown("**Top Risks**")
            st.markdown(items_markdown(item.get("top_risks", item.get("risk_factors", []))))
            st.markdown("**More Bullish If**")
            st.markdown(items_markdown(item.get("more_bullish_if", item.get("what_would_make_more_bullish", []))))
            st.markdown("**More Bearish If**")
            st.markdown(items_markdown(item.get("more_bearish_if", item.get("what_would_make_more_bearish", []))))


def _render_scenarios(scenarios: dict[str, Any]) -> None:
    st.markdown("### 3M Scenario Analysis")
    cols = st.columns(3)
    for col, key, title in zip(cols, ["base", "bull", "bear"], ["Base Case", "Bull Case", "Bear Case"]):
        scenario = safe_dict(safe_dict(scenarios).get(key))
        with col:
            st.markdown(f"#### {title}")
            st.markdown(items_markdown(scenario.get("conditions", scenario.get("scenario_conditions", []))))
            st.caption(safe_str(scenario.get("summary", scenario.get("system_implication", ""))))
            impacts = safe_dict(scenario.get("asset_impacts"))
            rows = [{"Asset": asset, "Impact": display_value(impacts.get(asset, scenario.get(asset, scenario.get(asset.replace("-USD", ""), "n/a"))))} for asset in ASSET_TICKERS]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_key_risks(risks: list[Any]) -> None:
    st.markdown("### Key Risks")
    rows = []
    for idx, risk in enumerate(safe_list(risks)[:5], start=1):
        item = safe_dict(risk)
        rows.append(
            {
                "Rank": display_value(item.get("rank", idx)),
                "Risk": item.get("risk") or item.get("title") or item.get("name") or "UNNAMED_RISK",
                "Status": item.get("status", "n/a"),
                "Description": safe_str(item.get("description", "")),
                "Assets": ", ".join(normalize_assets(item.get("affected_assets", item.get("assets", [])))),
                "Escalation Trigger": item.get("escalation_trigger") or item.get("trigger_to_escalate") or item.get("trigger_to_watch") or "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_change_view(change_view: dict[str, Any]) -> None:
    st.markdown("### What Would Change Our View?")
    tabs = st.tabs(ASSET_TICKERS)
    for tab, ticker in zip(tabs, ASSET_TICKERS):
        raw_item = safe_dict(change_view).get(ticker, {})
        item = safe_dict(raw_item)
        with tab:
            left, right = st.columns(2)
            with left:
                st.markdown("**Bullish Triggers**")
                st.markdown(items_markdown(item.get("bullish_triggers", [])))
            with right:
                st.markdown("**Bearish Triggers**")
                st.markdown(items_markdown(item.get("bearish_triggers", [])))
            if isinstance(raw_item, str):
                st.caption(raw_item)


def _render_diagnostics(result: CioRunResult) -> None:
    st.markdown("### First Run Diagnostics")
    st.caption(f"Model: {CIO_MODEL} | reasoning: {CIO_REASONING_EFFORT} | critic: {CIO_CRITIC_ENABLED}")
    st.json(
        {
            "status": result.status,
            "openai_api": result.usage.get("openai_api"),
            "analysis_source": result.analysis_source,
            "analyst_status": result.analyst_status,
            "critic_status": result.critic_status,
            "generated_at": result.generated_at,
            "next_scheduled_at": result.next_scheduled_at,
            "http_api_error_message": result.usage.get("analyst_error") or result.usage.get("critic_error"),
            "input_tokens": result.usage.get("token_usage", {}).get("input_tokens"),
            "cached_input_tokens": result.usage.get("token_usage", {}).get("cached_input_tokens"),
            "output_tokens": result.usage.get("token_usage", {}).get("output_tokens"),
            "estimated_cost": result.usage.get("estimated_cost"),
            "latency_seconds": result.usage.get("latency_seconds"),
            "monthly_cost_warning_usd": CIO_MONTHLY_COST_WARNING_USD,
        },
        expanded=False,
    )
    if result.analysis_source == "DETERMINISTIC_FALLBACK":
        st.warning("CIO View currently uses deterministic fallback. GPT synthesis was unavailable or has not been generated yet.")
    with st.expander("A. Snapshot Validation Result", expanded=True):
        st.json(result.snapshot.get("snapshot_validation_errors", []))
    with st.expander("B. Snapshot JSON sent to GPT", expanded=True):
        st.json(result.snapshot)
    with st.expander("C. Analyst status/result", expanded=True):
        st.caption(result.analyst_status)
        st.json({"raw": result.analyst_raw or result.analyst_result, "normalized": result.analyst_normalized or result.analyst_result})
    with st.expander("D. Critic status/result", expanded=True):
        st.caption(result.critic_status)
        st.json({"raw": result.critic_raw or result.critic_result or {}, "normalized": result.critic_normalized or result.critic_result or {}})
    with st.expander("E. Final result", expanded=True):
        st.json(result.final_result)
    with st.expander("F-I. Source, Token Usage, Estimated Cost, Validation Warnings", expanded=True):
        st.json(
            {
                "analysis_source": result.analysis_source,
                "usage": result.usage,
                "validation_warnings": result.snapshot.get("snapshot_validation_errors", []),
                "section_sources": result.usage.get("section_sources", safe_dict(result.final_result).get("section_sources", {})),
                "sections_using_fallback": result.usage.get("sections_using_fallback", []),
            }
        )


def _metric_card(label: Any, value: Any, detail: Any = "") -> None:
    st.markdown(
        f"""
<div style="padding:0.65rem 0; line-height:1.15;">
  <div style="font-size:0.72rem; color:#94a3b8; font-weight:700;">{escape(label)}</div>
  <div style="font-size:1.0rem; color:#f8fafc; font-weight:800;">{escape(display_value(value))}</div>
  <div style="font-size:0.72rem; color:#cbd5e1;">{escape(display_value(detail))}</div>
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


SCORE_PATHS = {
    "Alpha": [
        ("gold_regime", "gold_alpha"),
        ("btc_regime", "alpha"),
        ("asset_market_data", "SPY", "alpha_score"),
        ("asset_market_data", "QQQ", "alpha_score"),
        ("asset_market_data", "GLD", "alpha_score"),
        ("asset_market_data", "BTC-USD", "alpha_score"),
    ],
    "StructuralMacro": [("gold_regime", "structural_macro"), ("btc_regime", "structural_macro")],
    "ForwardMacroRisk": [("gold_regime", "forward_macro_risk"), ("btc_regime", "forward_macro_risk")],
    "TacticalFlow": [("gold_regime", "tactical_flow")],
    "FastTransitionRisk": [("market_regime", "fast_transition_risk")],
    "MacroTransitionRisk": [("market_regime", "macro_transition_risk")],
    "GlobalLiquidityScore": [("global_liquidity", "global_liquidity_score"), ("btc_regime", "global_liquidity_score")],
    "M2Impulse": [("global_liquidity", "m2_impulse")],
    "CBImpulse": [("global_liquidity", "cb_impulse")],
    "USNLImpulse": [("global_liquidity", "us_net_liquidity_impulse")],
}


def validate_cio_snapshot(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for field, paths in SCORE_PATHS.items():
        for path in paths:
            value = get_path(snapshot, path)
            if value is None:
                continue
            number = to_float(value)
            if not np.isfinite(number) or number < 0 or number > 100:
                errors.append({"field": field, "path": ".".join(path), "value": clean_value(value), "issue": "INVALID_SCORE_RANGE_0_100"})
                set_path(snapshot, path, None)
    for ticker in ASSET_TICKERS:
        asset = snapshot.get("asset_market_data", {}).get(ticker, {})
        if asset.get("current_price") is None:
            errors.append({"field": "AssetMarketData", "path": f"asset_market_data.{ticker}.current_price", "value": None, "issue": "MISSING_CURRENT_PRICE"})
    return errors


def get_path(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_path(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current: Any = data
    for part in path[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current[path[-1]] = value


def normalize_cio_response(raw: Any, snapshot: dict[str, Any], analysis_source: str) -> dict[str, Any]:
    raw_dict = safe_dict(raw)
    fallback = deterministic_cio_view(snapshot, "SECTION_FALLBACK")
    section_sources: dict[str, str] = {}
    validation_errors: list[str] = []

    overall = normalize_overall(raw_dict.get("overall_system_state", raw_dict), fallback["overall_system_state"])
    if not overall.get("label"):
        overall = dict(fallback["overall_system_state"])
        overall["overall_system_state_source"] = "DERIVED_FALLBACK"
        section_sources["overall_system_state"] = "FALLBACK"
        validation_errors.append("overall_system_state.label missing")
    else:
        section_sources["overall_system_state"] = "LLM" if analysis_source != "DETERMINISTIC_FALLBACK" else "FALLBACK"

    dimensions = normalize_dimensions(raw_dict.get("dimensions"), snapshot, fallback["dimensions"], section_sources, validation_errors)
    cross_signals = normalize_cross_signals(raw_dict.get("cross_regime_signals"), fallback["cross_regime_signals"], section_sources, validation_errors, analysis_source)
    outlook = normalize_asset_outlook_block(raw_dict.get("asset_outlook_3m"), snapshot, fallback["asset_outlook_3m"], section_sources, validation_errors, analysis_source)
    scenarios = normalize_scenarios(raw_dict.get("scenarios"), fallback["scenarios"], section_sources, validation_errors, analysis_source)
    risks = normalize_key_risks(raw_dict.get("key_risks"), fallback["key_risks"], section_sources, validation_errors, analysis_source)
    change_view = normalize_change_view(raw_dict.get("what_would_change_view"), fallback["what_would_change_view"], section_sources, validation_errors, analysis_source)

    data_notes = safe_list(raw_dict.get("data_quality_notes")) + safe_list(snapshot.get("data_quality", {}).get("notes"))
    if snapshot.get("data_quality", {}).get("status") == "PARTIAL_DATA":
        data_notes.extend(data_quality_reasons(snapshot))
    if validation_errors:
        data_notes.extend(validation_errors)

    return {
        "as_of_date": safe_str(raw_dict.get("as_of_date")) or snapshot.get("as_of_date"),
        "analysis_source": analysis_source,
        "overall_system_state": overall,
        "dimensions": dimensions,
        "cross_regime_signals": cross_signals,
        "asset_outlook_3m": outlook,
        "scenarios": scenarios,
        "key_risks": risks,
        "what_would_change_view": change_view,
        "critic_changes_summary": safe_list(raw_dict.get("critic_changes_summary")),
        "data_quality_notes": [safe_str(x) for x in data_notes if safe_str(x)],
        "section_sources": section_sources,
        "_validation_errors": validation_errors,
    }


def normalize_cio_result(result: dict[str, Any], snapshot: dict[str, Any], analysis_source: str) -> dict[str, Any]:
    return normalize_cio_response(result, snapshot, analysis_source)


def normalize_overall(raw: Any, fallback: dict[str, Any]) -> dict[str, str]:
    data = safe_dict(raw)
    label = first_alias(data, ["label", "state", "status", "name", "title"]) or fallback.get("label", "")
    summary = first_alias(data, ["summary", "assessment", "description", "reason", "rationale"]) or fallback.get("summary", "")
    return {"label": safe_str(label), "summary": safe_str(summary)}


def normalize_dimensions(raw: Any, snapshot: dict[str, Any], fallback: dict[str, Any], section_sources: dict[str, str], errors: list[str]) -> dict[str, dict[str, str]]:
    data = safe_dict(raw)
    out: dict[str, dict[str, str]] = {}
    used_fallback = False
    for key, fallback_item in fallback.items():
        item = safe_dict(data.get(key))
        if not item:
            item = fallback_item
            used_fallback = True
        explanation = first_alias(item, ["explanation", "reason", "rationale", "description", "summary"]) or fallback_item.get("explanation", "")
        normalized = {"state": safe_str(item.get("state") or fallback_item.get("state") or "UNKNOWN"), "explanation": safe_str(explanation)}
        if key == "liquidity":
            normalized["level"] = safe_str(item.get("level") or fallback_item.get("level") or state_from_liquidity_score(snapshot.get("global_liquidity", {}).get("global_liquidity_score")))
            normalized["direction"] = safe_str(item.get("direction") or fallback_item.get("direction") or snapshot.get("global_liquidity", {}).get("global_liquidity_direction") or "UNKNOWN")
            if not normalized["state"] or normalized["state"] == "UNKNOWN":
                normalized["state"] = liquidity_state_from_level_direction(normalized["level"], normalized["direction"])
        if not normalized["explanation"]:
            normalized["explanation"] = dimension_explanation(key, normalized, snapshot)
            used_fallback = True
        out[key] = normalized
    section_sources["dimensions"] = "FALLBACK" if used_fallback else "LLM"
    if used_fallback:
        errors.append("dimensions repaired with section-level fallback")
    return out


def normalize_cross_signals(raw: Any, fallback: list[Any], section_sources: dict[str, str], errors: list[str], analysis_source: str) -> list[dict[str, Any]]:
    items = safe_list(raw)
    if not items:
        section_sources["cross_regime_signals"] = "FALLBACK"
        errors.append("cross_regime_signals missing")
        items = fallback
    else:
        section_sources["cross_regime_signals"] = "LLM" if analysis_source != "DETERMINISTIC_FALLBACK" else "FALLBACK"
    out = []
    for item in items[:5]:
        data = safe_dict(item)
        if not data:
            data = {"signal": "UNNAMED_SIGNAL", "description": safe_str(item)}
        signal = first_alias(data, ["signal", "name", "title", "label"]) or "UNNAMED_SIGNAL"
        description = first_alias(data, ["description", "reason", "summary", "rationale"]) or ""
        assets = data.get("affected_assets", data.get("assets", data.get("affected", [])))
        out.append({"signal": safe_str(signal), "description": safe_str(description), "affected_assets": normalize_assets(assets)})
    return out


def normalize_asset_outlook_block(raw: Any, snapshot: dict[str, Any], fallback: dict[str, Any], section_sources: dict[str, str], errors: list[str], analysis_source: str) -> dict[str, dict[str, Any]]:
    data = safe_dict(raw)
    out = {}
    used_fallback = False
    for ticker in ASSET_TICKERS:
        item = safe_dict(data.get(ticker))
        if not item:
            item = fallback.get(ticker, fallback_asset_outlook(ticker, snapshot))
            used_fallback = True
        out[ticker] = normalize_asset_outlook(item, ticker, snapshot, analysis_source)
    section_sources["asset_outlook_3m"] = "FALLBACK" if used_fallback else "LLM"
    if used_fallback:
        errors.append("asset_outlook_3m repaired with section-level fallback")
    return out


def normalize_scenarios(raw: Any, fallback: dict[str, Any], section_sources: dict[str, str], errors: list[str], analysis_source: str) -> dict[str, dict[str, Any]]:
    data = safe_dict(raw)
    alias_map = {"base": ["base", "base_case"], "bull": ["bull", "bull_case"], "bear": ["bear", "bear_case"]}
    out: dict[str, dict[str, Any]] = {}
    used_fallback = False
    for key, aliases in alias_map.items():
        item = safe_dict(next((data.get(alias) for alias in aliases if alias in data), {}))
        fallback_item = fallback.get(key, {})
        if not item:
            item = fallback_item
            used_fallback = True
        conditions = item.get("conditions", item.get("scenario_conditions", []))
        summary = first_alias(item, ["summary", "description", "system_implication"]) or fallback_item.get("summary") or fallback_item.get("system_implication", "")
        impacts = safe_dict(item.get("asset_impacts", item.get("impacts", item.get("asset_outlook", {}))))
        if not impacts:
            impacts = {asset: item.get(asset, item.get(asset.replace("-USD", ""), fallback_item.get(asset, ""))) for asset in ASSET_TICKERS}
            used_fallback = True
        out[key] = {
            "conditions": [safe_str(x) for x in safe_list(conditions)[:5]],
            "summary": safe_str(summary),
            "asset_impacts": {asset: safe_str(impacts.get(asset, impacts.get(asset.replace("-USD", ""), ""))) for asset in ASSET_TICKERS},
        }
    section_sources["scenarios"] = "FALLBACK" if used_fallback else "LLM"
    if used_fallback:
        errors.append("scenarios repaired with section-level fallback")
    return out


def normalize_key_risks(raw: Any, fallback: list[Any], section_sources: dict[str, str], errors: list[str], analysis_source: str) -> list[dict[str, Any]]:
    items = safe_list(raw)
    if not items:
        section_sources["key_risks"] = "FALLBACK"
        errors.append("key_risks missing")
        items = fallback
    else:
        section_sources["key_risks"] = "LLM" if analysis_source != "DETERMINISTIC_FALLBACK" else "FALLBACK"
    out = []
    for idx, item in enumerate(items[:5], start=1):
        data = safe_dict(item)
        risk = first_alias(data, ["risk", "title", "name", "label"]) or "UNNAMED_RISK"
        trigger = first_alias(data, ["escalation_trigger", "trigger_to_escalate", "trigger", "watch_trigger", "escalation"]) or ""
        out.append(
            {
                "rank": int(to_float(data.get("rank")) if np.isfinite(to_float(data.get("rank"))) else idx),
                "risk": safe_str(risk),
                "status": safe_str(data.get("status") or "WATCH"),
                "description": safe_str(data.get("description") or data.get("summary") or data.get("rationale") or ""),
                "affected_assets": normalize_assets(data.get("affected_assets", data.get("assets", data.get("affected", [])))),
                "escalation_trigger": safe_str(trigger),
            }
        )
    return out


def normalize_change_view(raw: Any, fallback: dict[str, Any], section_sources: dict[str, str], errors: list[str], analysis_source: str) -> dict[str, dict[str, Any]]:
    data = safe_dict(raw)
    out = {}
    used_fallback = False
    for ticker in ASSET_TICKERS:
        value = data.get(ticker)
        fallback_item = safe_dict(fallback.get(ticker))
        if isinstance(value, str):
            out[ticker] = {"bullish_triggers": [], "bearish_triggers": [], "_raw_text": value}
            used_fallback = True
            continue
        item = safe_dict(value)
        if not item:
            item = fallback_item
            used_fallback = True
        bullish = item.get("bullish_triggers", item.get("more_bullish_if", item.get("what_would_make_more_bullish", [])))
        bearish = item.get("bearish_triggers", item.get("more_bearish_if", item.get("what_would_make_more_bearish", [])))
        out[ticker] = {"bullish_triggers": [safe_str(x) for x in safe_list(bullish)[:4]], "bearish_triggers": [safe_str(x) for x in safe_list(bearish)[:4]]}
    section_sources["what_would_change_view"] = "FALLBACK" if used_fallback else "LLM"
    if used_fallback:
        errors.append("what_would_change_view repaired with section-level fallback")
    return out


def normalize_asset_outlook(item: dict[str, Any], ticker: str, snapshot: dict[str, Any], source: str) -> dict[str, Any]:
    fallback = fallback_asset_outlook(ticker, snapshot)
    if not isinstance(item, dict):
        item = {}
    out = {**fallback, **item}
    out["ticker"] = ticker
    bias = out.get("bias") or out.get("bias_3m") or out.get("outlook") or out.get("state") or fallback.get("bias") or fallback.get("bias_3m") or "NEUTRAL"
    out["bias"] = safe_str(bias)
    out["bias_3m"] = out["bias"]
    out["top_supports"] = list_from_any(out.get("top_supports", out.get("supporting_factors", fallback["top_supports"])))[:4]
    out["top_risks"] = list_from_any(out.get("top_risks", out.get("risk_factors", fallback["top_risks"])))[:4]
    out["more_bullish_if"] = list_from_any(out.get("more_bullish_if", out.get("what_would_make_more_bullish", fallback["more_bullish_if"])))[:4]
    out["more_bearish_if"] = list_from_any(out.get("more_bearish_if", out.get("what_would_make_more_bearish", fallback["more_bearish_if"])))[:4]
    out["supporting_factors"] = out["top_supports"]
    out["risk_factors"] = out["top_risks"]
    out["what_would_make_more_bullish"] = out["more_bullish_if"]
    out["what_would_make_more_bearish"] = out["more_bearish_if"]
    confidence = to_float(out.get("confidence"))
    if not np.isfinite(confidence):
        confidence = fallback["confidence"]
    if source == "DETERMINISTIC_FALLBACK":
        confidence = min(confidence, 0.55)
    if snapshot.get("data_quality", {}).get("status") == "PARTIAL_DATA":
        confidence = min(confidence, 0.60)
    out["confidence"] = round(float(np.clip(confidence, 0.50, 0.85)), 2)
    return out


def list_from_any(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str) and "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]
    return [value]


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if isinstance(value, np.floating):
        number = float(value)
        return f"{number:.2f}".rstrip("0").rstrip(".") if math.isfinite(number) else ""
    return str(value)


def display_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "n/a"
        if abs(number) >= 1000:
            return f"{number:,.0f}"
        if abs(number) >= 10:
            return f"{number:.1f}"
        return f"{number:.2f}".rstrip("0").rstrip(".")
    return safe_str(value) or "n/a"


def first_alias(data: dict[str, Any], aliases: list[str]) -> Any:
    for key in aliases:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_assets(value: Any) -> list[str]:
    assets = [safe_str(item).strip() for item in safe_list(value)]
    return [asset for asset in assets if asset]


def liquidity_state_from_level_direction(level: str, direction: str) -> str:
    direction_upper = str(direction).upper()
    level_upper = str(level).upper()
    if "DETERIORATING" in direction_upper:
        return "DETERIORATING"
    if "NEGATIVE" in level_upper:
        return "NEGATIVE"
    if "SUPPORTIVE" in level_upper or "ACCELERATING" in direction_upper or "IMPROVING" in direction_upper:
        return "SUPPORTIVE"
    if "NEUTRAL" in level_upper:
        return "NEUTRAL"
    return "UNKNOWN"


def dimension_explanation(key: str, item: dict[str, Any], snapshot: dict[str, Any]) -> str:
    if key == "liquidity":
        return f"Liquidity level {item.get('level', 'UNKNOWN')} with direction {item.get('direction', 'UNKNOWN')}."
    if key == "systemic_stress":
        return f"Systemic stress state {item.get('state', 'UNKNOWN')} based on macro and fast transition risk."
    return f"{key.replace('_', ' ').title()} state is {item.get('state', 'UNKNOWN')} based on supplied CIO snapshot."


def data_quality_reasons(snapshot: dict[str, Any]) -> list[str]:
    quality = safe_dict(snapshot.get("data_quality"))
    reasons = []
    for label, key in [("missing", "missing_series"), ("stale", "stale_series"), ("partial", "partial_data"), ("invalid", "invalid_scores")]:
        values = safe_list(quality.get(key))
        if values:
            reasons.append(f"{label}: {', '.join(safe_str(v) for v in values[:6])}")
    validation = safe_list(snapshot.get("snapshot_validation_errors"))
    if validation:
        reasons.append(f"snapshot validation warnings: {len(validation)}")
    return reasons


def summarize_token_usage(usage: dict[str, Any]) -> dict[str, Any]:
    totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for key in ("analyst_usage", "critic_usage"):
        block = usage.get(key) or {}
        totals["input_tokens"] += int(block.get("input_tokens") or block.get("prompt_tokens") or 0)
        details = block.get("input_tokens_details") or block.get("prompt_tokens_details") or {}
        totals["cached_input_tokens"] += int(details.get("cached_tokens") or 0)
        totals["output_tokens"] += int(block.get("output_tokens") or block.get("completion_tokens") or 0)
        totals["total_tokens"] += int(block.get("total_tokens") or 0)
    if totals["total_tokens"] == 0:
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return totals


def estimate_usage_cost(token_usage: dict[str, Any]) -> float | None:
    if not token_usage or not token_usage.get("total_tokens"):
        return None
    # Conservative placeholder until model-specific billing table is configured.
    return None


def brief_error(exc: Exception) -> str:
    text = str(exc)
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        text = text.replace(key, "[redacted]")
    return text[:600]


def display_analysis_source(source: str) -> str:
    return {
        "LLM_ANALYST": "LLM",
        "LLM_CRITIC_CORRECTED": "Critic Corrected",
        "DETERMINISTIC_FALLBACK": "Fallback",
    }.get(str(source), str(source))


def is_score_like(value: Any) -> bool:
    number = to_float(value)
    return np.isfinite(number) and 0 <= number <= 100


def first_valid(row: dict[str, Any], columns: list[str]) -> Any:
    for column in columns:
        value = row.get(column)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            return value
    return None


def liquidity_direction_value(liquidity: dict[str, Any]) -> float:
    raw = to_float(liquidity.get("global_liquidity_direction_13w"))
    if np.isfinite(raw):
        return raw
    state = str(liquidity.get("global_liquidity_direction", "")).upper()
    if "DETERIORATING_FAST" in state:
        return -20.0
    if "DETERIORATING" in state:
        return -10.0
    if "ACCELERATING" in state or "IMPROVING" in state:
        return 10.0
    return 0.0


def state_from_liquidity_score(value: Any) -> str:
    score = to_float(value)
    if not np.isfinite(score):
        return "UNKNOWN"
    if score >= 65:
        return "SUPPORTIVE"
    if score >= 45:
        return "NEUTRAL"
    if score >= 35:
        return "DETERIORATING"
    return "NEGATIVE"


def market_trend_state(market: dict[str, Any]) -> str:
    final = str(market.get("final_state", "")).upper()
    structural = str(market.get("structural_regime", "")).upper()
    if "STRESS" in final:
        return "STRESS"
    if "CORRECTION" in final or "BEAR" in structural:
        return "CORRECTION"
    if "WARNING" in final and structural == "BULL":
        return "BULL_WITH_WARNING"
    if structural == "BULL":
        return "BULL"
    if "DETERIOR" in final:
        return "DETERIORATING"
    return "UNKNOWN"


def systemic_stress_state(market: dict[str, Any]) -> str:
    macro = to_float(market.get("macro_transition_risk"))
    fast = to_float(market.get("fast_transition_risk"))
    risk = np.nanmax([macro, fast])
    if not np.isfinite(risk):
        return "UNKNOWN"
    if risk >= 80:
        return "EXTREME"
    if risk >= 60:
        return "HIGH"
    if risk >= 35:
        return "ELEVATED"
    return "LOW"


def risk_appetite_state(market: dict[str, Any]) -> str:
    fast = to_float(market.get("fast_transition_risk"))
    if not np.isfinite(fast):
        return "UNKNOWN"
    if fast >= 45:
        return "NEGATIVE"
    if fast >= 25:
        return "NEUTRAL"
    return "POSITIVE"


def macro_item(macro: dict[str, Any], name: str) -> dict[str, Any]:
    for item in macro.get("items", []):
        if item.get("instrument") == name:
            return item
    return {}


def change_value(item: dict[str, Any], key: str = "3m") -> float:
    value = item.get(key)
    if isinstance(value, str):
        value = value.replace("%", "").replace("bp", "").replace(",", "").strip()
    return to_float(value)


def growth_dimension_state(macro: dict[str, Any]) -> str:
    pmi_m = to_float(macro_item(macro, "U.S. ISM Manufacturing PMI").get("current"))
    pmi_s = to_float(macro_item(macro, "U.S. ISM Services PMI").get("current"))
    claims_3m = change_value(macro_item(macro, "U.S. Initial Jobless Claims"))
    values = [v for v in [pmi_m, pmi_s] if np.isfinite(v)]
    if not values and not np.isfinite(claims_3m):
        return "UNKNOWN"
    avg = np.mean(values) if values else math.nan
    if np.isfinite(avg) and avg < 48:
        return "CONTRACTING"
    if np.isfinite(claims_3m) and claims_3m > 5:
        return "SLOWING"
    if np.isfinite(avg) and avg > 52:
        return "ACCELERATING"
    return "STABLE"


def inflation_dimension_state(macro: dict[str, Any]) -> str:
    breakeven = change_value(macro_item(macro, "U.S. 10-Year Breakeven Inflation Rate"))
    wti = change_value(macro_item(macro, "WTI Crude Oil"))
    if not np.isfinite(breakeven) and not np.isfinite(wti):
        return "UNKNOWN"
    if np.nanmax([breakeven, wti]) > 3:
        return "REACCELERATING"
    if np.nanmin([breakeven, wti]) < -3:
        return "FALLING"
    return "STABLE"


def rates_dimension_state(macro: dict[str, Any]) -> str:
    us2y = change_value(macro_item(macro, "U.S. 2-Year Treasury Yield"))
    real10 = change_value(macro_item(macro, "U.S. 10-Year Real Yield"))
    if not np.isfinite(us2y) and not np.isfinite(real10):
        return "UNKNOWN"
    if np.nanmax([us2y, real10]) > 0.10:
        return "TIGHTENING"
    if np.nanmin([us2y, real10]) < -0.10:
        return "EASING"
    return "NEUTRAL"


def financial_conditions_state(macro: dict[str, Any], market: dict[str, Any]) -> str:
    fast = to_float(market.get("fast_transition_risk"))
    vix = change_value(macro_item(macro, "CBOE Volatility Index"))
    move = change_value(macro_item(macro, "ICE BofA MOVE Index"))
    if np.isfinite(fast) and fast >= 60:
        return "TIGHT"
    if np.nanmax([vix, move]) > 10:
        return "TIGHTENING"
    if np.isfinite(fast) and fast < 25:
        return "EASY"
    return "NEUTRAL"


def credit_dimension_state(macro: dict[str, Any]) -> str:
    hy = change_value(macro_item(macro, "U.S. High Yield Option-Adjusted Spread"))
    ig = change_value(macro_item(macro, "U.S. Investment Grade Option-Adjusted Spread"))
    if not np.isfinite(hy) and not np.isfinite(ig):
        return "UNKNOWN"
    if np.nanmax([hy, ig]) > 0.10:
        return "DETERIORATING"
    if np.nanmax([hy, ig]) > 0.30:
        return "STRESSED"
    if np.nanmax([hy, ig]) <= 0:
        return "BENIGN"
    return "NORMAL"


def asset_fallback_bias(ticker: str, snapshot: dict[str, Any], alpha: float) -> str:
    liquidity = snapshot.get("global_liquidity", {})
    market = snapshot.get("market_regime", {})
    gold = snapshot.get("gold_regime", {})
    btc = snapshot.get("btc_regime", {})
    if ticker == "GLD":
        if to_float(gold.get("gold_alpha")) >= 60 and to_float(gold.get("tactical_flow")) >= 60:
            return "NEUTRAL_POSITIVE" if to_float(gold.get("structural_macro")) < 40 else "POSITIVE"
        return "NEUTRAL_NEGATIVE" if to_float(gold.get("gold_alpha")) < 50 else "NEUTRAL"
    if ticker == "BTC-USD":
        if to_float(btc.get("alpha")) >= 60 and str(btc.get("etf_flow_state")) == "POSITIVE" and liquidity_direction_value(liquidity) >= 0:
            return "POSITIVE"
        if "DETERIORATING" in str(btc.get("global_liquidity_direction")):
            return "NEUTRAL_NEGATIVE"
        return "NEUTRAL"
    if ticker == "QQQ":
        if liquidity_direction_value(liquidity) < -10 or to_float(market.get("macro_transition_risk")) > 40:
            return "NEUTRAL_NEGATIVE"
    if str(market.get("structural_regime", "")).upper() == "BULL" and np.isfinite(alpha) and alpha >= 50:
        return "NEUTRAL_POSITIVE"
    return "NEUTRAL"


def asset_specific_lists(ticker: str, snapshot: dict[str, Any]) -> tuple[list[str], list[str], list[str], list[str]]:
    market = snapshot.get("market_regime", {})
    liquidity = snapshot.get("global_liquidity", {})
    gold = snapshot.get("gold_regime", {})
    btc = snapshot.get("btc_regime", {})
    if ticker == "SPY":
        return (
            [f"Structural regime {market.get('structural_regime', 'UNKNOWN')}", f"Macro risk {market.get('macro_transition_risk', 'n/a')}", f"Systemic stress {systemic_stress_state(market)}"],
            [f"Liquidity direction {liquidity.get('global_liquidity_direction', 'UNKNOWN')}", f"Fast risk {market.get('fast_transition_risk', 'n/a')}", "Credit spread widening would change the risk profile"],
            ["GlobalLiquidityDirection > 0", "MacroTransitionRisk < 20", "DXY and US2Y weaken", "Breadth confirmations improve"],
            ["GlobalLiquidityScore < 40", "MacroTransitionRisk > 40", "HY/IG spreads widen", "FastRisk > 40"],
        )
    if ticker == "QQQ":
        return (
            [f"Structural equity backdrop {market.get('structural_regime', 'UNKNOWN')}", "Duration/growth beta benefits if yields ease", "Alpha/trend confirmation from table"],
            [f"Liquidity direction {liquidity.get('global_liquidity_direction', 'UNKNOWN')}", "10Y real yield or US2Y rising pressures duration", "DXY strength tightens financial conditions"],
            ["GlobalLiquidityDirection > 0", "10Y real yield falls", "US2Y falls", "DXY weakens"],
            ["Liquidity deteriorates further", "Real yield rises", "US2Y rises", "QQQ Alpha < 40"],
        )
    if ticker == "GLD":
        return (
            [f"Gold Alpha {gold.get('gold_alpha', 'n/a')}", f"Tactical Flow {gold.get('tactical_flow', 'n/a')}", f"ETF Flow Score {gold.get('etf_flow_score', 'n/a')}"],
            [f"Structural Macro {gold.get('structural_macro', 'n/a')}", f"Forward Macro Risk {gold.get('forward_macro_risk', 'n/a')}", "Flow support can fade if ETF/COT weaken"],
            ["Gold Alpha >= 70", "ETF Flow Score >= 70", "Real yield falling", "Forward Macro Risk < 60"],
            ["Gold Alpha < 50", "ETF Flow < 40", "Structural Macro < 40", "Forward Macro Risk remains high and flows weaken"],
        )
    return (
        [f"Halving phase {btc.get('halving_phase', 'UNKNOWN')}", f"BTC Alpha {btc.get('alpha', 'n/a')}", f"ETF flow state {btc.get('etf_flow_state', 'n/a')}", f"Funding 28D {btc.get('funding_28d', 'n/a')}"],
        [f"Liquidity direction {btc.get('global_liquidity_direction', 'UNKNOWN')}", f"Forward Macro Risk {btc.get('forward_macro_risk', 'n/a')}", "OI/funding leverage can rise without price confirmation"],
        ["Halving setup and liquidity improve together", "BTC Alpha > 60", "ETF flows improving", "OI stable after deleveraging"],
        ["Global liquidity deteriorates", "BTC Alpha < 40", "ETF flows weaken", "OI/funding leverage rises without price confirmation"],
    )


def asset_environment_text(ticker: str, market: dict[str, Any], liquidity: dict[str, Any], gold: dict[str, Any], btc: dict[str, Any]) -> str:
    if ticker == "GLD":
        return f"Gold-specific drivers: alpha {gold.get('gold_alpha', 'n/a')}, structural macro {gold.get('structural_macro', 'n/a')}, forward risk {gold.get('forward_macro_risk', 'n/a')}, tactical flow {gold.get('tactical_flow', 'n/a')}."
    if ticker == "BTC-USD":
        return f"BTC cycle/liquidity setup: {btc.get('halving_phase', 'UNKNOWN')} with liquidity direction {btc.get('global_liquidity_direction', 'UNKNOWN')} and ETF state {btc.get('etf_flow_state', 'n/a')}."
    if ticker == "QQQ":
        return f"Growth-duration setup with higher sensitivity to liquidity {liquidity.get('global_liquidity_direction', 'UNKNOWN')}, rates and DXY than SPY."
    return f"Equity broad-market setup: {market.get('structural_regime', 'UNKNOWN')} / {market.get('final_state', 'UNKNOWN')} with liquidity direction {liquidity.get('global_liquidity_direction', 'UNKNOWN')}."


def dynamic_bull_conditions(snapshot: dict[str, Any]) -> list[str]:
    conditions = ["Credit remains benign"]
    if liquidity_direction_value(snapshot.get("global_liquidity", {})) <= 0:
        conditions.append("GlobalLiquidityDirection turns positive")
    if to_float(snapshot.get("global_liquidity", {}).get("global_liquidity_score")) < 60:
        conditions.append("GlobalLiquidityScore > 60")
    conditions.extend(["US2Y / real yields decline", "DXY weakens"])
    return conditions[:5]


def dynamic_bear_conditions(snapshot: dict[str, Any]) -> list[str]:
    return ["GlobalLiquidityScore < 40", "Liquidity direction remains DETERIORATING_FAST", "MacroTransitionRisk > 40", "HY/IG OAS widens", "DXY / real yields rise"]


def fallback_confidence(snapshot: dict[str, Any]) -> float:
    if snapshot.get("data_quality", {}).get("status") == "PARTIAL_DATA":
        return 0.52
    return 0.55


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
                "analyst_raw": result.analyst_raw,
                "analyst_normalized": result.analyst_normalized,
                "critic_raw": result.critic_raw,
                "critic_normalized": result.critic_normalized,
                "analyst_result": result.analyst_result,
                "critic_result": result.critic_result,
                "final_result": result.final_result,
                "usage": result.usage,
                "status": result.status,
                "analysis_source": result.analysis_source,
                "analyst_status": result.analyst_status,
                "critic_status": result.critic_status,
                "generated_at": result.generated_at,
                "next_scheduled_at": result.next_scheduled_at,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def cio_result_from_cache(cache: dict[str, Any], status: str) -> CioRunResult:
    snapshot = safe_dict(cache.get("snapshot"))
    snapshot.setdefault("as_of_date", str(cache.get("generated_at") or datetime.now(timezone.utc).isoformat()))
    analysis_source = str(cache.get("analysis_source") or cache.get("final_result", {}).get("analysis_source") or "DETERMINISTIC_FALLBACK")
    final_result = normalize_cio_result(cache.get("final_result", {}), snapshot, analysis_source)
    analyst_result = normalize_cio_result(cache.get("analyst_result", {}), snapshot, "LLM_ANALYST" if cache.get("analyst_status") == "SUCCESS" else "DETERMINISTIC_FALLBACK")
    critic_result = normalize_cio_result(cache.get("critic_result", {}), snapshot, "LLM_CRITIC_CORRECTED") if cache.get("critic_result") else None
    return CioRunResult(
        snapshot=snapshot,
        analyst_result=analyst_result,
        critic_result=critic_result,
        final_result=final_result,
        usage=cache.get("usage", {}),
        status=status,
        analysis_source=analysis_source,
        analyst_status=str(cache.get("analyst_status") or cache.get("usage", {}).get("analyst_status") or "SKIPPED"),
        critic_status=str(cache.get("critic_status") or cache.get("usage", {}).get("critic_status") or "SKIPPED"),
        generated_at=str(cache.get("generated_at", "")),
        next_scheduled_at=str(cache.get("next_scheduled_at", "")),
        analyst_raw=cache.get("analyst_raw"),
        analyst_normalized=cache.get("analyst_normalized"),
        critic_raw=cache.get("critic_raw"),
        critic_normalized=cache.get("critic_normalized"),
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
