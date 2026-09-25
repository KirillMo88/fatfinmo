from __future__ import annotations

import html
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from business_cycle_tab import load_business_cycle_snapshot_cached, load_macro_surprises_snapshot_cached
from current_risk import classify_component_state
from funding_conditions import read_snapshot as read_funding_snapshot, read_wresbal_history
from financial_fragility_export import build_financial_fragility_validation_workbook
from global_dashboard import GlobalDashboardSnapshot, build_global_dashboard_snapshot
from liquidity_forecast import read_forecast_snapshot
from market_cycle_tab import load_market_cycle_snapshot_cached, load_spy_macro_outlook_cached
from spy_macro_outlook import SPY_MACRO_HORIZONS, score_direction, score_state
from rates_financial_conditions import read_snapshot as read_rates_snapshot
from treasury_fiscal_regime import read_snapshot as read_treasury_snapshot


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
STATE_COLORS = {
    "LOW": "#22c55e", "NORMAL": "#94a3b8", "CURRENT": "#22c55e",
    "EXPANSION": "#22c55e", "BOTTOMING": "#38bdf8", "BROAD EXPANSION": "#22c55e",
    "ELEVATED": "#facc15", "WATCH": "#facc15", "PEAKING": "#f59e0b",
    "HIGH": "#f97316", "ACUTE": "#ef4444", "CONTRACTION": "#ef4444",
    "BROAD CONTRACTION": "#ef4444", "STRONG DIVERGENCE": "#ef4444",
    "MILD DIVERGENCE": "#facc15", "ALIGNED": "#22c55e",
    "STALE": "#f59e0b", "PARTIAL DATA": "#facc15", "UNAVAILABLE": "#64748b",
}


def _empty() -> SimpleNamespace:
    return SimpleNamespace(
        current={}, history=pd.DataFrame(), outlook=pd.DataFrame(), asset_outlook=pd.DataFrame(),
        analogs=pd.DataFrame(), daily=pd.DataFrame(), weekly=pd.DataFrame(), status={},
    )


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}{suffix}" if np.isfinite(number) else "n/a"


def _pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number * 100:.{digits}f}%" if np.isfinite(number) else "n/a"


def _trillions(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "$" + f"{number / 1000.0:,.2f}T" if np.isfinite(number) else "n/a"


def _bool(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return "YES" if bool(value) else "NO"


def _color(state: Any) -> str:
    text = str(state).upper()
    if text in STATE_COLORS:
        return STATE_COLORS[text]
    for key, color in STATE_COLORS.items():
        if key in text:
            return color
    return "#64748b"


def _percentile_status_color(value: Any) -> str:
    text = str(value or "").upper()
    if "DECELERATION" in text:
        return "#ef4444"
    if "ACCELERATION" in text:
        return "#22c55e"
    return "#94a3b8"


def _final_regime_color(value: Any) -> str:
    text = str(value or "").upper()
    if any(token in text for token in ("WARNING", "CONTRACTION", "DRAIN", "TIGHT", "STRESS", "DETERIORATING", "RISK")):
        return "#ef4444"
    if any(token in text for token in ("SUPPORTIVE", "EXPANSION", "EASING", "IMPROVING", "LIQUIDITY_OK")):
        return "#22c55e"
    return "#94a3b8"


def _cycle_panel_color(extension: Any, momentum: Any) -> str:
    extension_text = str(extension or "").upper().replace("_", " ")
    try:
        momentum_value = float(momentum)
    except (TypeError, ValueError):
        momentum_value = np.nan
    if "OVERSOLD" in extension_text:
        return "#38bdf8"
    if "EXTREME" in extension_text or "OVEREXTENDED" in extension_text:
        return "#ef4444"
    if "EXTENDED" in extension_text:
        if np.isfinite(momentum_value) and momentum_value > 0:
            return "#facc15"
        if np.isfinite(momentum_value) and momentum_value < 0:
            return "#ef4444"
        return "#94a3b8"
    if "NORMAL" in extension_text:
        return "#22c55e" if np.isfinite(momentum_value) and momentum_value > 0 else "#94a3b8"
    return "#94a3b8"


def _macro_outlook_color(macro_outlook: Any) -> str:
    if macro_outlook is None:
        return "#94a3b8"
    try:
        score = float(macro_outlook.current.get("Macro_3M"))
    except (TypeError, ValueError, AttributeError):
        return "#94a3b8"
    if not np.isfinite(score):
        return "#94a3b8"
    if score <= 40:
        return "#ef4444"
    if score <= 50:
        return "#facc15"
    return "#22c55e"


def _historical_outlook_color(value: Any) -> str:
    try:
        percentage = float(value) * 100.0
    except (TypeError, ValueError):
        return "#94a3b8"
    if not np.isfinite(percentage):
        return "#94a3b8"
    if percentage < 5:
        return "#ef4444"
    if percentage <= 10:
        return "#facc15"
    return "#22c55e"


def _current_risk_color(state: Any, credit_confirmation: Any) -> str:
    text = str(state or "").upper().replace("_", " ")
    try:
        credit_value = float(credit_confirmation)
    except (TypeError, ValueError):
        credit_value = np.nan
    if "RED FLAG" in text and np.isfinite(credit_value) and credit_value >= 60:
        return "#ef4444"
    if "MODERATE" in text:
        return "#facc15"
    if "HIGH" in text or "RED FLAG" in text:
        return "#f97316"
    return "#94a3b8"


def _inflation_color(score: Any) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "#94a3b8"
    if not np.isfinite(value):
        return "#94a3b8"
    return "#ef4444" if value > 0.2 else "#22c55e"


def _economy_regime_color(value: Any) -> str:
    text = str(value or "").upper()
    if "GOLDILOCKS" in text or "REFLATION" in text:
        return "#22c55e"
    if "STAGFLATION" in text or "DISINFLATIONARY" in text or "SLOWDOWN" in text:
        return "#ef4444"
    return "#94a3b8"


def _rates_pressure_color(value: Any) -> str:
    try:
        pressure = float(value)
    except (TypeError, ValueError):
        return "#94a3b8"
    if not np.isfinite(pressure):
        return "#94a3b8"
    if pressure <= 0.5:
        return "#22c55e"
    if pressure <= 1:
        return "#facc15"
    return "#ef4444"


def _fiscal_color(value: Any) -> str:
    text = str(value or "").upper()
    if text == "POSITIVE":
        return "#22c55e"
    if text == "NEUTRAL":
        return "#94a3b8"
    return "#ef4444"


def _card(
    title: str,
    state: Any,
    rows: list[tuple[str, Any]],
    *,
    prominent: bool = False,
    accent_color: str | None = None,
) -> None:
    safe_state = html.escape(str(state))
    detail = "".join(
        "<div class='gd-row'><span>{}</span><strong>{}</strong></div>".format(
            html.escape(str(label)), html.escape(str(value))
        )
        for label, value in rows
    )
    st.markdown(
        f"""
        <div class="gd-card{' gd-prominent' if prominent else ''}" style="border-top-color:{html.escape(accent_color or _color(state))}">
          <div class="gd-title">{html.escape(title)}</div>
          <div class="gd-state">{safe_state}</div>
          <div class="gd-details">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _styles() -> None:
    st.markdown(
        """
        <style>
        .gd-card {border:1px solid #273244;border-top:3px solid #64748b;border-radius:6px;padding:12px 14px;
                  background:#111720;min-height:190px;margin-bottom:10px;overflow:hidden}
        .gd-prominent {min-height:160px;background:#121a24}
        .gd-title {font-size:.72rem;color:#8ea4bd;font-weight:800;text-transform:uppercase;margin-bottom:6px}
        .gd-state {font-size:1.08rem;color:#f8fafc;font-weight:800;line-height:1.22;margin-bottom:10px;overflow-wrap:anywhere}
        .gd-details {display:grid;gap:5px}
        .gd-row {display:grid;grid-template-columns:minmax(0,1fr) auto;column-gap:10px;align-items:start;font-size:.76rem;line-height:1.25}
        .gd-row span {color:#94a3b8;min-width:0}.gd-row strong {color:#e5edf6;text-align:right;max-width:180px;overflow-wrap:anywhere}
        .gd-flow {display:grid;grid-template-columns:1fr 42px 1fr 42px 1fr;align-items:center;margin:8px 0 14px}
        .gd-node {border:1px solid #334155;border-radius:6px;padding:13px;text-align:center;background:#111720;min-height:88px}
        .gd-node b {display:block;color:#f8fafc;font-size:.92rem;margin-top:5px;overflow-wrap:anywhere}
        .gd-node span {color:#8ea4bd;font-size:.7rem;font-weight:800}.gd-arrow {text-align:center;color:#64748b;font-size:1.4rem}
        .gd-policy {border-left:3px solid #38bdf8;padding:8px 12px;background:#111720;color:#dbeafe;font-weight:700;margin:0 0 12px}
        .gd-cycle-panel {border:1px solid #273244;border-top:3px solid #64748b;border-radius:6px;padding:12px 14px;
                         background:#111720;min-height:190px;margin-bottom:10px;overflow:hidden}
        .gd-cycle-title {font-size:.72rem;color:#8ea4bd;font-weight:800;text-transform:uppercase;margin-bottom:8px}
        .gd-cycle-body {display:grid;gap:5px;font-size:.76rem}
        .gd-cycle-row {display:grid;grid-template-columns:minmax(0,1fr) auto;column-gap:8px;align-items:baseline}
        .gd-cycle-row span {color:#cbd5e1;min-width:0}.gd-cycle-row strong {color:#f8fafc;text-align:right;white-space:nowrap;max-width:220px;overflow-wrap:anywhere}
        @media (max-width: 800px) {.gd-flow{grid-template-columns:1fr}.gd-arrow{transform:rotate(90deg)}.gd-card{min-height:auto}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_snapshots(api_key: str | None) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    loaded: dict[str, Any] = {}
    loaders = {
        "market": lambda: load_market_cycle_snapshot_cached(int(st.session_state.get("market_cycle_refresh_nonce", 0))),
        "business": lambda: load_business_cycle_snapshot_cached(api_key, int(st.session_state.get("business_cycle_refresh_nonce", 0))),
        "rates": read_rates_snapshot,
        "funding": read_funding_snapshot,
        "treasury": read_treasury_snapshot,
    }
    for name, loader in loaders.items():
        try:
            loaded[name] = loader()
        except Exception as exc:
            loaded[name] = _empty()
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    try:
        loaded["macro"] = load_macro_surprises_snapshot_cached(
            getattr(loaded["business"], "history", pd.DataFrame()),
            int(st.session_state.get("business_cycle_refresh_nonce", 0)),
        )
    except Exception as exc:
        loaded["macro"] = _empty()
        errors.append(f"macro surprises: {type(exc).__name__}: {exc}")
    try:
        loaded["forecast"], loaded["forecast_status"] = read_forecast_snapshot()
    except Exception as exc:
        loaded["forecast"], loaded["forecast_status"] = pd.DataFrame(), {}
        errors.append(f"liquidity forecast: {type(exc).__name__}: {exc}")
    return loaded, errors


def render_global_dashboard_tab(
    api_key: str | None,
    market_transition_snapshot: dict[str, Any],
    liquidity_regime: pd.DataFrame,
    transition_history_loader: Callable[[], pd.DataFrame] | None = None,
) -> None:
    _styles()
    st.subheader("Global Dashboard")
    loaded, errors = _load_snapshots(api_key)
    toolbar, export_col, spacer = st.columns([1.35, 2.6, 6.05])
    with toolbar:
        if st.button("Reload Dashboard", use_container_width=True):
            st.rerun()
    with export_col:
        export_requested = st.button(
            "Build Financial Fragility Export",
            use_container_width=True,
            key="financial_fragility_validation_export",
        )
    if export_requested:
        with st.spinner("Writing loaded production snapshots to XLSX..."):
            try:
                transition_history = (
                    transition_history_loader() if transition_history_loader is not None else pd.DataFrame()
                )
                payload, filename = build_financial_fragility_validation_workbook(
                    market_snapshot=loaded["market"],
                    liquidity_regime=liquidity_regime,
                    forecast_frame=loaded["forecast"],
                    business_snapshot=loaded["business"],
                    macro_snapshot=loaded["macro"],
                    rates_snapshot=loaded["rates"],
                    funding_snapshot=loaded["funding"],
                    treasury_snapshot=loaded["treasury"],
                    transition_snapshot=market_transition_snapshot,
                    transition_history=transition_history,
                    wresbal_history=read_wresbal_history(),
                )
                st.session_state["financial_fragility_export_payload"] = {
                    "bytes": payload,
                    "filename": filename,
                }
            except Exception as exc:
                st.warning(f"Financial Fragility export failed: {type(exc).__name__}: {exc}")
    export_payload = st.session_state.get("financial_fragility_export_payload")
    if isinstance(export_payload, dict) and export_payload.get("bytes"):
        st.download_button(
            "Download Financial Fragility Validation .xlsx",
            data=export_payload["bytes"],
            file_name=str(export_payload.get("filename") or "financial_fragility_validation_export.xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="financial_fragility_validation_download",
        )

    dashboard = build_global_dashboard_snapshot(
        liquidity_regime=liquidity_regime,
        forecast_frame=loaded["forecast"],
        forecast_status=loaded["forecast_status"],
        market_snapshot=loaded["market"],
        business_snapshot=loaded["business"],
        macro_snapshot=loaded["macro"],
        rates_snapshot=loaded["rates"],
        funding_snapshot=loaded["funding"],
        treasury_snapshot=loaded["treasury"],
        transition_snapshot=market_transition_snapshot,
    )
    loaded["liquidity_regime"] = liquidity_regime
    macro_outlook = None
    try:
        market_history = getattr(loaded["market"], "history", pd.DataFrame())
        if not market_history.empty:
            macro_outlook = load_spy_macro_outlook_cached(
                market_history,
                api_key,
                int(st.session_state.get("market_cycle_refresh_nonce", 0)),
            )
    except Exception as exc:
        errors.append(f"SPY Macro Outlook: {type(exc).__name__}: {exc}")
    as_of = pd.to_datetime(dashboard.fields.get("DashboardAsOf"), errors="coerce")
    st.caption(f"Dashboard as of {as_of.date() if pd.notna(as_of) else 'unavailable'} | Production outputs are shown without a combined global score.")
    if errors:
        st.warning("Some modules are unavailable; the rest of the dashboard remains active. " + " | ".join(errors))

    _render_executive(dashboard, macro_outlook)
    _render_diagnostics(dashboard, loaded)


def _render_cycle_panel(title: str, rows: list[tuple[str, str]], accent_color: str | None = None) -> None:
    body = "".join(
        f"<div class='gd-cycle-row'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in rows
    )
    st.markdown(
        f"""
        <div class="gd-cycle-panel" style="border-top-color:{html.escape(accent_color or '#64748b')}">
          <div class="gd-cycle-title">{html.escape(title)}</div>
          <div class="gd-cycle-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _cycle_phase(value: Any, percentile: Any) -> str:
    phase = str(value or "n/a").replace("_", " ").title()
    percentile_text = _fmt(percentile, 1)
    return phase if percentile_text == "n/a" else f"{phase} ({percentile_text})"


def _cycle_percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number * 100:.1f}%" if np.isfinite(number) else "n/a"


def _render_executive(d: GlobalDashboardSnapshot, macro_outlook: Any = None) -> None:
    lq, mk, ec, rt, fd, tr = d.liquidity, d.market, d.economy, d.rates, d.funding, d.treasury
    st.markdown("### Global Liquidity & Forecast")
    liquidity_columns = st.columns(4, gap="medium")
    with liquidity_columns[0]:
        _card(
            "Global Liquidity Score",
            _fmt(lq["GlobalLiquidityScore"], 1),
            [
                ("Final Regime", lq["GlobalLiquidityState"]),
                ("Status 13W", lq["GlobalLiquidityDirection"]),
                ("Status 26W", lq["GlobalLiquidityDirection26W"]),
                ("Status 52W", lq["GlobalLiquidityDirection52W"]),
                ("65M cycle Maturity", _fmt(lq["LiquidityCycleMaturity"], 0) + "%"),
                ("Liquidity Forecast Signal", lq["LiquidityForecastSignal"]),
                ("Near-Term Treasury Refinancing", _fmt(lq["NearTermTreasuryRefinancing"], 1)),
                ("Data Status", lq["LiquidityDataStatus"]),
            ],
            prominent=True,
            accent_color=_final_regime_color(lq["GlobalLiquidityState"]),
        )
    with liquidity_columns[1]:
        _card(
            "Global M2",
            _trillions(lq["GlobalM2Value"]),
            [
                ("Growth Percentile Status (52w)", lq["M2GrowthPercentileStatus"]),
                ("Fast Impulse Percentile Status (13w)", lq["M2FastImpulsePercentileStatus"]),
                ("Medium Impulse Percentile Status (26w)", lq["M2MediumImpulsePercentileStatus"]),
                ("Slow Impulse Percentile Status (39w)", lq["M2SlowImpulsePercentileStatus"]),
            ],
            accent_color=_percentile_status_color(lq["M2MediumImpulsePercentileStatus"]),
        )
    with liquidity_columns[2]:
        _card(
            "Global CB Assets",
            _trillions(lq["GlobalCBAssetsValue"]),
            [
                ("Growth Percentile Status (52w)", lq["CBGrowthPercentileStatus"]),
                ("Fast Impulse Percentile Status (13w)", lq["CBFastImpulsePercentileStatus"]),
                ("Medium Impulse Percentile Status (26w)", lq["CBMediumImpulsePercentileStatus"]),
                ("Slow Impulse Percentile Status (39w)", lq["CBSlowImpulsePercentileStatus"]),
            ],
            accent_color=_percentile_status_color(lq["CBMediumImpulsePercentileStatus"]),
        )
    with liquidity_columns[3]:
        _card(
            "US Net Liquidity",
            _trillions(lq["USNetLiquidityValue"]),
            [
                ("Growth Percentile Status (52w)", lq["USNLGrowthPercentileStatus"]),
                ("Fast Impulse Percentile Status (13w)", lq["USNLFastImpulsePercentileStatus"]),
                ("Medium Impulse Percentile Status (26w)", lq["USNLMediumImpulsePercentileStatus"]),
                ("Slow Impulse Percentile Status (39w)", lq["USNLSlowImpulsePercentileStatus"]),
            ],
            accent_color=_percentile_status_color(lq["USNLMediumImpulsePercentileStatus"]),
        )

    st.markdown("### Market Cycle")
    cycle_cols = st.columns(6, gap="medium")
    with cycle_cols[0]:
        _render_cycle_panel(
            "Structural Market Cycle",
            [
                ("SMA 200M Extension", _cycle_phase(mk["StructuralExtensionZone"], mk["StructuralExtensionPercentile"])),
                ("36M ROC 3MMA", _cycle_percent(mk["SPX_ROC36M_3MMA"])),
                ("ROC Momentum 12M", _cycle_percent(mk["StructuralROCMomentum_12M"])),
                ("Structural Market Cycle Maturity", _fmt(mk["StructuralMaturityPct"], 1) + "%"),
            ],
            accent_color=_cycle_panel_color(mk["StructuralExtensionZone"], mk["StructuralROCMomentum_12M"]),
        )
    with cycle_cols[1]:
        _render_cycle_panel(
            "Medium Term Market Cycle",
            [
                ("SMA 200W Extension", _cycle_phase(mk["SMA200WExtensionPhase"], mk["SMA200WExtensionPercentile"])),
                ("12M ROC 3MMA", _cycle_percent(mk["SPX_ROC12M_3MMA"])),
                ("ROC Momentum 3M", _cycle_percent(mk["SPX_ROC_Momentum_3M"])),
                ("LONG EXTENSION CYCLE MATURITY", _fmt(mk["LongCycleMaturityPct"], 1) + "%"),
                ("PRIMARY MARKET CYCLE MATURITY", _fmt(mk["PrimaryCycleMaturityPct"], 1) + "%"),
            ],
            accent_color=_cycle_panel_color(mk["SMA200WExtensionPhase"], mk["SPX_ROC_Momentum_3M"]),
        )
    with cycle_cols[2]:
        _render_cycle_panel(
            "Performance",
            [
                ("ROC 1M", _cycle_percent(mk["PerformanceROC1M"])),
                ("ROC 3M", _cycle_percent(mk["PerformanceROC3M"])),
                ("ROC 6M", _cycle_percent(mk["PerformanceROC6M"])),
                ("ROC 12M", _cycle_percent(mk["PerformanceROC12M"])),
            ],
        )
    with cycle_cols[3]:
        _render_cycle_panel(
            "Current Risk",
            [
                ("Status", mk["CurrentMarketRiskState"]),
                ("Breadth Risk", classify_component_state(mk["CurrentRiskBreadthRisk"])),
                ("RSI Divergence Risk", classify_component_state(mk["CurrentRiskRSIDivergenceRisk"])),
                ("VIX Risk", classify_component_state(mk["CurrentRiskVIXRisk"])),
                ("High Beta Risk", classify_component_state(mk["CurrentRiskHighBetaRisk"])),
                ("High Yield Risk", classify_component_state(mk["CurrentRiskHYRisk"])),
            ],
            accent_color=_current_risk_color(
                f"{mk['CurrentMarketRiskState']} {mk['CurrentRiskSignalClass']}",
                mk["CurrentRiskCreditConfirmation"],
            ),
        )
    with cycle_cols[4]:
        macro_rows = []
        if macro_outlook is not None:
            for horizon in SPY_MACRO_HORIZONS:
                score = macro_outlook.current.get(f"Macro_{horizon}")
                macro_rows.append((f"{horizon} State", f"{score_state(score)} ({_fmt(score, 1)}), {score_direction(macro_outlook, horizon)}"))
            macro_rows.append(("Financial Transmission", f"{_fmt(macro_outlook.current.get('Transmission_Score'), 1)} / 100"))
        else:
            macro_rows.append(("Status", "UNAVAILABLE"))
        _render_cycle_panel("Macro Outlook", macro_rows, accent_color=_macro_outlook_color(macro_outlook))
    with cycle_cols[5]:
        _card("Historical Outlook", _pct(mk["HistoricalOutlook_12M_Median"]), [
            ("12M P(positive)", _pct(mk["HistoricalOutlook_12M_PPositive"])),
            ("P(drawdown >15%)", _pct(mk["HistoricalOutlook_DD15"])),
            ("Independent N", mk["HistoricalOutlook_IndependentN"]),
            ("Confidence", mk["HistoricalOutlook_Confidence"]),
            ("Similarity / coverage", f"{_fmt(mk['HistoricalOutlook_AverageSimilarity'])} / {_pct(mk['HistoricalOutlook_Coverage'])}"),
        ], accent_color=_historical_outlook_color(mk["HistoricalOutlook_12M_Median"]))

    st.markdown("### Economy")
    cols = st.columns(3)
    with cols[0]:
        _card("Business Cycle", ec["BusinessCyclePhase"], [
            ("Direction", ec["BusinessCycleDirection"]),
            ("Transition zone", _bool(ec["BusinessCycleTransitionZone"])),
            ("Confidence", ec["BusinessCycleConfidence"]),
            ("Labor", ec["LaborCycleState"]),
        ])
    with cols[1]:
        _card("Inflation", ec["InflationDirection"], [
            ("Leading score", _fmt(ec["InflationLeadingScore"], 2)),
            ("Realized confirmation", ec["RealizedInflationConfirmation"]),
            ("Transition zone", _bool(ec["InflationTransitionZone"])),
        ], accent_color=_inflation_color(ec["InflationLeadingScore"]))
    with cols[2]:
        _card("Economy Regime", ec["EconomyRegime"], [
            ("Growth surprise", ec["GrowthSurpriseState"]),
            ("Inflation surprise", ec["InflationSurpriseState"]),
            ("Turning signal", ec["TurningSignal"]),
        ], accent_color=_economy_regime_color(ec["EconomyRegime"]))

    st.markdown("### Financial System")
    cols = st.columns([1.0, 1.0, 1.2])
    with cols[0]:
        _card("Rates & Financial Conditions", rt["Regime"], [
            ("Rates pressure", f"{_fmt(rt['RatesPressure'], 2)} / {rt['RatesDirection']}"),
            ("Core FC", f"{rt['CoreFCState']} / {rt['CoreFCStress']}"),
            ("Credit", f"{_fmt(rt['CreditState'], 2)} / {rt['CreditDirection']}"),
            ("Curve 26W", rt["YieldCurveState"]),
        ], accent_color=_rates_pressure_color(rt["RatesPressure"]))
    with cols[1]:
        _card("Funding Conditions", fd["FundingState"], [
            ("Direction", fd["FundingDirection"]),
            ("Money market stress", _fmt(fd["MoneyMarketStress"], 2)),
            ("Reserve pressure", _fmt(fd["ReservePressure"], 2)),
            ("Technical / persistent", f"{_bool(fd['TechnicalFundingFlag'])} / {_bool(fd['PersistentFundingFlag'])}"),
            ("Primary driver", fd["PrimaryDriver"]),
        ], accent_color="#22c55e" if str(fd["FundingState"]).upper() == "NORMAL" else "#ef4444")
    with cols[2]:
        _card("Financial Fragility", d.fragility["FinancialFragilityState"], [
            ("Direction", d.fragility["FinancialFragilityDirection"]),
            ("Macro / vulnerability", "n/a / n/a"),
            ("Market / funding stress", "n/a / n/a"),
            ("Transition risk", "n/a"),
            ("Status", "Production module unavailable"),
        ], prominent=True, accent_color="#22c55e" if str(d.fragility["FinancialFragilityState"]).upper() == "NORMAL" else "#ef4444")

    st.markdown("### Treasury & Fiscal")
    cols = st.columns(3)
    with cols[0]:
        _card("Treasury Liquidity", tr["TreasuryLiquidityState"], [
            ("Composite impulse", _fmt(tr["TreasuryLiquidityImpulse"], 2)),
            ("4W / 13W impulse", f"{_fmt(tr['TreasuryLiquidity4W'], 1)} / {_fmt(tr['TreasuryLiquidity13W'], 1)}"),
        ])
    with cols[1]:
        _card("Fiscal", tr["FiscalGrowthImpulse"], [
            ("Fiscal impulse", _fmt(tr["FiscalImpulse"], 2)),
            ("Deficit / GDP percentile", _pct(tr["DeficitGDPPercentile"])),
        ], accent_color=_fiscal_color(tr["FiscalGrowthImpulse"]))
    with cols[2]:
        _card("Financing Pressure", tr["FinancingPressure"], [
            ("Bill financing share", _pct(tr["BillFinancingShare"])),
            ("Duration supply percentile", _pct(tr["DurationSupplyProxy"])),
            ("Absorption", tr["AbsorptionState"]),
        ])
    st.markdown(f"<div class='gd-policy'>Policy Mix: {html.escape(str(tr['PolicyMix']))}</div>", unsafe_allow_html=True)

    st.markdown("### Cross-Cycle Sequence")
    cc = d.cross_cycle
    st.markdown(
        f"""
        <div class="gd-flow">
          <div class="gd-node"><span>GLOBAL LIQUIDITY</span><b>{html.escape(str(cc['LiquidityState']))}</b></div>
          <div class="gd-arrow">&rarr;</div>
          <div class="gd-node"><span>MARKET MOMENTUM</span><b>{html.escape(str(cc['MarketState']))}</b></div>
          <div class="gd-arrow">&rarr;</div>
          <div class="gd-node"><span>BUSINESS CYCLE</span><b>{html.escape(str(cc['BusinessState']))}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _card("Current Sequence State", cc["CrossCycleState"], [
        ("Observed liquidity -> market lag", cc["LiquidityToMarketLag"]),
        ("Historical median", cc["HistoricalLiquidityToMarketLag"]),
        ("Observed market -> business lag", cc["MarketToBusinessLag"]),
        ("Historical median", cc["HistoricalMarketToBusinessLag"]),
    ])

    st.markdown("### Key Divergences")
    st.dataframe(d.divergences, use_container_width=True, hide_index=True)

    st.markdown("### Forward Outlook")
    if d.forward_outlook.empty:
        st.info("Production analog outcomes are unavailable.")
    else:
        display = d.forward_outlook.copy()
        percent_cols = ["3M Median", "6M Median", "12M Median", "P(Positive)", "P25", "P75", "DD >15%", "DD >25%", "DD >45%", "Coverage"]
        for column in percent_cols:
            display[column] = pd.to_numeric(display[column], errors="coerce") * 100
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={column: st.column_config.NumberColumn(column, format="%.1f%%") for column in percent_cols}
            | {"Avg Similarity": st.column_config.NumberColumn("Avg Similarity", format="%.1f")},
        )
        st.caption("All assets use the same production Market Cycle analog episodes. Statistics after 3M/6M/12M medians are shown for the 12M horizon.")


def _line_chart(frame: pd.DataFrame, date_col: str, series: list[tuple[str, str, str]], title: str) -> go.Figure:
    fig = go.Figure()
    for column, label, color in series:
        if column not in frame:
            continue
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(frame[date_col], errors="coerce"), y=pd.to_numeric(frame[column], errors="coerce"),
            mode="lines", name=label, line={"color": color, "width": 1.8},
            hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>{html.escape(label)}: %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title=title, height=300, margin={"l": 45, "r": 15, "t": 45, "b": 35},
        paper_bgcolor="#0e131b", plot_bgcolor="#0e131b", font={"color": "#e5edf6", "size": 11},
        hovermode="closest", legend={"orientation": "h", "y": -0.2},
    )
    fig.update_xaxes(gridcolor="#273244")
    fig.update_yaxes(gridcolor="#273244")
    return fig


def _render_diagnostics(d: GlobalDashboardSnapshot, loaded: dict[str, Any]) -> None:
    st.markdown("### Detailed Diagnostics")
    with st.expander("Global Liquidity", expanded=False):
        left, right = st.columns(2)
        regime = loaded.get("liquidity_regime", pd.DataFrame())
        forecast = loaded.get("forecast", pd.DataFrame())
        if regime.empty:
            st.dataframe(pd.DataFrame([d.liquidity]), use_container_width=True, hide_index=True)
        else:
            with left:
                st.plotly_chart(_line_chart(regime.tail(260), "date", [("global_liquidity_score", "Liquidity Score", "#38bdf8"), ("long_cycle_value", "65M Cycle", "#facc15")], "Liquidity Score and 65M Cycle"), use_container_width=True, config=PLOT_CONFIG)
            with right:
                st.plotly_chart(_line_chart(forecast.tail(260), "Date", [("LiquidityPressureScore", "Pressure", "#fb7185"), ("PolicyResponseScore", "Policy Response", "#34d399")], "Pressure vs Policy Response"), use_container_width=True, config=PLOT_CONFIG)
    with st.expander("Market Cycle", expanded=False):
        st.dataframe(pd.DataFrame([d.market]), use_container_width=True, hide_index=True)
    with st.expander("Business Cycle", expanded=False):
        current = getattr(loaded.get("business"), "current", {}) or {}
        keys = ["BusinessCycleLevel", "BusinessCycleMomentum", "SurveyScore", "LaborScore", "ProductionScore", "DemandIncomeScore", "BusinessCycleState", "BusinessCycleTransitionZone", "BusinessCycleConfidence"]
        st.dataframe(pd.DataFrame([{key: current.get(key) for key in keys}]), use_container_width=True, hide_index=True)
    with st.expander("Inflation", expanded=False):
        current = getattr(loaded.get("business"), "current", {}) or {}
        keys = ["MarketPricingScore", "ModelImpliedInflationScore", "SurveyInflationScore", "InflationDirectionScore", "InflationState", "RealizedInflationDirection", "InflationConfirmationStatus", "EconomyRegime"]
        st.dataframe(pd.DataFrame([{key: current.get(key) for key in keys}]), use_container_width=True, hide_index=True)
    with st.expander("Macro Surprises", expanded=False):
        current = getattr(loaded.get("macro"), "current", {}) or {}
        keys = ["GrowthSurpriseScore", "PMIContribution", "RetailSalesContribution", "InitialJoblessClaimsContribution", "CPIContribution", "TurningSignal", "Persistence"]
        st.dataframe(pd.DataFrame([{key: current.get(key) for key in keys}]), use_container_width=True, hide_index=True)
    with st.expander("Rates & Financial Conditions", expanded=False):
        st.dataframe(pd.DataFrame([d.rates]), use_container_width=True, hide_index=True)
    with st.expander("Funding Conditions", expanded=False):
        st.dataframe(pd.DataFrame([d.funding]), use_container_width=True, hide_index=True)
    with st.expander("Treasury & Fiscal", expanded=False):
        st.dataframe(pd.DataFrame([d.treasury]), use_container_width=True, hide_index=True)
    with st.expander("Financial Fragility", expanded=False):
        st.info("A standalone production Financial Fragility framework is not present in the application. No proxy or simple average is substituted.")
    with st.expander("Historical Outlook / Analog Diagnostics", expanded=False):
        market = loaded.get("market", _empty())
        st.dataframe(getattr(market, "outlook", pd.DataFrame()), use_container_width=True, hide_index=True)
        analogs = getattr(market, "analogs", pd.DataFrame())
        columns = ["Date", "Similarity", "Coverage", "StructuralSimilarity", "MomentumSimilarity", "SMA200WSimilarity", "CurrentRiskSimilarity", "PositioningSimilarity"]
        if not analogs.empty:
            st.dataframe(analogs[[column for column in columns if column in analogs]].head(50), use_container_width=True, hide_index=True)
    with st.expander("Data Freshness", expanded=False):
        freshness = d.freshness.copy()
        freshness["As Of"] = pd.to_datetime(freshness["As Of"], errors="coerce").dt.strftime("%Y-%m-%d")
        st.dataframe(freshness, use_container_width=True, hide_index=True)
