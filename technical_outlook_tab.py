from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from technical_outlook.config import CORE_ASSETS
from technical_outlook.service import analyze_requested_asset
from technical_outlook.storage import (
    read_chart_bars,
    read_latest_snapshot,
    read_manifest,
    read_settings,
    write_settings,
)


def render_technical_outlook_tab(available_tickers: list[str] | None = None) -> None:
    st.markdown("## Technical Outlook")
    settings = read_settings()
    enabled = st.toggle(
        "Use LLM Interpretation",
        value=bool(settings.get("use_llm_interpretation", False)),
        help="Quant updates nightly. When enabled, LLM interpretation updates only during the Friday-night overnight run.",
        key="technical_outlook_llm_enabled",
    )
    if enabled != bool(settings.get("use_llm_interpretation", False)):
        write_settings({"use_llm_interpretation": enabled})
        st.caption("Setting saved. The next canonical LLM refresh is the Friday-night overnight update.")
    st.caption("Quant Engine is the source of truth. Scenario probabilities and all displayed levels are deterministic model outputs.")

    manifest = read_manifest()
    statuses = {item.get("ticker"): item for item in manifest.get("assets", [])}
    for ticker in CORE_ASSETS:
        snapshot = read_latest_snapshot(ticker)
        if not snapshot:
            st.markdown(f"### {ticker}")
            status = statuses.get(ticker, {})
            st.warning(status.get("stale_reason") or "No Technical Outlook snapshot yet. The overnight worker will create it.")
            continue
        render_technical_outlook_asset(snapshot, status=statuses.get(ticker))

    st.divider()
    st.markdown("## Additional Asset Analysis")
    choices = sorted({str(ticker).strip().upper() for ticker in (available_tickers or []) if str(ticker).strip()})
    ticker = st.selectbox("Ticker", [""] + choices, index=0, key="technical_outlook_additional_ticker")
    if st.button("Analyze", key="technical_outlook_analyze", disabled=not ticker):
        try:
            with st.spinner(f"Running the full Technical Outlook model for {ticker}..."):
                snapshot = analyze_requested_asset(ticker, allow_scheduled_llm=False)
            st.session_state["technical_outlook_additional_result"] = snapshot["ticker"]
        except Exception as exc:
            st.error(f"{ticker}: analysis failed — {type(exc).__name__}: {exc}")
    selected = st.session_state.get("technical_outlook_additional_result")
    if selected:
        snapshot = read_latest_snapshot(str(selected))
        if snapshot:
            render_technical_outlook_asset(snapshot, heading_level=3)


def render_technical_outlook_asset(
    snapshot: dict[str, Any],
    *,
    status: dict[str, Any] | None = None,
    heading_level: int = 2,
) -> None:
    ticker = str(snapshot.get("ticker", "Asset"))
    st.markdown(f"{'#' * heading_level} {ticker}")
    if status and status.get("status") not in {None, "CURRENT"}:
        st.warning(f"{status.get('status')}: {status.get('stale_reason', 'using the last valid snapshot')}")
    final = snapshot.get("final_state") or {}
    top = st.columns(5)
    top[0].metric("Current Price", _price(snapshot.get("price")))
    top[1].metric("Structural Trend", final.get("structural_trend", "N/A"))
    top[2].metric("Momentum", final.get("momentum", "N/A"))
    top[3].metric("Elliott Phase", final.get("elliott_phase", "UNRESOLVED"))
    top[4].metric("6M Bias", final.get("six_month_bias", "N/A"))
    detail = st.columns(4)
    detail[0].metric("Elliott Wave State", final.get("elliott_wave_state", snapshot.get("elliott_current_wave_state", "N/A")))
    detail[1].metric("Extension", final.get("extension", "N/A"))
    detail[2].metric("Confidence", final.get("confidence", "N/A"))
    detail[3].metric("MTF Regime", final.get("multi_timeframe_regime", "N/A"))
    st.caption(
        f"Quantitative Analysis Updated: {snapshot.get('quant_updated_at', 'N/A')}  |  "
        f"LLM Interpretation Updated: {snapshot.get('llm_updated_at') or 'N/A'}  |  "
        f"Model: {snapshot.get('model_version', 'N/A')}"
    )

    overlay_cols = st.columns(4)
    show_primary = overlay_cols[0].checkbox("Elliott Primary", value=True, key=f"to_primary_{ticker}")
    show_alternative = overlay_cols[1].checkbox("Elliott Alternative", value=False, key=f"to_alt_{ticker}")
    show_levels = overlay_cols[2].checkbox("Support / Resistance", value=False, key=f"to_levels_{ticker}")
    show_profile = overlay_cols[3].checkbox("Volume Profile", value=False, key=f"to_profile_{ticker}")
    if show_primary and show_alternative:
        st.caption("Primary and Alternative are both visible by explicit selection.")

    chart_cols = st.columns(2)
    for column, timeframe, label in ((chart_cols[0], "1M", "MONTHLY"), (chart_cols[1], "1W", "WEEKLY")):
        bars = read_chart_bars(ticker, str(snapshot["snapshot_id"]), timeframe)
        with column:
            st.markdown(f"#### {label}")
            if bars.empty:
                st.info("Chart data unavailable.")
            else:
                candidate_key = "elliott_major_primary" if timeframe == "1M" else "elliott_primary"
                alternative_key = "elliott_major_alternative" if timeframe == "1M" else "elliott_alternative"
                chart = build_technical_chart(
                    bars,
                    snapshot,
                    primary=snapshot.get(candidate_key) if show_primary else None,
                    alternative=snapshot.get(alternative_key) if show_alternative else None,
                    show_levels=show_levels,
                    show_profile=show_profile,
                )
                st.altair_chart(chart, use_container_width=True)

    _render_analysis(snapshot)
    st.divider()


def build_technical_chart(
    bars: pd.DataFrame,
    snapshot: dict[str, Any],
    *,
    primary: dict[str, Any] | None,
    alternative: dict[str, Any] | None,
    show_levels: bool,
    show_profile: bool,
) -> alt.VConcatChart:
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["direction"] = np.where(frame["close"] >= frame["open"], "up", "down")
    base = alt.Chart(frame).encode(
        x=alt.X("timestamp:T", axis=alt.Axis(title=None, format="%b %Y", labelFontSize=8)),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Date"), alt.Tooltip("open:Q", format=",.2f"),
            alt.Tooltip("high:Q", format=",.2f"), alt.Tooltip("low:Q", format=",.2f"), alt.Tooltip("close:Q", format=",.2f"),
        ],
    )
    wick = base.mark_rule().encode(
        y=alt.Y("low:Q", scale=alt.Scale(zero=False), title="Price"), y2="high:Q",
        color=alt.Color("direction:N", scale=alt.Scale(domain=["up", "down"], range=["#22c55e", "#ef4444"]), legend=None),
    )
    body = base.mark_bar(size=3).encode(
        y=alt.Y("open:Q", scale=alt.Scale(zero=False)), y2="close:Q",
        color=alt.Color("direction:N", scale=alt.Scale(domain=["up", "down"], range=["#22c55e", "#ef4444"]), legend=None),
    )
    layers: list[Any] = [wick, body]
    colors = {"sma50": "#f59e0b", "sma100": "#14b8a6", "sma200": "#60a5fa"}
    for name, color in colors.items():
        if name in frame and frame[name].notna().any():
            layers.append(base.mark_line(color=color, strokeWidth=1.5).encode(y=alt.Y(f"{name}:Q", scale=alt.Scale(zero=False))))
    if show_levels:
        zones = pd.DataFrame(snapshot.get("support_resistance") or [])
        if not zones.empty:
            zones["color"] = np.where(zones["role"] == "SUPPORT", "#22c55e", "#ef4444")
            layers.append(alt.Chart(zones).mark_rect(opacity=0.10).encode(y="low:Q", y2="high:Q", color=alt.Color("color:N", scale=None, legend=None)))
    if show_profile:
        profile = snapshot.get("volume_profile") or {}
        profile_levels = []
        if profile.get("poc") is not None:
            profile_levels.append({"price": profile["poc"], "kind": "POC"})
        profile_levels.extend({"price": value, "kind": "HVN"} for value in profile.get("hvns", []))
        profile_levels.extend({"price": value, "kind": "LVN"} for value in profile.get("lvns", []))
        if profile_levels:
            layers.append(
                alt.Chart(pd.DataFrame(profile_levels)).mark_rule(strokeDash=[5, 3], opacity=0.8).encode(
                    y="price:Q", color=alt.Color("kind:N", scale=alt.Scale(domain=["POC", "HVN", "LVN"], range=["#facc15", "#a78bfa", "#64748b"]))
                )
            )
    layers.extend(_elliott_layers(primary, "#ffffff"))
    layers.extend(_elliott_layers(alternative, "#f97316"))
    price_chart = alt.layer(*layers).properties(height=320).interactive()

    indicator_layers = []
    if "rsi14" in frame:
        indicator_layers.append(alt.Chart(frame).mark_line(color="#a78bfa").encode(x="timestamp:T", y=alt.Y("rsi14:Q", title="RSI")))
    indicator = alt.layer(*indicator_layers).properties(height=70) if indicator_layers else alt.Chart(frame).mark_line(opacity=0).encode(x="timestamp:T", y="close:Q").properties(height=10)
    return alt.vconcat(price_chart, indicator, spacing=3).resolve_scale(x="shared")


def _elliott_layers(candidate: dict[str, Any] | None, color: str) -> list[Any]:
    if not candidate or not candidate.get("waves"):
        return []
    points = pd.DataFrame([
        {"date": item.get("pivot_time"), "price": item.get("price"), "label": item.get("wave_label"), "status": item.get("wave_status")}
        for item in candidate["waves"] if item.get("pivot_time") and item.get("price") is not None
    ])
    if points.empty:
        return []
    points["date"] = pd.to_datetime(points["date"], errors="coerce")
    line = alt.Chart(points).mark_line(color=color, strokeWidth=1.6).encode(x="date:T", y=alt.Y("price:Q", scale=alt.Scale(zero=False)))
    labels = alt.Chart(points).mark_text(color=color, dy=-10, fontWeight="bold").encode(x="date:T", y="price:Q", text="label:N")
    return [line, labels]


def _render_analysis(snapshot: dict[str, Any]) -> None:
    st.markdown("### Technical Analysis")
    st.info(_interpretation_text(snapshot))
    st.markdown("#### Market Structure")
    st.dataframe(pd.DataFrame([
        _structure_row("Monthly", snapshot.get("monthly_structure"), snapshot.get("monthly_moving_averages")),
        _structure_row("Weekly", snapshot.get("weekly_structure"), snapshot.get("weekly_moving_averages")),
    ]), hide_index=True, use_container_width=True)

    st.markdown("#### Elliott Structure")
    elliott_cols = st.columns(2)
    with elliott_cols[0]:
        _render_elliott_candidate("PRIMARY COUNT", snapshot.get("elliott_primary") or {}, snapshot.get("elliott_confidence"))
    with elliott_cols[1]:
        _render_elliott_candidate("ALTERNATIVE COUNT", snapshot.get("elliott_alternative") or {}, snapshot.get("elliott_confidence"))
    parent = snapshot.get("elliott_major_primary") or {}
    st.caption(f"Major parent: {parent.get('label', 'UNRESOLVED')} | Parent/child: {(snapshot.get('elliott_parent_child_map') or {}).get('primary_compatibility', 'N/A')}")

    st.markdown("#### Momentum")
    monthly = snapshot.get("monthly_indicators") or {}
    weekly = snapshot.get("weekly_indicators") or {}
    st.dataframe(pd.DataFrame([
        {"Metric": "RSI14", "Monthly": _number(monthly.get("rsi14")), "Weekly": _number(weekly.get("rsi14")), "Interpretation": weekly.get("rsi_regime")},
        {"Metric": "MACD", "Monthly": _number(monthly.get("macd")), "Weekly": _number(weekly.get("macd")), "Interpretation": f"{weekly.get('macd_zero_state')} / {weekly.get('macd_histogram')}"},
        {"Metric": "ROC12", "Monthly": _number(monthly.get("roc12")), "Weekly": _number(weekly.get("roc12")), "Interpretation": weekly.get("roc_state")},
        {"Metric": "PPO200 / Extension", "Monthly": _number(monthly.get("extension200")), "Weekly": _number(weekly.get("extension200")), "Interpretation": weekly.get("extension_state")},
    ]), hide_index=True, use_container_width=True)
    divergences = snapshot.get("divergences") or []
    st.caption("Detected Divergences")
    if divergences:
        st.dataframe(pd.DataFrame([{ "Type": item.get("type"), "Timeframe": item.get("timeframe"), "Status": "ACTIVE" if item.get("active") else "INACTIVE", "Magnitude": item.get("magnitude")} for item in divergences]), hide_index=True, use_container_width=True)
    else:
        st.caption("No active structural RSI/MACD divergence detected.")

    st.markdown("#### Volume / Volume Profile")
    volume = snapshot.get("volume_state") or {}
    profile = snapshot.get("volume_profile") or {}
    st.dataframe(pd.DataFrame([{
        "Volume Trend": volume.get("trend"), "Event": volume.get("event"), "Current / Average": _number(volume.get("current_vs_average")),
        "POC": _number(profile.get("poc")), "Nearest HVN": _nearest(profile.get("hvns"), snapshot.get("price")),
        "Nearest LVN": _nearest(profile.get("lvns"), snapshot.get("price")), "Value Area": _range(profile.get("value_area")),
    }]), hide_index=True, use_container_width=True)

    st.markdown("#### Key Levels")
    zones = snapshot.get("support_resistance") or []
    if zones:
        st.dataframe(pd.DataFrame([{
            "Role": zone.get("role"), "Zone": f"{zone.get('low', 0):,.2f}–{zone.get('high', 0):,.2f}", "Confluence": zone.get("confluence"),
            "Sources": ", ".join(zone.get("sources") or []), "Timeframes": ", ".join(zone.get("timeframes") or []),
        } for zone in zones]), hide_index=True, use_container_width=True)

    st.markdown("#### Scenario Matrix")
    st.dataframe(pd.DataFrame([{
        "Scenario": item.get("scenario"), "Probability": f"{item.get('probability')}%", "Trigger": item.get("trigger"),
        "Expected Path": " → ".join(_price(value) for value in item.get("expected_path", [])), "Target": _range_list(item.get("target_zone")), "Invalidation": item.get("invalidation"),
    } for item in snapshot.get("scenarios", [])]), hide_index=True, use_container_width=True)

    st.markdown("#### Forecast Horizons")
    horizon_cols = st.columns(3)
    for column, key, title in zip(horizon_cols, ("short_term", "medium_term", "six_month"), ("SHORT TERM", "MEDIUM TERM", "6M OUTLOOK")):
        item = (snapshot.get("horizons") or {}).get(key, {})
        with column:
            st.markdown(f"**{title} — {item.get('range', '')}**")
            st.markdown(f"**{item.get('state', 'N/A')}**")
            st.caption(item.get("explanation", ""))

    st.markdown("#### Confirmation / Invalidation")
    st.dataframe(pd.DataFrame(snapshot.get("confirmation_matrix") or []), hide_index=True, use_container_width=True)
    with st.expander("Historical Analogs"):
        analogs = snapshot.get("historical_analogs") or {}
        st.markdown(f"Similar Historical Periods: **{analogs.get('sample_size', 0)}**")
        if analogs.get("warning"):
            st.warning(analogs["warning"])
        returns = analogs.get("returns") or {}
        st.dataframe(pd.DataFrame([
            {"Statistic": "Mean", **{horizon: _number((returns.get(horizon) or {}).get("mean")) for horizon in ("3M", "6M", "12M")}},
            {"Statistic": "Median", **{horizon: _number((returns.get(horizon) or {}).get("median")) for horizon in ("3M", "6M", "12M")}},
        ]), hide_index=True, use_container_width=True)
        st.caption(f"Future Maximum Drawdown Probability: {analogs.get('drawdown_probabilities', {})}")


def _render_elliott_candidate(title: str, candidate: dict[str, Any], confidence: Any) -> None:
    st.markdown(f"**{title}**")
    st.markdown(f"{candidate.get('label', 'UNRESOLVED')}")
    st.caption(
        f"Score {candidate.get('score', 0):.1f}/100 | Confidence {confidence or 'LOW'} | "
        f"Current Wave {candidate.get('current_wave', 'N/A')} | {candidate.get('wave_state', 'POTENTIAL')}"
    )
    st.caption(f"Targets: {candidate.get('fib_targets') or 'N/A'} | Invalidation: {candidate.get('invalidation') or 'N/A'}")


def _structure_row(label: str, structure: Any, ma: Any) -> dict[str, Any]:
    structure = structure or {}
    ma = ma or {}
    return {"Timeframe": label, "State": structure.get("state"), "Swings": structure.get("sequence"), "MA Structure": ma.get("ordering"), "Explanation": structure.get("explanation")}


def _interpretation_text(snapshot: dict[str, Any]) -> str:
    if snapshot.get("llm_enabled") and snapshot.get("llm_interpretation"):
        return str((snapshot["llm_interpretation"] or {}).get("summary") or snapshot.get("deterministic_narrative"))
    return str(snapshot.get("deterministic_narrative") or "")


def _price(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "N/A"


def _number(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "N/A"


def _nearest(values: Any, price: Any) -> str:
    try:
        numbers = [float(value) for value in values or []]
        return _price(min(numbers, key=lambda value: abs(value - float(price)))) if numbers else "N/A"
    except Exception:
        return "N/A"


def _range(value: Any) -> str:
    return f"{_price((value or {}).get('low'))}–{_price((value or {}).get('high'))}" if isinstance(value, dict) else "N/A"


def _range_list(value: Any) -> str:
    return f"{_price(value[0])}–{_price(value[1])}" if isinstance(value, list) and len(value) >= 2 else "N/A"
