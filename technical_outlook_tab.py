from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from technical_outlook.config import CORE_ASSETS
from technical_outlook.service import analyze_requested_asset, run_llm_now
from technical_outlook.storage import (
    read_chart_bars,
    read_latest_snapshot,
    read_manifest,
    read_settings,
    write_settings,
)


def render_technical_outlook_tab(available_tickers: list[str] | None = None) -> None:
    _render_technical_outlook_tab(available_tickers, legacy=False)


def render_technical_outlook_v0_tab(available_tickers: list[str] | None = None) -> None:
    """Render the pre-Structural-Weekly-S/R view against the current persisted snapshot."""
    _render_technical_outlook_tab(available_tickers, legacy=True)


def _render_technical_outlook_tab(
    available_tickers: list[str] | None,
    *,
    legacy: bool,
) -> None:
    variant = "technical_outlook_v0" if legacy else "technical_outlook"
    st.markdown("## Technical Outlook v0" if legacy else "## Technical Outlook")
    settings = read_settings()
    enabled = st.toggle(
        "Enable Scheduled LLM",
        value=bool(settings.get("use_llm_interpretation", False)),
        help="When enabled, the scheduled LLM interpretation refresh runs on Friday night. Run LLM remains available at any time.",
        key=f"{variant}_llm_enabled",
    )
    if enabled != bool(settings.get("use_llm_interpretation", False)):
        write_settings({"use_llm_interpretation": enabled})
        st.caption("Setting saved. The next canonical LLM refresh is the Friday-night overnight update.")
    if legacy:
        st.caption(
            "Frozen pre-Structural-Weekly-S/R view. Support / Resistance, Key Levels and scenarios use the original Daily S/R output only. "
            "The compact LLM request fix is retained."
        )
    else:
        st.caption("Quant calculations use Weekly + Daily bars. Elliott Structure is assigned by the LLM; prices, levels and probabilities remain deterministic outputs.")

    manifest = read_manifest()
    statuses = {item.get("ticker"): item for item in manifest.get("assets", [])}
    for ticker in CORE_ASSETS:
        snapshot = read_latest_snapshot(ticker)
        if not snapshot:
            st.markdown(f"### {ticker}")
            status = statuses.get(ticker, {})
            st.warning(status.get("stale_reason") or "No Technical Outlook snapshot yet. The overnight worker will create it.")
            continue
        render_technical_outlook_asset(snapshot, status=statuses.get(ticker), legacy=legacy)

    st.divider()
    st.markdown("## Additional Asset Analysis")
    choices = sorted({str(ticker).strip().upper() for ticker in (available_tickers or []) if str(ticker).strip()})
    ticker = st.selectbox("Ticker", [""] + choices, index=0, key=f"{variant}_additional_ticker")
    if st.button("Analyze", key=f"{variant}_analyze", disabled=not ticker):
        try:
            with st.spinner(f"Running the full Technical Outlook model for {ticker}..."):
                snapshot = analyze_requested_asset(ticker, allow_scheduled_llm=False)
            st.session_state[f"{variant}_additional_result"] = snapshot["ticker"]
        except Exception as exc:
            st.error(f"{ticker}: analysis failed — {type(exc).__name__}: {exc}")
    selected = st.session_state.get(f"{variant}_additional_result")
    if selected:
        snapshot = read_latest_snapshot(str(selected))
        if snapshot:
            render_technical_outlook_asset(snapshot, heading_level=3, legacy=legacy)


def render_technical_outlook_asset(
    snapshot: dict[str, Any],
    *,
    status: dict[str, Any] | None = None,
    heading_level: int = 2,
    legacy: bool = False,
) -> None:
    ticker = str(snapshot.get("ticker", "Asset"))
    widget_prefix = "to_v0" if legacy else "to"
    notice_key = "technical_outlook_llm_notice_v0" if legacy else "technical_outlook_llm_notice"
    run_llm_key = f"technical_outlook_run_llm_v0_{ticker}" if legacy else f"technical_outlook_run_llm_{ticker}"
    st.markdown(f"{'#' * heading_level} {ticker}")
    if st.session_state.get(notice_key) == ticker:
        st.success(f"LLM analysis updated for {ticker}.")
        del st.session_state[notice_key]
    if status and status.get("status") not in {None, "CURRENT"}:
        st.warning(f"{status.get('status')}: {status.get('stale_reason', 'using the last valid snapshot')}")
    final = snapshot.get("final_state") or {}
    elliott = _llm_elliott(snapshot)
    elliott_primary = elliott.get("primary") or {}
    top = st.columns(5)
    top[0].metric("Current Price", _price(snapshot.get("price")))
    top[1].metric("Structural Trend", final.get("structural_trend", "N/A"))
    top[2].metric("Momentum", final.get("momentum", "N/A"))
    top[3].metric("Elliott Phase (LLM)", elliott_primary.get("label", "AWAITING LLM"))
    top[4].metric("6M Bias", final.get("six_month_bias", "N/A"))
    detail = st.columns(4)
    detail[0].metric("Elliott Wave State", elliott.get("current_wave_state", "AWAITING LLM"))
    detail[1].metric("Extension", final.get("extension", "N/A"))
    detail[2].metric("Quant Confidence", final.get("confidence", "N/A"))
    detail[3].metric("MTF Regime", final.get("multi_timeframe_regime", "N/A"))
    st.caption(
        f"Quantitative Analysis Updated: {snapshot.get('quant_updated_at', 'N/A')}  |  "
        f"LLM Interpretation Updated: {snapshot.get('llm_updated_at') or 'N/A'}  |  "
        f"Model: {snapshot.get('model_version', 'N/A')}"
    )
    if st.button("Run LLM", key=run_llm_key, type="primary"):
        try:
            with st.spinner(f"Running LLM Technical Outlook for {ticker}..."):
                run_llm_now(ticker)
            st.session_state[notice_key] = ticker
            st.rerun()
        except Exception as exc:
            st.error(f"{ticker}: LLM analysis failed — {type(exc).__name__}: {exc}")

    overlay_cols = st.columns(4 if legacy else 5)
    show_primary = overlay_cols[0].checkbox("Elliott Primary", value=True, key=f"{widget_prefix}_primary_{ticker}")
    show_alternative = overlay_cols[1].checkbox("Elliott Alternative", value=False, key=f"{widget_prefix}_alt_{ticker}")
    show_levels = overlay_cols[2].checkbox(
        "Support / Resistance" if legacy else "Daily Support / Resistance",
        value=True,
        key=f"{widget_prefix}_levels_{ticker}",
    )
    if legacy:
        show_weekly_levels = False
        show_profile = overlay_cols[3].checkbox("Volume Profile", value=False, key=f"{widget_prefix}_profile_{ticker}")
    else:
        show_weekly_levels = overlay_cols[3].checkbox("Weekly Support / Resistance", value=True, key=f"{widget_prefix}_weekly_levels_{ticker}")
        show_profile = overlay_cols[4].checkbox("Volume Profile", value=False, key=f"{widget_prefix}_profile_{ticker}")
    if show_primary and show_alternative:
        st.caption("Primary and Alternative are both visible by explicit selection.")

    chart_cols = st.columns(2)
    for column, timeframe, label in ((chart_cols[0], "1W", "WEEKLY"), (chart_cols[1], "1D", "DAILY")):
        bars = read_chart_bars(ticker, str(snapshot["snapshot_id"]), timeframe)
        with column:
            st.markdown(f"#### {label}")
            if bars.empty:
                st.info("Chart data unavailable.")
            else:
                chart = build_technical_chart(
                    bars,
                    snapshot,
                    primary=elliott.get("primary") if show_primary else None,
                    alternative=elliott.get("alternative") if show_alternative else None,
                    show_levels=show_levels,
                    show_profile=show_profile,
                    show_weekly_levels=show_weekly_levels,
                )
                st.altair_chart(chart, use_container_width=True)

    _render_analysis(snapshot, legacy=legacy)
    st.divider()


def build_technical_chart(
    bars: pd.DataFrame,
    snapshot: dict[str, Any],
    *,
    primary: dict[str, Any] | None,
    alternative: dict[str, Any] | None,
    show_levels: bool,
    show_profile: bool,
    show_weekly_levels: bool = False,
) -> alt.VConcatChart:
    frame = bars.tail(500).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    frame = frame.dropna(subset=["timestamp"])
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
    for visible, level_set in ((show_levels, "daily"), (show_weekly_levels, "weekly")):
        if not visible:
            continue
        zones = _visible_support_resistance(snapshot, level_set=level_set)
        if not zones.empty:
            layers.append(
                alt.Chart(zones).mark_rect().encode(
                    y="low:Q",
                    y2="high:Q",
                    color=alt.Color("color:N", scale=None, legend=None),
                    opacity=alt.Opacity("zone_opacity:Q", scale=None, legend=None),
                    tooltip=[
                        alt.Tooltip("role:N", title="Zone"),
                        alt.Tooltip("confluence:N", title="Confluence"),
                        alt.Tooltip("low:Q", title="Low", format=",.2f"),
                        alt.Tooltip("high:Q", title="High", format=",.2f"),
                        alt.Tooltip("level_set:N", title="Timeframe"),
                    ],
                )
            )
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
    start = frame["timestamp"].min()
    end = frame["timestamp"].max()
    layers.extend(_elliott_layers(primary, "#ffffff", start, end))
    layers.extend(_elliott_layers(alternative, "#f97316", start, end))
    price_chart = alt.layer(*layers).properties(height=320)

    indicator_layers = []
    if "rsi14" in frame:
        indicator_layers.append(alt.Chart(frame).mark_line(color="#a78bfa").encode(x="timestamp:T", y=alt.Y("rsi14:Q", title="RSI")))
    indicator = alt.layer(*indicator_layers).properties(height=70) if indicator_layers else alt.Chart(frame).mark_line(opacity=0).encode(x="timestamp:T", y="close:Q").properties(height=10)
    return alt.vconcat(price_chart, indicator, spacing=3).resolve_scale(x="shared")


def _elliott_layers(candidate: dict[str, Any] | None, color: str, start: pd.Timestamp, end: pd.Timestamp) -> list[Any]:
    if not candidate or not candidate.get("waves"):
        return []
    points = pd.DataFrame([
        {"date": item.get("pivot_time"), "price": item.get("price"), "label": item.get("wave_label"), "status": item.get("wave_status")}
        for item in candidate["waves"] if item.get("pivot_time") and item.get("price") is not None
    ])
    if points.empty:
        return []
    points["date"] = pd.to_datetime(points["date"], errors="coerce", utc=True).dt.tz_convert(None)
    points = points.loc[points["date"].between(start, end, inclusive="both")]
    if points.empty:
        return []
    line = alt.Chart(points).mark_line(color=color, strokeWidth=1.6).encode(x="date:T", y=alt.Y("price:Q", scale=alt.Scale(zero=False)))
    labels = alt.Chart(points).mark_text(color=color, dy=-10, fontWeight="bold").encode(x="date:T", y="price:Q", text="label:N")
    return [line, labels]


def _visible_support_resistance(snapshot: dict[str, Any], *, level_set: str = "daily") -> pd.DataFrame:
    weekly = str(level_set).lower() == "weekly"
    key = "weekly_support_resistance" if weekly else "daily_support_resistance"
    fallback = [] if weekly else snapshot.get("support_resistance")
    zones = pd.DataFrame(snapshot.get(key) or fallback or [])
    if zones.empty or "confluence" not in zones or "role" not in zones:
        return pd.DataFrame()
    zones = zones.loc[zones["confluence"].isin(["HIGH", "VERY_HIGH"])].copy()
    if zones.empty:
        return zones
    if weekly:
        zones["color"] = np.where(zones["role"] == "SUPPORT", "#00d4ff", "#ffb020")
        zones["zone_opacity"] = np.where(zones["confluence"] == "VERY_HIGH", 0.48, 0.32)
        zones["level_set"] = "WEEKLY"
    else:
        zones["color"] = np.where(zones["role"] == "SUPPORT", "#00ff88", "#ff4d5a")
        zones["zone_opacity"] = np.where(zones["confluence"] == "VERY_HIGH", 0.42, 0.26)
        zones["level_set"] = "DAILY"
    return zones


def _render_analysis(snapshot: dict[str, Any], *, legacy: bool = False) -> None:
    st.markdown("### Technical Analysis")
    st.markdown("#### System Summary")
    st.info(str(snapshot.get("deterministic_narrative") or "System summary is not available."))
    st.markdown("#### LLM Summary")
    _render_llm_summary(snapshot)
    st.markdown("#### Market Structure")
    st.dataframe(pd.DataFrame([
        _structure_row("Weekly", snapshot.get("weekly_structure"), snapshot.get("weekly_moving_averages")),
        _structure_row("Daily", snapshot.get("daily_structure"), snapshot.get("daily_moving_averages")),
    ]), hide_index=True, use_container_width=True)

    st.markdown("#### Elliott Structure — LLM")
    elliott = _llm_elliott(snapshot)
    elliott_cols = st.columns(2)
    with elliott_cols[0]:
        _render_elliott_candidate("PRIMARY COUNT", elliott.get("primary") or {}, elliott.get("confidence"))
    with elliott_cols[1]:
        _render_elliott_candidate("ALTERNATIVE COUNT", elliott.get("alternative") or {}, elliott.get("confidence"))
    st.caption(f"Source: LLM | Current wave state: {elliott.get('current_wave_state', 'AWAITING LLM')}")

    st.markdown("#### Momentum")
    weekly = snapshot.get("weekly_indicators") or {}
    daily = snapshot.get("daily_indicators") or {}
    st.dataframe(pd.DataFrame([
        {"Metric": "RSI14", "Weekly": _number(weekly.get("rsi14")), "Daily": _number(daily.get("rsi14")), "Interpretation": daily.get("rsi_regime")},
        {"Metric": "MACD", "Weekly": _number(weekly.get("macd")), "Daily": _number(daily.get("macd")), "Interpretation": f"{daily.get('macd_zero_state')} / {daily.get('macd_histogram')}"},
        {"Metric": "ROC12", "Weekly": _number(weekly.get("roc12")), "Daily": _number(daily.get("roc12")), "Interpretation": daily.get("roc_state")},
        {"Metric": "PPO200 / Extension", "Weekly": _number(weekly.get("extension200")), "Daily": _number(daily.get("extension200")), "Interpretation": daily.get("extension_state")},
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
    use_weekly = False
    if not legacy:
        level_timeframe = st.radio(
            "S/R analysis timeframe",
            ("Weekly", "Daily"),
            horizontal=True,
            key=f"technical_outlook_level_timeframe_{snapshot.get('ticker', 'asset')}",
        )
        use_weekly = level_timeframe == "Weekly"
    content = _analysis_content(snapshot, legacy=legacy, use_weekly=use_weekly)
    zones = content["zones"]
    if zones:
        st.dataframe(pd.DataFrame([{
            "Role": zone.get("role"), "Zone": f"{zone.get('low', 0):,.2f}–{zone.get('high', 0):,.2f}",
            "Distance from Price %": _zone_distance_from_price(snapshot.get("price"), zone), "Confluence": zone.get("confluence"),
            "Sources": ", ".join(zone.get("sources") or []), "Timeframes": ", ".join(zone.get("timeframes") or []),
        } for zone in zones]), hide_index=True, use_container_width=True)
    elif not legacy:
        st.info(f"No HIGH or VERY_HIGH {level_timeframe.lower()} confluence zones are available.")

    st.markdown("#### Scenario Matrix")
    scenarios = content["scenarios"]
    st.dataframe(pd.DataFrame([{
        "Scenario": item.get("scenario"), "Probability": f"{item.get('probability')}%", "Trigger": item.get("trigger"),
        "Expected Path": " → ".join(_price(value) for value in item.get("expected_path", [])), "Target": _range_list(item.get("target_zone")), "Invalidation": item.get("invalidation"),
    } for item in scenarios]), hide_index=True, use_container_width=True)

    st.markdown("#### Forecast Horizons")
    horizon_cols = st.columns(3)
    for column, key, title in zip(horizon_cols, ("short_term", "medium_term", "six_month"), ("SHORT TERM", "MEDIUM TERM", "6M OUTLOOK")):
        item = (snapshot.get("horizons") or {}).get(key, {})
        with column:
            st.markdown(f"**{title} — {item.get('range', '')}**")
            st.markdown(f"**{item.get('state', 'N/A')}**")
            st.caption(item.get("explanation", ""))

    st.markdown("#### Confirmation / Invalidation")
    confirmation = content["confirmation"]
    st.dataframe(pd.DataFrame(confirmation), hide_index=True, use_container_width=True)
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


def _analysis_content(
    snapshot: dict[str, Any],
    *,
    legacy: bool,
    use_weekly: bool,
) -> dict[str, list[dict[str, Any]]]:
    if legacy:
        return {
            "zones": list(snapshot.get("support_resistance") or []),
            "scenarios": list(snapshot.get("scenarios") or []),
            "confirmation": list(snapshot.get("confirmation_matrix") or []),
        }
    zones = (
        snapshot.get("weekly_support_resistance")
        if use_weekly
        else snapshot.get("daily_support_resistance") or snapshot.get("support_resistance")
    ) or []
    scenarios = (
        snapshot.get("weekly_scenarios")
        if use_weekly
        else snapshot.get("daily_scenarios") or snapshot.get("scenarios")
    ) or []
    confirmation = (
        snapshot.get("weekly_confirmation_matrix")
        if use_weekly
        else snapshot.get("daily_confirmation_matrix") or snapshot.get("confirmation_matrix")
    ) or []
    return {
        "zones": [zone for zone in zones if zone.get("confluence") in {"HIGH", "VERY_HIGH"}][:20],
        "scenarios": list(scenarios),
        "confirmation": list(confirmation),
    }


def _render_elliott_candidate(title: str, candidate: dict[str, Any], confidence: Any) -> None:
    st.markdown(f"**{title}**")
    st.markdown(f"{candidate.get('label', 'AWAITING LLM')}")
    st.caption(
        f"Confidence {confidence or 'N/A'} | Current Wave {candidate.get('current_wave', 'N/A')} | "
        f"{candidate.get('wave_state', 'UNRESOLVED')}"
    )
    st.caption(f"Targets: {candidate.get('targets') or 'N/A'} | Invalidation: {candidate.get('invalidation') or 'N/A'}")
    if candidate.get("rationale"):
        st.caption(candidate["rationale"])


def _structure_row(label: str, structure: Any, ma: Any) -> dict[str, Any]:
    structure = structure or {}
    ma = ma or {}
    return {"Timeframe": label, "State": structure.get("state"), "Swings": structure.get("sequence"), "MA Structure": ma.get("ordering"), "Explanation": structure.get("explanation")}


def _render_llm_summary(snapshot: dict[str, Any]) -> None:
    interpretation = snapshot.get("llm_interpretation") or {}
    if not interpretation:
        st.warning("LLM summary is not available. Click Run LLM to generate Elliott Structure and block-by-block interpretation.")
        return
    st.success(str(interpretation.get("summary") or "LLM summary is empty."))
    labels = (
        ("Market Structure", "market_structure_summary"),
        ("Elliott Structure", "elliott_summary"),
        ("Momentum", "momentum_summary"),
        ("Volume / Volume Profile", "volume_profile_summary"),
        ("Key Levels", "key_levels_summary"),
        ("Scenario Matrix", "scenario_summary"),
        ("Forecast Horizons", "forecast_horizons_summary"),
        ("Confirmation / Invalidation", "confirmation_invalidation_summary"),
        ("Historical Analogs", "historical_analogs_summary"),
    )
    st.dataframe(
        pd.DataFrame([{"Analysis Block": label, "LLM Summary": interpretation.get(field, "")} for label, field in labels]),
        hide_index=True,
        use_container_width=True,
    )
    if interpretation.get("risk_factors"):
        st.caption("Risk Factors: " + " | ".join(interpretation["risk_factors"]))
    if interpretation.get("key_confirmation_points"):
        st.caption("Key Confirmation Points: " + " | ".join(interpretation["key_confirmation_points"]))


def _llm_elliott(snapshot: dict[str, Any]) -> dict[str, Any]:
    interpretation = snapshot.get("llm_interpretation") or {}
    value = interpretation.get("elliott_structure")
    return value if isinstance(value, dict) else {}


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


def _zone_distance_from_price(current_price: Any, zone: dict[str, Any]) -> str:
    try:
        average_price = (float(zone["low"]) + float(zone["high"])) / 2.0
        if average_price == 0.0:
            return "N/A"
        distance = float(current_price) / average_price - 1.0
        return f"{distance:+.2%}"
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return "N/A"
