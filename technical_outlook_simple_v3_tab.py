from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from technical_outlook_simple_v3.config import CONFIG, CORE_ASSETS, CORE_ASSET_KEYS
from technical_outlook_simple_v3.service import analyze_requested_asset, run_llm_now
from technical_outlook_simple_v3.chart import CHART_CONFIG, build_reference_chart
from technical_outlook_simple_v3.storage import (
    read_chart_bars,
    read_latest_snapshot,
    read_manifest,
    read_settings,
    write_settings,
)


def render_technical_outlook_simple_v3_tab(available_tickers: list[str] | None = None) -> None:
    st.markdown("## Technical Outlook v3")
    st.caption(
        "SIMPLE v3: independent Weekly strategic and Daily tactical analysis. Key Point classes use capped "
        "family scores (HIGH / MID / LOW); the 6M Scenario Matrix uses Weekly Quant evidence only."
    )
    settings = read_settings()
    enabled = st.toggle(
        "Enable Scheduled LLM",
        value=bool(settings.get("use_llm_interpretation", False)),
        help="Optional commentary refreshes during the Friday-night overnight run. It cannot change Quant outputs.",
        key="simple_v3_llm_enabled",
    )
    if enabled != bool(settings.get("use_llm_interpretation", False)):
        write_settings({"use_llm_interpretation": enabled})
        st.caption("SIMPLE v3 LLM setting saved.")
    manifest = read_manifest()
    statuses = {item.get("ticker"): item for item in manifest.get("assets", [])}
    for ticker in CORE_ASSET_KEYS:
        snapshot = read_latest_snapshot(ticker)
        if snapshot:
            render_simple_v3_asset(snapshot, status=statuses.get(ticker))
        else:
            st.markdown(f"### {ticker}")
            st.warning((statuses.get(ticker) or {}).get("stale_reason") or "No SIMPLE v3 snapshot yet. The overnight worker will create it.")

    st.divider()
    st.markdown("## Additional Asset Analysis")
    choices = sorted({str(value).strip().upper() for value in (available_tickers or []) if str(value).strip()})
    ticker = st.selectbox("Ticker", [""] + choices, key="simple_v3_additional_ticker")
    if st.button("Analyze", key="simple_v3_analyze", disabled=not ticker):
        try:
            with st.spinner(f"Running Technical Outlook v3 for {ticker}..."):
                result = analyze_requested_asset(ticker, allow_scheduled_llm=False)
            st.session_state["simple_v3_additional_result"] = result["ticker"]
        except Exception as exc:
            st.error(f"{ticker}: analysis failed — {type(exc).__name__}: {exc}")
    selected = st.session_state.get("simple_v3_additional_result")
    if selected:
        snapshot = read_latest_snapshot(str(selected))
        if snapshot:
            render_simple_v3_asset(snapshot, heading_level=3)


def render_simple_v3_asset(
    snapshot: dict[str, Any], *, status: dict[str, Any] | None = None, heading_level: int = 2,
) -> None:
    ticker = str(snapshot.get("ticker", "Asset"))
    st.markdown(f"{'#' * heading_level} {ticker}")
    if status and status.get("status") not in {None, "CURRENT"}:
        st.warning(f"{status.get('status')}: {status.get('stale_reason', 'using the last valid snapshot')}")
    final = snapshot.get("final_state") or {}
    top = st.columns(6)
    top[0].metric("Current Price", _price(snapshot.get("price")))
    top[1].metric("Structural Trend", final.get("structural_trend", "N/A"))
    top[2].metric("Weekly Momentum", final.get("weekly_momentum", "N/A"))
    top[3].metric("6M Bias", final.get("six_month_bias", "N/A"))
    top[4].metric("Confidence", final.get("confidence", "N/A"))
    top[5].metric("Swing Config", snapshot.get("swing_config_source", "N/A"))
    st.caption(
        f"Quant Updated: {snapshot.get('quant_updated_at', 'N/A')} | LLM Updated: {snapshot.get('llm_updated_at') or 'N/A'} | "
        f"Model: {snapshot.get('model_version')} | S/R: {snapshot.get('sr_engine_version')} | Scenario: {snapshot.get('scenario_engine_version')}"
    )
    if st.button("Run LLM", key=f"simple_v3_run_llm_{ticker}", type="primary"):
        try:
            with st.spinner(f"Running optional SIMPLE v3 commentary for {ticker}..."):
                run_llm_now(ticker)
            st.success(f"LLM commentary updated for {ticker}.")
            st.rerun()
        except Exception as exc:
            st.error(f"{ticker}: LLM analysis failed — {type(exc).__name__}: {exc}")

    interpretation = snapshot.get("llm_interpretation") or {}
    elliott = interpretation.get("elliott_structure") or {}

    st.markdown("### System Summary")
    st.info(_format_numeric_text(str(snapshot.get("deterministic_narrative") or "System summary unavailable.")))
    st.markdown("### LLM Commentary")
    if interpretation:
        st.success(_format_numeric_text(str(interpretation.get("summary") or "LLM summary is empty.")))
        commentary_frame = pd.DataFrame([
            {"Block": "Market Structure", "Commentary": _format_numeric_text(interpretation.get("market_structure_summary", ""))},
            {"Block": "Elliott", "Commentary": _format_numeric_text(interpretation.get("elliott_summary", ""))},
            {"Block": "Momentum", "Commentary": _format_numeric_text(interpretation.get("momentum_summary", ""))},
            {"Block": "Volume Profile", "Commentary": _format_numeric_text(interpretation.get("volume_profile_summary", ""))},
            {"Block": "Key Levels", "Commentary": _format_numeric_text(interpretation.get("key_levels_summary", ""))},
            {"Block": "Scenario Matrix", "Commentary": _format_numeric_text(interpretation.get("scenario_summary", ""))},
        ])
        st.dataframe(
            commentary_frame,
            column_config={
                # None means content-sized in Streamlit: the Block column
                # follows its longest label instead of taking a fixed share.
                "Block": st.column_config.TextColumn(width=None),
                "Commentary": st.column_config.TextColumn(width=None),
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("Optional LLM commentary is not available. Quant levels and scenarios are complete without it.")

    controls = st.columns(4)
    show_primary = controls[0].checkbox("Elliott Primary", value=False, key=f"simple_v3_primary_{ticker}")
    show_alternative = controls[1].checkbox("Elliott Alternative", value=False, key=f"simple_v3_alt_{ticker}")
    show_levels = controls[2].checkbox("Support / Resistance", value=True, key=f"simple_v3_sr_{ticker}")
    st.markdown("**Zone sources**")
    source_controls = st.columns(4)
    source_filters = {
        "SWING_STRUCTURE": source_controls[0].checkbox("Swing", value=False, key=f"simple_v3_swing_{ticker}"),
        "VOLUME_ACCEPTANCE": source_controls[1].checkbox("Volume", value=True, key=f"simple_v3_volume_{ticker}"),
        "FIBONACCI": source_controls[2].checkbox("Fibonacci", value=False, key=f"simple_v3_fibonacci_{ticker}"),
        "MOVING_AVERAGE": source_controls[3].checkbox("Moving Average", value=True, key=f"simple_v3_ma_{ticker}"),
    }
    st.caption("Swing: orange · Volume: pink · Strategic Fibonacci: violet · Tactical Fibonacci: green · SMA 50 / 100 / 200: blue / teal / red")
    enabled_sources = {family for family, enabled_source in source_filters.items() if enabled_source}
    history_window = snapshot.get("history_window") or {}
    chart_cols = st.columns(2)
    for column, timeframe, title, zone_key in (
        (chart_cols[0], "1W", "WEEKLY", "weekly_zones"),
        (chart_cols[1], "1D", "DAILY", "daily_zones"),
    ):
        bars = read_chart_bars(ticker, str(snapshot.get("snapshot_id")), timeframe)
        with column:
            st.markdown(f"#### {title}")
            available_bars = history_window.get("weekly_available" if timeframe == "1W" else "daily_available")
            insufficient = history_window.get("weekly_insufficient" if timeframe == "1W" else "daily_insufficient")
            if insufficient:
                st.caption(f"Fixed window: {available_bars or 0} completed bars available (less than 300).")
            key_point_filter = st.radio(
                "Key Point Zones",
                CONFIG["display"]["key_point_filter_options"],
                index=2,
                horizontal=True,
                key=f"simple_v3_key_point_filter_{ticker}_{timeframe}",
            )
            if bars.empty:
                st.info("Chart data unavailable.")
            else:
                chart = build_simple_v3_chart(
                    bars,
                    filter_chart_zones(snapshot.get(zone_key) or [], enabled_sources) if show_levels else [],
                    primary=elliott.get("primary") if show_primary else None,
                    alternative=elliott.get("alternative") if show_alternative else None,
                    timeframe=title,
                    ticker=str(snapshot.get("provider_symbol") or ticker),
                    key_point_filter=key_point_filter,
                    pivots=snapshot.get("weekly_pivots" if timeframe == "1W" else "daily_pivots") or [],
                    profile=snapshot.get("weekly_volume_profile" if timeframe == "1W" else "daily_volume_profile") or {},
                    fibonacci=snapshot.get("weekly_fibonacci_framework" if timeframe == "1W" else "daily_fibonacci_framework") or {},
                    enabled_sources=enabled_sources,
                )
                st.plotly_chart(chart, use_container_width=True, theme=None, config=CHART_CONFIG,
                                key=f"simple_v3_chart_{ticker}_{timeframe}")

    _render_v3_analysis_modules(snapshot, interpretation)

    st.markdown("### Weekly Key Levels")
    _render_zone_table(snapshot.get("weekly_zones") or [], classes={"HIGH", "MID", "LOW"})
    st.markdown("### Daily Key Levels")
    _render_zone_table(snapshot.get("daily_zones") or [], classes={"HIGH", "MID", "LOW"})
    st.markdown("### Weekly 6M Scenario Matrix")
    scenarios = snapshot.get("weekly_scenario_matrix") or []
    st.dataframe(pd.DataFrame([{
        "Scenario": item.get("scenario"),
        "Probability": f"{_number(item.get('probability'))}%",
        "Confidence": item.get("confidence"),
        "Trigger": item.get("trigger"),
        "Primary Target": _target(item.get("primary_target")),
        "Extended Target": _target(item.get("extended_target")),
        "Structural Target": _target(item.get("structural_target")),
        "Invalidation": item.get("invalidation"),
    } for item in scenarios]), hide_index=True, use_container_width=True)
    st.markdown("### Expected 6M Path")
    st.markdown(" → ".join(_path_item(item) for item in snapshot.get("weekly_expected_path") or []))
    _render_v3_forecast_and_confirmation(snapshot)

    st.markdown("### Weekly / Daily Momentum")
    weekly = snapshot.get("weekly_momentum_summary") or {}
    daily = snapshot.get("daily_momentum_summary") or {}
    st.dataframe(pd.DataFrame([
        {"Metric": "Classification", "Weekly": weekly.get("classification"), "Daily": daily.get("classification")},
        {"Metric": "Trajectory", "Weekly": weekly.get("trajectory"), "Daily": daily.get("trajectory")},
        {"Metric": "RSI14", "Weekly": _number(weekly.get("rsi14")), "Daily": _number(daily.get("rsi14"))},
        {"Metric": "MACD", "Weekly": _number(weekly.get("macd")), "Daily": _number(daily.get("macd"))},
        {"Metric": "ROC12", "Weekly": _number(weekly.get("roc12")), "Daily": _number(daily.get("roc12"))},
    ]), hide_index=True, use_container_width=True)
    with st.expander("Historical Analogs"):
        analogs = snapshot.get("historical_analogs") or {}
        st.caption(f"Sample size: {analogs.get('sample_size', 0)} | {analogs.get('warning') or 'Causal sample available'}")
        st.json(_round_numeric_values({"returns": analogs.get("returns"), "drawdown_probabilities": analogs.get("drawdown_probabilities")}))
    st.divider()


def _render_v3_analysis_modules(snapshot: dict[str, Any], interpretation: dict[str, Any]) -> None:
    """Render the diagnostic modules carried forward from the original outlook.

    These are deliberately read-only views over the SIMPLE v3 snapshot.  They
    do not invoke the legacy model or alter the v3 chart/zone calculations.
    """
    st.markdown("### Market Structure")
    st.dataframe(pd.DataFrame([
        _v3_structure_row("Weekly", snapshot.get("weekly_structure"), snapshot.get("weekly_moving_averages")),
        _v3_structure_row("Daily", snapshot.get("daily_structure"), snapshot.get("daily_moving_averages")),
    ]), hide_index=True, use_container_width=True)

    st.markdown("### Elliott Structure — LLM")
    elliott = (interpretation or {}).get("elliott_structure") or {}
    elliott_cols = st.columns(2)
    with elliott_cols[0]:
        _render_v3_elliott_candidate("PRIMARY COUNT", elliott.get("primary") or {}, elliott.get("confidence"))
    with elliott_cols[1]:
        _render_v3_elliott_candidate("ALTERNATIVE COUNT", elliott.get("alternative") or {}, elliott.get("confidence"))
    st.caption(f"Source: LLM | Current wave state: {elliott.get('current_wave_state', 'AWAITING LLM')}")

    st.markdown("### Momentum")
    weekly = snapshot.get("weekly_momentum_summary") or snapshot.get("weekly_indicators") or {}
    daily = snapshot.get("daily_momentum_summary") or snapshot.get("daily_indicators") or {}
    st.dataframe(pd.DataFrame([
        {"Metric": "RSI14", "Weekly": _number(weekly.get("rsi14")), "Daily": _number(daily.get("rsi14")), "Interpretation": daily.get("rsi_regime")},
        {"Metric": "MACD", "Weekly": _number(weekly.get("macd")), "Daily": _number(daily.get("macd")), "Interpretation": f"{daily.get('macd_zero_state')} / {daily.get('macd_histogram')}"},
        {"Metric": "ROC12", "Weekly": _number(weekly.get("roc12")), "Daily": _number(daily.get("roc12")), "Interpretation": daily.get("roc_state")},
        {"Metric": "PPO200 / Extension", "Weekly": _number(weekly.get("extension200")), "Daily": _number(daily.get("extension200")), "Interpretation": daily.get("extension_state")},
    ]), hide_index=True, use_container_width=True)
    divergences = snapshot.get("divergences") or []
    st.caption("Detected Divergences")
    if divergences:
        st.dataframe(pd.DataFrame([
            {"Type": item.get("type"), "Timeframe": item.get("timeframe"),
             "Status": "ACTIVE" if item.get("active") else "INACTIVE", "Magnitude": item.get("magnitude")}
            for item in divergences
        ]), hide_index=True, use_container_width=True)
    else:
        st.caption("No structural RSI/MACD divergence detected.")

    st.markdown("### Volume / Volume Profile")
    volume_rows = []
    for label, state_key, profile_key in (
        ("Weekly", "weekly_volume_state", "weekly_volume_profile"),
        ("Daily", "daily_volume_state", "daily_volume_profile"),
    ):
        state = snapshot.get(state_key) or {}
        profile = snapshot.get(profile_key) or {}
        volume_rows.append({
            "Timeframe": label,
            "Volume Trend": state.get("trend"),
            "Event": state.get("event"),
            "Current / Average": _number(state.get("current_vs_average")),
            "POC": _number(profile.get("poc")),
            "Nearest HVN": _nearest_v3_profile(profile.get("hvns") or profile.get("local_peaks"), snapshot.get("price")),
            "Nearest LVN": "N/A",
            "Value Area": _profile_range(profile.get("poc_zone") or profile.get("range")),
        })
    st.dataframe(pd.DataFrame(volume_rows), hide_index=True, use_container_width=True)



def _render_v3_forecast_and_confirmation(snapshot: dict[str, Any]) -> None:
    st.markdown("### Forecast Horizons")
    horizons = _v3_horizons(snapshot)
    horizon_cols = st.columns(3)
    for column, key, title in zip(horizon_cols, ("short_term", "medium_term", "six_month"), ("SHORT TERM", "MEDIUM TERM", "6M OUTLOOK")):
        item = horizons[key]
        with column:
            st.markdown(f"**{title} — {item['range']}**")
            st.markdown(f"**{item['state']}**")
            st.caption(item["explanation"])

    st.markdown("### Confirmation / Invalidation")
    st.dataframe(pd.DataFrame(_v3_confirmation(snapshot)), hide_index=True, use_container_width=True)


def _v3_structure_row(label: str, structure: Any, moving_average: Any) -> dict[str, Any]:
    structure = structure if isinstance(structure, dict) else {}
    moving_average = moving_average if isinstance(moving_average, dict) else {}
    return {
        "Timeframe": label,
        "State": structure.get("state", "N/A"),
        "Swings": structure.get("sequence", "N/A"),
        "MA Structure": moving_average.get("ordering", "N/A"),
        "Explanation": structure.get("explanation", "N/A"),
    }


def _render_v3_elliott_candidate(title: str, candidate: dict[str, Any], confidence: Any) -> None:
    st.markdown(f"**{title}**")
    st.markdown(f"**{_format_numeric_text(candidate.get('label', 'AWAITING LLM'))}**")
    st.caption(
        f"Confidence {confidence or 'N/A'} | Current wave {_format_numeric_text(candidate.get('current_wave', 'N/A'))} | "
        f"{candidate.get('wave_state', 'UNRESOLVED')}"
    )
    st.caption(
        f"Targets: {_format_numeric_text(candidate.get('targets', 'N/A'))} | "
        f"Invalidation: {_format_numeric_text(candidate.get('invalidation', 'N/A'))}"
    )


def _v3_horizons(snapshot: dict[str, Any]) -> dict[str, dict[str, str]]:
    daily = snapshot.get("daily_structure") or {}
    daily_momentum = snapshot.get("daily_momentum_summary") or {}
    scenarios = snapshot.get("weekly_scenario_matrix") or []
    dominant = max(scenarios, key=lambda item: float(item.get("probability") or 0.0), default={})
    probability = dominant.get("probability", "N/A")
    scenario_name = dominant.get("scenario", "N/A")
    return {
        "short_term": {
            "range": "1–4 weeks",
            "state": str(daily_momentum.get("classification", "N/A")),
            "explanation": f"Daily momentum is {daily_momentum.get('classification', 'N/A')} and {daily_momentum.get('trajectory', 'N/A')}.",
        },
        "medium_term": {
            "range": "1–3 months",
            "state": str(daily.get("state", "N/A")),
            "explanation": f"Daily market structure is {daily.get('state', 'N/A')} ({daily.get('sequence', 'N/A')}).",
        },
        "six_month": {
            "range": "3–6 months",
            "state": str(scenario_name),
            "explanation": f"Weekly quantitative priority scenario is {scenario_name}; probability is {_number(probability)}%.",
        },
    }


def _v3_confirmation(snapshot: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in snapshot.get("weekly_scenario_matrix") or []:
        scenario = str(item.get("scenario", "Scenario"))
        trigger = item.get("trigger")
        invalidation = item.get("invalidation")
        if trigger:
            rows.append({"event": str(trigger), "interpretation": f"{scenario} scenario confirmed"})
        if invalidation:
            rows.append({"event": str(invalidation), "interpretation": f"{scenario} scenario invalidated"})
    return rows or [{"event": "N/A", "interpretation": "No scenario confirmation events available."}]


def _nearest_v3_profile(values: Any, current: Any) -> str:
    current_value = _finite(current)
    candidates: list[float] = []
    for value in values if isinstance(values, list) else []:
        raw = value.get("center") if isinstance(value, dict) else value
        number = _finite(raw)
        if number is not None:
            candidates.append(number)
    if not candidates:
        return "N/A"
    selected = min(candidates, key=lambda item: abs(item - current_value)) if current_value is not None else candidates[0]
    return _number(selected)


def _profile_range(value: Any) -> str:
    if not isinstance(value, dict):
        return "N/A"
    low = value.get("low")
    high = value.get("high")
    if _finite(low) is None or _finite(high) is None:
        return "N/A"
    return f"{_number(low)}–{_number(high)}"


SOURCE_COLORS = {
    "SWING_STRUCTURE": "#f97316",
    "VOLUME_ACCEPTANCE": "#06b6d4",
    "FIBONACCI": "#a855f7",
    "MOVING_AVERAGE": "#facc15",
    "MULTI_SOURCE": "#f8fafc",
    "UNKNOWN": "#94a3b8",
}

SOURCE_LABELS = {
    "SWING_STRUCTURE": "Swing",
    "VOLUME_ACCEPTANCE": "Volume",
    "FIBONACCI": "Fibonacci",
    "MOVING_AVERAGE": "Moving Average",
}


def zone_source_color(zone: dict[str, Any]) -> str:
    """Return a source-based overlay color for a chart zone."""
    families = sorted({str(value).upper() for value in (zone.get("source_families") or []) if value})
    if len(families) == 1:
        return SOURCE_COLORS.get(families[0], SOURCE_COLORS["UNKNOWN"])
    if len(families) > 1:
        return SOURCE_COLORS["MULTI_SOURCE"]
    return SOURCE_COLORS["UNKNOWN"]


def _zone_source_label(zone: dict[str, Any]) -> str:
    families = sorted({str(value).upper() for value in (zone.get("source_families") or []) if value})
    return " + ".join(SOURCE_LABELS.get(family, family) for family in families) or "Unknown"


def visible_zone_frame(zones: list[dict[str, Any]], *, timeframe: str, key_point_filter: str = "ALL") -> pd.DataFrame:
    rows = [
        zone for zone in zones
        if not zone.get("hidden_by_60pct_filter") and _key_point_zone_visible(zone, key_point_filter)
    ]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["color"] = frame.apply(lambda row: zone_source_color(row.to_dict()), axis=1)
    frame["source_label"] = frame.apply(lambda row: _zone_source_label(row.to_dict()), axis=1)
    if "key_point_class" not in frame:
        frame["key_point_class"] = frame.get("confluence_class", "LOW")
    else:
        frame["key_point_class"] = frame["key_point_class"].fillna(frame.get("confluence_class", "LOW"))
    frame["zone_opacity"] = np.select(
        [
            frame["key_point_class"].eq("HIGH"),
            frame["key_point_class"].eq("MID"),
            frame["key_point_class"].eq("MEDIUM"),
        ],
        [0.50, 0.34, 0.34],
        default=0.24,
    )
    frame["level_set"] = timeframe
    return frame


def _key_point_zone_visible(zone: dict[str, Any], key_point_filter: str) -> bool:
    if str(key_point_filter).upper() == "OFF":
        return False
    score_value = zone.get("key_point_score")
    if score_value is None or not np.isfinite(float(score_value or 0)) or float(score_value or 0) <= 0:
        score_value = zone.get("total_score", zone.get("quality_score", 0.0))
    score = float(score_value or 0.0)
    zone_class = str(zone.get("key_point_class") or zone.get("confluence_class") or "").upper()
    selected = str(key_point_filter).upper()
    if selected == "HIGH":
        return zone_class in {"HIGH", "VERY_HIGH"} if zone_class else score > 150.0
    if selected in {"HIGH + MID", "HIGH+MID"}:
        return zone_class in {"MID", "MEDIUM", "HIGH", "VERY_HIGH"} if zone_class else score >= 75.0
    return True


def filter_chart_zones(
    zones: list[dict[str, Any]],
    enabled_sources: set[str],
) -> list[dict[str, Any]]:
    """Keep zones that contain at least one enabled source family.

    A zone can contain multiple families; it remains on the chart when any of
    those families is enabled.  This makes the source switches useful for
    inspecting the contribution of each family without changing the stored
    quant snapshot or scenario calculations.
    """
    if not enabled_sources:
        return []
    selected = {str(value).upper() for value in enabled_sources}
    return [
        zone for zone in zones
        if selected.intersection({str(value).upper() for value in (zone.get("source_families") or [])})
    ]


def build_simple_v3_chart(
    bars: pd.DataFrame,
    zones: list[dict[str, Any]],
    *,
    primary: dict[str, Any] | None,
    alternative: dict[str, Any] | None,
    timeframe: str,
    key_point_filter: str = "OFF",
    pivots: list[dict[str, Any]] | None = None,
    profile: dict[str, Any] | None = None,
    fibonacci: dict[str, Any] | None = None,
    enabled_sources: set[str] | None = None,
    ticker: str = "",
) -> Any:
    return build_reference_chart(
        bars, zones, primary=primary, alternative=alternative, timeframe=timeframe,
        key_point_filter=key_point_filter, pivots=pivots, profile=profile,
        fibonacci=fibonacci, enabled_sources=enabled_sources, ticker=ticker,
    )


def _render_zone_table(zones: list[dict[str, Any]], *, classes: set[str]) -> None:
    visible = [zone for zone in zones if zone.get("key_point_class", zone.get("confluence_class")) in classes and not zone.get("hidden_by_60pct_filter")]
    if not visible:
        class_label = " / ".join(sorted(classes))
        st.caption(f"No {class_label} zones currently qualify.")
        return
    family_names = {
        "SWING_STRUCTURE": "Swing", "VOLUME_ACCEPTANCE": "Volume",
        "FIBONACCI": "Fibonacci", "MOVING_AVERAGE": "Moving Average",
    }
    st.dataframe(pd.DataFrame([{
        "Role": zone.get("role"),
        "Zone": f"{float(zone.get('low', 0)):,.2f}–{float(zone.get('high', 0)):,.2f}",
        "Class": zone.get("key_point_class", zone.get("confluence_class")),
        "Score": _number(zone.get("key_point_score", zone.get("quality_score"))),
        "Pivot Score": _number(zone.get("pivot_score")),
        "SMA Score": _number(zone.get("sma_score")),
        "Volume Score": _number(zone.get("volume_score")),
        "Fibonacci Score": _number(zone.get("fibonacci_score")),
        "Strength": zone.get("strength_class"),
        "Drivers": " | ".join(zone.get("drivers") or []),
        "Distance %": _percent(zone.get("distance_pct")),
    } for zone in visible]), hide_index=True, use_container_width=True)


def _target(value: Any) -> str:
    if not isinstance(value, dict):
        return "N/A"
    if value.get("label") and not value.get("range"):
        return str(value["label"])
    range_ = value.get("range")
    if isinstance(range_, list) and len(range_) >= 2:
        return f"{_price(range_[0])}–{_price(range_[1])} ({value.get('source', 'N/A')})"
    return "N/A"


def _path_item(value: Any) -> str:
    if isinstance(value, list) and len(value) >= 2:
        return f"{_price(value[0])}–{_price(value[1])}"
    return _price(value)


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


_DECIMAL_TOKEN = re.compile(r"(?<![A-Za-z0-9_])([+-]?\d+\.\d+)(?![A-Za-z0-9_])")


def _format_numeric_text(value: Any) -> str:
    """Format decimal tokens embedded in narrative/LLM text to two places."""
    text = "" if value is None else str(value)

    def replace(match: re.Match[str]) -> str:
        try:
            return f"{float(match.group(1)):.2f}"
        except (TypeError, ValueError):
            return match.group(1)

    return _DECIMAL_TOKEN.sub(replace, text)


def _round_numeric_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _round_numeric_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_numeric_values(item) for item in value]
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return round(float(value), 2)
    return value


def _percent(value: Any) -> str:
    try:
        return f"{float(value):+.2%}"
    except Exception:
        return "N/A"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None
