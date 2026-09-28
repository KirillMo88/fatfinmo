from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from elliott_waves.config import ANALYSIS_WINDOWS, ASSET_SPECS, DEFAULT_ANALYSIS_WINDOW
from elliott_waves.export import build_json_export, build_xlsx_export
from elliott_waves.storage import read_chart_bars, read_manifest, read_quotes, read_snapshot
from elliott_waves.summary import build_summary


ELLIOTT_PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}
ANALYSIS_OPTIONS = list(ANALYSIS_WINDOWS)


def render_elliott_waves_tab() -> None:
    st.subheader("Eliot waves")
    manifest = read_manifest()
    if not manifest:
        st.info("PENDING_INITIAL_COMPUTE — запустите фоновое задание Elliott Waves. Тяжёлый расчёт не выполняется при открытии вкладки.")
        return

    manifest_rows = {row.get("canonical_asset_id"): row for row in manifest.get("assets", [])}
    control_cols = st.columns([1.45, 2.1, 1.45, 1.25, 1.25])
    with control_cols[0]:
        chart_timeframe = st.radio("Бары", ["1D", "1W", "1M"], index=1, horizontal=True, key="elliott_chart_timeframe")
    with control_cols[1]:
        analysis_window = st.radio(
            "Период анализа",
            ANALYSIS_OPTIONS,
            index=ANALYSIS_OPTIONS.index(DEFAULT_ANALYSIS_WINDOW),
            horizontal=True,
            key="elliott_analysis_window",
        )
    with control_cols[2]:
        price_scale = st.radio("Шкала", ["Linear", "Log"], horizontal=True, key="elliott_price_scale")

    snapshots: dict[str, dict[str, Any]] = {}
    for asset_id in ASSET_SPECS:
        row = manifest_rows.get(asset_id, {})
        window_row = (row.get("windows") or {}).get(analysis_window, {})
        snapshot_id = window_row.get("snapshot_id")
        if not snapshot_id and analysis_window == DEFAULT_ANALYSIS_WINDOW:
            snapshot_id = row.get("snapshot_id")
        if snapshot_id:
            snapshot = read_snapshot(asset_id, snapshot_id)
            if snapshot:
                snapshots[asset_id] = snapshot

    view_settings = _current_view_settings(snapshots, chart_timeframe, analysis_window, price_scale)
    with control_cols[3]:
        xlsx_bytes = build_xlsx_export(manifest, snapshots, view_settings) if snapshots else b""
        st.download_button(
            "Экспорт XLSX",
            data=xlsx_bytes,
            file_name=f"elliott_waves_{manifest.get('manifest_id', 'snapshot')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=not bool(snapshots),
            use_container_width=True,
            key="elliott_export_xlsx",
        )
    with control_cols[4]:
        json_bytes = build_json_export(manifest, snapshots, view_settings) if snapshots else b""
        st.download_button(
            "Экспорт JSON",
            data=json_bytes,
            file_name=f"elliott_waves_{manifest.get('manifest_id', 'snapshot')}.json",
            mime="application/json",
            disabled=not bool(snapshots),
            use_container_width=True,
            key="elliott_export_json",
        )

    st.caption(
        f"Manifest {manifest.get('manifest_id', 'n/a')} · published {manifest.get('published_at', 'n/a')} · "
        f"анализ {analysis_window} загружается из готового снимка; переключатели не запускают расчёт"
    )
    quotes = read_quotes().get("assets", {})
    ordered = list(ASSET_SPECS)
    for row_start in range(0, len(ordered), 2):
        columns = st.columns(2, gap="medium")
        for column, asset_id in zip(columns, ordered[row_start : row_start + 2]):
            with column:
                _render_asset_card(
                    asset_id,
                    manifest_rows.get(asset_id, {}),
                    snapshots.get(asset_id),
                    quotes.get(asset_id, {}),
                    chart_timeframe,
                    analysis_window,
                    price_scale,
                )


def _render_asset_card(
    asset_id: str,
    manifest_row: dict[str, Any],
    snapshot: dict[str, Any] | None,
    quote: dict[str, Any],
    chart_timeframe: str,
    analysis_window: str,
    price_scale: str,
) -> None:
    st.markdown(f"### {asset_id}")
    if snapshot is None:
        status = manifest_row.get("status", "PENDING_INITIAL_COMPUTE")
        reason = manifest_row.get("stale_reason") or "Снимок ещё не опубликован."
        st.error(f"{status}: {reason}")
        return

    price = quote.get("price")
    price_text = f"{float(price):,.2f}" if price is not None and np.isfinite(float(price)) else "n/a"
    freshness = f"analytics {snapshot.get('as_of', 'n/a')}"
    quote_status = quote.get("status", "NO_QUOTE")
    st.caption(
        f"Цена: {price_text} ({quote_status}, {quote.get('observed_at', 'n/a')}) · "
        f"{freshness} · base {snapshot.get('base_timeframe')} · {snapshot.get('provider_label')}"
    )
    if manifest_row.get("status") != "CURRENT":
        st.warning(f"{manifest_row.get('status')}: {manifest_row.get('stale_reason')}")

    scenarios = snapshot.get("root_scenarios", [])
    scenario_options = {scenario.get("selection_label", scenario.get("scenario_id")): scenario.get("scenario_id") for scenario in scenarios}
    scenario_key = f"elliott_scenario_{asset_id}"
    stored_scenario = st.session_state.get(scenario_key)
    if stored_scenario is not None and stored_scenario not in scenario_options:
        st.warning("Ранее выбранный сценарий отсутствует в новом снимке; показан текущий основной вариант.")
        del st.session_state[scenario_key]
    if len(scenario_options) > 1:
        selected_label = st.selectbox("Сценарий", list(scenario_options), key=scenario_key)
        scenario_id = scenario_options[selected_label]
    elif len(scenario_options) == 1:
        selected_label, scenario_id = next(iter(scenario_options.items()))
        st.caption(f"Сценарий: {selected_label}")
        st.session_state[scenario_key] = selected_label
    else:
        scenario_id = None
        st.info("UNRESOLVED — допустимый активный сценарий не найден.")

    node = _selected_node(snapshot, scenario_id)
    option_cols = st.columns([1, 1, 1, 1.35, 1, 1])
    with option_cols[0]:
        show_major = st.toggle("Major", value=True, key=f"elliott_major_{asset_id}")
    with option_cols[1]:
        show_intermediate = st.toggle("Intermediate", value=True, key=f"elliott_intermediate_{asset_id}")
    with option_cols[2]:
        show_minor = st.toggle("Minor", value=True, key=f"elliott_minor_{asset_id}")
    with option_cols[3]:
        label_mode = st.selectbox("Подписи", ["Auto", "Full", "Minimal"], key=f"elliott_labels_{asset_id}")
    with option_cols[4]:
        show_targets = st.toggle("Цели", value=True, key=f"elliott_targets_{asset_id}")
    with option_cols[5]:
        show_channels = st.toggle("Каналы", value=False, key=f"elliott_channels_{asset_id}")

    bars = read_chart_bars(asset_id, snapshot["snapshot_id"], chart_timeframe)
    if bars.empty:
        st.error(f"Для {asset_id} нет свечей {chart_timeframe} в периоде {analysis_window}.")
    else:
        fig = _build_chart(
            snapshot,
            scenario_id,
            node,
            bars,
            price_scale,
            {"Major": show_major, "Intermediate": show_intermediate, "Minor": show_minor},
            label_mode,
            chart_timeframe,
            show_targets,
            show_channels,
        )
        st.plotly_chart(fig, use_container_width=True, config=ELLIOTT_PLOTLY_CONFIG, key=f"elliott_chart_{asset_id}")

    st.markdown(build_summary(snapshot, scenario_id, node.get("node_id") if node else None, label_mode))
    active_path = _active_path(snapshot, scenario_id)
    if active_path:
        st.caption(
            "Активный путь: "
            + " → ".join(
                f"{item.get('degree')} · {item.get('pattern_type')}"
                for item in active_path
            )
        )
    with st.expander("Подробности", expanded=False):
        _render_details(snapshot, scenario_id, node)


def _build_chart(
    snapshot: dict[str, Any],
    scenario_id: str | None,
    node: dict[str, Any] | None,
    bars: pd.DataFrame,
    price_scale: str,
    visible_degrees: dict[str, bool],
    label_mode: str,
    chart_timeframe: str,
    show_targets: bool,
    show_channels: bool,
) -> go.Figure:
    d = bars.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp", "open", "high", "low", "close"])
    fig = go.Figure(
        go.Candlestick(
            x=d["timestamp"],
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            name=snapshot.get("canonical_asset_id"),
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
            increasing_fillcolor="#14532d",
            decreasing_fillcolor="#7f1d1d",
        )
    )
    if "is_closed" in d.columns:
        partial = d.loc[~d["is_closed"].fillna(False).astype(bool)]
        if not partial.empty:
            fig.add_trace(
                go.Scatter(
                    x=partial["timestamp"],
                    y=partial["close"],
                    mode="markers+text",
                    text=["незакрытая"] * len(partial),
                    textposition="bottom center",
                    marker={"symbol": "diamond-open", "size": 9, "color": "#f59e0b"},
                    name="Незакрытый период",
                    hovertemplate="Незакрытая свеча<br>%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>",
                )
            )
    if node:
        active_ids = {item.get("node_id") for item in _active_path(snapshot, scenario_id)}
        visible_nodes = _visible_wave_nodes(
            snapshot,
            scenario_id,
            visible_degrees,
            label_mode,
            chart_timeframe,
        )
        degree_style = {
            "Major": {"color": "#38bdf8", "width": 2.8, "marker": 8},
            "Intermediate": {"color": "#a78bfa", "width": 1.9, "marker": 6},
            "Minor": {"color": "#94a3b8", "width": 1.15, "marker": 4},
        }
        for visible_node in visible_nodes[:120]:
            labels = visible_node.get("labels") or []
            if not labels:
                continue
            ratio_text = ", ".join(
                f"{ratio.get('ratio_id')}={float(ratio.get('value')):.3f}"
                for ratio in visible_node.get("ratios", [])
                if ratio.get("value") is not None
            ) or "n/a"
            preliminary_reason = ", ".join(visible_node.get("unknown_requirements", [])[:3]) or "нет"
            x = [pd.Timestamp(label["pivot_time"]) for label in labels]
            y = [float(label["price"]) for label in labels]
            is_forming = visible_node.get("endpoint_status") == "FORMING"
            degree = str(visible_node.get("degree") or visible_node.get("relative_degree") or "Minor")
            style = degree_style.get(degree, degree_style["Minor"])
            is_active = visible_node.get("node_id") in active_ids
            is_historical = visible_node.get("map_status") in {"HISTORICAL", "COMPLETED", "SUPERSEDED"}
            opacity = 1.0 if is_active else (0.52 if is_historical else 0.72)
            text = [_display_label(label.get("label"), degree) for label in labels]
            if label_mode == "Minimal" and not is_active:
                text = [None] * len(labels)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers+text",
                    text=text,
                    textposition="top center",
                    name=f"{degree} · {visible_node.get('pattern_type')}",
                    opacity=opacity,
                    line={"color": style["color"], "width": style["width"] + (0.6 if is_active else 0), "dash": "dash" if is_forming else "solid"},
                    marker={"size": style["marker"] + (1 if is_active else 0)},
                    customdata=[
                        [
                            visible_node.get("pattern_type"),
                            degree,
                            label.get("status"),
                            label.get("confirmed_at"),
                            visible_node.get("verified_depth"),
                            ratio_text,
                            preliminary_reason,
                        ]
                        for label in labels
                    ],
                    hovertemplate=(
                        "Дата: %{x|%Y-%m-%d}<br>Цена: %{y:,.2f}<br>"
                        "Модель: %{customdata[0]}<br>Степень: %{customdata[1]}<br>"
                        "Статус: %{customdata[2]}<br>Подтверждено: %{customdata[3]}<br>"
                        "verified_depth: %{customdata[4]}<br>Соотношения: %{customdata[5]}<br>"
                        "Предварительно, пока: %{customdata[6]}<extra></extra>"
                    ),
                )
            )
        if show_targets:
            colors = ["rgba(250,204,21,0.14)", "rgba(56,189,248,0.12)", "rgba(167,139,250,0.12)"]
            visible_targets = [target for target in node.get("targets", []) if target.get("status") != "EXCLUDED"]
            for idx, target in enumerate(visible_targets[:6]):
                fig.add_hrect(
                    y0=target.get("price_low"),
                    y1=target.get("price_high"),
                    fillcolor=colors[idx % len(colors)],
                    line_width=0,
                    annotation_text=f"{target.get('role')} {target.get('coefficient')}",
                    annotation_position="top right",
                )
        invalidation = node.get("invalidation") or {}
        if invalidation.get("level") is not None:
            fig.add_hline(
                y=float(invalidation["level"]),
                line={"color": "#fb7185", "dash": "dot", "width": 1.4},
                annotation_text="Отмена",
                annotation_position="bottom right",
            )
        if show_channels:
            for channel in node.get("channels", []):
                anchors = channel.get("anchors") or []
                if len(anchors) == 2:
                    for line_index, (x_values, y_values) in enumerate(_channel_lines(channel, d["timestamp"].max())):
                        fig.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=y_values,
                                mode="lines",
                                line={"color": "#94a3b8", "dash": "dot", "width": 1},
                                name=channel.get("kind") if line_index == 0 else f"{channel.get('kind')} parallel",
                                hoverinfo="skip",
                                showlegend=line_index == 0,
                            )
                        )

    fig.update_layout(
        height=480,
        margin={"l": 12, "r": 74, "t": 18, "b": 18},
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font={"color": "#e5e7eb", "size": 11},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.03, "x": 0},
        xaxis={
            "rangeslider": {"visible": False},
            "showgrid": False,
            "range": [d["timestamp"].min(), d["timestamp"].max()],
        },
        yaxis={"side": "right", "type": "log" if price_scale == "Log" else "linear", "gridcolor": "rgba(148,163,184,0.14)"},
    )
    return fig


def _render_details(snapshot: dict[str, Any], scenario_id: str | None, node: dict[str, Any] | None) -> None:
    metadata = {
        "source": f"{snapshot.get('provider_label')} / {snapshot.get('provider_symbol')}",
        "instrument_type": snapshot.get("instrument_type"),
        "provenance": snapshot.get("provenance_note"),
        "snapshot_id": snapshot.get("snapshot_id"),
        "scenario_id": scenario_id,
        "base_timeframe": snapshot.get("base_timeframe"),
        "data_version": snapshot.get("data_version"),
        "engine_version": snapshot.get("engine_version"),
        "availability_mode": snapshot.get("availability_mode"),
        "quality_flags": ", ".join(snapshot.get("quality_flags", [])),
        "display_warnings": " | ".join(snapshot.get("display_warnings", [])),
    }
    st.dataframe(pd.DataFrame([metadata]), use_container_width=True, hide_index=True)
    if not node:
        st.write(snapshot.get("unresolved_reasons", []))
        return
    st.write(
        {
            "model": node.get("pattern_type"),
            "role": node.get("role_in_parent"),
            "geometry_status": node.get("geometry_status"),
            "context_status": node.get("context_status"),
            "subdivision_status": node.get("subdivision_status"),
            "verified_depth": node.get("verified_depth"),
            "verification_coverage": node.get("verification_coverage"),
            "unknown_requirements": node.get("unknown_requirements"),
            "invalidation": node.get("invalidation"),
        }
    )
    checks = pd.DataFrame(node.get("rule_checks", []))
    if not checks.empty:
        st.dataframe(checks, use_container_width=True, hide_index=True)


def _walk_tree(root: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    stack = [root]
    while stack:
        current = stack.pop(0)
        node_id = str(current.get("node_id"))
        if node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(current)
        stack.extend(nodes[child] for child in current.get("children", []) if child in nodes)
    return ordered


def _channel_lines(channel: dict[str, Any], visible_end: pd.Timestamp) -> list[tuple[list[pd.Timestamp], list[float]]]:
    anchors = channel.get("anchors") or []
    if len(anchors) != 2:
        return []
    first_time = pd.Timestamp(anchors[0]["pivot_time"])
    second_time = pd.Timestamp(anchors[1]["pivot_time"])
    first_price = float(anchors[0]["price"])
    second_price = float(anchors[1]["price"])
    seconds = (second_time - first_time).total_seconds()
    if seconds == 0:
        return []
    end_time = max(pd.Timestamp(visible_end), second_time)
    slope = (second_price - first_price) / seconds
    end_price = first_price + slope * (end_time - first_time).total_seconds()
    lines = [([first_time, end_time], [first_price, end_price])]
    parallel = channel.get("parallel_through")
    if parallel:
        parallel_time = pd.Timestamp(parallel["pivot_time"])
        parallel_price = float(parallel["price"])
        parallel_end = parallel_price + slope * (end_time - parallel_time).total_seconds()
        lines.append(([parallel_time, end_time], [parallel_price, parallel_end]))
    return lines


def _selected_node(snapshot: dict[str, Any], scenario_id: str | None) -> dict[str, Any] | None:
    scenarios = {row.get("scenario_id"): row for row in snapshot.get("root_scenarios", [])}
    scenario = scenarios.get(scenario_id or snapshot.get("main_root_scenario_id"))
    if not scenario:
        return None
    active_path = scenario.get("active_path") or []
    node_id = active_path[-1] if active_path else None
    return next((row for row in snapshot.get("wave_nodes", []) if row.get("node_id") == node_id), None)


def _active_path(snapshot: dict[str, Any], scenario_id: str | None) -> list[dict[str, Any]]:
    scenarios = {row.get("scenario_id"): row for row in snapshot.get("root_scenarios", [])}
    scenario = scenarios.get(scenario_id or snapshot.get("main_root_scenario_id"))
    nodes = {row.get("node_id"): row for row in snapshot.get("wave_nodes", [])}
    return [nodes[node_id] for node_id in (scenario or {}).get("active_path", []) if node_id in nodes]


def _display_label(label: Any, degree: str) -> str:
    value = str(label or "")
    return f"({value})" if degree == "Major" else value


def _visible_wave_nodes(
    snapshot: dict[str, Any],
    scenario_id: str | None,
    visible_degrees: dict[str, bool],
    label_mode: str,
    chart_timeframe: str,
) -> list[dict[str, Any]]:
    scenarios = {row.get("scenario_id"): row for row in snapshot.get("root_scenarios", [])}
    scenario = scenarios.get(scenario_id or snapshot.get("main_root_scenario_id"))
    nodes = {row.get("node_id"): row for row in snapshot.get("wave_nodes", [])}
    active_ids = {node.get("node_id") for node in _active_path(snapshot, scenario_id)}
    selected = [nodes[node_id] for node_id in (scenario or {}).get("node_ids", []) if node_id in nodes]
    selected = [node for node in selected if visible_degrees.get(str(node.get("degree")), False)]
    if label_mode == "Minimal":
        return [node for node in selected if node.get("node_id") in active_ids]
    if label_mode == "Auto" and chart_timeframe in {"1W", "1M"}:
        return [
            node
            for node in selected
            if node.get("degree") != "Minor" or node.get("node_id") in active_ids
        ]
    return selected


def _current_view_settings(
    snapshots: dict[str, dict[str, Any]],
    chart_timeframe: str,
    analysis_window: str,
    price_scale: str,
) -> dict[str, dict[str, Any]]:
    settings: dict[str, dict[str, Any]] = {}
    for asset_id, snapshot in snapshots.items():
        scenario_label = st.session_state.get(f"elliott_scenario_{asset_id}")
        scenario_map = {row.get("selection_label"): row.get("scenario_id") for row in snapshot.get("root_scenarios", [])}
        scenario_id = scenario_map.get(scenario_label) or snapshot.get("main_root_scenario_id")
        node = _selected_node(snapshot, scenario_id)
        settings[asset_id] = {
            "snapshot_id": snapshot.get("snapshot_id"),
            "scenario_id": scenario_id,
            "focus_node_id": node.get("node_id") if node else None,
            "chart_timeframe": chart_timeframe,
            "analysis_window": analysis_window,
            "visible_degrees": {
                "Major": st.session_state.get(f"elliott_major_{asset_id}", True),
                "Intermediate": st.session_state.get(f"elliott_intermediate_{asset_id}", True),
                "Minor": st.session_state.get(f"elliott_minor_{asset_id}", True),
            },
            "label_mode": st.session_state.get(f"elliott_labels_{asset_id}", "Auto"),
            "price_scale": price_scale,
            "show_targets": st.session_state.get(f"elliott_targets_{asset_id}", True),
            "show_channels": st.session_state.get(f"elliott_channels_{asset_id}", False),
        }
    return settings
