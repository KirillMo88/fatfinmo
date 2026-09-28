from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd

from .storage import read_base_bars, read_chart_bars
from .summary import build_summary


SHEET_NAMES = [
    "Readme",
    "Metadata",
    "Parameters",
    "Bars",
    "Pivots",
    "Scenarios",
    "WaveNodes",
    "RuleChecks",
    "Ratios",
    "Targets",
    "Channels",
    "Events",
]


def build_json_export(
    manifest: dict[str, Any],
    snapshots: dict[str, dict[str, Any]],
    view_settings: dict[str, dict[str, Any]],
) -> bytes:
    assets: dict[str, Any] = {}
    for asset_id, snapshot in snapshots.items():
        settings = view_settings.get(asset_id, {})
        timeframe = settings.get("chart_timeframe", "1W")
        bars = read_chart_bars(asset_id, snapshot["snapshot_id"], timeframe)
        assets[asset_id] = {
            "snapshot": snapshot,
            "view": settings,
            "summary": build_summary(
                snapshot,
                settings.get("scenario_id"),
                settings.get("focus_node_id"),
                settings.get("visible_degree", "Auto"),
            ),
            "display_bars": _records(bars),
        }
    payload = {"manifest": manifest, "assets": assets}
    return json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default).encode("utf-8")


def build_xlsx_export(
    manifest: dict[str, Any],
    snapshots: dict[str, dict[str, Any]],
    view_settings: dict[str, dict[str, Any]],
) -> bytes:
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in SHEET_NAMES}
    tables["Readme"] = [
        {"Field": "Purpose", "Value": "Elliott Wave Engine V2 snapshot export"},
        {"Field": "Manifest ID", "Value": manifest.get("manifest_id")},
        {"Field": "Published at", "Value": manifest.get("published_at")},
        {"Field": "Important", "Value": "FibFit is diagnostic, not a probability. Targets have no target date."},
        {"Field": "Data lineage", "Value": "Base analytical bars and selected chart bars are distinguished by bar_role and timeframe."},
    ]
    for asset_id, snapshot in snapshots.items():
        settings = view_settings.get(asset_id, {})
        scenario_id = settings.get("scenario_id") or snapshot.get("main_scenario_id")
        tables["Metadata"].append(
            {
                "manifest_id": manifest.get("manifest_id"),
                "asset_id": asset_id,
                "snapshot_id": snapshot.get("snapshot_id"),
                "scenario_id": scenario_id,
                "summary": build_summary(snapshot, scenario_id, settings.get("focus_node_id"), settings.get("visible_degree", "Auto")),
                **{key: snapshot.get(key) for key in [
                    "provider_symbol", "source_id", "instrument_type", "currency", "price_unit",
                    "session_calendar", "source_timezone", "adjustment_mode", "base_timeframe", "as_of",
                    "created_at", "data_cutoff", "availability_mode", "engine_version", "rule_profile",
                    "parameter_hash", "data_version", "history_start", "history_end", "bar_count",
                ]},
                "chart_timeframe": settings.get("chart_timeframe"),
                "date_range": settings.get("date_range"),
                "price_scale": settings.get("price_scale"),
                "visible_degree": settings.get("visible_degree"),
                "quality_flags": snapshot.get("quality_flags"),
                "provenance_note": snapshot.get("provenance_note"),
            }
        )
        for key, value in snapshot.get("parameters", {}).items():
            tables["Parameters"].append({"asset_id": asset_id, "snapshot_id": snapshot.get("snapshot_id"), "parameter": key, "value": value})

        base = read_base_bars(asset_id, snapshot["snapshot_id"]).copy()
        if not base.empty:
            base["bar_role"] = "ANALYTICAL_BASE"
            tables["Bars"].extend(_records(base.assign(asset_id=asset_id, snapshot_id=snapshot["snapshot_id"])))
        timeframe = settings.get("chart_timeframe", "1W")
        chart = read_chart_bars(asset_id, snapshot["snapshot_id"], timeframe).copy()
        if not chart.empty:
            chart["bar_role"] = "SELECTED_DISPLAY"
            tables["Bars"].extend(_records(chart.assign(asset_id=asset_id, snapshot_id=snapshot["snapshot_id"])))

        for stream in snapshot.get("pivot_streams", []):
            for pivot in stream.get("pivots", []):
                tables["Pivots"].append({"asset_id": asset_id, "snapshot_id": snapshot["snapshot_id"], "stream_k": stream.get("k"), "branch": stream.get("branch"), **pivot})
        for scenario in snapshot.get("scenarios", []):
            tables["Scenarios"].append({"asset_id": asset_id, "snapshot_id": snapshot["snapshot_id"], **scenario})
        for node in snapshot.get("nodes", []):
            compact = {key: value for key, value in node.items() if key not in {"rule_checks", "ratios", "targets", "channels"}}
            tables["WaveNodes"].append({"asset_id": asset_id, "snapshot_id": snapshot["snapshot_id"], **compact})
            for check in node.get("rule_checks", []):
                tables["RuleChecks"].append({"asset_id": asset_id, "snapshot_id": snapshot["snapshot_id"], "node_id": node.get("node_id"), **check})
            for ratio in node.get("ratios", []):
                tables["Ratios"].append({"asset_id": asset_id, "snapshot_id": snapshot["snapshot_id"], "node_id": node.get("node_id"), **ratio})
            for target in node.get("targets", []):
                tables["Targets"].append({"asset_id": asset_id, "snapshot_id": snapshot["snapshot_id"], "node_id": node.get("node_id"), **target})
            for channel in node.get("channels", []):
                tables["Channels"].append({"asset_id": asset_id, "snapshot_id": snapshot["snapshot_id"], "node_id": node.get("node_id"), **channel})
        for event in snapshot.get("events", []):
            tables["Events"].append({"asset_id": asset_id, "snapshot_id": snapshot["snapshot_id"], **event})

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm:ss") as writer:
        for sheet_name in SHEET_NAMES:
            frame = pd.DataFrame(tables[sheet_name])
            if frame.empty:
                frame = pd.DataFrame({"Status": ["No rows"]})
            frame = _excel_safe(frame)
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, max(0, len(frame)), max(0, len(frame.columns) - 1))
            for idx, column in enumerate(frame.columns):
                width = min(60, max(12, len(str(column)) + 2))
                worksheet.set_column(idx, idx, width)
    return output.getvalue()


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return frame.replace({np.nan: None}).to_dict(orient="records")


def _excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            out[column] = out[column].map(
                lambda value: json.dumps(value, ensure_ascii=False, default=_json_default)
                if isinstance(value, (dict, list, tuple))
                else value
            )
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return str(value)
