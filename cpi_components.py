from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


BLS_CPI_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
CPI_RAW_CACHE_PATH = Path("persistent") / "cpi_components" / "raw_monthly.csv"
CPI_CACHE_META_PATH = Path("persistent") / "cpi_components" / "cache_meta.json"
CPI_CACHE_TTL_SECONDS = 24 * 60 * 60
CPI_HISTORY_YEARS = 5

# The first four rows are the fixed-weight, mutually exclusive headline CPI
# components. CPI and Core CPI are independent benchmark indexes.
CPI_SERIES: dict[str, dict[str, Any]] = {
    "Food": {"series_id": "CUUR0000SAF1", "weight_2026": 13.698, "color": "#f59e0b", "group": "component"},
    "Energy": {"series_id": "CUUR0000SA0E", "weight_2026": 6.383, "color": "#38bdf8", "group": "component"},
    "Shelter": {"series_id": "CUUR0000SAH1", "weight_2026": 35.625, "color": "#a78bfa", "group": "component"},
    "All other items": {"series_id": "CUUR0000SA0L12E", "weight_2026": 44.294, "color": "#22c55e", "group": "component"},
    "CPI": {"series_id": "CUUR0000SA0", "weight_2026": 100.000, "color": "#f8fafc", "group": "benchmark"},
    "Core CPI": {"series_id": "CUUR0000SA0L1E", "weight_2026": 79.919, "color": "#fb7185", "group": "benchmark"},
}
CPI_HORIZONS: dict[str, int] = {"3M": 3, "6M": 6, "9M": 9, "12M": 12, "24M": 24, "36M": 36}
CPI_HORIZON_SUBTITLES = {
    "3M": "3-month annualized inflation rate",
    "6M": "6-month annualized inflation rate",
    "9M": "9-month annualized inflation rate",
    "12M": "12-month inflation rate (YoY)",
    "24M": "24-month annualized inflation rate",
    "36M": "36-month annualized inflation rate",
}
CPI_RAW_COLUMNS = ["Series_ID", "Date", "Value"]


class CPIDataError(RuntimeError):
    """Raised when BLS CPI data cannot be parsed or a horizon is unavailable."""


def get_bls_api_key(api_key: str | None = None) -> str | None:
    key = str(api_key or os.environ.get("BLS_API_KEY") or "").strip()
    if key:
        return key
    try:
        import streamlit as st

        secret = st.secrets.get("BLS_API_KEY", "")
        return str(secret).strip() or None
    except Exception:
        return None


def parse_bls_cpi_payload(payload: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise CPIDataError("BLS returned a malformed response.")
    if payload.get("status") != "REQUEST_SUCCEEDED":
        messages = payload.get("message") or []
        details = "; ".join(str(message) for message in messages) or "unknown API error"
        raise CPIDataError(f"BLS API request failed: {details}")

    results = payload.get("Results")
    if isinstance(results, list):
        series_rows = [row for result in results if isinstance(result, dict) for row in result.get("series", [])]
    elif isinstance(results, dict):
        series_rows = results.get("series", [])
    else:
        raise CPIDataError("BLS response is missing Results.series.")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for series in series_rows:
        if not isinstance(series, dict):
            continue
        series_id = str(series.get("seriesID", "")).strip()
        if series_id not in {meta["series_id"] for meta in CPI_SERIES.values()}:
            continue
        seen.add(series_id)
        for item in series.get("data", []):
            period = str(item.get("period", ""))
            if not re.fullmatch(r"M(?:0[1-9]|1[0-2])", period):
                continue
            try:
                date = pd.Timestamp(year=int(item["year"]), month=int(period[1:]), day=1)
                value = float(item["value"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if np.isfinite(value):
                records.append({"Series_ID": series_id, "Date": date, "Value": value})

    expected = {meta["series_id"] for meta in CPI_SERIES.values()}
    missing = sorted(expected - seen)
    if missing:
        raise CPIDataError("BLS response is missing series: " + ", ".join(missing))
    frame = pd.DataFrame(records, columns=CPI_RAW_COLUMNS)
    if frame.empty:
        raise CPIDataError("BLS response contained no valid monthly CPI index values.")
    return normalize_cpi_raw_data(frame)


def normalize_cpi_raw_data(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=CPI_RAW_COLUMNS)
    out = frame.copy()
    out["Series_ID"] = out["Series_ID"].astype(str).str.strip().str.upper()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    valid_ids = {meta["series_id"] for meta in CPI_SERIES.values()}
    out = out.loc[out["Series_ID"].isin(valid_ids)].dropna(subset=CPI_RAW_COLUMNS)
    return out.sort_values(["Series_ID", "Date"]).drop_duplicates(["Series_ID", "Date"], keep="last")[CPI_RAW_COLUMNS].reset_index(drop=True)


def fetch_bls_cpi_raw_data(
    api_key: str | None = None,
    *,
    now: Any = None,
    post: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    end_year = int(current.year)
    start_year = end_year - CPI_HISTORY_YEARS
    request_body: dict[str, Any] = {
        "seriesid": [meta["series_id"] for meta in CPI_SERIES.values()],
        "startyear": str(start_year),
        "endyear": str(end_year),
        "catalog": False,
        "calculations": False,
        "annualaverage": False,
        "aspects": False,
    }
    key = get_bls_api_key(api_key)
    if key:
        request_body["registrationkey"] = key
    request_post = post or httpx.post
    try:
        response = request_post(
            BLS_CPI_API_URL,
            json=request_body,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        response.raise_for_status()
        return parse_bls_cpi_payload(response.json())
    except httpx.HTTPError as exc:
        raise CPIDataError(f"BLS API HTTP request failed: {exc}") from exc
    except (ValueError, TypeError) as exc:
        raise CPIDataError(f"BLS API returned malformed JSON: {exc}") from exc


def read_cpi_cache(cache_path: Path = CPI_RAW_CACHE_PATH) -> pd.DataFrame:
    try:
        return normalize_cpi_raw_data(pd.read_csv(cache_path, parse_dates=["Date"]))
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return pd.DataFrame(columns=CPI_RAW_COLUMNS)


def _write_cpi_cache(frame: pd.DataFrame, cache_path: Path, meta_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    normalize_cpi_raw_data(frame).to_csv(temp_path, index=False, date_format="%Y-%m-%d")
    temp_path.replace(cache_path)
    meta_path.write_text(
        json.dumps({"fetched_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )


def latest_common_cpi_month(raw: pd.DataFrame) -> pd.Timestamp | None:
    normalized = normalize_cpi_raw_data(raw)
    if normalized.empty:
        return None
    pivot = normalized.pivot(index="Date", columns="Series_ID", values="Value")
    required = [meta["series_id"] for meta in CPI_SERIES.values()]
    if any(series_id not in pivot.columns for series_id in required):
        return None
    common = pivot[required].dropna(how="any")
    return pd.Timestamp(common.index.max()) if not common.empty else None


def _cache_is_stale(raw: pd.DataFrame, now: Any = None) -> bool:
    reference = latest_common_cpi_month(raw)
    if reference is None:
        return True
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current_month = current.tz_localize(None).to_period("M") if current.tzinfo else current.to_period("M")
    reference_month = reference.to_period("M")
    return reference_month < current_month - 2


def load_cpi_raw_history(
    api_key: str | None = None,
    *,
    cache_path: Path = CPI_RAW_CACHE_PATH,
    meta_path: Path = CPI_CACHE_META_PATH,
    now: Any = None,
    force_refresh: bool = False,
    fetcher: Callable[..., pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cached = read_cpi_cache(cache_path)
    if cached.empty:
        cache_age = None
    else:
        try:
            cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            fetched_at = pd.Timestamp(cache_meta["fetched_at_utc"])
            current_time = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
            if current_time.tzinfo is None:
                current_time = current_time.tz_localize("UTC")
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.tz_localize("UTC")
            cache_age = max((current_time - fetched_at).total_seconds(), 0.0)
        except (OSError, ValueError, KeyError, TypeError):
            cache_age = None
    if not cached.empty and not force_refresh and cache_age is not None and cache_age < CPI_CACHE_TTL_SECONDS:
        return cached, {
            "source": "Persistent BLS cache",
            "used_cache_fallback": False,
            "stale": _cache_is_stale(cached, now),
            "error": None,
        }
    try:
        fetch = fetcher or fetch_bls_cpi_raw_data
        fresh = normalize_cpi_raw_data(fetch(api_key=api_key, now=now))
        if fresh.empty:
            raise CPIDataError("BLS returned no usable CPI observations.")
        combined = pd.concat([cached, fresh], ignore_index=True)
        combined = normalize_cpi_raw_data(combined)
        try:
            _write_cpi_cache(combined, cache_path, meta_path)
        except OSError:
            # A read-only persistent volume must not block use of fresh data.
            pass
        return combined, {
            "source": "U.S. Bureau of Labor Statistics Public Data API v2",
            "used_cache_fallback": False,
            "stale": _cache_is_stale(combined, now),
            "error": None,
        }
    except Exception as exc:
        if cached.empty:
            detail = str(exc)
            if get_bls_api_key(api_key) is None:
                detail += " Configure BLS_API_KEY in the application environment or Streamlit secrets if your BLS access requires registration."
            return cached, {
                "source": "Unavailable",
                "used_cache_fallback": False,
                "stale": True,
                "error": detail,
            }
        return cached, {
            "source": "Persistent BLS cache",
            "used_cache_fallback": True,
            "stale": _cache_is_stale(cached, now),
            "error": str(exc),
        }


def calculate_cpi_breakdown(raw: pd.DataFrame, horizon: str | int) -> pd.DataFrame:
    if isinstance(horizon, str):
        if horizon not in CPI_HORIZONS:
            raise ValueError(f"Unsupported CPI horizon: {horizon}")
        months = CPI_HORIZONS[horizon]
        horizon_label = horizon
    else:
        months = int(horizon)
        if months not in CPI_HORIZONS.values():
            raise ValueError(f"Unsupported CPI horizon: {horizon}")
        horizon_label = f"{months}M"

    normalized = normalize_cpi_raw_data(raw)
    reference_month = latest_common_cpi_month(normalized)
    if reference_month is None:
        raise CPIDataError("CPI series do not have a common observation month.")
    comparison_month = reference_month - pd.DateOffset(months=months)
    pivot = normalized.pivot(index="Date", columns="Series_ID", values="Value").sort_index()
    required = [meta["series_id"] for meta in CPI_SERIES.values()]
    if comparison_month not in pivot.index or pivot.loc[comparison_month, required].isna().any():
        raise CPIDataError("Insufficient CPI history for selected period")

    current_values = pivot.loc[reference_month, required]
    comparison_values = pivot.loc[comparison_month, required]
    rows: list[dict[str, Any]] = []
    for name, meta in CPI_SERIES.items():
        series_id = meta["series_id"]
        current_value = float(current_values[series_id])
        comparison_value = float(comparison_values[series_id])
        if current_value <= 0 or comparison_value <= 0:
            raise CPIDataError("Insufficient CPI history for selected period")
        inflation = ((current_value / comparison_value) ** (12.0 / months) - 1.0) * 100.0
        rows.append(
            {
                "Category": name,
                "Series_ID": series_id,
                "Weight_2026": float(meta["weight_2026"]),
                "Group": meta["group"],
                "InflationRate": float(inflation),
                "CurrentIndex": current_value,
                "ComparisonIndex": comparison_value,
                "ReferenceMonth": reference_month,
                "ComparisonMonth": comparison_month,
                "Horizon": horizon_label,
            }
        )
    return pd.DataFrame(rows)


def build_cpi_components_chart(breakdown: pd.DataFrame, horizon: str) -> go.Figure:
    if horizon not in CPI_HORIZONS:
        raise ValueError(f"Unsupported CPI horizon: {horizon}")
    data = breakdown.set_index("Category").reindex(CPI_SERIES).reset_index()
    positions = [0, 1, 2, 3, 5, 6]
    labels = [f"{name}<br>{CPI_SERIES[name]['weight_2026']:.1f}%" for name in CPI_SERIES]
    colors = [CPI_SERIES[name]["color"] for name in CPI_SERIES]
    values = pd.to_numeric(data["InflationRate"], errors="coerce").to_numpy(dtype=float)
    contributions = [
        value * float(weight) / 100 if group == "component" and np.isfinite(value) else np.nan
        for value, weight, group in zip(values, data["Weight_2026"], data["Group"])
    ]
    bar_labels = [
        f"{value:.1f}% ({contribution:+.2f} pp)" if np.isfinite(contribution) else f"{value:.1f}%"
        for value, contribution in zip(values, contributions)
    ]
    customdata = np.column_stack(
        [
            data["Category"].astype(str),
            data["Weight_2026"].map(lambda value: f"{float(value):.1f}%"),
            data["Horizon"].astype(str),
            data["CurrentIndex"].map(lambda value: f"{float(value):.3f}"),
            data["ComparisonIndex"].map(lambda value: f"{float(value):.3f}"),
            pd.to_datetime(data["ReferenceMonth"]).dt.strftime("%b %Y"),
            data["Series_ID"].astype(str),
            [f"{value:+.2f} pp" if np.isfinite(value) else "Not applicable" for value in contributions],
        ]
    )
    ref_month = pd.Timestamp(data["ReferenceMonth"].iloc[0]).strftime("%b %Y")
    subtitle = CPI_HORIZON_SUBTITLES[horizon]
    rate_label = "YoY inflation" if horizon == "12M" else "Annualized inflation"
    value_min = float(np.nanmin(values))
    value_max = float(np.nanmax(values))
    padding = max((value_max - value_min) * 0.14, 0.6)
    y_min = min(value_min - padding, -0.6)
    y_max = max(value_max + padding, 0.6)

    fig = go.Figure(
        go.Bar(
            x=positions,
            y=values,
            width=0.72,
            marker={"color": colors, "line": {"color": "#0f131a", "width": 0.8}},
            text=bar_labels,
            textposition="outside",
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>2026 weight: %{customdata[1]}<br>Period: %{customdata[2]}<br>"
                f"{rate_label}: %{{y:.1f}}%<br>Current index: %{{customdata[3]}}<br>"
                "Index at comparison month: %{customdata[4]}<br>Observation: %{customdata[5]}<br>"
                "Estimated headline CPI contribution: %{customdata[7]}<br>"
                "BLS series: %{customdata[6]}<extra></extra>"
            ),
            name="Annualized inflation",
        )
    )
    fig.add_shape(
        type="line", x0=4, x1=4, y0=0, y1=1, xref="x", yref="paper",
        line={"color": "#64748b", "dash": "dot", "width": 1},
    )
    fig.add_hline(y=0, line={"color": "#cbd5e1", "width": 1.2})
    fig.add_annotation(x=1.5, y=1.04, xref="x", yref="paper", text="COMPONENTS", showarrow=False,
                       font={"size": 10, "color": "#94a3b8"})
    fig.add_annotation(x=5.5, y=1.04, xref="x", yref="paper", text="BENCHMARKS", showarrow=False,
                       font={"size": 10, "color": "#94a3b8"})
    fig.update_layout(
        title={"text": f"CPI Components Breakdown<br><sup>{subtitle} · Latest observation: {ref_month}</sup>", "x": 0.01, "xanchor": "left"},
        template="plotly_dark", height=390, paper_bgcolor="#0f131a", plot_bgcolor="#0f131a",
        font={"color": "#e5e7eb", "size": 11}, margin={"l": 62, "r": 28, "t": 86, "b": 70},
        showlegend=False,
    )
    fig.update_xaxes(
        tickmode="array", tickvals=positions, ticktext=labels, range=[-0.7, 6.7],
        showgrid=False, zeroline=False, color="#cbd5e1", linecolor="#475569",
    )
    fig.update_yaxes(
        title="Annualized inflation rate (%)", range=[y_min, y_max],
        showgrid=True, gridcolor="#263241", zeroline=False, color="#cbd5e1", linecolor="#475569",
    )
    return fig

