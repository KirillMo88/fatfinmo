from __future__ import annotations

from typing import Any
import html

import numpy as np
import pandas as pd


def build_top_analytics(
    current_risk: Any,
    liquidity_history: pd.DataFrame,
    vix_history: pd.DataFrame,
    move_history: pd.DataFrame,
    funding_history: pd.DataFrame,
) -> list[tuple[str, str, list[str]]]:
    """Build the five metrics displayed above the app's view selector."""
    if isinstance(current_risk, dict):
        risk_status = _format_text(current_risk.get("CurrentMarketRiskState"))
        risk_components = _high_risk_components(current_risk)
    else:
        risk_status = _format_text(current_risk)
        risk_components = []
    liquidity = _dated_numeric_series(liquidity_history, "global_liquidity_score")
    score = _last_value(liquidity)
    roc_1m = _relative_change(liquidity, 28)
    roc_3m = _relative_change(liquidity, 91)
    roc_6m = _relative_change(liquidity, 183)

    vix_value, vix_change_1w, vix_change_4w, vix_percentile = _volatility_summary(_dated_numeric_series(vix_history, "VIX"))
    move_value, move_change_1w, move_change_4w, move_percentile = _volatility_summary(_dated_numeric_series(move_history, "MOVE"))
    funding = _latest_text(funding_history, "FundingState")

    return [
        ("Current Risk", risk_status, ["Market Cycle", f"High+ components: {', '.join(risk_components) or 'None'}"]),
        (
            "Global Liquidity Score",
            _format_number(score),
            [
                f"ROC 1M {_format_percent(roc_1m)}",
                f"ROC 3M {_format_percent(roc_3m)}",
                f"ROC 6M {_format_percent(roc_6m)}",
            ],
        ),
        (
            "VIX",
            vix_value,
            [
                f"1W change {_format_signed(vix_change_1w)} pts",
                f"4W change {_format_signed(vix_change_4w)} pts",
                f"5Y percentile {_format_percentile(vix_percentile)}",
            ],
        ),
        (
            "MOVE",
            move_value,
            [
                f"1W change {_format_signed(move_change_1w)} pts",
                f"4W change {_format_signed(move_change_4w)} pts",
                f"5Y percentile {_format_percentile(move_percentile)}",
            ],
        ),
        ("Funding Stress", funding, ["Funding Conditions"]),
    ]


def render_top_analytics(slots: list[Any], metrics: list[tuple[str, str, list[str]]]) -> None:
    for slot, (label, value, detail_lines) in zip(slots, metrics):
        with slot.container():
            details_html = "".join(
                f"<div style='font-size:.78rem;color:#cbd5e1'>{html.escape(line)}</div>"
                for line in detail_lines
            )
            st_html = (
                "<div style='padding-top:1.15rem;line-height:1.12;'>"
                f"<div style='font-size:.78rem;color:#94a3b8;font-weight:700'>{html.escape(label)}</div>"
                f"<div style='font-size:1.035rem;color:#f8fafc;font-weight:800'>{html.escape(value)}</div>"
                f"{details_html}"
                "</div>"
            )
            import streamlit as st

            st.markdown(st_html, unsafe_allow_html=True)


def _dated_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame is None or frame.empty or column not in frame.columns:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(frame[column], errors="coerce")
    if "Date" in frame.columns:
        dates = pd.to_datetime(frame["Date"], errors="coerce")
    elif "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
    else:
        dates = pd.to_datetime(frame.index, errors="coerce")
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates))
    series = series[~series.index.isna()].dropna().sort_index()
    return series[~series.index.duplicated(keep="last")]


def _relative_change(series: pd.Series, days: int) -> float | None:
    if len(series) < 2:
        return None
    current = float(series.iloc[-1])
    historical = series.loc[series.index <= series.index[-1] - pd.Timedelta(days=days)]
    if historical.empty:
        return None
    previous = float(historical.iloc[-1])
    if not np.isfinite(previous) or not np.isfinite(current) or previous == 0:
        return None
    return current / previous - 1.0


def _volatility_summary(series: pd.Series) -> tuple[str, float | None, float | None, float | None]:
    if series.empty:
        return "N/A", None, None, None
    current = float(series.iloc[-1])
    change_1w = current - float(series.iloc[-6]) if len(series) >= 6 else None
    change_4w = current - float(series.iloc[-21]) if len(series) >= 21 else None
    cutoff = series.index[-1] - pd.DateOffset(years=5)
    history = series.loc[series.index >= cutoff]
    full_window = not history.empty and history.index[0] <= cutoff + pd.Timedelta(days=7)
    percentile = float(history.le(current).mean() * 100.0) if full_window and len(history) >= 1000 else None
    return f"{current:.2f}", change_1w, change_4w, percentile


def _high_risk_components(current: dict[str, Any]) -> list[str]:
    component_labels = (
        ("CurrentRiskDrawdownRiskState", "Drawdown Risk"),
        ("CurrentRiskPriceCycleVulnerabilityRiskState", "Price-Cycle Vulnerability Risk"),
        ("CurrentRiskBreadthRiskState", "Breadth Risk"),
        ("CurrentRiskRSIDivergenceRiskState", "RSI Divergence Risk"),
        ("CurrentRiskVIXRiskState", "VIX Risk"),
        ("CurrentRiskHighBetaRiskState", "High Beta Risk"),
        ("CurrentRiskHYRiskState", "High Yield Risk"),
    )
    high_states = {"HIGH", "HIGH RISK", "RED FLAG", "EXTREME", "CRITICAL"}
    return [
        label
        for state_key, label in component_labels
        if _format_text(current.get(state_key)).upper() in high_states
    ]


def _last_value(series: pd.Series) -> float | None:
    return float(series.iloc[-1]) if not series.empty and np.isfinite(series.iloc[-1]) else None


def _latest_text(frame: pd.DataFrame, column: str) -> str:
    if frame is None or frame.empty or column not in frame.columns:
        return "N/A"
    values = frame[column].dropna()
    return _format_text(values.iloc[-1]) if not values.empty else "N/A"


def _format_text(value: Any) -> str:
    return str(value).replace("_", " ").strip() if value is not None and not pd.isna(value) else "N/A"


def _format_number(value: float | None) -> str:
    return f"{value:.1f}" if value is not None and np.isfinite(value) else "N/A"


def _format_percent(value: float | None) -> str:
    return f"{value:+.1%}" if value is not None and np.isfinite(value) else "N/A"


def _format_signed(value: float | None) -> str:
    return f"{value:+.2f}" if value is not None and np.isfinite(value) else "N/A"


def _format_percentile(value: float | None) -> str:
    return f"{value:.0f}%" if value is not None and np.isfinite(value) else "N/A"
