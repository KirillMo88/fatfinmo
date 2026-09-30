"""Fixed-window chart styled after the user's light-background reference."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import CONFIG
from .volume_profile import build_volume_profile


FAMILY_COLORS = {
    "SWING_STRUCTURE": "#ffa000", "VOLUME_ACCEPTANCE": "#ee4d9b",
    "FIBONACCI": "#9655ff", "MOVING_AVERAGE": "#008f88",
}
CHART_CONFIG = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "responsive": True}


def build_reference_chart(
    bars: pd.DataFrame, zones: list[dict[str, Any]], *,
    primary: dict[str, Any] | None, alternative: dict[str, Any] | None,
    timeframe: str, key_point_filter: str = "OFF",
    pivots: list[dict[str, Any]] | None = None,
    profile: dict[str, Any] | None = None,
    fibonacci: dict[str, Any] | None = None,
    enabled_sources: set[str] | None = None, ticker: str = "",
) -> go.Figure:
    frame = bars.tail(int(CONFIG["chart"]["bars"])).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    frame = frame.dropna(subset=["timestamp"])
    enabled = set(FAMILY_COLORS) if enabled_sources is None else set(enabled_sources)
    fig = go.Figure()
    if frame.empty:
        fig.update_layout(template="plotly_white", height=480, paper_bgcolor="white", plot_bgcolor="white")
        return fig
    start, end = frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1]
    profile = profile if profile and profile.get("bins") else build_volume_profile(frame, timeframe=timeframe)
    has_profile = "VOLUME_ACCEPTANCE" in enabled and profile.get("status") == "AVAILABLE" and bool(profile.get("bins"))
    price_end = 0.875 if has_profile else 1.0
    current = float(frame.iloc[-1]["close"])

    def band(low: float, high: float, color: str, *, name: str = "", legend: bool = False, opacity: float = 0.2, across_profile: bool = False) -> None:
        for index, (left, right) in enumerate([(0.0, price_end)] + ([(0.89, 1.0)] if across_profile and has_profile else [])):
            fig.add_shape(type="rect", xref="paper", yref="y", x0=left, x1=right,
                          y0=low, y1=high, fillcolor=color, opacity=opacity,
                          line=dict(color=color, width=0.6), layer="below", name=name,
                          showlegend=legend and index == 0, legendrank=20 if "Strategic" in name else 40)

    def level(price: float, name: str, color: str, dash: str, width: float, rank: int, legend: bool = True) -> None:
        fig.add_shape(type="line", xref="paper", yref="y", x0=0, x1=1, y0=price, y1=price,
                      line=dict(color=color, width=width, dash=dash), layer="above",
                      name=name, showlegend=legend, legendrank=rank)

    if "SWING_STRUCTURE" in enabled:
        cfg = CONFIG["clustering"]["weekly" if timeframe.upper() == "WEEKLY" else "daily"]
        atr = pd.to_numeric(pd.Series([frame.iloc[-1].get("atr14")]), errors="coerce").iloc[0]
        if not np.isfinite(atr) or atr <= 0:
            atr = current * (0.04 if timeframe.upper() == "WEEKLY" else 0.02)
        tolerance = min(max(current * cfg["base_price_fraction"], atr * cfg["atr_multiplier"]), current * cfg["radius_cap_fraction"])
        confirmed = []
        for pivot in pivots or []:
            date = pd.to_datetime(pivot.get("pivot_time"), errors="coerce", utc=True)
            if pivot.get("status") != "CONFIRMED" or pd.isna(date):
                continue
            date = date.tz_convert(None)
            if start <= date <= end:
                price = float(pivot["price"])
                band(price - tolerance, price + tolerance, "#ffb938", opacity=0.18)
                confirmed.append((date, price, pivot.get("kind")))
    else:
        confirmed = []

    fib = fibonacci or {}
    if "FIBONACCI" in enabled:
        for key, color, fill, rank in (("strategic", "#9655ff", "#ad8cff", 10), ("tactical", "#008c82", "#86d992", 30)):
            if key == "tactical" and fib.get("tactical_suppressed"):
                continue
            seen = set()
            for item in (fib.get(key) or {}).get("levels") or []:
                kind = item.get("type")
                if kind in seen:
                    continue
                seen.add(kind)
                if kind == "0.382":
                    level(float(item["price"]), f"{key.title()} Fibonacci 0.382", color, "dashdot", 1.8, rank)
                elif kind == "0.500_0.618":
                    band(float(item["low"]), float(item["high"]), fill,
                         name=f"{key.title()} Fibonacci Zone 0.500–0.618", legend=True, opacity=0.25, across_profile=True)

    for zone in zones:
        score_value = zone.get("key_point_score")
        if score_value is None or not np.isfinite(float(score_value or 0)) or float(score_value or 0) <= 0:
            score_value = zone.get("total_score", zone.get("quality_score", 0))
        score = float(score_value or 0)
        zone_class = str(zone.get("key_point_class") or zone.get("confluence_class") or "").upper()
        selected = key_point_filter.upper().replace(" ", "")
        if selected == "OFF" or zone.get("hidden_by_60pct_filter"):
            continue
        if selected == "HIGH":
            if zone_class:
                if zone_class not in {"HIGH", "VERY_HIGH"}:
                    continue
            elif score <= 150:
                continue
        elif selected == "HIGH+MID":
            if zone_class:
                if zone_class not in {"MID", "MEDIUM", "HIGH", "VERY_HIGH"}:
                    continue
            elif score < 75:
                continue
        families = set(zone.get("source_families") or [])
        if not families.intersection(enabled):
            continue
        color = FAMILY_COLORS.get(next(iter(families)), "#778899") if len(families) == 1 else "#718096"
        band(float(zone["low"]), float(zone["high"]), color, opacity=0.2 if score > 150 else 0.12)

    if has_profile:
        bins = profile["bins"]
        fig.add_trace(go.Bar(
            x=[item["volume"] for item in bins], y=[item["center"] for item in bins],
            width=[(item["high"] - item["low"]) * 0.96 for item in bins],
            orientation="h", xaxis="x2", yaxis="y", base=0,
            marker=dict(color="#ee4d9b", line=dict(color="white", width=0.5)),
            opacity=0.92, showlegend=False, name="Volume Profile",
            customdata=[[item["low"], item["high"]] for item in bins],
            hovertemplate="Price %{customdata[0]:,.2f}–%{customdata[1]:,.2f}<br>Volume %{x:,.0f}<extra>Volume Profile</extra>",
        ))
        level(float(profile["poc"]), f"POC {profile['poc']:,.2f}", "#111111", "solid", 2.0, 50)
        for index, peak in enumerate(profile.get("local_peaks") or []):
            level(float(peak["center"]), "Local volume peak", "#555555", "dash", 1.1, 60, legend=index == 0)

    fig.add_trace(go.Candlestick(
        x=frame["timestamp"], open=frame["open"], high=frame["high"], low=frame["low"], close=frame["close"],
        increasing=dict(line=dict(color="#00c853", width=0.8), fillcolor="#00c853"),
        decreasing=dict(line=dict(color="#ff3838", width=0.8), fillcolor="#ff3838"),
        name="Price", showlegend=False, whiskerwidth=0,
    ))
    if "MOVING_AVERAGE" in enabled:
        for index, (column, color) in enumerate((("sma50", "#0066ff"), ("sma100", "#00bfae"), ("sma200", "#ff2020"))):
            if column in frame and frame[column].notna().any():
                fig.add_trace(go.Scatter(x=frame["timestamp"], y=frame[column], mode="lines",
                                        line=dict(color=color, width=1.7 if column == "sma50" else 1.4),
                                        name=f"SMA {column[3:]}", legendrank=70 + index,
                                        hovertemplate="%{y:,.2f}<extra>%{fullData.name}</extra>"))
    if confirmed:
        fig.add_trace(go.Scatter(x=[p[0] for p in confirmed], y=[p[1] for p in confirmed],
                                mode="markers", marker=dict(color="#ffa000", size=6.5),
                                name="Confirmed swing pivots", legend="legend2", legendrank=1,
                                hovertemplate="%{x|%d %b %Y}<br>%{y:,.2f}<extra>Confirmed swing pivot</extra>"))
    if "FIBONACCI" in enabled:
        for key, label, color, symbol in (
            ("ath", "Latest ATH", "#f000ef", "triangle-up"),
            ("strategic_anchor", "Strategic Fibonacci anchor", "#963cff", "triangle-down"),
            ("tactical_anchor", "Tactical Fibonacci anchor", "#00857d", "triangle-down"),
        ):
            if key == "tactical_anchor" and fib.get("tactical_suppressed"):
                continue
            point = fib.get(key) or {}
            date = pd.to_datetime(point.get("date") or point.get("pivot_time"), errors="coerce", utc=True)
            if pd.notna(date) and point.get("price") is not None and start <= date.tz_convert(None) <= end:
                fig.add_trace(go.Scatter(x=[date.tz_convert(None)], y=[point["price"]], mode="markers",
                                        marker=dict(color=color, symbol=symbol, size=9), name=label, legend="legend2",
                                        hovertemplate="%{x|%d %b %Y}<br>%{y:,.2f}<extra>%{fullData.name}</extra>"))
    for candidate, name, color in ((primary, "Elliott Primary", "#243b68"), (alternative, "Elliott Alternative", "#d97706")):
        waves = (candidate or {}).get("waves") or []
        points = [(pd.to_datetime(p.get("pivot_time"), errors="coerce", utc=True), p) for p in waves]
        points = [(date.tz_convert(None), p) for date, p in points if pd.notna(date) and start <= date.tz_convert(None) <= end and p.get("price") is not None]
        if points:
            fig.add_trace(go.Scatter(x=[date for date, _ in points], y=[p["price"] for _, p in points],
                                    text=[p.get("wave_label") for _, p in points], mode="lines+text",
                                    textposition="top center", line=dict(color=color, width=1.4), name=name, legend="legend2"))

    y_values = [frame["low"], frame["high"]]
    if "MOVING_AVERAGE" in enabled:
        y_values.extend(frame[col] for col in ("sma50", "sma100", "sma200") if col in frame)
    values = pd.to_numeric(pd.concat(y_values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    ymin, ymax = float(values.min()), float(values.max())
    padding = max((ymax - ymin) * 0.04, ymax * 0.002)
    axis = dict(showline=True, mirror=True, linecolor="#454545", linewidth=1,
                gridcolor="#e9e9e9", gridwidth=1, zeroline=False, ticks="outside",
                tickcolor="#555555", ticklen=3, tickfont=dict(size=12, color="#222222"), fixedrange=True)
    legend = dict(bgcolor="rgba(255,255,255,0.84)", bordercolor="#d6d6d6", borderwidth=0.7,
                  font=dict(size=9, color="#222222"), xanchor="left", yanchor="top", y=0.985,
                  itemclick=False, itemdoubleclick=False, traceorder="normal")
    title = f"{ticker + ' ' if ticker else ''}{timeframe.title()} ({start.year}–{end.year})"
    fig.update_layout(
        template="plotly_white", height=480, autosize=True,
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(family="Arial", color="#222222", size=12),
        margin=dict(l=48, r=12, t=68, b=36), dragmode=False, hovermode="closest",
        title=dict(text=title + "<br><sup>Horizontal Volume Profile, Swing Zones, Pivots and Fibonacci</sup>", x=0.45, xanchor="center", font=dict(size=14)),
        xaxis=dict(**axis, domain=[0, price_end], type="date", range=[start, end],
                   rangeslider=dict(visible=False), tickformat="%Y" if timeframe.upper() == "WEEKLY" else "%b %Y", nticks=7, title=None),
        yaxis=dict(**axis, range=[ymin - padding, ymax + padding], title=dict(text="Price", font=dict(size=12))),
        legend=dict(**legend, x=0.008), legend2=dict(**legend, x=0.40 if has_profile else 0.46),
        bargap=0.04,
    )
    if has_profile:
        fig.update_layout(xaxis2=dict(domain=[0.89, 1], anchor="y", range=[0, max(p["volume"] for p in profile["bins"]) * 1.04],
                                     fixedrange=True, showgrid=False, showticklabels=False, zeroline=False,
                                     showline=True, mirror=True, linecolor="#454545", linewidth=1))
        fig.add_shape(type="rect", xref="paper", yref="paper", x0=0.89, x1=1, y0=0, y1=1,
                      line=dict(color="#454545", width=1), fillcolor="rgba(0,0,0,0)")
        fig.add_annotation(xref="paper", yref="paper", x=0.945, y=1.02, text="Volume<br>Profile",
                           showarrow=False, xanchor="center", yanchor="bottom", font=dict(size=12, color="#222222"))
    return fig
