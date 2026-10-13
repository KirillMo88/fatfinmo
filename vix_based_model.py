from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from finance_core import download_completed_ohlcv


VIX_MODEL_START = pd.Timestamp("2003-08-01")
VIX_MODEL_CHART_START = pd.Timestamp("2003-08-01")
VXX_TICKER = "VXX"
VXX_INITIAL_CAPITAL = 1_000.0
VXX_TRANSACTION_COST = 0.0005
VXX_TAKE_PROFIT_MULTIPLE = 1.70
VXX_TRAILING_STOP_MULTIPLE = 0.80
VXX_MAX_HOLDING_SESSIONS = 40
VIX_MODEL_COLORS = {
    "spx": "#ff9800",
    "vix": "#2786d8",
    "daily": "#2786d8",
    "weekly": "#2786d8",
    "sma50": "#f97316",
    "sma200": "#22c55e",
    "rapid": "#f97316",
    "red": "rgba(239, 68, 68, 0.15)",
    "positive": "rgba(56, 189, 248, 0.16)",
    "grid": "#344054",
    "text": "#e5e7eb",
    "zero": "#64748b",
}


@dataclass
class VIXModelData:
    frame: pd.DataFrame
    events: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass
class VXXStrategyResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]
    diagnostics: dict[str, Any]


def _empty_model_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Date", "SPX_Close", "VIX_Close", "SMA50D", "SMA200D", "DailySpread",
            "SMA50W", "SMA200W", "WeeklySpread", "WeeklyRegime", "RiskOff",
            "DailyCrossUp", "DailyRapid", "WeeklyRapid", "StrategyAWeight", "StrategyBWeight",
            "ForcedEntryActive", "InitialEntrySPX", "LastSignal", "LastSignalType",
        ]
    )


def _market_weekly_vix(daily: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Build completed weekly VIX bars from daily observations, including short holiday weeks."""
    if daily.empty or not {"Date", "VIX_Close"}.issubset(daily.columns):
        return pd.DataFrame(columns=["Date", "VIX_Close"])

    values = daily[["Date", "VIX_Close"]].dropna().copy()
    values["Date"] = pd.to_datetime(values["Date"], errors="coerce")
    values["VIX_Close"] = pd.to_numeric(values["VIX_Close"], errors="coerce")
    values = values.dropna().sort_values("Date").drop_duplicates("Date", keep="last")
    if values.empty:
        return pd.DataFrame(columns=["Date", "VIX_Close"])

    values = values.set_index("Date")
    # Group by exchange week and take the final observed session. Past shortened
    # weeks (e.g. Thursday close before Thanksgiving) are complete observations.
    weekly = values["VIX_Close"].resample("W-FRI").last().dropna().rename("VIX_Close").to_frame()
    last_date = pd.Timestamp(values.index.max()).normalize()
    last_label = pd.Timestamp(weekly.index.max()).normalize()

    if now is None:
        now = pd.Timestamp.now(tz="UTC").tz_convert("America/New_York")
    elif now.tzinfo is None:
        now = now.tz_localize("UTC").tz_convert("America/New_York")
    else:
        now = now.tz_convert("America/New_York")

    current_week_label = (
        now.normalize() + pd.Timedelta(days=4 - now.weekday())
    ).tz_localize(None).normalize()
    # Do not consume a current-week candle until the Friday close. This keeps
    # partial Mon-Thu weekly bars out while allowing a holiday-shortened week to
    # complete as soon as the calendar week has ended.
    current_week_is_closed = now.weekday() >= 4 and (now.weekday() > 4 or now.hour >= 16)
    if last_label == current_week_label and not current_week_is_closed:
        weekly = weekly.loc[weekly.index < current_week_label]

    out = weekly.reset_index().rename(columns={weekly.index.name or "index": "WeekLabel"})
    # The as-of date is the actual final trading observation, rather than the
    # nominal Friday label, so short weeks are available after their last close.
    source_dates = values.index.to_series().resample("W-FRI").last().dropna()
    out["Date"] = source_dates.reindex(pd.DatetimeIndex(out["WeekLabel"])).to_numpy()
    return out[["Date", "VIX_Close"]].dropna().sort_values("Date").reset_index(drop=True)


def _deduplicated_signal(mask: pd.Series, minimum_gap: int) -> pd.Series:
    result = pd.Series(False, index=mask.index)
    last_position: int | None = None
    for position, active in enumerate(mask.fillna(False).to_numpy(dtype=bool)):
        if active and (last_position is None or position - last_position >= minimum_gap):
            result.iloc[position] = True
            last_position = position
    return result


def _deduplicated_signal_by_calendar_days(
    mask: pd.Series,
    dates: pd.Series,
    minimum_days: int,
) -> pd.Series:
    """Keep active signals separated by a minimum number of calendar days."""
    result = pd.Series(False, index=mask.index)
    normalized_dates = pd.to_datetime(dates, errors="coerce")
    last_date: pd.Timestamp | None = None
    for position, (date, active) in enumerate(
        zip(normalized_dates, mask.fillna(False).to_numpy(dtype=bool))
    ):
        if not active or pd.isna(date):
            continue
        current_date = pd.Timestamp(date).normalize()
        if last_date is None or (current_date - last_date).days >= minimum_days:
            result.iloc[position] = True
            last_date = current_date
    return result


def _compute_state_machine(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy().reset_index(drop=True)
    out["WeeklyRegime"] = ""
    valid_weekly = pd.to_numeric(out["WeeklySpread"], errors="coerce")
    red = False
    regimes: list[str] = []
    for spread in valid_weekly:
        if not np.isfinite(spread):
            regimes.append("")
            continue
        if not red and spread >= 0.0:
            red = True
        elif red and spread <= -10.0:
            red = False
        regimes.append("RED" if red else "WHITE")
    out["WeeklyRegime"] = regimes
    out["RiskOff"] = (out["WeeklyRegime"] == "RED") & (out["DailySpread"] > 0)
    out["StrategyAWeight"] = np.where(out["RiskOff"], 0.0, 1.0)

    n = len(out)
    weights = np.ones(n, dtype=float)
    active_history = np.zeros(n, dtype=bool)
    initial_history = np.full(n, np.nan, dtype=float)
    last_signal_types = np.full(n, "", dtype=object)
    events: list[dict[str, Any]] = []
    active = False
    initial_spx = np.nan
    episode_start: pd.Timestamp | None = None
    reached = [False, False]
    for i, row in out.iterrows():
        date = pd.Timestamp(row["Date"])
        spread = pd.to_numeric(pd.Series([row["DailySpread"]]), errors="coerce").iloc[0]
        close = pd.to_numeric(pd.Series([row["SPX_Close"]]), errors="coerce").iloc[0]
        is_red = row["WeeklyRegime"] == "RED"
        baseline = float(row["StrategyAWeight"])

        if active and (spread <= 0.0 or not is_red or baseline >= 1.0):
            old_weight = weights[i - 1] if i else 1.0
            if old_weight < 1.0 - 1e-10:
                reason = "Daily Spread Normalization" if spread <= 0.0 else "Weekly Regime Normalization"
                event = _trade_event(date, "Remaining Cash Purchase", row, old_weight, 1.0, initial_spx, episode_start)
                event["Trigger"] = reason
                events.append(event)
                last_signal_types[i] = reason
            active = False
            initial_spx = np.nan
            episode_start = None
            reached = [False, False]
            weights[i] = baseline
            active_history[i] = False
            continue

        if not active and is_red and spread > 85.0 and baseline < 1.0 - 1e-10:
            active = True
            initial_spx = float(close)
            episode_start = date
            reached = [False, False]
            weights[i] = 0.70
            events.append(_trade_event(date, "Forced Initial Entry 70%", row, baseline, 0.70, initial_spx, episode_start))
            last_signal_types[i] = "Forced Initial Entry 70%"
            active_history[i] = True
            initial_history[i] = initial_spx
            continue

        if active:
            old_weight = weights[i - 1] if i else 0.70
            target = max(0.70, old_weight)
            for j, (drawdown, tranche, label) in enumerate(
                ((0.10, 0.15, "Additional Entry −10%"), (0.20, 0.15, "Additional Entry −20%"))
            ):
                if not reached[j] and close <= initial_spx * (1.0 - drawdown) + 1e-10:
                    reached[j] = True
                    target = min(1.0, target + tranche)
                    events.append(_trade_event(date, label, row, old_weight if target < 1.0 else max(0.70, target - tranche), target, initial_spx, episode_start))
                    last_signal_types[i] = label
            weights[i] = min(1.0, target)
            active_history[i] = True
            initial_history[i] = initial_spx
        else:
            weights[i] = baseline
            active_history[i] = False

    out["StrategyBWeight"] = weights
    out["ForcedEntryActive"] = active_history
    out["InitialEntrySPX"] = initial_history
    out["LastSignalType"] = last_signal_types
    out["LastSignal"] = np.where(out["LastSignalType"].astype(str).ne(""), out["LastSignalType"], "")

    # Include baseline defensive transitions in the audit trail for both modes.
    for i in range(1, len(out)):
        row = out.iloc[i]
        date = pd.Timestamp(row["Date"])
        current_a = float(row["StrategyAWeight"])
        previous_a = float(out.iloc[i - 1]["StrategyAWeight"])
        if abs(current_a - previous_a) > 1e-10:
            signal = "Structural Risk-Off" if current_a < previous_a else (
                "Daily Spread Normalization" if float(row["DailySpread"]) <= 0.0 else "Weekly Regime Normalization"
            )
            events.append(_trade_event(date, signal, row, previous_a, current_a, np.nan, None, strategy="Strategy A — Defensive"))

        # Strategy B follows the same continuous defensive baseline whenever a
        # staged forced-entry episode is not active.
        if not bool(row["ForcedEntryActive"]):
            current_b = float(row["StrategyBWeight"])
            previous_b = float(out.iloc[i - 1]["StrategyBWeight"])
            if abs(current_b - previous_b) > 1e-10 and not bool(out.iloc[i - 1]["ForcedEntryActive"]):
                signal = "Structural Risk-Off" if current_b < previous_b else (
                    "Daily Spread Normalization" if float(row["DailySpread"]) <= 0.0 else "Weekly Regime Normalization"
                )
                events.append(_trade_event(date, signal, row, previous_b, current_b, np.nan, None))

    if events:
        events.sort(key=lambda event: (pd.Timestamp(event["Date"]), str(event["Strategy"]), str(event["Signal type"])))
    return out, pd.DataFrame(events)


def _trade_event(
    date: pd.Timestamp,
    signal_type: str,
    row: pd.Series,
    previous_weight: float,
    new_weight: float,
    initial_spx: float,
    episode_start: pd.Timestamp | None,
    strategy: str = "Strategy B — Forced Entry",
) -> dict[str, Any]:
    return {
        "Date": date,
        "Strategy": strategy,
        "Signal type": signal_type,
        "Weekly Regime": row.get("WeeklyRegime", ""),
        "Daily Spread (%)": row.get("DailySpread", np.nan),
        "SPX Price": row.get("SPX_Close", np.nan),
        "Previous SPX Weight": previous_weight,
        "New SPX Weight": new_weight,
        "Transaction Cost (%)": 0.0005 * abs(new_weight - previous_weight) * 100.0,
        "Execution status": "Reference close-to-close model",
        "Trigger": signal_type,
        "Initial Entry SPX": initial_spx,
        "Episode Start": episode_start,
    }


def build_vix_model_data(daily_source: pd.DataFrame) -> VIXModelData:
    if daily_source is None or daily_source.empty or "SPX_Close" not in daily_source or "VIX" not in daily_source:
        return VIXModelData(_empty_model_frame(), pd.DataFrame(), {"status": "missing", "message": "SPX or VIX daily data is unavailable."})

    source = daily_source.copy()
    if "Date" in source.columns:
        dates = pd.to_datetime(source.pop("Date"), errors="coerce")
    else:
        dates = pd.to_datetime(source.index, errors="coerce")
    base = pd.DataFrame(
        {
            "Date": dates,
            "SPX_Close": pd.to_numeric(source["SPX_Close"], errors="coerce").to_numpy(),
            "VIX_Close": pd.to_numeric(source["VIX"], errors="coerce").to_numpy(),
        }
    ).dropna(subset=["Date", "SPX_Close", "VIX_Close"])
    base = base.loc[base["SPX_Close"] > 0].sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    if base.empty:
        return VIXModelData(_empty_model_frame(), pd.DataFrame(), {"status": "missing", "message": "No aligned SPX/VIX observations."})

    base["SMA50D"] = base["VIX_Close"].rolling(50, min_periods=50).mean()
    base["SMA200D"] = base["VIX_Close"].rolling(200, min_periods=200).mean()
    base["DailySpread"] = 100.0 * (base["SMA50D"] / base["SMA200D"] - 1.0)
    base["DailyCrossUp"] = (base["DailySpread"].shift(1) <= 0.0) & (base["DailySpread"] > 0.0)

    weekly = _market_weekly_vix(base[["Date", "VIX_Close"]])
    weekly["SMA50W"] = weekly["VIX_Close"].rolling(50, min_periods=50).mean()
    weekly["SMA200W"] = weekly["VIX_Close"].rolling(200, min_periods=200).mean()
    weekly["WeeklySpread"] = 100.0 * (weekly["SMA50W"] / weekly["SMA200W"] - 1.0)
    weekly = weekly.dropna(subset=["WeeklySpread"])
    if weekly.empty:
        base["SMA50W"] = np.nan
        base["SMA200W"] = np.nan
        base["WeeklySpread"] = np.nan
    else:
        base = pd.merge_asof(
            base.sort_values("Date"),
            weekly[["Date", "SMA50W", "SMA200W", "WeeklySpread"]].sort_values("Date"),
            on="Date",
            direction="backward",
        )

    daily_delta50 = base["DailySpread"] - base["DailySpread"].shift(50)
    daily_delta15 = base["DailySpread"] - base["DailySpread"].shift(15)
    daily_rapid_condition = base["DailySpread"].between(-10.0, 0.0) & (daily_delta50 >= 3.0) & (daily_delta15 > 0.0)
    base["DailyRapid"] = _deduplicated_signal(daily_rapid_condition, 63)

    weekly["WeeklyRapidCondition"] = (
        weekly["WeeklySpread"].between(-10.0, 0.0)
        & (weekly["WeeklySpread"] - weekly["WeeklySpread"].shift(13) >= 3.0)
        & (weekly["WeeklySpread"] - weekly["WeeklySpread"].shift(4) > 0.0)
    )
    weekly["WeeklyRapid"] = _deduplicated_signal(weekly["WeeklyRapidCondition"], 13)
    weekly_rapid_dates = set(weekly.loc[weekly["WeeklyRapid"], "Date"].tolist())
    base["WeeklyRapid"] = base["Date"].isin(weekly_rapid_dates)

    base, events = _compute_state_machine(base)
    chart_base = base.loc[base["Date"] >= VIX_MODEL_CHART_START].copy().reset_index(drop=True)
    diagnostics = {
        "status": "ok",
        "first_date": base["Date"].min(),
        "last_date": base["Date"].max(),
        "observations": int(len(base)),
        "chart_observations": int(len(chart_base)),
        "missing_spx": int(pd.to_numeric(source["SPX_Close"], errors="coerce").isna().sum()),
        "aligned_observations": int(len(base)),
        "first_daily_spread": base["DailySpread"].first_valid_index(),
        "first_weekly_spread": base["WeeklySpread"].first_valid_index(),
    }
    return VIXModelData(chart_base, events, diagnostics)


def _signal_spans(frame: pd.DataFrame, mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    dates = pd.to_datetime(frame["Date"]).reset_index(drop=True)
    active = mask.fillna(False).to_numpy(dtype=bool)
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start: int | None = None
    for i, value in enumerate(active):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(active) - 1):
            end = i if value and i == len(active) - 1 else i - 1
            spans.append((dates.iloc[start], dates.iloc[end]))
            start = None
    return spans


def build_vix_primary_chart(frame: pd.DataFrame, date_range: str = "MAX") -> go.Figure:
    if frame.empty:
        fig = go.Figure()
        fig.add_annotation(text="VIX / SPX history is unavailable", showarrow=False)
        return fig

    chart = frame.copy().reset_index(drop=True)
    date_values = pd.to_datetime(chart["Date"])
    start = pd.Timestamp("2003-01-01")
    if date_range != "MAX":
        years = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}.get(date_range)
        if years:
            start = max(start, date_values.max() - pd.DateOffset(years=years))
    visible = date_values >= start
    chart = chart.loc[visible].reset_index(drop=True)
    if chart.empty:
        return go.Figure()
    x = pd.to_datetime(chart["Date"])
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[2.6, 1.3, 1.5, 1.5],
        subplot_titles=(f"SPX / VIX / daily spread / weekly spread — 2003–{date_values.max().year}", "", "", ""),
    )
    custom = np.column_stack(
        [chart["WeeklyRegime"].astype(str), chart["StrategyAWeight"] * 100.0, chart["StrategyBWeight"] * 100.0]
    )
    figure.add_trace(
        go.Scatter(
            x=x, y=np.log(chart["SPX_Close"]), name="log SPX", mode="lines",
            line={"color": VIX_MODEL_COLORS["spx"], "width": 2.2},
            legend="legend",
            customdata=custom,
            hovertemplate="%{x|%Y-%m-%d}<br>ln(SPX): %{y:.4f}<br>Weekly regime: %{customdata[0]}<br>Strategy A target: %{customdata[1]:.0f}% SPX<br>Strategy B target: %{customdata[2]:.0f}% SPX<extra></extra>",
        ), row=1, col=1,
    )
    for left, right in _signal_spans(chart, chart["WeeklyRegime"].eq("RED")):
        figure.add_vrect(x0=left, x1=right, fillcolor=VIX_MODEL_COLORS["red"], line_width=0, layer="below", row=1, col=1)
    for date in x[chart["DailyCrossUp"].fillna(False).to_numpy(dtype=bool)]:
        figure.add_vline(x=date, line={"color": "rgba(22, 163, 74, 0.75)", "width": 1.0}, row=1, col=1)

    # VIX and moving averages. Independent >=25 episodes are context only.
    vix25 = chart["VIX_Close"] >= 25.0
    for left, right in _signal_spans(chart, vix25):
        figure.add_vrect(x0=left, x1=right, fillcolor="rgba(14, 165, 233, 0.12)", line_width=0, layer="below", row=2, col=1)
    for col, name, color, width in [
        ("VIX_Close", "VIX daily close", VIX_MODEL_COLORS["vix"], 1.25),
        ("SMA50D", "SMA50D", VIX_MODEL_COLORS["sma50"], 1.6),
        ("SMA200D", "SMA200D", VIX_MODEL_COLORS["sma200"], 1.6),
    ]:
        figure.add_trace(go.Scatter(x=x, y=chart[col], name=name, mode="lines", line={"color": color, "width": width}, legend="legend2", hovertemplate=f"%{{x|%Y-%m-%d}}<br>{name}: %{{y:.2f}}<extra></extra>"), row=2, col=1)
    figure.add_hline(y=25, line={"color": "#38bdf8", "dash": "dash", "width": 1}, row=2, col=1, annotation_text="VIX 25", annotation_position="top right")
    if chart["DailyRapid"].any():
        rapid = chart["DailyRapid"].fillna(False)
        figure.add_trace(go.Scatter(x=x[rapid], y=chart.loc[rapid, "VIX_Close"], name="Daily rapid approach", mode="markers", legend="legend2", marker={"symbol": "diamond", "size": 8, "color": VIX_MODEL_COLORS["rapid"]}, hovertemplate="%{x|%Y-%m-%d}<br>Rapid approach<br>VIX: %{y:.2f}<extra></extra>"), row=2, col=1)
    independent = chart["VIX_Close"].ge(25) & ~chart["VIX_Close"].shift(1).ge(25).fillna(False)
    if independent.any():
        figure.add_trace(go.Scatter(x=x[independent], y=chart.loc[independent, "VIX_Close"], name="VIX ≥25 onset", mode="markers", legend="legend2", marker={"symbol": "circle", "size": 6, "color": "#0369a1"}, hovertemplate="%{x|%Y-%m-%d}<br>Independent VIX ≥25 episode<extra></extra>"), row=2, col=1)

    daily_spread = chart["DailySpread"]
    figure.add_trace(go.Scatter(x=x, y=daily_spread, name="100 × (SMA50D / SMA200D − 1)", mode="lines", legend="legend3", line={"color": VIX_MODEL_COLORS["daily"], "width": 1.5}, hovertemplate="%{x|%Y-%m-%d}<br>Daily spread: %{y:.2f}%<extra></extra>"), row=3, col=1)
    daily_rapid = chart["DailyRapid"].fillna(False)
    figure.add_trace(go.Scatter(x=x[daily_rapid], y=daily_spread[daily_rapid], name="Daily rapid approach", mode="markers", legend="legend3", marker={"symbol": "diamond", "size": 8, "color": VIX_MODEL_COLORS["rapid"]}, hovertemplate="%{x|%Y-%m-%d}<br>Daily rapid approach: %{y:.2f}%<extra></extra>"), row=3, col=1)

    weekly_spread = chart["WeeklySpread"]
    positive_weekly_spread = weekly_spread.where(weekly_spread > 0.0)
    figure.add_trace(go.Scatter(x=x, y=positive_weekly_spread, name="Positive weekly spread", mode="lines", line={"width": 0, "color": "rgba(56, 189, 248, 0)"}, fill="tozeroy", fillcolor=VIX_MODEL_COLORS["positive"], hoverinfo="skip", showlegend=False), row=4, col=1)
    figure.add_trace(go.Scatter(x=x, y=weekly_spread, name="100 × (SMA50W / SMA200W − 1)", mode="lines", legend="legend4", line={"color": VIX_MODEL_COLORS["weekly"], "width": 1.6}, hovertemplate="%{x|%Y-%m-%d}<br>Weekly spread: %{y:.2f}%<br>Regime: %{customdata}<extra></extra>", customdata=chart["WeeklyRegime"]), row=4, col=1)
    weekly_rapid = chart["WeeklyRapid"].fillna(False)
    figure.add_trace(go.Scatter(x=x[weekly_rapid], y=weekly_spread[weekly_rapid], name="Weekly rapid approach", mode="markers", legend="legend4", marker={"symbol": "diamond", "size": 8, "color": VIX_MODEL_COLORS["rapid"]}, hovertemplate="%{x|%Y-%m-%d}<br>Weekly rapid approach: %{y:.2f}%<extra></extra>"), row=4, col=1)

    for row in (3, 4):
        figure.add_hline(y=0, line={"color": VIX_MODEL_COLORS["zero"], "width": 1}, row=row, col=1)
        figure.add_hline(y=-10, line={"color": "#60a5fa", "dash": "dash", "width": 1}, row=row, col=1)
    figure.update_yaxes(title_text="log SPX", row=1, col=1)
    figure.update_yaxes(title_text="VIX index", row=2, col=1)
    figure.update_yaxes(title_text="Daily spread,<br>% of SMA200", row=3, col=1)
    figure.update_yaxes(title_text="Weekly spread,<br>% of SMA200", row=4, col=1)
    figure.update_xaxes(title_text="Date", row=4, col=1)
    figure.update_xaxes(range=[x.min(), x.max()], showgrid=False, row=1, col=1)
    for row in (2, 3, 4):
        figure.update_xaxes(matches="x", showgrid=False, row=row, col=1)
    figure.update_layout(
        height=820,
        margin={"l": 40, "r": 24, "t": 70, "b": 38},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": VIX_MODEL_COLORS["text"], "size": 11},
        hovermode="x unified",
        showlegend=True,
        legend={"x": 0.02, "y": 0.995, "xref": "paper", "yref": "paper", "bgcolor": "rgba(255,255,255,0.75)", "font": {"color": "#111827"}},
        legend2={"x": 0.02, "y": 0.60, "xref": "paper", "yref": "paper", "bgcolor": "rgba(255,255,255,0.75)", "font": {"color": "#111827"}},
        legend3={"x": 0.02, "y": 0.37, "xref": "paper", "yref": "paper", "bgcolor": "rgba(255,255,255,0.75)", "font": {"color": "#111827"}},
        legend4={"x": 0.02, "y": 0.12, "xref": "paper", "yref": "paper", "bgcolor": "rgba(255,255,255,0.75)", "font": {"color": "#111827"}},
        dragmode="pan",
    )
    for axis in ("xaxis", "xaxis2", "xaxis3", "xaxis4", "yaxis", "yaxis2", "yaxis3", "yaxis4"):
        figure.layout[axis].showgrid = axis.startswith("y")
        if axis.startswith("y"):
            figure.layout[axis].gridcolor = VIX_MODEL_COLORS["grid"]
            figure.layout[axis].zerolinecolor = VIX_MODEL_COLORS["grid"]
    return figure


def _evaluate_strategy(frame: pd.DataFrame, weight_column: str) -> tuple[dict[str, float], pd.DataFrame]:
    data = frame.loc[frame["Date"] >= VIX_MODEL_START].copy().reset_index(drop=True)
    data = data.dropna(subset=["SPX_Close", weight_column]).reset_index(drop=True)
    if len(data) < 2:
        return {}, pd.DataFrame()
    prices = pd.to_numeric(data["SPX_Close"], errors="coerce")
    raw_returns = prices.pct_change(fill_method=None).fillna(0.0).to_numpy(dtype=float)
    target = pd.to_numeric(data[weight_column], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    previous = np.r_[1.0, target[:-1]]
    costs = 0.0005 * np.abs(target - previous)
    net_returns = previous * raw_returns - costs
    nav = np.cumprod(1.0 + net_returns)
    peak = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    drawdown = nav / peak - 1.0
    years = len(nav) / 252.0
    cagr = nav[-1] ** (1.0 / years) - 1.0 if nav[-1] > 0 and years > 0 else np.nan
    volatility = float(np.std(net_returns, ddof=1) * np.sqrt(252.0))
    sharpe = float(np.mean(net_returns) / np.std(net_returns, ddof=1) * np.sqrt(252.0)) if np.std(net_returns, ddof=1) > 0 else np.nan
    metrics = {
        "CAGR (%)": float(cagr * 100.0),
        "Annualized Volatility (%)": volatility * 100.0,
        "Maximum Drawdown (%)": float(drawdown.min() * 100.0),
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": float(cagr / abs(drawdown.min())) if drawdown.min() < 0 else np.nan,
        "Average SPX Exposure (%)": float(previous.mean() * 100.0),
        "Time in Cash (%)": float(np.mean(previous < 1.0 - 1e-10) * 100.0),
        "Risk-Off Episodes": float(_count_episodes(data["RiskOff"].fillna(False))),
        "Forced-Entry Episodes": float(_count_episodes(data["ForcedEntryActive"].fillna(False))) if weight_column == "StrategyBWeight" else 0.0,
        "Transaction Costs (%)": float(costs.sum() * 100.0),
        "Final NAV": float(nav[-1]),
    }
    curve = pd.DataFrame({"Date": data["Date"].to_numpy(), "NAV": nav, "Drawdown (%)": drawdown * 100.0, "Net Return": net_returns, "Exposure (%)": previous * 100.0})
    return metrics, curve


def _count_episodes(values: pd.Series) -> int:
    active = values.astype(bool).to_numpy()
    return int(np.sum(active & ~np.r_[False, active[:-1]]))


@st.cache_data(show_spinner=False, ttl=3_600)
def load_vxx_ohlc() -> pd.DataFrame:
    """Load completed, split-adjusted VXX daily bars from Yahoo Finance."""
    return download_completed_ohlcv(VXX_TICKER, period="max")


def _normalize_vxx_ohlc(source: pd.DataFrame) -> pd.DataFrame:
    columns = ["Date", "Open", "High", "Low", "Close"]
    if source is None or source.empty:
        return pd.DataFrame(columns=columns)
    frame = source.copy()
    if "Date" in frame.columns:
        dates = pd.to_datetime(frame["Date"], errors="coerce")
    else:
        dates = pd.to_datetime(frame.index, errors="coerce")
    out = pd.DataFrame({"Date": np.asarray(dates)})
    for column in columns[1:]:
        if column not in frame.columns:
            return pd.DataFrame(columns=columns)
        out[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy()
    out = (
        out.dropna(subset=columns)
        .loc[lambda values: (values[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    return out


def backtest_vxx_strategy(
    vix_frame: pd.DataFrame,
    vxx_ohlc: pd.DataFrame,
    initial_capital: float = VXX_INITIAL_CAPITAL,
) -> VXXStrategyResult:
    """Run the independent, close-signal/next-close VXX strategy state machine."""
    empty_daily = pd.DataFrame(
        columns=[
            "Date", "VXX_Open", "VXX_High", "VXX_Low", "VXX_Close", "NAV",
            "DrawdownPct", "Position", "EligibleEntrySignal", "EntryExecuted",
            "ExitExecuted", "ExitReason",
        ]
    )
    empty_trades = pd.DataFrame(
        columns=[
            "Entry Date", "Entry Signal Date", "Entry Price", "Exit Date", "Exit Price",
            "Exit Reason", "Holding Sessions", "Gross Return (%)", "Net Return (%)",
            "Entry Cost ($)", "Exit Cost ($)", "Capital Before ($)", "Capital After ($)",
        ]
    )
    required_signal_columns = {"Date", "DailyRapid", "VIX_Close", "WeeklyRegime"}
    if vix_frame is None or vix_frame.empty or not required_signal_columns.issubset(vix_frame.columns):
        return VXXStrategyResult(
            empty_daily,
            empty_trades,
            {},
            {"status": "missing", "message": "VIX signal history is unavailable."},
        )
    prices = _normalize_vxx_ohlc(vxx_ohlc)
    if prices.empty:
        return VXXStrategyResult(
            empty_daily,
            empty_trades,
            {},
            {"status": "missing", "message": "Completed VXX OHLC history is unavailable from Yahoo Finance."},
        )

    signal_columns = ["Date", "DailyRapid", "VIX_Close", "WeeklyRegime"]
    if "DailySpread" in vix_frame.columns:
        signal_columns.append("DailySpread")
    signals = vix_frame[signal_columns].copy()
    signals["Date"] = pd.to_datetime(signals["Date"], errors="coerce")
    signals["VIX_Close"] = pd.to_numeric(signals["VIX_Close"], errors="coerce")
    signals["DailyRapid"] = signals["DailyRapid"].fillna(False).astype(bool)
    signals = signals.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    if "DailySpread" in signals.columns:
        daily_spread = pd.to_numeric(signals["DailySpread"], errors="coerce")
        rapid_condition = (
            daily_spread.between(-10.0, 0.0)
            & (daily_spread - daily_spread.shift(50)).ge(3.0)
            & (daily_spread - daily_spread.shift(15)).gt(0.0)
        )
        signals["VXXRapid"] = _deduplicated_signal_by_calendar_days(
            rapid_condition,
            signals["Date"],
            63,
        )
        vxx_rapid_candidates = int(rapid_condition.sum())
        vxx_signal_source = "DailySpread conditions with 63-calendar-day cooldown"
    else:
        # Backward-compatible path for compact synthetic test fixtures. The
        # production VIX model always supplies DailySpread.
        signals["VXXRapid"] = signals["DailyRapid"]
        vxx_rapid_candidates = int(signals["DailyRapid"].sum())
        vxx_signal_source = "precomputed DailyRapid fallback"

    signals["EligibleEntrySignal"] = (
        signals["VXXRapid"]
        & signals["VIX_Close"].lt(13.0)
        & signals["WeeklyRegime"].eq("WHITE")
    )
    eligible_all = signals.loc[signals["EligibleEntrySignal"]].copy()

    # A signal is actionable only when VXX itself already has a completed
    # history.  Otherwise ``searchsorted`` would carry an old VIX signal
    # forward—potentially for years—to the first available Yahoo VXX bar.
    first_price_date = pd.Timestamp(prices["Date"].iloc[0])
    eligible = eligible_all.loc[eligible_all["Date"].ge(first_price_date)].copy()
    discarded_signals_before_vxx_history = int(len(eligible_all) - len(eligible))

    price_dates = pd.DatetimeIndex(prices["Date"])
    executions: dict[int, list[pd.Timestamp]] = {}
    pending_after_history: list[pd.Timestamp] = []
    for signal_date in eligible["Date"]:
        execution_index = int(price_dates.searchsorted(pd.Timestamp(signal_date), side="right"))
        if execution_index >= len(prices):
            pending_after_history.append(pd.Timestamp(signal_date))
        else:
            executions.setdefault(execution_index, []).append(pd.Timestamp(signal_date))

    signal_lookup = signals.set_index("Date").to_dict("index")
    capital = float(initial_capital)
    shares = 0.0
    entry_date: pd.Timestamp | None = None
    entry_signal_date: pd.Timestamp | None = None
    entry_price = np.nan
    capital_before_entry = np.nan
    entry_cost = np.nan
    take_profit = np.nan
    highest_high = np.nan
    holding_sessions = 0
    trade_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    skipped_signals_in_position = 0

    for index, row in prices.iterrows():
        date = pd.Timestamp(row["Date"])
        open_price = float(row["Open"])
        high_price = float(row["High"])
        low_price = float(row["Low"])
        close_price = float(row["Close"])
        was_in_position = shares > 0.0
        entry_executed = False
        exit_executed = False
        exit_reason = ""

        if was_in_position:
            holding_sessions += 1
            active_stop = VXX_TRAILING_STOP_MULTIPLE * float(highest_high)
            exit_price = np.nan
            if open_price < active_stop:
                exit_price = open_price
                exit_reason = "TRAILING STOP — GAP"
            elif open_price > take_profit:
                exit_price = open_price
                exit_reason = "TAKE PROFIT — GAP"
            else:
                stop_touched = low_price <= active_stop
                target_touched = high_price >= take_profit
                if stop_touched:
                    # Stop-first is the conservative rule when both levels are
                    # reachable and intraday event order is unknown.
                    exit_price = active_stop
                    exit_reason = "TRAILING STOP"
                elif target_touched:
                    exit_price = take_profit
                    exit_reason = "TAKE PROFIT"
                elif holding_sessions >= VXX_MAX_HOLDING_SESSIONS:
                    exit_price = close_price
                    exit_reason = "TIME EXIT"

            if np.isfinite(exit_price):
                gross_proceeds = shares * float(exit_price)
                exit_cost = gross_proceeds * VXX_TRANSACTION_COST
                capital = gross_proceeds - exit_cost
                gross_return = float(exit_price) / float(entry_price) - 1.0
                net_return = capital / float(capital_before_entry) - 1.0
                trade_rows.append(
                    {
                        "Entry Date": entry_date,
                        "Entry Signal Date": entry_signal_date,
                        "Entry Price": float(entry_price),
                        "Exit Date": date,
                        "Exit Price": float(exit_price),
                        "Exit Reason": exit_reason,
                        "Holding Sessions": int(holding_sessions),
                        "Gross Return (%)": gross_return * 100.0,
                        "Net Return (%)": net_return * 100.0,
                        "Entry Cost ($)": float(entry_cost),
                        "Exit Cost ($)": float(exit_cost),
                        "Capital Before ($)": float(capital_before_entry),
                        "Capital After ($)": float(capital),
                    }
                )
                shares = 0.0
                entry_date = None
                entry_signal_date = None
                entry_price = np.nan
                capital_before_entry = np.nan
                entry_cost = np.nan
                take_profit = np.nan
                highest_high = np.nan
                holding_sessions = 0
                exit_executed = True
            else:
                highest_high = max(float(highest_high), high_price)

        candidate_signals = executions.get(index, [])
        if candidate_signals:
            if was_in_position:
                skipped_signals_in_position += len(candidate_signals)
            elif shares <= 0.0:
                entry_signal_date = candidate_signals[0]
                capital_before_entry = float(capital)
                entry_cost = capital_before_entry * VXX_TRANSACTION_COST
                investable_capital = capital_before_entry - entry_cost
                entry_price = close_price
                shares = investable_capital / entry_price
                capital = 0.0
                entry_date = date
                take_profit = entry_price * VXX_TAKE_PROFIT_MULTIPLE
                # The entry is executed at the close; the day's earlier high is
                # not part of the post-entry trailing history.
                highest_high = entry_price
                holding_sessions = 0
                entry_executed = True

        signal_row = signal_lookup.get(date, {})
        nav = shares * close_price if shares > 0.0 else capital
        daily_rows.append(
            {
                "Date": date,
                "VXX_Open": open_price,
                "VXX_High": high_price,
                "VXX_Low": low_price,
                "VXX_Close": close_price,
                "NAV": float(nav),
                "Position": "VXX" if shares > 0.0 else "CASH",
                "EligibleEntrySignal": bool(signal_row.get("EligibleEntrySignal", False)),
                "EntryExecuted": entry_executed,
                "ExitExecuted": exit_executed,
                "ExitReason": exit_reason,
            }
        )

    daily = pd.DataFrame(daily_rows)
    daily["DrawdownPct"] = (daily["NAV"] / daily["NAV"].cummax() - 1.0) * 100.0
    trades = pd.DataFrame(trade_rows, columns=empty_trades.columns)
    daily_returns = pd.to_numeric(daily["NAV"], errors="coerce").pct_change(fill_method=None).fillna(0.0)
    daily_std = float(daily_returns.std(ddof=1))
    elapsed_years = max((len(daily) - 1) / 252.0, 0.0)
    final_nav = float(daily["NAV"].iloc[-1])
    cagr = (
        (final_nav / float(initial_capital)) ** (1.0 / elapsed_years) - 1.0
        if elapsed_years > 0 and final_nav > 0 and initial_capital > 0
        else np.nan
    )
    trade_returns = pd.to_numeric(trades.get("Net Return (%)", pd.Series(dtype="float64")), errors="coerce")
    metrics = {
        "Total Trades": float(len(trades)),
        "Winning Trades": float((trade_returns > 0.0).sum()),
        "Losing Trades": float((trade_returns <= 0.0).sum()),
        "Average Trade Return (%)": float(trade_returns.mean()) if not trade_returns.empty else np.nan,
        "Median Trade Return (%)": float(trade_returns.median()) if not trade_returns.empty else np.nan,
        "CAGR (%)": float(cagr * 100.0) if np.isfinite(cagr) else np.nan,
        "Maximum Drawdown (%)": float(daily["DrawdownPct"].min()),
        "Sharpe Ratio": (
            float(daily_returns.mean() / daily_std * np.sqrt(252.0))
            if daily_std > 0.0
            else np.nan
        ),
        "Final Portfolio Capital ($)": final_nav,
    }
    current_position = "VXX" if shares > 0.0 else "CASH"
    diagnostics: dict[str, Any] = {
        "status": "ok",
        "source": "Yahoo Finance VXX, auto-adjusted completed daily OHLC",
        "first_date": daily["Date"].min(),
        "last_date": daily["Date"].max(),
        "raw_rapid_candidates": vxx_rapid_candidates,
        "raw_rapid_signals": int(signals["VXXRapid"].sum()),
        "vxx_signal_source": vxx_signal_source,
        "cooldown_days": 63,
        "eligible_entry_signals": int(len(eligible)),
        "discarded_signals_before_vxx_history": discarded_signals_before_vxx_history,
        "skipped_signals_in_position": int(skipped_signals_in_position),
        "current_position": current_position,
        "pending_signal_date": pending_after_history[-1] if pending_after_history and shares <= 0.0 else None,
        "entry_date": entry_date,
        "entry_price": float(entry_price) if np.isfinite(entry_price) else np.nan,
        "take_profit": float(take_profit) if np.isfinite(take_profit) else np.nan,
        "active_trailing_stop": (
            float(VXX_TRAILING_STOP_MULTIPLE * highest_high)
            if shares > 0.0 and np.isfinite(highest_high)
            else np.nan
        ),
        "holding_sessions": int(holding_sessions),
    }
    return VXXStrategyResult(daily, trades, metrics, diagnostics)


def vxx_strategy_positioning(vix_frame: pd.DataFrame, vxx_ohlc: pd.DataFrame) -> dict[str, Any]:
    """Return the VXX V1.0 exposure and completed-week regime for Portfolio."""
    weekly_regime = "N/A"
    if vix_frame is not None and not vix_frame.empty and "WeeklyRegime" in vix_frame.columns:
        regimes = vix_frame["WeeklyRegime"].astype(str).str.upper()
        regimes = regimes[regimes.isin(["RED", "WHITE"])]
        if not regimes.empty:
            weekly_regime = str(regimes.iloc[-1])

    result = backtest_vxx_strategy(vix_frame, vxx_ohlc)
    if result.diagnostics.get("status") != "ok" or result.daily.empty:
        return {
            "position_pct": np.nan,
            "current_position": "DATA_INCOMPLETE",
            "weekly_regime": weekly_regime,
            "status": "DATA_INCOMPLETE",
            "message": str(result.diagnostics.get("message", "VXX strategy history is unavailable.")),
        }

    current_position = str(result.diagnostics.get("current_position", "CASH")).strip().upper()
    return {
        "position_pct": 0.0 if current_position == "CASH" else 100.0,
        "current_position": current_position,
        "weekly_regime": weekly_regime,
        "status": "READY",
        "signal_date": result.diagnostics.get("last_date"),
    }


def _render_vxx_strategy(vix_frame: pd.DataFrame) -> None:
    st.markdown("### VXX Trading Strategy V1.0")
    with st.spinner("Loading completed VXX daily bars from Yahoo Finance..."):
        vxx_ohlc = load_vxx_ohlc()
    result = backtest_vxx_strategy(vix_frame, vxx_ohlc)
    if result.diagnostics.get("status") != "ok" or result.daily.empty:
        st.warning(result.diagnostics.get("message", "VXX strategy history is unavailable."))
        return

    metrics = result.metrics
    position = str(result.diagnostics.get("current_position", "CASH"))
    final_capital = float(metrics.get("Final Portfolio Capital ($)", np.nan))
    summary = (
        f"VXX Strategy · {position} · {int(metrics.get('Total Trades', 0))} closed trades · "
        f"Final capital ${final_capital:,.2f}"
    )
    with st.expander(summary, expanded=False):
        metric_rows = [
            ("Current Position", position),
            ("Total Trades", f"{int(metrics.get('Total Trades', 0))}"),
            ("Wins / Losses", f"{int(metrics.get('Winning Trades', 0))} / {int(metrics.get('Losing Trades', 0))}"),
            ("Final Capital", f"${final_capital:,.2f}"),
            ("CAGR", f"{metrics.get('CAGR (%)', np.nan):.2f}%"),
            ("Maximum Drawdown", f"{metrics.get('Maximum Drawdown (%)', np.nan):.2f}%"),
            ("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', np.nan):.2f}"),
            ("Average / Median Trade", f"{metrics.get('Average Trade Return (%)', np.nan):.2f}% / {metrics.get('Median Trade Return (%)', np.nan):.2f}%"),
        ]
        columns = st.columns(4)
        for index, (label, value) in enumerate(metric_rows):
            with columns[index % len(columns)]:
                st.metric(label, value)

        if position == "VXX":
            st.write(
                {
                    "Entry Date": pd.Timestamp(result.diagnostics["entry_date"]).strftime("%Y-%m-%d"),
                    "Entry Price": f"${result.diagnostics['entry_price']:,.4f}",
                    "Take Profit": f"${result.diagnostics['take_profit']:,.4f}",
                    "Active Trailing Stop": f"${result.diagnostics['active_trailing_stop']:,.4f}",
                    "Completed Holding Sessions": result.diagnostics["holding_sessions"],
                }
            )
        elif result.diagnostics.get("pending_signal_date") is not None:
            st.info(
                "Eligible signal after the latest completed VXX bar; entry is pending at the next completed VXX close."
            )

        nav_col, drawdown_col = st.columns(2)
        with nav_col:
            nav_fig = go.Figure()
            nav_fig.add_trace(
                go.Scatter(
                    x=result.daily["Date"],
                    y=result.daily["NAV"],
                    mode="lines",
                    name="VXX strategy NAV",
                    line={"color": "#38bdf8", "width": 1.8},
                )
            )
            _style_model_chart(nav_fig, "VXX Strategy Portfolio NAV", height=340)
            nav_fig.update_yaxes(title_text="Portfolio value, $")
            st.plotly_chart(nav_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        with drawdown_col:
            drawdown_fig = go.Figure()
            drawdown_fig.add_trace(
                go.Scatter(
                    x=result.daily["Date"],
                    y=result.daily["DrawdownPct"],
                    mode="lines",
                    name="Drawdown",
                    line={"color": "#ef4444", "width": 1.6},
                    fill="tozeroy",
                    fillcolor="rgba(239, 68, 68, 0.12)",
                )
            )
            _style_model_chart(drawdown_fig, "VXX Strategy Drawdown", height=340)
            drawdown_fig.update_yaxes(title_text="Drawdown, %")
            st.plotly_chart(drawdown_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

        st.markdown("#### VXX Trade History")
        if result.trades.empty:
            st.info("No completed VXX trades in the available Yahoo Finance history.")
        else:
            display_trades = result.trades.sort_values("Entry Date", ascending=False).copy()
            st.dataframe(display_trades, use_container_width=True, hide_index=True)
            st.download_button(
                "Export VXX trade history CSV",
                display_trades.to_csv(index=False).encode("utf-8"),
                file_name="vxx_trading_strategy_v1_trade_history.csv",
                mime="text/csv",
                key="vxx_strategy_trade_history_download",
            )
        st.caption(
            f"Single full-history backtest: {pd.Timestamp(result.diagnostics['first_date']).date()}–"
            f"{pd.Timestamp(result.diagnostics['last_date']).date()}. Yahoo Finance auto-adjusted completed VXX OHLC; "
            "$1,000 initial capital; 5 bp on entry and 5 bp on exit; 0% cash return. "
            "Signals execute at the next VXX close. Intraday exits use stop-first ordering when TP and stop are both reachable."
        )


def _render_performance(frame: pd.DataFrame) -> None:
    strategies = [
        ("SPX Buy & Hold", np.ones(len(frame), dtype=float)),
        ("Strategy A — Defensive", frame["StrategyAWeight"].to_numpy(dtype=float)),
        ("Strategy B — Forced Entry", frame["StrategyBWeight"].to_numpy(dtype=float)),
    ]
    metric_table: dict[str, dict[str, float]] = {}
    curves: dict[str, pd.DataFrame] = {}
    proxy = frame.copy()
    proxy["RiskOff"] = proxy["RiskOff"].fillna(False)
    proxy["ForcedEntryActive"] = proxy["ForcedEntryActive"].fillna(False)
    for label, weights in strategies:
        temp = proxy.copy()
        temp["SelectedWeight"] = weights
        metrics, curve = _evaluate_strategy(temp, "SelectedWeight")
        if label == "Strategy B — Forced Entry":
            metrics["Forced-Entry Episodes"] = float(_count_episodes(proxy["ForcedEntryActive"]))
        elif label == "Strategy A — Defensive":
            metrics["Forced-Entry Episodes"] = 0.0
        else:
            metrics["Risk-Off Episodes"] = 0.0
            metrics["Forced-Entry Episodes"] = 0.0
        metric_table[label] = metrics
        curves[label] = curve
    if not metric_table["SPX Buy & Hold"]:
        st.info("Not enough aligned history to calculate backtest statistics.")
        return
    stats = pd.DataFrame(metric_table)
    st.markdown("#### Performance Comparison")
    st.dataframe(stats, use_container_width=True)

    curve_col, drawdown_col = st.columns(2)
    with curve_col:
        fig = go.Figure()
        for name, curve in curves.items():
            fig.add_trace(go.Scatter(x=curve["Date"], y=curve["NAV"], mode="lines", name=name))
        _style_model_chart(fig, "Portfolio NAV", height=360)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
    with drawdown_col:
        fig = go.Figure()
        for name, curve in curves.items():
            fig.add_trace(go.Scatter(x=curve["Date"], y=curve["Drawdown (%)"], mode="lines", name=name))
        _style_model_chart(fig, "Portfolio Drawdown", height=360)
        fig.update_yaxes(title_text="Drawdown (%)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def _style_model_chart(fig: go.Figure, title: str, height: int = 360) -> None:
    fig.update_layout(
        title=title,
        height=height,
        margin={"l": 24, "r": 16, "t": 50, "b": 18},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": VIX_MODEL_COLORS["text"], "size": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.16},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=VIX_MODEL_COLORS["grid"], zerolinecolor=VIX_MODEL_COLORS["grid"])


def render_vix_based_model(daily_source: pd.DataFrame) -> None:
    st.markdown("### VIX Based Model")
    model = build_vix_model_data(daily_source)
    if model.diagnostics.get("status") != "ok" or model.frame.empty:
        st.warning(model.diagnostics.get("message", "VIX Based Model history is unavailable."))
        return
    full = model.frame.copy()
    latest = full.iloc[-1]
    current_spread = pd.to_numeric(pd.Series([latest.get("DailySpread")]), errors="coerce").iloc[0]
    current_weekly = pd.to_numeric(pd.Series([latest.get("WeeklySpread")]), errors="coerce").iloc[0]
    forced_enabled = bool(st.session_state.get("vix_model_forced_entry_enabled", False))
    selected_weight = float(latest["StrategyBWeight"] if forced_enabled else latest["StrategyAWeight"])
    current_active = bool(latest["ForcedEntryActive"] and forced_enabled)
    compact = (
        f"VIX Based Model · Weekly {latest['WeeklyRegime'] or 'N/A'} · Daily {current_spread:.2f}% · "
        f"Target {selected_weight:.0%} SPX · Forced Entry {'ON' if forced_enabled else 'OFF'}"
    )
    range_choice = st.radio("Chart range", ["1Y", "3Y", "5Y", "10Y", "MAX"], index=4, horizontal=True, key="vix_based_model_range")
    fig = build_vix_primary_chart(full, range_choice)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": True, "displaylogo": False, "responsive": True, "scrollZoom": False},
    )
    st.caption("Weekly indicators use only completed exchange weeks; the final available session sets the week close, including shortened holiday weeks. Red shading is structural; green lines are daily upward crossovers. Rapid-approach markers are informational only.")

    _render_vxx_strategy(full)

    with st.expander(compact, expanded=False):
        forced_enabled = st.toggle(
            "Forced SPX Entry (>85% Daily Spread)",
            value=forced_enabled,
            key="vix_model_forced_entry_enabled",
            help="OFF: Strategy A — Defensive. ON: Strategy B — staged 70% / 15% / 15% re-entry.",
        )
        selected_weight = float(latest["StrategyBWeight"] if forced_enabled else latest["StrategyAWeight"])
        current_active = bool(latest["ForcedEntryActive"] and forced_enabled)
        active_strategy = "Strategy B — Staged Re-entry 70% / 15% / 15%" if forced_enabled else "Strategy A — Defensive"
        st.markdown(f"**Active strategy:** {active_strategy}")
        cols = st.columns(4)
        values = [
            ("Weekly Regime", str(latest["WeeklyRegime"] or "N/A")),
            ("Weekly Spread", f"{current_weekly:.2f}%" if np.isfinite(current_weekly) else "N/A"),
            ("Daily Spread", f"{current_spread:.2f}%" if np.isfinite(current_spread) else "N/A"),
            ("RiskOff", str(bool(latest["RiskOff"]))),
            ("Forced Entry Enabled", str(forced_enabled)),
            ("Forced Entry Active", str(current_active)),
            ("SPX Target Weight", f"{selected_weight:.0%}"),
            ("Cash Target Weight", f"{1.0 - selected_weight:.0%}"),
        ]
        for i, (label, value) in enumerate(values):
            with cols[i % len(cols)]:
                st.metric(label, value)
        if forced_enabled:
            entry_price = latest.get("InitialEntrySPX")
            next_level = "None"
            if current_active and np.isfinite(pd.to_numeric(pd.Series([entry_price]), errors="coerce").iloc[0]):
                entry_price = float(entry_price)
                reached = model.events.loc[model.events["Initial Entry SPX"].eq(entry_price)] if not model.events.empty else pd.DataFrame()
                buys = reached[reached["Signal type"].str.contains("Additional Entry", na=False)] if not reached.empty else pd.DataFrame()
                done = set(buys["Signal type"].tolist())
                next_level = "−10%" if "Additional Entry −10%" not in done else ("−20%" if "Additional Entry −20%" not in done else "Normalization to 100%")
                entry_text = f"{entry_price:,.2f}"
            else:
                entry_text = "N/A"
            st.write(
                {
                    "Initial Forced Entry SPX": entry_text,
                    "Next Purchase Level": next_level,
                    "Last Signal": _last_signal_summary(model.events, forced_enabled, latest["Date"]),
                    "Last Updated": pd.Timestamp(latest["Date"]).strftime("%Y-%m-%d"),
                }
            )
        else:
            st.caption("Forced-entry fields are not applicable while Strategy A is active.")

        _render_performance(full)
        st.markdown("#### Trade History")
        if model.events.empty:
            st.info("No model transitions in the available backtest history.")
        else:
            events = model.events.drop(columns=["Initial Entry SPX", "Episode Start"], errors="ignore").copy()
            st.dataframe(events.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
            st.download_button(
                "Export trade history CSV",
                events.to_csv(index=False).encode("utf-8"),
                file_name="vix_based_model_trade_history.csv",
                mime="text/csv",
            )
        st.caption(
            "Reference backtest: SPX price index (no dividends), 0% cash return, 5 bp × absolute exposure change; "
            "signals at completed close apply to the following close-to-close interval. Sharpe uses a 0% risk-free rate, "
            "sample standard deviation and 252 trading days."
        )
        st.caption(
            f"Aligned data: {model.diagnostics['observations']:,} sessions, "
            f"{pd.Timestamp(model.diagnostics['first_date']).date()}–{pd.Timestamp(model.diagnostics['last_date']).date()}. "
            "The V10/V14 source daily-bar CSVs are not part of the app bundle; exact regression reconciliation requires the same historical source bars."
        )


def _last_signal_summary(events: pd.DataFrame, forced_enabled: bool, as_of: Any) -> str:
    if events.empty:
        return "None"
    strategy = "Strategy B — Forced Entry" if forced_enabled else "Strategy A — Defensive"
    eligible = events.loc[(events["Strategy"] == strategy) & (pd.to_datetime(events["Date"]) <= pd.Timestamp(as_of))]
    if eligible.empty:
        return "None"
    last = eligible.sort_values("Date").iloc[-1]
    return f"{pd.Timestamp(last['Date']).strftime('%Y-%m-%d')} · {last['Signal type']}"
