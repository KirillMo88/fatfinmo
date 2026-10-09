from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .trading_system import (
    build_daily_observations,
    expanding_regression,
    instrument_for_regime,
    monthly_ratio_series,
    run_backtest,
    run_state_machine,
    silver_platinum_deviation,
)


REGIME_COLORS = {
    "GOLD": "rgba(212, 175, 55, 0.13)",
    "SILVER": "rgba(190, 198, 210, 0.14)",
    "PLATINUM": "rgba(78, 139, 215, 0.15)",
}
MODE_OPTIONS = ("1x", "3x")


def current_model_regime(market_data: dict[str, Any]) -> str:
    """Return the latest completed-week regime from the strategy state machine."""
    daily = market_data.get("daily", {})
    monthly = market_data.get("monthly", {})
    gold_daily = daily.get("GOLD", pd.DataFrame())
    silver_daily = daily.get("SILVER", pd.DataFrame())
    if gold_daily is None or gold_daily.empty or silver_daily is None or silver_daily.empty:
        return "DATA_INCOMPLETE"
    history, _, _, _ = _cached_signal_history(
        gold_daily,
        silver_daily,
        daily.get("PLATINUM", pd.DataFrame()),
        monthly.get("GOLD", pd.DataFrame()),
        monthly.get("SILVER", pd.DataFrame()),
        monthly.get("PLATINUM", pd.DataFrame()),
        _current_week_key(),
    )
    if history is None or history.empty:
        return "NOT_READY"
    return str(history.iloc[-1].get("regime", "NOT_READY"))


def render_gold_silver_trading_system(
    snapshot: Any,
    selected_range: str,
    market_data: dict[str, Any],
) -> None:
    st.subheader("Gold / Silver Trading System")
    st.caption("Expanding Regression 1998–Current | Weekly Signals")
    st.caption("Historical assumption: CASH until December 2014, then GOLD; GOLD → Silver/Platinum is allowed again from June 2019 at +2σ. The −2σ exit to CASH remains active.")

    mode = st.segmented_control("Investment mode", MODE_OPTIONS, default="1x", key="gs_trading_mode")
    mode = mode or "1x"
    monthly = market_data.get("monthly", {})
    daily = market_data.get("daily", {})
    gold_monthly = monthly.get("GOLD", pd.DataFrame())
    silver_monthly = monthly.get("SILVER", pd.DataFrame())
    platinum_monthly = monthly.get("PLATINUM", pd.DataFrame())
    gold_daily = daily.get("GOLD", pd.DataFrame())
    silver_daily = daily.get("SILVER", pd.DataFrame())
    platinum_daily = daily.get("PLATINUM", pd.DataFrame())

    history, events, monthly_gs, monthly_model = _cached_signal_history(
        gold_daily,
        silver_daily,
        platinum_daily,
        gold_monthly,
        silver_monthly,
        platinum_monthly,
        _current_week_key(),
    )
    data_status = market_data.get("status", {})
    incomplete = [name for name, frame in (("Gold daily", gold_daily), ("Silver daily", silver_daily)) if frame is None or frame.empty]
    if incomplete:
        st.warning(f"Signal calculation unavailable: missing {' and '.join(incomplete)} TradingView daily observations.")
        _render_data_sources(data_status)
        return
    # Re-rendering a different mode reuses the cached source frames and the
    # same regime series; only the instrument map and resulting P&L change.
    prices: dict[str, pd.DataFrame] = {
        "GOLD": gold_daily,
        "SILVER": silver_daily,
        "PLATINUM": platinum_daily,
        **market_data.get("leveraged", {}),
    }
    backtest = run_backtest(prices, history, events, mode)
    currency_warning = market_data.get("currency_warning")
    if currency_warning:
        st.info(currency_warning)

    current_status = _current_status(history, events, market_data, mode)
    _render_current_status(current_status, mode, backtest)
    if pd.notna(current_status.get("last_data")):
        data_age = (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timestamp(current_status["last_data"]).normalize()).days
        if data_age > 5:
            st.warning(f"The latest underlying daily observation is {data_age} days old; treat the current status as stale.")
    if backtest["status"] == "READY" and pd.notna(backtest.get("latest_data_date")):
        aligned_age = (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timestamp(backtest["latest_data_date"]).normalize()).days
        if aligned_age > 5:
            st.warning(f"The latest common strategy-price observation is {aligned_age} days old; the performance curve is stale.")

    if backtest["status"] != "READY":
        st.warning(backtest["message"])
        if mode == "3x":
            _render_data_sources(data_status)
        return

    toggle_left, toggle_right = st.columns(2)
    with toggle_left:
        show_signals = st.toggle("Show signal markers", value=True, key="gs_show_signals")
    with toggle_right:
        show_positions = st.toggle("Shade executed positions", value=True, key="gs_show_positions")

    gold_weekly = market_data.get("gold_weekly", pd.Series(dtype=float))
    if gold_weekly is None or gold_weekly.empty:
        gold_weekly = _gold_weekly_from_snapshot(snapshot)
    chart_history = _build_chart_history(history, monthly_gs, monthly_model)
    figure = _build_figure(chart_history, events, backtest, gold_weekly, selected_range, mode, show_signals, show_positions)
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})
    _render_performance(backtest, mode)
    with st.expander("Annual Performance", expanded=False):
        if backtest["annual"].empty:
            st.info("Annual performance is not available yet.")
        else:
            st.dataframe(
                backtest["annual"],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Strategy Return": st.column_config.NumberColumn(format="%.2f%%"),
                    "Strategy MDD": st.column_config.NumberColumn(format="%.2f%%"),
                    "Gold Return": st.column_config.NumberColumn(format="%.2f%%"),
                    "Gold MDD": st.column_config.NumberColumn(format="%.2f%%"),
                    "Excess Return": st.column_config.NumberColumn(format="%.2f%%"),
                },
            )
    _render_data_sources(data_status)


def _current_status(
    history: pd.DataFrame,
    events: pd.DataFrame,
    market_data: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    if history is None or history.empty:
        return {"regime": "NOT_READY", "instrument": "N/A", "ratio": np.nan, "zscore": np.nan}
    latest = history.iloc[-1]
    regime = str(latest.get("regime", "NOT_READY"))
    ratio = _number(latest.get("gold_silver_ratio"))
    zscore = _number(latest.get("zscore"))
    candidates: list[tuple[float, str, float]] = []
    if regime == "CASH":
        candidates = [(float(latest.get("upper_1", np.nan)), "+1σ: GOLD", 1.0)]
    elif regime == "GOLD":
        candidates = [
            (float(latest.get("upper_2", np.nan)), "+2σ: Late-Metals", 2.0),
            (float(latest.get("lower_2", np.nan)), "−2σ: CASH", -2.0),
        ]
    elif regime in {"SILVER", "PLATINUM"}:
        candidates = [
            (float(latest.get("lower_1", np.nan)), "−1σ: GOLD", -1.0),
            (float(latest.get("lower_2", np.nan)), "−2σ: CASH", -2.0),
        ]
    candidates = [item for item in candidates if np.isfinite(item[0]) and item[0] > 0]
    next_threshold = min(candidates, key=lambda item: abs(ratio / item[0] - 1)) if candidates and np.isfinite(ratio) else None
    last_signal = pd.NaT
    if events is not None and not events.empty:
        last_signal = pd.to_datetime(events["signal_date"], errors="coerce").max()
    source_dates = [
        frame.index.max()
        for frame in market_data.get("daily", {}).values()
        if frame is not None and not frame.empty
    ]
    return {
        "regime": regime,
        "instrument": instrument_for_regime(regime, mode),
        "ratio": ratio,
        "zscore": zscore,
        "threshold_label": next_threshold[1] if next_threshold else "N/A",
        "threshold": next_threshold[0] if next_threshold else np.nan,
        "distance_pct": abs(ratio / next_threshold[0] - 1.0) * 100.0 if next_threshold and np.isfinite(ratio) else np.nan,
        "last_signal": last_signal,
        "last_data": max(source_dates) if source_dates else pd.NaT,
        "signal_date": history.index[-1],
    }


@st.cache_data(show_spinner=False)
def _cached_signal_history(
    gold_daily: pd.DataFrame,
    silver_daily: pd.DataFrame,
    platinum_daily: pd.DataFrame,
    gold_monthly: pd.DataFrame,
    silver_monthly: pd.DataFrame,
    platinum_monthly: pd.DataFrame,
    current_week_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    as_of = pd.Timestamp(current_week_key)
    monthly_gs = monthly_ratio_series(gold_monthly, silver_monthly, as_of=as_of, name="gold_silver_ratio")
    model = expanding_regression(monthly_gs)
    monthly_sp = silver_platinum_deviation(silver_monthly, platinum_monthly, as_of=as_of)
    observations = build_daily_observations(gold_daily, silver_daily, platinum_daily, model, monthly_sp)
    history, events, _ = run_state_machine(observations, as_of=as_of)
    return history, events, monthly_gs, model


def _build_chart_history(
    daily_history: pd.DataFrame,
    monthly_ratio: pd.Series,
    monthly_model: pd.DataFrame,
) -> pd.DataFrame:
    """Show monthly historical context before daily synchronized bars begin."""
    if daily_history is None or daily_history.empty or monthly_ratio is None or monthly_ratio.empty:
        return daily_history.copy() if daily_history is not None else pd.DataFrame()
    monthly = pd.DataFrame({"gold_silver_ratio": pd.to_numeric(monthly_ratio, errors="coerce")})
    if monthly_model is not None and not monthly_model.empty:
        model_columns = ["mean", "sigma", "upper_1", "upper_2", "lower_1", "lower_2"]
        monthly = monthly.join(monthly_model[model_columns], how="left")
    else:
        monthly["mean"] = np.nan
        monthly["sigma"] = np.nan
        for column in ("upper_1", "upper_2", "lower_1", "lower_2"):
            monthly[column] = np.nan
    monthly["zscore"] = (monthly["gold_silver_ratio"] - monthly["mean"]) / monthly["sigma"].replace(0, np.nan)
    monthly["regime"] = "NOT_READY"
    monthly.index = pd.to_datetime(monthly.index)
    daily = daily_history.copy()
    daily.index = pd.to_datetime(daily.index)
    monthly = monthly.loc[monthly.index < daily.index.min()]
    return pd.concat([monthly, daily], axis=0).sort_index()


def _current_week_key() -> str:
    current = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return current.to_period("W-FRI").start_time.strftime("%Y-%m-%d")


def _render_current_status(
    status: dict[str, Any],
    mode: str,
    backtest: dict[str, Any],
) -> None:
    cols = st.columns(7)
    regime = str(status.get("regime", "NOT_READY"))
    _metric(cols[0], "Current regime", regime)
    _metric(cols[1], "Current instrument", str(status.get("instrument", "N/A")))
    ratio = _number(status.get("ratio"))
    _metric(cols[2], "Gold / Silver", f"{ratio:.4f}" if np.isfinite(ratio) else "N/A")
    zscore = _number(status.get("zscore"))
    _metric(cols[3], "Current Z-score", f"{zscore:+.2f}" if np.isfinite(zscore) else "N/A")
    next_label = str(status.get("threshold_label", "N/A"))
    distance = _number(status.get("distance_pct"))
    _metric(cols[4], "Next threshold", f"{next_label} · {distance:.1f}%" if np.isfinite(distance) else "N/A")
    last_signal = status.get("last_signal", pd.NaT)
    _metric(cols[5], "Last signal", "N/A" if pd.isna(last_signal) else last_signal.strftime("%Y-%m-%d"))
    last_date = status.get("last_data", pd.NaT)
    _metric(cols[6], "Last data update", "N/A" if pd.isna(last_date) else pd.Timestamp(last_date).strftime("%Y-%m-%d"))
    if mode == "3x" and backtest["status"] == "READY":
        st.caption(f"3x backtest starts {pd.Timestamp(backtest['start_date']):%Y-%m-%d}, based on the first common valid 3GOL.L / 3SIL.L price history. Returns use the actual funds; no extra leverage multiplier applied.")


def _build_figure(
    history: pd.DataFrame,
    events: pd.DataFrame,
    backtest: dict[str, Any],
    gold_weekly: pd.Series,
    selected_range: str,
    mode: str,
    show_signals: bool,
    show_positions: bool,
) -> go.Figure:
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.571, 0.143, 0.143, 0.143],
        subplot_titles=("Gold / Silver Ratio and Expanding Regression", "Gold Price (USD/oz)", "Portfolio Capital (USD)", f"Strategy Drawdown · Max {backtest['max_drawdown_pct']:.2f}%"),
    )
    curve = backtest["curve"]
    close_curve = curve.loc[curve["phase"].eq("Close")].copy()
    start, end = pd.Timestamp(backtest["start_date"]), pd.Timestamp(backtest["end_date"])
    years = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}.get(selected_range)
    range_start = end - pd.DateOffset(years=years) if years else (history.index.min() if not history.empty else start)
    active = history.loc[(history.index >= range_start) & (history.index <= end)].copy()
    for column, label, color in (
        ("upper_1", "+1σ", "#e4a0a0"),
        ("upper_2", "+2σ", "#ff7474"),
        ("lower_2", "−2σ", "#8ed2a6"),
        ("lower_1", "−1σ", "#a2dfb7"),
    ):
        fill = "tonexty" if column in {"upper_2", "lower_1"} else None
        fill_color = "rgba(235, 90, 90, .13)" if column == "upper_2" else "rgba(75, 190, 115, .13)"
        figure.add_trace(
            go.Scatter(
                x=active.index,
                y=active[column],
                name=label,
                mode="lines",
                line={"color": color, "width": 1, "dash": "dot"},
                fill=fill,
                fillcolor=fill_color if fill else None,
                connectgaps=False,
                legendgroup="channel",
            ),
            row=1,
            col=1,
        )
    figure.add_trace(go.Scatter(x=active.index, y=active["mean"], name="Regression Mean", mode="lines", line={"color": "#f2cf67", "width": 2}, connectgaps=False), row=1, col=1)
    figure.add_trace(go.Scatter(x=active.index, y=active["gold_silver_ratio"], name="Gold / Silver Ratio", mode="lines", line={"color": "#f3f5f7", "width": 1.5}, connectgaps=False), row=1, col=1)

    if show_positions and not close_curve.empty:
        position_values = close_curve.copy()
        position_values["chart_regime"] = position_values["regime"].replace({"PLATINUM": "SILVER"}) if mode == "3x" else position_values["regime"]
        states = position_values["chart_regime"].astype(str).to_list()
        dates = pd.to_datetime(position_values["date"]).to_list()
        boundaries = [0] + [i for i in range(1, len(states)) if states[i] != states[i - 1]] + [len(states)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            state = states[left]
            color = REGIME_COLORS.get(state)
            if not color:
                continue
            x0 = dates[left]
            x1 = dates[right] if right < len(dates) else dates[-1] + pd.Timedelta(days=1)
            figure.add_vrect(x0=x0, x1=x1, fillcolor=color, line_width=0, layer="below", row=1, col=1)

    if show_signals and backtest.get("executed_events") is not None and not backtest["executed_events"].empty:
        marker_rows: list[dict[str, Any]] = []
        for _, execution in backtest["executed_events"].iterrows():
            for event in execution.get("signal_events", []):
                marker_rows.append({**event, "execution_date": execution["execution_date"]})
        if marker_rows:
            markers = pd.DataFrame(marker_rows)
            marker_y = pd.to_numeric(markers["ratio"], errors="coerce")
            colors = ["#ffd166" if str(value).endswith("GOLD") else "#62c7ef" if "PLATINUM" in str(value) else "#c0c7d1" if "SILVER" in str(value) else "#b8a3ee" for value in markers["to_regime"]]
            labels = [f"{r.from_regime} → {r.to_regime}" for r in markers.itertuples()]
            selected_assets = [instrument_for_regime(str(value), mode) for value in markers["to_regime"]]
            custom = np.column_stack(
                [
                    pd.to_datetime(markers["signal_date"]).dt.strftime("%Y-%m-%d"),
                    labels,
                    markers["zscore"].map(lambda value: f"{_number(value):+.2f}" if np.isfinite(_number(value)) else "N/A"),
                    markers["threshold"].astype(str),
                    selected_assets,
                    pd.to_datetime(markers["execution_date"]).dt.strftime("%Y-%m-%d"),
                    markers["sequence_ambiguous"].map(lambda value: "Yes — sequence ambiguous" if bool(value) else "No"),
                ]
            )
            figure.add_trace(
                go.Scatter(
                    x=pd.to_datetime(markers["signal_date"]),
                    y=marker_y,
                    mode="markers",
                    name="Signals",
                    marker={"size": 9, "symbol": "diamond", "color": colors, "line": {"color": "#101820", "width": 1}},
                    customdata=custom,
                    hovertemplate="Signal: %{customdata[0]}<br>%{customdata[1]}<br>Ratio: %{y:.4f}<br>Z-score: %{customdata[2]}<br>Threshold: %{customdata[3]}<br>Selected asset: %{customdata[4]}<br>Modeled execution: %{customdata[5]}<br>%{customdata[6]}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    gold = pd.to_numeric(gold_weekly, errors="coerce") if gold_weekly is not None else pd.Series(dtype=float)
    gold.index = pd.to_datetime(gold.index, errors="coerce")
    gold = gold.loc[gold.index.notna() & (gold.index >= range_start) & (gold.index <= end)].dropna()
    figure.add_trace(go.Scatter(x=gold.index, y=gold, name="Gold spot", mode="lines", line={"color": "#dfbd55", "width": 1.5}, connectgaps=False), row=2, col=1)

    figure.add_trace(go.Scatter(x=curve["date"], y=curve["equity"], name="Strategy Equity", mode="lines", line={"color": "#66d19e", "width": 2}, customdata=np.column_stack([curve["regime"], curve["equity"] / float(backtest["initial_capital"]) - 1]), hovertemplate="%{x|%Y-%m-%d}<br>Strategy: $%{y:,.2f}<br>Position: %{customdata[0]}<br>Cumulative return: %{customdata[1]:+.2%}<extra></extra>"), row=3, col=1)
    figure.add_trace(go.Scatter(x=curve["date"], y=curve["gold_equity"], name="Gold Buy & Hold", mode="lines", line={"color": "#c2a344", "width": 1.5}, customdata=curve["gold_equity"] / float(backtest["initial_capital"]) - 1, hovertemplate="%{x|%Y-%m-%d}<br>Gold benchmark: $%{y:,.2f}<br>Cumulative return: %{customdata:+.2%}<extra></extra>"), row=3, col=1)
    figure.add_trace(go.Bar(x=curve["date"], y=curve["drawdown"].clip(upper=0), name="Strategy Drawdown", marker_color="#e76f6f", customdata=curve["regime"], hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<br>Position: %{customdata}<extra></extra>"), row=4, col=1)
    figure.add_hline(y=0, line={"color": "#9ba3ad", "width": 1}, row=4, col=1)

    figure.update_xaxes(range=[range_start, end], showgrid=True, gridcolor="rgba(130,145,160,.18)", linecolor="rgba(150,160,170,.25)")
    for axis in ("xaxis2", "xaxis3", "xaxis4"):
        figure.layout[axis].matches = "x"
    figure.update_yaxes(showgrid=True, gridcolor="rgba(130,145,160,.18)", zeroline=False)
    figure.update_yaxes(title_text="Ratio", row=1, col=1)
    figure.update_yaxes(title_text="USD/oz", row=2, col=1)
    figure.update_yaxes(title_text="USD", row=3, col=1)
    figure.update_yaxes(title_text="Drawdown %", row=4, col=1)
    figure.update_layout(
        height=900,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 30, "r": 15, "t": 45, "b": 15},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        hovermode="x unified",
        barmode="overlay",
    )
    return figure


def _render_performance(backtest: dict[str, Any], mode: str) -> None:
    st.markdown("#### Performance Summary")
    state = backtest["time_in_state_pct"]
    columns = 6 if mode == "3x" else 7
    labels = [
        ("Initial Capital", _money(backtest["initial_capital"])),
        ("Final Capital", _money(backtest["final_capital"])),
        ("Total Return", _pct(backtest["total_return_pct"])),
        ("CAGR", _pct(backtest["cagr_pct"])),
        ("Max Drawdown", _pct(backtest["max_drawdown_pct"])),
        ("Calmar Ratio", _number_text(backtest["calmar_ratio"], 2)),
        ("Trades", str(backtest["number_of_trades"])),
        ("Time in GOLD", _pct(state.get("GOLD", np.nan))),
        ("Time in SILVER", _pct(state.get("SILVER", np.nan) + (state.get("PLATINUM", 0.0) if mode == "3x" else 0.0))),
    ]
    if mode == "1x":
        labels.append(("Time in PLATINUM", _pct(state.get("PLATINUM", np.nan))))
    labels.extend(
        [
            ("Time in CASH", _pct(state.get("CASH", np.nan))),
            ("Gold Buy & Hold CAGR", _pct(backtest["gold_cagr_pct"])),
            ("Gold Buy & Hold MDD", _pct(backtest["gold_max_drawdown_pct"])),
        ]
    )
    for offset in range(0, len(labels), columns):
        row = st.columns(columns)
        for target, (label, value) in zip(row, labels[offset : offset + columns]):
            target.metric(label, value)


def _render_data_sources(status: dict[str, str]) -> None:
    with st.expander("Data sources and quality", expanded=False):
        st.caption("Ratio signals use synchronized daily closes from TradingView (computed in-app; no ratio high/low synthesized). Each fully completed week is confirmed using the observed daily-close extrema. Weekly-only fallback is not used.")
        st.caption("1x spot returns use TradingView daily OHLC. 3x returns use actual yfinance auto-adjusted 3GOL.L / 3SIL.L prices converted to USD. Missing observations are not forward-filled.")
        if status:
            st.dataframe(pd.DataFrame([{"Series": key, "Status": value} for key, value in status.items()]), hide_index=True, use_container_width=True)


def _gold_weekly_from_snapshot(snapshot: Any) -> pd.Series:
    history = getattr(snapshot, "history", pd.DataFrame())
    if history is None or history.empty or not {"date", "gold_price"}.issubset(history.columns):
        return pd.Series(dtype=float, name="gold_price")
    values = history[["date", "gold_price"]].copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce")
    values["gold_price"] = pd.to_numeric(values["gold_price"], errors="coerce")
    return values.dropna().drop_duplicates("date", keep="last").set_index("date")["gold_price"]


def _next_threshold(row: pd.Series, regime: str) -> tuple[str, float]:
    ratio = _number(row.get("gold_silver_ratio"))
    choices = []
    if regime == "CASH":
        choices = [("+1σ: GOLD", _number(row.get("upper_1")))]
    elif regime == "GOLD":
        choices = [("+2σ: Late-Metals", _number(row.get("upper_2"))), ("−2σ: CASH", _number(row.get("lower_2")))]
    elif regime in {"SILVER", "PLATINUM"}:
        choices = [("−1σ: GOLD", _number(row.get("lower_1"))), ("−2σ: CASH", _number(row.get("lower_2")))]
    choices = [(label, level) for label, level in choices if np.isfinite(level) and level > 0]
    if not choices or not np.isfinite(ratio):
        return "N/A", np.nan
    label, level = min(choices, key=lambda item: abs(ratio / item[1] - 1.0))
    return label, abs(ratio / level - 1.0) * 100.0


def _metric(column: Any, label: str, value: str) -> None:
    column.metric(label, value)


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _pct(value: Any) -> str:
    number = _number(value)
    return f"{number:+.2f}%" if np.isfinite(number) else "N/A"


def _money(value: Any) -> str:
    number = _number(value)
    return f"${number:,.2f}" if np.isfinite(number) else "N/A"


def _number_text(value: Any, decimals: int = 2) -> str:
    number = _number(value)
    return f"{number:.{decimals}f}" if np.isfinite(number) else "N/A"
