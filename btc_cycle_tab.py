from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from btc_cycle import (
    BTC_CYCLE_HORIZONS,
    BTC_LIQUIDITY_CYCLE_MONTHS,
    BTC_RANGE_OPTIONS,
    BTC_PROJECTED_HALVING,
    HALVING_BASES,
    LIQUIDITY_MODIFIERS,
    SECONDARY_WEIGHTS,
    build_btc_cycle_history,
    build_btc_gold_ratio_history,
    btc_cycle_time_range,
    btc_cycle_export_xlsx,
    btc_cycle_validation,
    build_btc_modular_cycle_forecast,
    liquidity_cycle_position,
)
from btc_halving_price_forecast import (
    DEFAULT_BTC_BOTTOM_2026,
    DEFAULT_HALVING_TO_TOP_MULTIPLIERS,
    build_btc_halving_price_forecast,
    historical_return_table,
    historical_timing_table,
)


BTC_CYCLE_COLORS = {
    "EARLY_EXPANSION": "#22c55e",
    "LATE_EXPANSION": "#84cc16",
    "CYCLE_TOP_RISK": "#f97316",
    "BEAR_DELEVERAGING": "#ef4444",
    "BOTTOMING_TRANSITION": "#38bdf8",
    "ACCUMULATION_PRE_HALVING": "#a78bfa",
}
LIQUIDITY_PHASE_COLORS = {
    "RECOVERY_ACCELERATION": "#38bdf8",
    "ACCELERATING_EXPANSION": "#22c55e",
    "DECELERATING_EXPANSION": "#facc15",
    "ACCELERATING_CONTRACTION": "#ef4444",
}
SCORE_BANDS = [
    (0.0, 20.0, "#7f1d1d"),
    (20.0, 40.0, "#c2410c"),
    (40.0, 60.0, "#64748b"),
    (60.0, 80.0, "#16a34a"),
    (80.0, 95.0, "#15803d"),
]
BTC_CYCLE_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def render_btc_cycle_tab(
    btc_weekly: pd.DataFrame,
    global_m2_cycle: pd.DataFrame,
    macro_weekly: pd.DataFrame,
    global_liquidity_score: pd.DataFrame | None = None,
    btc_etf_flow_history: pd.DataFrame | None = None,
) -> None:
    st.subheader("BTC Cycle")
    st.caption("Halving cycle, Global M2 liquidity cycle, and macro conditions for BTC")
    history = build_btc_cycle_history(
        btc_weekly,
        global_m2_cycle,
        macro_weekly,
        global_liquidity_score=global_liquidity_score,
    )
    if history.empty:
        st.warning("BTC Cycle inputs are unavailable.")
        return
    validation_errors = btc_cycle_validation(history)
    if validation_errors:
        st.error("BTC Cycle validation failed: " + "; ".join(validation_errors))
        return

    historical = history.loc[~history["Projected"].astype(bool)].copy()
    current = historical.iloc[-1]
    _render_score_cards(current)
    st.markdown("### BTC Macro Score Table")
    st.dataframe(_score_table(current), hide_index=True, use_container_width=True)
    _render_drivers(current)
    next_cycle_scenario = _render_next_cycle_controls()
    halving_forecast = _halving_forecast_from_session_state()
    scenario_targets = {
        "Conservative": "BTC_CycleTop_Conservative",
        "Base": "BTC_CycleTop_Base",
        "Strong Liquidity": "BTC_CycleTop_StrongLiquidity",
    }
    next_cycle_forecast = build_btc_modular_cycle_forecast(
        btc_weekly,
        target_peak=halving_forecast[scenario_targets[next_cycle_scenario]],
        scenario=next_cycle_scenario,
    )
    st.session_state["btc_next_cycle_forecast"] = next_cycle_forecast.copy()

    time_range = st.radio(
        "Time range",
        BTC_RANGE_OPTIONS,
        horizontal=True,
        index=4,
        key="btc_cycle_time_range",
    )
    st.plotly_chart(
        _build_btc_price_halving_figure(history, time_range, next_cycle_forecast),
        use_container_width=True,
        config=BTC_CYCLE_PLOTLY_CONFIG,
    )
    st.plotly_chart(
        _build_structural_cycles_figure(history, time_range),
        use_container_width=True,
        config=BTC_CYCLE_PLOTLY_CONFIG,
    )
    horizon = st.radio(
        "Macro score horizon",
        list(BTC_CYCLE_HORIZONS),
        horizontal=True,
        index=0,
        key="btc_cycle_macro_horizon",
    )
    st.plotly_chart(
        _build_macro_score_figure(history, horizon, time_range),
        use_container_width=True,
        config=BTC_CYCLE_PLOTLY_CONFIG,
    )
    gold_weekly = _load_btc_cycle_gold_weekly()
    btc_gold_ratio = build_btc_gold_ratio_history(btc_weekly, gold_weekly)
    st.plotly_chart(
        _build_btc_gold_ratio_figure(history, btc_gold_ratio, time_range),
        use_container_width=True,
        config=BTC_CYCLE_PLOTLY_CONFIG,
    )
    if btc_gold_ratio.empty:
        st.caption("BTC/Gold ratio is unavailable because aligned BTC and GOLD observations were not returned.")
    etf_flow_figure = _build_btc_etf_flow_intensity_figure(history, btc_etf_flow_history, time_range)
    if etf_flow_figure.data:
        st.plotly_chart(
            etf_flow_figure,
            use_container_width=True,
            config=BTC_CYCLE_PLOTLY_CONFIG,
        )
    else:
        st.info("No BTC ETF flow history for the selected time range.")
    _render_btc_halving_price_forecast()
    _render_halving_diagnostic(current)
    _render_liquidity_diagnostic(current)
    _render_secondary_diagnostic(current)
    _render_model_details(current)
    st.download_button(
        "Download BTC Cycle.xlsx",
        data=btc_cycle_export_xlsx(history, next_cycle_forecast),
        file_name="btc_cycle.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="btc_cycle_xlsx_download",
    )


@st.cache_data(show_spinner=False, ttl=21600)
def _load_btc_cycle_gold_weekly() -> pd.Series:
    from gold_regime.service import load_gold_mcp_weekly

    return load_gold_mcp_weekly()


def _halving_forecast_from_session_state() -> dict[str, Any]:
    return build_btc_halving_price_forecast(
        st.session_state.get("btc_bottom_price_2026", DEFAULT_BTC_BOTTOM_2026),
        {
            "Conservative": st.session_state.get(
                "btc_halving_top_conservative", DEFAULT_HALVING_TO_TOP_MULTIPLIERS["Conservative"]
            ),
            "Base": st.session_state.get("btc_halving_top_base", DEFAULT_HALVING_TO_TOP_MULTIPLIERS["Base"]),
            "Strong liquidity": st.session_state.get(
                "btc_halving_top_strong_liquidity", DEFAULT_HALVING_TO_TOP_MULTIPLIERS["Strong liquidity"]
            ),
        },
    )


def _render_btc_halving_price_forecast() -> dict[str, Any]:
    tooltip = (
        "BTC Halving Price Forecast estimates the next cycle top in two stages. First, the 2026 bottom is "
        "projected to the 2028 halving using the historical average Bottom → Halving multiple. Second, three "
        "Halving → Top scenarios are applied to estimate the next cycle top."
    )
    st.markdown(
        '<h3 style="margin: 0.6rem 0 0.35rem; font-size: 1.15rem; font-weight: 700;">'
        f'BTC Halving Price Forecast <span title="{html.escape(tooltip, quote=True)}" '
        'aria-label="Forecast methodology" style="font-size: 0.8rem; color: #94a3b8; cursor: help;">ⓘ</span></h3>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Historical return multiples**")
        _render_forecast_table(historical_return_table(), height=160)
    with right:
        st.markdown("**Historical cycle timing**")
        _render_forecast_table(historical_timing_table(), height=160)

    historical_average = build_btc_halving_price_forecast()["BTC_Avg_BottomToHalving"]
    average_row = pd.DataFrame(
        [{"Assumption": "Average Bottom → Halving", "Value": f"{historical_average:.2f}x"}]
    )
    _render_forecast_table(average_row, height=82)

    st.markdown("**Halving → Top**")
    conservative_col, base_col, liquidity_col = st.columns(3)
    with conservative_col:
        conservative = st.number_input(
            "Conservative",
            min_value=0.01,
            max_value=25.0,
            value=float(DEFAULT_HALVING_TO_TOP_MULTIPLIERS["Conservative"]),
            step=0.05,
            format="%.2f",
            key="btc_halving_top_conservative",
        )
    with base_col:
        base = st.number_input(
            "Base",
            min_value=0.01,
            max_value=25.0,
            value=float(DEFAULT_HALVING_TO_TOP_MULTIPLIERS["Base"]),
            step=0.05,
            format="%.2f",
            key="btc_halving_top_base",
        )
    with liquidity_col:
        strong_liquidity = st.number_input(
            "Strong liquidity",
            min_value=0.01,
            max_value=25.0,
            value=float(DEFAULT_HALVING_TO_TOP_MULTIPLIERS["Strong liquidity"]),
            step=0.05,
            format="%.2f",
            key="btc_halving_top_strong_liquidity",
        )

    st.markdown("**Price Forecast**")
    bottom_label, bottom_input = st.columns([3, 1])
    with bottom_label:
        st.markdown("Bottom price 2026")
    with bottom_input:
        bottom_price = st.number_input(
            "Bottom price 2026",
            min_value=1,
            value=int(DEFAULT_BTC_BOTTOM_2026),
            step=1000,
            format="%d",
            label_visibility="collapsed",
            key="btc_bottom_price_2026",
        )

    forecast = build_btc_halving_price_forecast(
        bottom_price,
        {
            "Conservative": conservative,
            "Base": base,
            "Strong liquidity": strong_liquidity,
        },
    )
    st.session_state["btc_halving_price_forecast"] = forecast.copy()
    forecast_rows = pd.DataFrame(
        [
            {"Price Forecast": "Price at halving", "USD": _format_usd(forecast["BTC_PriceAtHalving"])},
            {"Price Forecast": "Cycle top price", "USD": ""},
            {"Price Forecast": "    Conservative decay", "USD": _format_usd(forecast["BTC_CycleTop_Conservative"])},
            {"Price Forecast": "    Base", "USD": _format_usd(forecast["BTC_CycleTop_Base"])},
            {"Price Forecast": "    Strong liquidity", "USD": _format_usd(forecast["BTC_CycleTop_StrongLiquidity"])},
            {"Price Forecast": "Average forecast", "USD": _format_usd(forecast["BTC_CycleTop_Average"])},
        ]
    )
    _render_forecast_table(forecast_rows, height=275, highlight_metric="Average forecast")
    st.caption(_halving_forecast_summary(forecast))
    return forecast


def _render_next_cycle_controls() -> str:
    tooltip = (
        "The Next Cycle model is based on the average log-price shape of two historical BTC modular cycles: "
        "02 Nov 2018–25 Nov 2022 and 27 Sep 2022–26 Jun 2026. The historical shape is normalized and rescaled "
        "so that the projected cycle peak equals the selected BTC Halving Price Forecast scenario target."
    )
    title, selector = st.columns([3, 1])
    with title:
        st.markdown(
            '<h3 style="margin: 0.6rem 0 0.35rem; font-size: 1.15rem; font-weight: 700;">'
            f'Next Cycle <span title="{html.escape(tooltip, quote=True)}" aria-label="Next Cycle methodology" '
            'style="font-size: 0.8rem; color: #94a3b8; cursor: help;">ⓘ</span></h3>',
            unsafe_allow_html=True,
        )
    with selector:
        return st.selectbox(
            "Forecast scenario",
            ["Conservative", "Base", "Strong Liquidity"],
            index=1,
            key="btc_next_cycle_scenario",
        )


def _render_forecast_table(
    frame: pd.DataFrame,
    height: int,
    highlight_metric: str | None = None,
) -> None:
    styler = frame.style.set_properties(
        **{
            "padding": "3px 7px",
            "border-bottom": "1px solid #263241",
            "white-space": "nowrap",
            "font-size": "0.82rem",
        }
    ).set_table_styles(
        [
            {"selector": "th", "props": [("border-bottom", "1px solid #64748b"), ("font-weight", "700")]},
            {"selector": "td", "props": [("border-right", "1px solid #263241")]},
        ]
    )
    if highlight_metric is not None:
        styler = styler.apply(
            lambda row: [
                "background-color: #16324a; color: #eff6ff; font-weight: 700"
                if row.iloc[0] == highlight_metric
                else ""
                for _ in row
            ],
            axis=1,
        )
    st.dataframe(styler, hide_index=True, use_container_width=True, height=height)


def _halving_forecast_summary(forecast: dict[str, Any]) -> str:
    bottom_k = forecast["BTC_Bottom_2026"] / 1000.0
    halving_k = forecast["BTC_PriceAtHalving"] / 1000.0
    cycle_tops_k = [
        forecast["BTC_CycleTop_Conservative"] / 1000.0,
        forecast["BTC_CycleTop_Base"] / 1000.0,
        forecast["BTC_CycleTop_StrongLiquidity"] / 1000.0,
    ]
    low_k, high_k = min(cycle_tops_k), max(cycle_tops_k)
    base_k = forecast["BTC_CycleTop_Base"] / 1000.0
    multipliers = [
        forecast["BTC_HalvingToTop_Conservative"],
        forecast["BTC_HalvingToTop_Base"],
        forecast["BTC_HalvingToTop_StrongLiquidity"],
    ]
    return (
        f"Based on an assumed 2026 bottom of ${bottom_k:,.1f}k and the historical average Bottom → Halving "
        f"multiple of {forecast['BTC_Avg_BottomToHalving']:.2f}x, the model estimates a BTC price near "
        f"${halving_k:,.0f}k at the 2028 halving. Applying Halving → Top scenarios of "
        f"{min(multipliers):.2f}x–{max(multipliers):.2f}x "
        f"gives a projected cycle-top range of approximately ${low_k:,.0f}k–${high_k:,.0f}k, with a base case "
        f"near ${base_k:,.0f}k."
    )


def _format_usd(value: float) -> str:
    return f"{value:,.0f}"


def _render_score_cards(current: pd.Series) -> None:
    columns = st.columns(4)
    for column, horizon in zip(columns, BTC_CYCLE_HORIZONS):
        with column:
            score = current.get(f"BTC_MACRO_{horizon}")
            state = str(current.get(f"BTC_MACRO_{horizon}_State", "DATA_INCOMPLETE"))
            value = "n/a" if not np.isfinite(_number(score)) else f"{_number(score):.1f} / 95"
            st.metric(horizon, value, state)
            st.caption(
                f"Halving: {_phase_label(current.get(f'HalvingPhase_{horizon}'))}\n\n"
                f"Liquidity: {_phase_label(current.get(f'LiquidityPhase_{horizon}'))}"
            )


def _score_table(current: pd.Series) -> pd.DataFrame:
    rows = []
    for horizon in BTC_CYCLE_HORIZONS:
        halving_phase = str(current.get(f"HalvingPhase_{horizon}", "DATA_INCOMPLETE"))
        liquidity_phase = str(current.get(f"LiquidityPhase_{horizon}", "DATA_INCOMPLETE"))
        score = _number(current.get(f"BTC_MACRO_{horizon}"))
        macro = _number(current.get(f"SecondaryMacro_{horizon}"))
        rows.append(
            {
                "Horizon": horizon,
                "Halving Phase": _phase_label(halving_phase),
                "Halving Base": _fmt(current.get(f"HalvingBase_{horizon}")),
                "Liquidity Phase": _phase_label(liquidity_phase),
                "Liquidity Modifier": _fmt_signed(current.get(f"LiquidityCycleModifier_{horizon}")),
                "Secondary Macro": _fmt(macro),
                "Secondary Modifier": _fmt_signed(current.get(f"SecondaryMacroModifier_{horizon}")),
                "Final BTC Macro Score": _fmt(score),
                "State": current.get(f"BTC_MACRO_{horizon}_State", "DATA_INCOMPLETE"),
                "Description": _description(halving_phase, liquidity_phase, macro),
            }
        )
    return pd.DataFrame(rows)


def _description(halving: str, liquidity: str, secondary: float) -> str:
    macro_tone = "supportive" if secondary > 55 else "restrictive" if secondary < 45 else "mixed"
    return (
        f"BTC is in {_phase_label(halving)} while Global M2 is expected to be in "
        f"{_phase_label(liquidity)}; secondary macro conditions are {macro_tone}."
    )


def _render_drivers(current: pd.Series) -> None:
    horizon = "3M"
    weights = SECONDARY_WEIGHTS[horizon]
    candidates = [
        ("Halving cycle", _number(current.get(f"HalvingBase_{horizon}")) - 50.0, _phase_label(current.get(f"HalvingPhase_{horizon}"))),
        ("Global M2 cycle", _number(current.get(f"LiquidityCycleModifier_{horizon}")), _phase_label(current.get(f"LiquidityPhase_{horizon}"))),
        ("DXY", weights[0] * (_number(current.get("DXYBull")) - 50.0), _direction_label(current.get("DXYBull"), "weaker USD")),
        ("US2Y", weights[1] * (_number(current.get("US2YBull")) - 50.0), _direction_label(current.get("US2YBull"), "falling short rates")),
        ("Real yields", weights[2] * (_number(current.get("RealYieldBull")) - 50.0), _direction_label(current.get("RealYieldBull"), "falling real yields")),
    ]
    candidates = [item for item in candidates if np.isfinite(item[1]) and item[1] != 0]
    positive = sorted((item for item in candidates if item[1] > 0), key=lambda item: item[1], reverse=True)
    negative = sorted((item for item in candidates if item[1] < 0), key=lambda item: abs(item[1]), reverse=True)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Positive Drivers")
        _driver_list(positive)
    with right:
        st.markdown("#### Negative Drivers")
        _driver_list(negative)


def _driver_list(drivers: list[tuple[str, float, str]]) -> None:
    if not drivers:
        st.caption("No material drivers in this direction.")
        return
    for name, contribution, detail in drivers[:4]:
        st.markdown(f"**{name}** · {detail} · {contribution:+.1f}")


def _build_btc_price_halving_figure(
    history: pd.DataFrame,
    time_range: str = "MAX",
    next_cycle_forecast: pd.DataFrame | None = None,
) -> go.Figure:
    range_start, range_end, include_forecast = btc_cycle_time_range(history, time_range)
    show_next_cycle = include_forecast and next_cycle_forecast is not None and not next_cycle_forecast.empty
    if show_next_cycle:
        range_end = pd.Timestamp(next_cycle_forecast["NextCycle_EndDate"].iloc[0])
    visible = history.loc[history["Date"].between(range_start, range_end)].copy()
    if not include_forecast:
        visible = visible.loc[~visible["Projected"].astype(bool)]
    observed = visible.loc[~visible["Projected"].astype(bool)].dropna(subset=["BTC_Price"]).copy()
    latest_date = pd.Timestamp(observed["Date"].max())
    fig = go.Figure()
    _add_phase_bands(fig, visible, "Current_Halving_Phase", BTC_CYCLE_COLORS, range_end, opacity=0.12)
    fig.add_trace(
        go.Scatter(
            x=observed["Date"],
            y=observed["BTC_Price"],
            mode="lines",
            name="BTC Actual",
            line={"color": "#f7931a", "width": 2.0},
            customdata=observed[["Current_Halving_Phase", "Current_Halving_Progress"]].to_numpy(),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>BTC: $%{y:,.0f}<br>Halving phase: %{customdata[0]}<br>Progress: %{customdata[1]:.1f}%<extra></extra>",
        )
    )
    liquidity = observed.dropna(subset=["GlobalLiquidityScore"])
    if not liquidity.empty:
        fig.add_trace(
            go.Scatter(
                x=liquidity["Date"],
                y=liquidity["GlobalLiquidityScore"],
                mode="lines",
                name="Global Liquidity Score",
                yaxis="y2",
                line={"color": "#38bdf8", "width": 1.8},
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Global Liquidity Score: %{y:.1f}<extra></extra>",
            )
        )
    if show_next_cycle:
        model = next_cycle_forecast.copy()
        scenario = str(model["NextCycle_Scenario"].iloc[0])
        peak_date = pd.Timestamp(model["NextCycle_PeakDate"].iloc[0])
        peak_price = float(model["NextCycle_PeakPrice"].iloc[0])
        end_date = pd.Timestamp(model["NextCycle_EndDate"].iloc[0])
        fig.add_vrect(
            x0=latest_date,
            x1=end_date,
            fillcolor="#94a3b8",
            opacity=0.045,
            line_width=0,
            layer="below",
        )
        model_tooltip_columns = [
            "NextCycle_ProgressPct",
            "HalvingPhase",
            "DistanceToProjectedHalvingDays",
            "DistanceToModelCycleTopDays",
            "HistoricalProjectedFlag",
            "ProgressSince2028HalvingTooltip",
        ]
        fig.add_trace(
            go.Scatter(
                x=model["Date"],
                y=model["NextCycle_ModelPrice"],
                mode="lines",
                name=f"Next Cycle Model — {scenario}",
                opacity=0.88,
                line={"color": "#38bdf8", "width": 2.2, "dash": "dash"},
                customdata=model[model_tooltip_columns].to_numpy(),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>Model BTC Price: $%{y:,.0f}<br>"
                    "Model Cycle Progress: %{customdata[0]:.1f}%<br>Halving Cycle Phase: %{customdata[1]}<br>"
                    "Distance to Projected Halving: %{customdata[2]:.0f} days<br>"
                    "Distance to Model Cycle Top: %{customdata[3]:.0f} days<br>"
                    "Historical / Projected: %{customdata[4]}<br>"
                    "Progress since 2028 Halving: %{customdata[5]}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[peak_date],
                y=[peak_price],
                mode="markers",
                name="Model Cycle Top",
                marker={"color": "#facc15", "size": 9, "line": {"color": "#0f131a", "width": 1}},
                hovertemplate="Model Cycle Top: %{x|%d %b %Y}<br>Price: $%{y:,.0f}<extra></extra>",
            )
        )
        fig.add_annotation(
            x=peak_date,
            y=peak_price,
            text=f"Model Cycle Top<br>{peak_date:%d %b %Y} · ${peak_price:,.0f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-38,
            font={"size": 9, "color": "#fde68a"},
        )
        fig.add_trace(
            go.Scatter(
                x=[end_date],
                y=[float(model["NextCycle_ModelPrice"].iloc[-1])],
                mode="markers",
                name="Next Modular Cycle End",
                marker={"color": "#cbd5e1", "size": 7, "symbol": "diamond"},
                hovertemplate="Next Modular Cycle End: %{x|%d %b %Y}<br>Model price: $%{y:,.0f}<extra></extra>",
            )
        )
        fig.add_annotation(
            x=end_date,
            y=0.99,
            yref="paper",
            text="PROJECTED",
            showarrow=False,
            xanchor="right",
            font={"size": 9, "color": "#cbd5e1"},
        )
    for event in pd.DatetimeIndex(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]):
        if event >= observed["Date"].min():
            fig.add_vline(x=event, line={"color": "#38bdf8", "width": 1, "dash": "dot"})
            fig.add_annotation(x=event, y=1, yref="paper", text=f"Halving {event.year}", showarrow=False, yanchor="bottom", font={"size": 9, "color": "#bae6fd"})
    if range_start <= BTC_PROJECTED_HALVING <= range_end:
        fig.add_vline(x=BTC_PROJECTED_HALVING, line={"color": "#a78bfa", "width": 1.5, "dash": "dash"})
        fig.add_annotation(x=BTC_PROJECTED_HALVING, y=0.96, yref="paper", text="Projected Halving · Apr 2028", showarrow=False, yanchor="top", font={"size": 10, "color": "#ddd6fe"})
        if show_next_cycle:
            model_dates = pd.to_datetime(next_cycle_forecast["Date"])
            log_price_at_halving = np.interp(
                BTC_PROJECTED_HALVING.value,
                model_dates.astype("int64").to_numpy(),
                np.log(pd.to_numeric(next_cycle_forecast["NextCycle_ModelPrice"], errors="coerce").to_numpy()),
            )
            model_price_at_halving = float(np.exp(log_price_at_halving))
            fig.add_trace(
                go.Scatter(
                    x=[BTC_PROJECTED_HALVING],
                    y=[model_price_at_halving],
                    mode="markers",
                    name="Model Price at Halving",
                    showlegend=False,
                    marker={"color": "#c4b5fd", "size": 7, "symbol": "diamond"},
                    hovertemplate="Projected Halving · Apr 2028<br>Model BTC Price: $%{y:,.0f}<extra></extra>",
                )
            )
    _add_latest_marker(fig, latest_date)
    fig.update_yaxes(type="log", title="BTC USD")
    if len(fig.data) > 1:
        fig.update_layout(
            yaxis2={
                "title": "Global Liquidity Score",
                "overlaying": "y",
                "side": "right",
                "range": [0, 100],
                "showgrid": False,
                "color": "#38bdf8",
            },
            margin={"l": 58, "r": 80, "t": 55, "b": 45},
        )
    fig.update_xaxes(range=[range_start, range_end], tickformat="%Y")
    return _style_btc_cycle_fig(fig, "BTC — Log Price & Halving Cycle", 440)


def _build_structural_cycles_figure(history: pd.DataFrame, time_range: str = "MAX") -> go.Figure:
    range_start, range_end, include_forecast = btc_cycle_time_range(history, time_range)
    visible = history.loc[history["Date"].between(range_start, range_end)].copy()
    if not include_forecast:
        visible = visible.loc[~visible["Projected"].astype(bool)]
    observed = visible.loc[~visible["Projected"].astype(bool)].copy()
    latest_date = pd.Timestamp(observed["Date"].max())
    fig = go.Figure()
    _add_phase_bands(fig, visible, "Current_Halving_Phase", BTC_CYCLE_COLORS, range_end, opacity=0.08)
    projected_phases = visible.loc[visible["Projected"].astype(bool)]
    if include_forecast:
        _add_phase_bands(fig, projected_phases, "Current_Liquidity_Phase", LIQUIDITY_PHASE_COLORS, range_end, opacity=0.22)
    line = visible.dropna(subset=["GlobalM2PrimaryCycle"])
    actual_line = line.loc[~line["GlobalM2CycleProjected"].astype(bool)]
    projected_line = line.loc[line["GlobalM2CycleProjected"].astype(bool)] if include_forecast else line.iloc[0:0]
    if not actual_line.empty:
        fig.add_trace(
            go.Scatter(
                x=actual_line["Date"],
                y=actual_line["GlobalM2PrimaryCycle"],
                mode="lines",
                name="Global M2 Primary Liquidity Cycle",
                line={"color": "#38bdf8", "width": 2.2},
                hovertemplate="Date: %{x|%Y-%m}<br>Global M2 Primary Liquidity Cycle: %{y:.2f}<extra></extra>",
            )
        )
    if not projected_line.empty:
        fig.add_trace(
            go.Scatter(
                x=projected_line["Date"],
                y=projected_line["GlobalM2PrimaryCycle"],
                mode="lines",
                name="Projected Global M2 Cycle",
                line={"color": "#38bdf8", "width": 2.2, "dash": "dash"},
                hovertemplate="Date: %{x|%Y-%m}<br>Projected Global M2 Primary Liquidity Cycle: %{y:.2f}<extra></extra>",
            )
        )
    trough = liquidity_cycle_position(latest_date, latest_date)["next_trough"]
    if include_forecast and pd.notna(trough) and latest_date < trough <= range_end:
        fig.add_vline(x=trough, line={"color": "#facc15", "width": 1.5, "dash": "dash"})
        fig.add_annotation(x=trough, y=0.98, yref="paper", text="Projected Liquidity Trough", showarrow=False, yanchor="top", font={"size": 10, "color": "#fde68a"})
    _add_latest_marker(fig, latest_date)
    if include_forecast:
        fig.add_annotation(x=latest_date + (range_end - latest_date) / 2, y=0.05, yref="paper", text="PROJECTED CYCLE", showarrow=False, font={"size": 10, "color": "#cbd5e1"})
    fig.add_hline(y=0, line={"color": "#64748b", "dash": "dot", "width": 1})
    fig.update_yaxes(title="Normalized Global M2 cycle")
    fig.update_xaxes(range=[range_start, range_end], tickformat="%Y")
    return _style_btc_cycle_fig(fig, "BTC Structural Cycles — Halving vs Global M2 Liquidity", 390)


def _build_macro_score_figure(history: pd.DataFrame, horizon: str, time_range: str = "MAX") -> go.Figure:
    range_start, range_end, include_forecast = btc_cycle_time_range(history, time_range)
    visible = history.loc[history["Date"].between(range_start, range_end)].copy()
    historical = visible.loc[~visible["Projected"].astype(bool)].copy()
    projected = visible.loc[visible["Projected"].astype(bool)].copy() if include_forecast else visible.iloc[0:0]
    score_column = f"BTC_MACRO_{horizon}"
    tooltip_columns = [
        "BTC_Price",
        f"HalvingPhase_{horizon}",
        f"HalvingBase_{horizon}",
        f"LiquidityPhase_{horizon}",
        f"LiquidityCycleModifier_{horizon}",
        f"SecondaryMacro_{horizon}",
        f"SecondaryMacroModifier_{horizon}",
        f"BTC_MACRO_{horizon}_State",
        "Historical_Projected_Flag",
    ]
    fig = go.Figure()
    for low, high, color in SCORE_BANDS:
        fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.10, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=historical["Date"],
            y=historical[score_column],
            mode="lines",
            name=f"BTC Macro {horizon}",
            line={"color": "#38bdf8", "width": 2.2},
            customdata=historical[tooltip_columns].to_numpy(),
            hovertemplate=_score_hover_template(horizon),
        )
    )
    if not projected.empty:
        fig.add_trace(
            go.Scatter(
                x=projected["Date"],
                y=projected[score_column],
                mode="lines",
                name="Projected",
                line={"color": "#38bdf8", "width": 2.2, "dash": "dash"},
                customdata=projected[tooltip_columns].to_numpy(),
                hovertemplate=_score_hover_template(horizon),
            )
        )
    _add_latest_marker(fig, pd.Timestamp(historical["Date"].max()))
    fig.update_yaxes(range=[0, 100], title="Score")
    fig.update_xaxes(range=[range_start, range_end], tickformat="%Y")
    return _style_btc_cycle_fig(fig, "BTC Macro Score", 360)


def _build_btc_gold_ratio_figure(
    history: pd.DataFrame,
    ratio_history: pd.DataFrame,
    time_range: str = "MAX",
) -> go.Figure:
    range_start, range_end, _ = btc_cycle_time_range(history, time_range)
    visible = ratio_history.loc[ratio_history["Date"].between(range_start, range_end)].copy()
    fig = go.Figure()
    if not visible.empty:
        fig.add_trace(
            go.Scatter(
                x=visible["Date"],
                y=visible["BTC_GOLD_Ratio"],
                mode="lines",
                name="BTC / Gold",
                line={"color": "#facc15", "width": 2.1},
                customdata=visible[["BTC_Price", "Gold_Price"]].to_numpy(),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>BTC / Gold: %{y:,.2f} oz per BTC<br>"
                    "BTC: $%{customdata[0]:,.0f}<br>Gold: $%{customdata[1]:,.2f}/oz<extra></extra>"
                ),
            )
        )
    fig.update_yaxes(title="Gold oz per BTC")
    fig.update_xaxes(range=[range_start, range_end], tickformat="%Y")
    return _style_btc_cycle_fig(fig, "BTC / Gold", 340)


def _build_btc_etf_flow_intensity_figure(
    history: pd.DataFrame,
    flow_history: pd.DataFrame | None,
    time_range: str = "MAX",
) -> go.Figure:
    range_start, range_end, _ = btc_cycle_time_range(history, time_range)
    required = {"date", "ETF_Flow_Intensity_4W", "ETF_Flow_3Y_Pctl"}
    data = pd.DataFrame(columns=sorted(required))
    if flow_history is not None and not flow_history.empty and required.issubset(flow_history.columns):
        data = flow_history.loc[:, ["date", "ETF_Flow_Intensity_4W", "ETF_Flow_3Y_Pctl"]].copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True).dt.tz_localize(None)
        for column in ("ETF_Flow_Intensity_4W", "ETF_Flow_3Y_Pctl"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=["date"]).loc[lambda frame: frame["date"].between(range_start, range_end)]
        data = data.sort_values("date")

    fig = go.Figure()
    if not data.empty:
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["ETF_Flow_Intensity_4W"],
                mode="lines",
                name="4W Flow Intensity",
                line={"color": "#22d3ee", "width": 1.8},
                hovertemplate="Week: %{x|%Y-%m-%d}<br>4W Flow Intensity: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["ETF_Flow_3Y_Pctl"],
                mode="lines",
                name="3Y Percentile",
                yaxis="y2",
                line={"color": "#facc15", "width": 1.8, "dash": "dash"},
                hovertemplate="Week: %{x|%Y-%m-%d}<br>3Y Percentile: %{y:.0f}<extra></extra>",
            )
        )
        fig.add_hline(y=0, line={"color": "#64748b", "dash": "dot", "width": 1})

    fig.update_xaxes(range=[range_start, range_end], tickformat="%Y")
    fig.update_yaxes(title="4W Flow Intensity, normalized")
    fig = _style_btc_cycle_fig(fig, "BTC ETF Fund Flows - 4W Flow Intensity and Trailing 3Y Percentile", 320)
    fig.update_layout(
        yaxis2={
            "title": "Trailing 3Y Percentile",
            "overlaying": "y",
            "side": "right",
            "range": [0, 100],
            "showgrid": False,
            "color": "#cbd5e1",
            "linecolor": "#475569",
        },
        margin={"l": 58, "r": 82, "t": 62, "b": 45},
    )
    if not data.empty:
        latest_date = pd.Timestamp(data["date"].max())
        fig.add_annotation(
            text=f"Last Updated: {latest_date:%Y-%m-%d} - CURRENT",
            xref="paper",
            yref="paper",
            x=0,
            y=1.12,
            showarrow=False,
            font={"color": "#cbd5e1", "size": 10},
            xanchor="left",
        )
    return fig


def _score_hover_template(horizon: str) -> str:
    return (
        "Date: %{x|%Y-%m-%d}<br>BTC Price: $%{customdata[0]:,.0f}<br>"
        f"BTC Macro {horizon}: %{{y:.1f}}<br>"
        "Halving Phase: %{customdata[1]}<br>Halving Base: %{customdata[2]:.1f}<br>"
        "Liquidity Phase: %{customdata[3]}<br>Liquidity Modifier: %{customdata[4]:+.1f}<br>"
        "Secondary Macro: %{customdata[5]:.1f}<br>Secondary Modifier: %{customdata[6]:+.1f}<br>"
        "Final State: %{customdata[7]}<br>Type: %{customdata[8]}<extra></extra>"
    )


def _add_phase_bands(
    fig: go.Figure,
    frame: pd.DataFrame,
    phase_column: str,
    colors: dict[str, str],
    end_date: pd.Timestamp,
    opacity: float,
) -> None:
    if frame is None or frame.empty or phase_column not in frame:
        return
    data = frame[["Date", phase_column]].dropna().sort_values("Date").reset_index(drop=True)
    if data.empty:
        return
    group_id = data[phase_column].ne(data[phase_column].shift()).cumsum()
    for _, group in data.groupby(group_id, sort=False):
        phase = str(group[phase_column].iloc[0])
        color = colors.get(phase)
        if not color:
            continue
        start = pd.Timestamp(group["Date"].iloc[0])
        end = pd.Timestamp(data.loc[group.index[-1] + 1, "Date"]) if group.index[-1] + 1 < len(data) else end_date
        fig.add_vrect(x0=start, x1=min(end, end_date), fillcolor=color, opacity=opacity, line_width=0, layer="below")


def _add_latest_marker(fig: go.Figure, latest_date: pd.Timestamp) -> None:
    fig.add_vline(x=latest_date, line={"color": "#f8fafc", "width": 1, "dash": "dot"})
    fig.add_annotation(x=latest_date, y=0.03, yref="paper", text="Latest Data", showarrow=False, yanchor="bottom", font={"size": 9, "color": "#f8fafc"})


def _style_btc_cycle_fig(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="#0f131a",
        plot_bgcolor="#0f131a",
        font={"color": "#e5e7eb", "size": 11},
        margin={"l": 58, "r": 30, "t": 55, "b": 45},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "top", "y": -0.14, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(showgrid=False, color="#cbd5e1", linecolor="#475569", ticks="outside")
    fig.update_yaxes(showgrid=True, gridcolor="#263241", color="#cbd5e1", linecolor="#475569", ticks="outside")
    return fig


def _render_halving_diagnostic(current: pd.Series) -> None:
    st.markdown("### Halving Diagnostic")
    rows = [
        ("Current Halving Phase", _phase_label(current.get("Current_Halving_Phase"))),
        ("Current Halving Progress", f"{_fmt(current.get('Current_Halving_Progress'))}%"),
        ("Previous Halving", _date(current.get("Previous_Halving"))),
        ("Next Projected Halving", "Apr 2028"),
    ]
    rows.extend((f"Expected {horizon}", _phase_label(current.get(f"HalvingPhase_{horizon}"))) for horizon in BTC_CYCLE_HORIZONS)
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]), hide_index=True, use_container_width=True)


def _render_liquidity_diagnostic(current: pd.Series) -> None:
    st.markdown("### Liquidity Cycle Diagnostic")
    latest_date = pd.Timestamp(current["Date"])
    position = liquidity_cycle_position(latest_date, latest_date)
    rows = [
        ("Canonical Series", "Global M2 Primary Liquidity Cycle"),
        ("Current Liquidity Phase", _phase_label(current.get("Current_Liquidity_Phase"))),
        ("Current Cycle Progress", f"{_fmt(current.get('Current_Liquidity_Progress'))}%"),
        ("Current Trough Anchor", _date(position.get("anchor"))),
        ("Cycle Length", f"{BTC_LIQUIDITY_CYCLE_MONTHS} months"),
        ("Projected Next Trough", _date(position.get("next_trough"))),
    ]
    rows.extend((f"Expected {horizon}", _phase_label(current.get(f"LiquidityPhase_{horizon}"))) for horizon in BTC_CYCLE_HORIZONS)
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]), hide_index=True, use_container_width=True)


def _render_secondary_diagnostic(current: pd.Series) -> None:
    st.markdown("### Secondary Macro Diagnostic")
    rows = [
        ("DXYBull", _fmt(current.get("DXYBull"))),
        ("US2YBull", _fmt(current.get("US2YBull"))),
        ("RealYieldBull", _fmt(current.get("RealYieldBull"))),
    ]
    for horizon in BTC_CYCLE_HORIZONS:
        rows.append((f"SecondaryMacro_{horizon}", _fmt(current.get(f"SecondaryMacro_{horizon}"))))
        rows.append((f"SecondaryMacroModifier_{horizon}", _fmt_signed(current.get(f"SecondaryMacroModifier_{horizon}"))))
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]), hide_index=True, use_container_width=True)


def _render_model_details(current: pd.Series) -> None:
    with st.expander("Model Details"):
        rows = []
        for horizon in BTC_CYCLE_HORIZONS:
            rows.append(
                {
                    "Horizon": horizon,
                    "Halving Phase": current.get(f"HalvingPhase_{horizon}"),
                    "Halving Base": current.get(f"HalvingBase_{horizon}"),
                    "Liquidity Phase": current.get(f"LiquidityPhase_{horizon}"),
                    "Liquidity Modifier": current.get(f"LiquidityCycleModifier_{horizon}"),
                    "DXYBull": current.get("DXYBull"),
                    "US2YBull": current.get("US2YBull"),
                    "RealYieldBull": current.get("RealYieldBull"),
                    "Secondary Macro": current.get(f"SecondaryMacro_{horizon}"),
                    "Secondary Modifier": current.get(f"SecondaryMacroModifier_{horizon}"),
                    "Score before clamp": current.get(f"BTC_MACRO_{horizon}_BeforeClamp"),
                    "Final Score": current.get(f"BTC_MACRO_{horizon}"),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _direction_label(value: Any, favorable: str) -> str:
    score = _number(value)
    if not np.isfinite(score):
        return "unavailable"
    return favorable if score >= 55 else "unfavorable" if score <= 45 else "mixed"


def _phase_label(value: Any) -> str:
    return str(value or "DATA_INCOMPLETE").replace("_", " ").title()


def _fmt(value: Any) -> str:
    number = _number(value)
    return "n/a" if not np.isfinite(number) else f"{number:.1f}"


def _fmt_signed(value: Any) -> str:
    number = _number(value)
    return "n/a" if not np.isfinite(number) else f"{number:+.1f}"


def _date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "n/a" if pd.isna(parsed) else pd.Timestamp(parsed).strftime("%d-%b-%Y")


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
