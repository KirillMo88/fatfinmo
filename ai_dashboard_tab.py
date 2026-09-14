from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai_dashboard import (
    AI_UNIVERSE,
    AIDashboardData,
    add_relative_benchmark_returns,
    ai_breadth_history,
    ai_tickers,
    calculate_ai_dashboard,
    group_components_table,
    group_rows_for_benchmark,
    load_ai_fundamentals,
    load_ai_price_history,
    normalized_group_history,
    normalized_ticker_history,
    period_start_date,
)


AI_PRICE_TTL_SECONDS = 21600
AI_FUNDAMENTALS_TTL_SECONDS = 43200
AI_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
PERF_COLS = ["Perf 1D", "Perf 1W", "Perf 1M", "Perf 3M", "Perf 6M", "Perf 12M", "Perf 3Y", "Perf 5Y", "Perf 10Y"]
MOMENTUM_PERFORMANCE_OPTIONS = {
    "1M": "Perf 1M",
    "3M": "Perf 3M",
    "6M": "Perf 6M",
    "12M": "Perf 12M",
}
RELATIVE_COLS = [
    "Relative Group 1M",
    "Relative Group 3M",
    "Relative Group 12M",
    "Relative Benchmark 1M",
    "Relative Benchmark 3M",
    "Relative Benchmark 6M",
    "Relative Benchmark 12M",
]
RATIO_PERCENT_COLS = [
    "Price vs SMA200D",
    "Quarterly Revenue Growth YoY",
    "Revenue Growth 3Y",
    "Operating Margin TTM",
    "Price vs ATH",
    "Median Quarterly Revenue Growth YoY",
    "Median Revenue Growth 3Y",
    "Median Operating Margin TTM",
    "Median Price vs ATH",
]
MONEY_COLS = ["Market Cap", "Total Market Cap"]
MULTIPLE_COLS = [
    "Trailing P/E",
    "Forward P/E",
    "PEG Ratio",
    "Price/Sales",
    "Median Trailing P/E",
    "Median Forward P/E",
    "Median PEG",
    "Median Price/Sales",
]
COUNT_COLS = ["Companies"]


@st.cache_data(show_spinner=True, ttl=AI_PRICE_TTL_SECONDS)
def load_ai_prices_cached(refresh_nonce: int = 0) -> dict[str, pd.DataFrame]:
    _ = refresh_nonce
    return load_ai_price_history(ai_tickers(include_benchmarks=True))


@st.cache_data(show_spinner=True, ttl=AI_FUNDAMENTALS_TTL_SECONDS)
def load_ai_fundamentals_cached(refresh_nonce: int = 0) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    _ = refresh_nonce
    return load_ai_fundamentals(ai_tickers())


@st.cache_data(show_spinner=True, ttl=AI_PRICE_TTL_SECONDS)
def build_ai_dashboard_cached(price_nonce: int = 0, fundamentals_nonce: int = 0) -> AIDashboardData:
    prices = load_ai_prices_cached(price_nonce)
    fundamentals, errors = load_ai_fundamentals_cached(fundamentals_nonce)
    return calculate_ai_dashboard(prices=prices, fundamentals=fundamentals, fundamentals_errors=errors)


def render_ai_dashboard_tab() -> None:
    st.subheader("AI Dashboard")
    if "ai_dashboard_refresh_nonce" not in st.session_state:
        st.session_state["ai_dashboard_refresh_nonce"] = 0
    if "ai_dashboard_fundamentals_refresh_nonce" not in st.session_state:
        st.session_state["ai_dashboard_fundamentals_refresh_nonce"] = 0

    c1, c2, c3 = st.columns([1.1, 1.1, 4.8])
    with c1:
        weighting = st.segmented_control(
            "Group weighting",
            ["Equal Weighted", "Market Cap Weighted"],
            default="Equal Weighted",
            key="ai_group_weighting",
        )
    with c2:
        benchmark = st.segmented_control("Benchmark", ["QQQ", "SPY"], default="QQQ", key="ai_benchmark")
    with c3:
        if st.button("Refresh AI Dashboard", use_container_width=False, key="ai_dashboard_refresh"):
            st.session_state["ai_dashboard_refresh_nonce"] += 1
            st.session_state["ai_dashboard_fundamentals_refresh_nonce"] += 1
            st.rerun()

    data = build_ai_dashboard_cached(
        st.session_state["ai_dashboard_refresh_nonce"],
        st.session_state["ai_dashboard_fundamentals_refresh_nonce"],
    )
    companies = add_relative_benchmark_returns(data.companies, data.benchmark_returns.get(benchmark, {}))
    groups = group_rows_for_benchmark(companies, weighting or "Equal Weighted", data.benchmark_returns.get(benchmark, {}))

    st.caption(f"Refresh status: {fmt_datetime(data.generated_at)} UTC · prices cached for 6 hours · fundamentals cached for 12 hours · benchmark: {benchmark}")
    if data.fundamentals_errors:
        with st.expander("Fundamentals warnings", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [{"Ticker": ticker, "Error": error} for ticker, error in sorted(data.fundamentals_errors.items())]
                ),
                use_container_width=True,
                hide_index=True,
            )

    overview_tab, groups_tab, companies_tab, fundamentals_tab, charts_tab = st.tabs(
        ["Overview", "Groups", "Companies", "Fundamentals", "Charts"]
    )
    with overview_tab:
        render_ai_overview(data, companies, groups, benchmark)
    with groups_tab:
        render_ai_groups(data, companies, groups)
    with companies_tab:
        render_ai_companies(companies)
    with fundamentals_tab:
        render_ai_fundamentals(companies, groups)
    with charts_tab:
        render_ai_charts(data, companies, groups, benchmark)


def render_ai_overview(data: AIDashboardData, companies: pd.DataFrame, groups: pd.DataFrame, benchmark: str) -> None:
    st.markdown("### AI Group Performance Heatmap")
    performance_table = groups[["Group", *PERF_COLS]].copy()
    st.dataframe(
        style_performance(performance_table, PERF_COLS),
        use_container_width=True,
        hide_index=True,
        height=dataframe_auto_height(performance_table),
    )

    st.markdown("### Group Condition")
    condition_cols = ["Group", "SMA200W Percentile", "SMA200D Robust Z 36M", "Price vs SMA200D", "% Above SMA50", "% Above SMA200", "% Positive 1M", "% Positive 3M", "% Positive 12M"]
    condition_table = groups[condition_cols].copy()
    st.dataframe(
        style_condition(condition_table),
        use_container_width=True,
        hide_index=True,
        height=dataframe_auto_height(condition_table),
    )

    st.markdown("### Group Fundamentals")
    fundamental_cols = [
        "Group",
        "Total Market Cap",
        "Median Forward P/E",
        "Median Price/Sales",
        "Median Quarterly Revenue Growth YoY",
        "Median Revenue Growth 3Y",
        "Median Operating Margin TTM",
        "Median Price vs ATH",
    ]
    fundamentals_table = groups[fundamental_cols].copy()
    st.dataframe(
        format_table(fundamentals_table),
        use_container_width=True,
        hide_index=True,
        height=dataframe_auto_height(fundamentals_table),
    )

    st.markdown(f"### AI Group Performance Chart vs {benchmark}")
    default_groups = groups.sort_values("Perf 3M", ascending=False, na_position="last")["Group"].head(5).tolist()
    selected_groups = st.multiselect(
        "Groups",
        list(AI_UNIVERSE.keys()),
        default=default_groups,
        key="ai_overview_groups",
    )
    period = st.radio("Period", ["1M", "3M", "6M", "12M", "3Y", "5Y"], index=3, horizontal=True, key="ai_overview_period")
    fig = build_group_performance_fig(data, selected_groups, [benchmark], period)
    st.plotly_chart(fig, use_container_width=True, config=AI_PLOTLY_CONFIG)

    leaders = companies.sort_values("Perf 3M", ascending=False, na_position="last").head(10)
    laggards = companies.sort_values("Perf 3M", ascending=True, na_position="last").head(10)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Current Leaders")
        leaders_table = leaders[["Group", "Company", "Ticker", "Perf 1M", "Perf 3M", "Perf 12M", "Relative Benchmark 3M"]].copy()
        st.dataframe(format_table(leaders_table), use_container_width=True, hide_index=True, height=dataframe_auto_height(leaders_table))
    with c2:
        st.markdown("### Current Laggards")
        laggards_table = laggards[["Group", "Company", "Ticker", "Perf 1M", "Perf 3M", "Perf 12M", "Relative Benchmark 3M"]].copy()
        st.dataframe(format_table(laggards_table), use_container_width=True, hide_index=True, height=dataframe_auto_height(laggards_table))


def render_ai_groups(data: AIDashboardData, companies: pd.DataFrame, groups: pd.DataFrame) -> None:
    _ = data
    st.markdown("### Groups")
    for idx, group_name in enumerate(AI_UNIVERSE.keys()):
        if idx:
            st.divider()
        render_ai_group_block(companies, groups, group_name)


def render_ai_group_block(companies: pd.DataFrame, groups: pd.DataFrame, group_name: str) -> None:
    selected = groups[groups["Group"].eq(group_name)].tail(1)
    if selected.empty:
        st.markdown(f"### {group_name}")
        st.info("No group data available.")
        return
    row = selected.iloc[0]
    st.markdown(f"### {group_name}")
    cols = st.columns(8)
    metrics = [
        ("Companies", fmt_number(row.get("Companies"), 0), ""),
        ("Total Market Cap", fmt_money(row.get("Total Market Cap")), ""),
        ("Return 1M", fmt_pct_points(row.get("Perf 1M")), ""),
        ("Return 3M", fmt_pct_points(row.get("Perf 3M")), ""),
        ("Return 12M", fmt_pct_points(row.get("Perf 12M")), ""),
        ("SMA200W %ile", fmt_number(row.get("SMA200W Percentile"), 0), overextension_state(row.get("SMA200W Percentile"))),
        ("Robust Z", fmt_number(row.get("SMA200D Robust Z 36M"), 2), ""),
        ("% Above SMA200", fmt_pct_points(row.get("% Above SMA200")), ""),
    ]
    for col, (label, value, detail) in zip(cols, metrics):
        with col:
            render_metric(label, value, detail)

    table = group_components_table(companies, groups, group_name)
    cols_to_show = [
        "Company",
        "Ticker",
        "Market Cap",
        *PERF_COLS,
        "Price vs SMA200D",
        "SMA200D Robust Z 36M",
        "SMA200W Percentile",
        "Relative Group 1M",
        "Relative Group 3M",
        "Relative Group 12M",
        "Price vs ATH",
    ]
    st.markdown("#### Components")
    components_table = table[cols_to_show].copy()
    st.dataframe(
        style_performance(format_table_source(components_table), [*PERF_COLS, "Relative Group 1M", "Relative Group 3M", "Relative Group 12M"]),
        use_container_width=True,
        hide_index=True,
        height=dataframe_auto_height(components_table),
    )


def render_ai_companies(companies: pd.DataFrame) -> None:
    st.markdown("### Companies")
    filtered = apply_company_filters(companies, prefix="ai_companies")
    cols = [
        "Group",
        "Company",
        "Ticker",
        "Market Cap",
        *PERF_COLS,
        "Price vs SMA200D",
        "SMA200D Robust Z 36M",
        "SMA200W Percentile",
        "Relative Group 1M",
        "Relative Group 3M",
        "Relative Group 12M",
        "Relative Benchmark 1M",
        "Relative Benchmark 3M",
        "Relative Benchmark 12M",
        "Trailing P/E",
        "Forward P/E",
        "PEG Ratio",
        "Price/Sales",
        "Quarterly Revenue Growth YoY",
        "Revenue Growth 3Y",
        "Operating Margin TTM",
        "Price vs ATH",
        "ATH Date",
    ]
    st.dataframe(format_table(filtered[cols].copy()), use_container_width=True, hide_index=True)


def render_ai_fundamentals(companies: pd.DataFrame, groups: pd.DataFrame) -> None:
    mode = st.segmented_control("Mode", ["Group", "Company"], default="Group", key="ai_fundamentals_mode")
    if mode == "Company":
        display = apply_company_filters(companies, prefix="ai_fundamentals")
        cols = [
            "Group",
            "Company",
            "Ticker",
            "Market Cap",
            "Trailing P/E",
            "Forward P/E",
            "PEG Ratio",
            "Price/Sales",
            "Quarterly Revenue Growth YoY",
            "Revenue Growth 3Y",
            "Operating Margin TTM",
            "Price vs ATH",
            "ATH Date",
        ]
    else:
        display = groups.copy()
        cols = [
            "Group",
            "Total Market Cap",
            "Median Trailing P/E",
            "Median Forward P/E",
            "Median PEG",
            "Median Price/Sales",
            "Median Quarterly Revenue Growth YoY",
            "Median Revenue Growth 3Y",
            "Median Operating Margin TTM",
            "Median Price vs ATH",
        ]
    st.dataframe(format_table(display[cols].copy()), use_container_width=True, hide_index=True)


def render_ai_charts(data: AIDashboardData, companies: pd.DataFrame, groups: pd.DataFrame, benchmark: str) -> None:
    st.markdown("### Group Performance")
    c1, c2 = st.columns([2, 1])
    with c1:
        selected_groups = st.multiselect("Groups", list(AI_UNIVERSE.keys()), default=list(AI_UNIVERSE.keys())[:4], key="ai_chart_groups")
    with c2:
        period = st.selectbox("Period", ["1M", "3M", "6M", "12M", "3Y", "5Y"], index=3, key="ai_chart_group_period")
    st.plotly_chart(build_group_performance_fig(data, selected_groups, ["SPY", "QQQ"], period), use_container_width=True, config=AI_PLOTLY_CONFIG)

    st.markdown("### Company vs Group")
    ticker_labels = [f"{row.Ticker} - {row.Company}" for row in companies.itertuples()]
    selected_label = st.selectbox("Company", ticker_labels, index=0, key="ai_company_vs_group_ticker")
    ticker = selected_label.split(" - ", 1)[0]
    company_row = companies[companies["Ticker"].eq(ticker)].iloc[0]
    period = st.selectbox("Chart period", ["1M", "3M", "6M", "12M", "3Y", "5Y"], index=3, key="ai_company_vs_group_period")
    st.plotly_chart(build_company_vs_group_fig(data, ticker, str(company_row["Group"]), period, benchmark), use_container_width=True, config=AI_PLOTLY_CONFIG)

    st.markdown("### Growth vs Valuation")
    c1, c2 = st.columns(2)
    with c1:
        x_metric = st.selectbox("X", ["Forward P/E", "Trailing P/E", "Price/Sales", "PEG Ratio"], index=0, key="ai_growth_val_x")
    with c2:
        y_metric = st.selectbox("Y", ["Quarterly Revenue Growth YoY", "Revenue Growth 3Y", "Operating Margin TTM"], index=1, key="ai_growth_val_y")
    st.plotly_chart(build_scatter_fig(companies, x_metric, y_metric, "Growth vs Valuation"), use_container_width=True, config=AI_PLOTLY_CONFIG)

    st.markdown("### Growth vs Profitability")
    st.plotly_chart(build_scatter_fig(companies, "Revenue Growth 3Y", "Operating Margin TTM", "Growth vs Profitability", size_metric="Market Cap"), use_container_width=True, config=AI_PLOTLY_CONFIG)

    st.markdown("### Momentum vs Overextension")
    c1, c2 = st.columns([1, 1])
    with c1:
        mode = st.segmented_control("Mode", ["Companies", "Groups"], default="Companies", key="ai_momentum_extension_mode")
    with c2:
        performance_period = st.segmented_control(
            "Performance",
            list(MOMENTUM_PERFORMANCE_OPTIONS.keys()),
            default="12M",
            key="ai_momentum_extension_performance",
        )
    momentum_source = companies if mode == "Companies" else groups
    st.plotly_chart(
        build_momentum_extension_fig(momentum_source, mode, MOMENTUM_PERFORMANCE_OPTIONS.get(performance_period or "12M", "Perf 12M")),
        use_container_width=True,
        config=AI_PLOTLY_CONFIG,
    )

    st.markdown("### Breadth")
    breadth_group = st.selectbox("Breadth group", list(AI_UNIVERSE.keys()), index=0, key="ai_breadth_group")
    breadth_period = st.selectbox("Breadth period", ["6M", "12M", "3Y", "5Y"], index=2, key="ai_breadth_period")
    st.plotly_chart(build_breadth_fig(data, breadth_group, breadth_period), use_container_width=True, config=AI_PLOTLY_CONFIG)


def apply_company_filters(companies: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = companies.copy()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        groups = st.multiselect("Group", list(AI_UNIVERSE.keys()), default=[], key=f"{prefix}_groups")
    with c2:
        ticker_text = st.text_input("Ticker", value="", key=f"{prefix}_ticker")
    with c3:
        perf_min = st.number_input("Min Perf 3M", value=-999.0, step=5.0, key=f"{prefix}_perf_min")
    with c4:
        max_extension = st.number_input("Max SMA200W Percentile", value=100.0, min_value=0.0, max_value=100.0, step=5.0, key=f"{prefix}_extension")
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        min_mcap = st.number_input("Min Market Cap, $B", value=0.0, step=10.0, key=f"{prefix}_mcap")
    with c6:
        max_fpe = st.number_input("Max Forward P/E", value=999.0, step=5.0, key=f"{prefix}_fpe")
    with c7:
        min_growth = st.number_input("Min Rev Growth 3Y %", value=-999.0, step=5.0, key=f"{prefix}_growth")
    with c8:
        min_margin = st.number_input("Min Operating Margin %", value=-999.0, step=5.0, key=f"{prefix}_margin")

    if groups:
        out = out[out["Group"].isin(groups)]
    if ticker_text.strip():
        needle = ticker_text.strip().upper()
        out = out[out["Ticker"].astype(str).str.upper().str.contains(needle, regex=False)]
    out = out[pd.to_numeric(out["Perf 3M"], errors="coerce").fillna(-9999) >= perf_min]
    out = out[pd.to_numeric(out["SMA200W Percentile"], errors="coerce").fillna(9999) <= max_extension]
    out = out[pd.to_numeric(out["Market Cap"], errors="coerce").fillna(0.0) >= min_mcap * 1_000_000_000.0]
    out = out[pd.to_numeric(out["Forward P/E"], errors="coerce").fillna(9999) <= max_fpe]
    out = out[pd.to_numeric(out["Revenue Growth 3Y"], errors="coerce").fillna(-9999) * 100.0 >= min_growth]
    out = out[pd.to_numeric(out["Operating Margin TTM"], errors="coerce").fillna(-9999) * 100.0 >= min_margin]
    return out


def build_group_performance_fig(data: AIDashboardData, groups: list[str], benchmarks: list[str], period: str) -> go.Figure:
    start = period_start_date(data.prices, period)
    fig = go.Figure()
    for group in groups:
        series = normalized_group_history(data.prices, group, start)
        if series.empty:
            continue
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=group))
    for benchmark in benchmarks:
        series = normalized_ticker_history(data.prices, benchmark, start, benchmark)
        if series.empty:
            continue
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=benchmark, line={"dash": "dash"}))
    fig.update_yaxes(title="Normalized performance, start = 100")
    return style_ai_fig(fig, "Cumulative Normalized Performance")


def build_company_vs_group_fig(data: AIDashboardData, ticker: str, group: str, period: str, benchmark: str) -> go.Figure:
    start = period_start_date(data.prices, period)
    fig = go.Figure()
    for series in [
        normalized_ticker_history(data.prices, ticker, start, ticker),
        normalized_group_history(data.prices, group, start).rename(f"{group} EW"),
        normalized_ticker_history(data.prices, benchmark, start, benchmark),
    ]:
        if not series.empty:
            fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=series.name))
    fig.update_yaxes(title="Normalized performance, start = 100")
    return style_ai_fig(fig, f"{ticker} vs {group} and {benchmark}")


def build_scatter_fig(df: pd.DataFrame, x_metric: str, y_metric: str, title: str, size_metric: str | None = None) -> go.Figure:
    d = df.dropna(subset=[x_metric, y_metric]).copy()
    marker: dict[str, Any] = {"size": 11, "opacity": 0.82, "line": {"width": 0.5, "color": "#0f172a"}}
    if size_metric and size_metric in d.columns:
        size = pd.to_numeric(d[size_metric], errors="coerce")
        marker["size"] = (8 + 28 * (size / size.max()).fillna(0.0)).clip(8, 36)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pd.to_numeric(d[x_metric], errors="coerce"),
            y=pd.to_numeric(d[y_metric], errors="coerce"),
            mode="markers+text",
            text=d["Ticker"],
            textposition="top center",
            marker=marker,
            customdata=d[["Company", "Group", "Market Cap"]],
            hovertemplate=(
                "Company: %{customdata[0]}<br>Group: %{customdata[1]}<br>"
                f"{x_metric}: %{{x:.2f}}<br>{y_metric}: %{{y:.2f}}<br>"
                "Market Cap: %{customdata[2]:,.0f}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title=x_metric)
    fig.update_yaxes(title=y_metric)
    return style_ai_fig(fig, title)


def build_momentum_extension_fig(df: pd.DataFrame, mode: str, performance_col: str = "Perf 12M") -> go.Figure:
    source = df.copy()
    if performance_col not in source.columns:
        source[performance_col] = np.nan
    percentile_col = f"{performance_col} Percentile"
    source[percentile_col] = cross_sectional_percentile(source[performance_col])
    label_col = "Ticker" if mode == "Companies" and "Ticker" in source.columns else "Group"
    fig = go.Figure()
    d = source.dropna(subset=[percentile_col, "SMA200W Percentile"])
    fig.add_trace(
        go.Scatter(
            x=d[percentile_col],
            y=d["SMA200W Percentile"],
            mode="markers+text",
            text=d[label_col],
            textposition="top center",
            marker={"size": 11, "opacity": 0.82},
            customdata=pd.concat(
                [
                    d[performance_col].rename("performance"),
                    d["Group"].rename("group") if "Group" in d.columns else pd.Series("", index=d.index, name="group"),
                ],
                axis=1,
            ),
            hovertemplate=(
                "Name: %{text}<br>"
                "Group: %{customdata[1]}<br>"
                f"{performance_col}: %{{customdata[0]:.1f}}%<br>"
                f"{performance_col} Percentile: %{{x:.1f}}<br>"
                "SMA200W Percentile: %{y:.1f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=50, line={"dash": "dash", "color": "#94a3b8"})
    fig.add_hline(y=50, line={"dash": "dash", "color": "#94a3b8"})
    fig.update_xaxes(title=f"{performance_col} Percentile", range=[0, 100])
    fig.update_yaxes(title="SMA200W Percentile", range=[0, 100])
    return style_ai_fig(fig, "Momentum vs Overextension")


def cross_sectional_percentile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=numeric.index)
    return numeric.rank(pct=True, method="average") * 100.0


def build_breadth_fig(data: AIDashboardData, group: str, period: str) -> go.Figure:
    start = period_start_date(data.prices, period)
    breadth = ai_breadth_history(data.prices, group, start)
    fig = go.Figure()
    if breadth.empty:
        return style_ai_fig(fig, f"{group} Breadth")
    fig.add_trace(go.Scatter(x=breadth["Date"], y=breadth["% Above SMA50"], mode="lines", name="% Above SMA50", line={"color": "#22c55e"}))
    fig.add_trace(go.Scatter(x=breadth["Date"], y=breadth["% Above SMA200"], mode="lines", name="% Above SMA200", line={"color": "#38bdf8"}))
    fig.add_trace(
        go.Scatter(
            x=breadth["Date"],
            y=breadth["Group Performance"],
            mode="lines",
            name="Group Performance",
            yaxis="y2",
            line={"color": "#facc15", "dash": "dash"},
        )
    )
    fig.update_layout(yaxis={"title": "Breadth %", "range": [0, 100]}, yaxis2={"title": "Performance", "overlaying": "y", "side": "right", "showgrid": False})
    return style_ai_fig(fig, f"{group} Breadth")


def style_ai_fig(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        height=380,
        paper_bgcolor="#0f131a",
        plot_bgcolor="#0f131a",
        font={"color": "#e5e7eb", "size": 11},
        margin={"l": 58, "r": 72, "t": 60, "b": 44},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "top", "y": -0.16, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(tickformat="%b'%y", showgrid=False, zeroline=False, color="#cbd5e1", linecolor="#475569")
    fig.update_yaxes(showgrid=True, gridcolor="#263241", zeroline=False, color="#cbd5e1", linecolor="#475569")
    return fig


def render_metric(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
<div style="padding: 0.65rem 0; line-height: 1.15;">
  <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700;">{html.escape(label)}</div>
  <div style="font-size: 1rem; color: #f8fafc; font-weight: 800;">{html.escape(value)}</div>
  <div style="font-size: 0.72rem; color: #cbd5e1;">{html.escape(detail)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def dataframe_auto_height(df: pd.DataFrame, row_height: int = 34, header_height: int = 42, padding: int = 14) -> int:
    rows = max(len(df), 1)
    return header_height + padding + rows * row_height


def style_performance(df: pd.DataFrame, cols: list[str]) -> pd.io.formats.style.Styler:
    out = format_table_source(df)
    styler = out.style.format(formatters_for(out), na_rep="N/A")
    for col in cols:
        if col in out.columns:
            styler = apply_table_gradient(styler, out, col, inverse=False)
    return styler


def style_condition(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    out = format_table_source(df)
    styler = out.style.format(formatters_for(out), na_rep="N/A")
    for col in ["SMA200W Percentile", "SMA200D Robust Z 36M", "Price vs SMA200D"]:
        if col in out.columns:
            styler = apply_table_gradient(styler, out, col, inverse=True)
    for col in ["% Above SMA50", "% Above SMA200", "% Positive 1M", "% Positive 3M", "% Positive 12M"]:
        if col in out.columns:
            styler = apply_table_gradient(styler, out, col, inverse=False)
    return styler


def format_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    out = format_table_source(df)
    return out.style.format(formatters_for(out), na_rep="N/A")


def format_table_source(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def formatters_for(df: pd.DataFrame) -> dict[str, Any]:
    formatters: dict[str, Any] = {}
    for col in df.columns:
        if col in PERF_COLS or col in RELATIVE_COLS:
            formatters[col] = fmt_return_points
        elif col in RATIO_PERCENT_COLS:
            formatters[col] = fmt_ratio_percent
        elif col in MONEY_COLS:
            formatters[col] = fmt_money
        elif col in MULTIPLE_COLS:
            formatters[col] = lambda value: fmt_number(value, 2)
        elif col in COUNT_COLS:
            formatters[col] = lambda value: fmt_number(value, 0)
        elif col in {"% Above SMA50", "% Above SMA200", "% Positive 1M", "% Positive 3M", "% Positive 12M"}:
            formatters[col] = fmt_return_points
        elif col in {"SMA200W Percentile", "Perf 12M Percentile"}:
            formatters[col] = lambda value: fmt_number(value, 1)
        elif col == "SMA200D Robust Z 36M":
            formatters[col] = lambda value: fmt_number(value, 2)
        else:
            formatters[col] = format_value
    return formatters


def apply_table_gradient(
    styler: pd.io.formats.style.Styler,
    df: pd.DataFrame,
    col: str,
    inverse: bool,
) -> pd.io.formats.style.Styler:
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return styler
    vmin = float(values.min())
    vmax = float(values.max())
    return styler.map(
        lambda value, lo=vmin, hi=vmax, inv=inverse: table_gradient_cell_style(value, lo, hi, inv),
        subset=[col],
    )


def table_gradient_cell_style(value: Any, vmin: float, vmax: float, inverse: bool = False) -> str:
    numeric = to_float(value)
    if not np.isfinite(numeric):
        return ""
    if vmax == vmin:
        return "background-color: #fff3bf; color: #111827;"
    t = (numeric - vmin) / (vmax - vmin)
    if inverse:
        t = 1.0 - t
    t = min(max(t, 0.0), 1.0)
    r = round(248 + t * (74 - 248))
    g = round(113 + t * (222 - 113))
    b = round(113 + t * (128 - 113))
    return f"background-color: rgb({r}, {g}, {b}); color: #111827;"


def format_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return "YES" if bool(value) else "NO"
    return str(value)


def fmt_return_points(value: Any) -> str:
    numeric = to_float(value)
    if not np.isfinite(numeric):
        return "N/A"
    return f"{numeric:.1f}%"


def fmt_ratio_percent(value: Any) -> str:
    numeric = to_float(value)
    if not np.isfinite(numeric):
        return "N/A"
    return f"{numeric * 100.0:.1f}%"


def fmt_number(value: Any, decimals: int = 1) -> str:
    numeric = to_float(value)
    if not np.isfinite(numeric):
        return "N/A"
    return f"{numeric:,.{decimals}f}"


def fmt_money(value: Any) -> str:
    numeric = to_float(value)
    if not np.isfinite(numeric):
        return "N/A"
    if abs(numeric) >= 1_000_000_000_000:
        return f"${numeric / 1_000_000_000_000:.2f}T"
    return f"${numeric / 1_000_000_000:.1f}B"


def fmt_pct_points(value: Any) -> str:
    numeric = to_float(value)
    if not np.isfinite(numeric):
        return "N/A"
    return f"{numeric:.1f}%"


def fmt_datetime(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "N/A"


def overextension_state(value: Any) -> str:
    numeric = to_float(value)
    if not np.isfinite(numeric):
        return "N/A"
    if numeric >= 90:
        return "Extreme Overbought"
    if numeric >= 75:
        return "Overbought"
    if numeric >= 60:
        return "Elevated"
    if numeric >= 40:
        return "Neutral"
    if numeric >= 25:
        return "Moderately Cheap"
    if numeric >= 10:
        return "Oversold"
    return "Extreme Oversold"


def to_float(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan
