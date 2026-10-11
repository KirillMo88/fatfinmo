from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode, JsCode

from world_economy import load_world_economy_data


@st.cache_data(show_spinner="Loading official BIS and IMF data…", ttl=6 * 60 * 60)
def _load_snapshot(refresh_nonce: int = 0) -> dict[str, Any]:
    return load_world_economy_data(force_refresh=refresh_nonce > 0)


def render_world_economy_tab() -> None:
    st.title("World Economy")
    refresh_col, _ = st.columns([1, 5])
    with refresh_col:
        refresh = st.button("Refresh World Economy", key="world_economy_refresh", use_container_width=True)
    if refresh:
        _load_snapshot.clear()
    try:
        snapshot = _load_snapshot(int(refresh))
    except Exception as exc:
        st.error(f"World Economy data is unavailable: {exc}")
        return

    table = snapshot["table"].copy()
    history = snapshot["history"].copy()
    diagnostics = snapshot["diagnostics"]
    year = int(snapshot["reference_year"])
    metadata = snapshot.get("metadata", {})
    if diagnostics.get("warning"):
        st.warning(diagnostics["warning"])
    st.caption(
        f"Reference year: {year} · WEO vintage: {metadata.get('weo_vintage', 'unknown')} · "
        f"BIS retrieved: {metadata.get('bis_retrieved_at_utc', 'unknown')} · "
        f"IMF retrieved: {metadata.get('weo_retrieved_at_utc', 'unknown')}"
    )

    _render_global_position(table, year)
    st.divider()
    _render_country_trajectory(history, table, year)
    st.divider()
    _render_comparison_table(table, year, metadata)

    with st.expander("Data coverage, sources & validation"):
        left, right = st.columns(2)
        left.metric("BIS economies", diagnostics.get("bis_countries_total", 0))
        left.metric("Mapped BIS economies", diagnostics.get("bis_countries_mapped", 0))
        left.metric("Countries with reference-year GDP + fiscal balance", diagnostics.get("countries_with_reference_gdp_and_fiscal", 0))
        right.write(f"**BIS series:** {diagnostics.get('bis_series_definition')}")
        right.write(f"**IMF indicators:** {', '.join(diagnostics.get('imf_indicators', []))}")
        right.write(f"**WEO vintage:** {diagnostics.get('imf_vintage') or 'unknown'}")
        right.write(f"**BIS latest quarters:** {diagnostics.get('latest_bis_quarters', {})}")
        unmapped = diagnostics.get("bis_unmapped_economies", [])
        if unmapped:
            st.warning(f"Unmapped BIS economies (excluded from country rankings): {unmapped}")
        st.caption("BIS source: data.bis.org/topics/TOTAL_CREDIT · IMF source: data.imf.org/Datasets/WEO")


def _render_global_position(table: pd.DataFrame, year: int) -> None:
    st.subheader("Global Debt & Fiscal Position")
    control_col, size_col = st.columns([1, 1])
    with control_col:
        country_filter = st.segmented_control(
            "Countries", ["All Countries", "Top 10 GDP"], default="All Countries", key="we_countries_filter"
        )
    with size_col:
        bubble_mode = st.segmented_control(
            "Bubble Size", ["Nominal GDP", "Real GDP Growth 3Y"], default="Nominal GDP", key="we_bubble_mode"
        )
    visible = table.copy() if country_filter == "All Countries" else table.nsmallest(10, "gdp_rank").copy()
    chart_data = visible.loc[
        visible["total_debt_gdp_pct"].notna() & visible["fiscal_balance_pct"].notna()
    ].copy()
    if chart_data.empty:
        st.info("No countries have both a BIS debt observation and IMF fiscal balance for this view.")
        return
    if bubble_mode == "Nominal GDP":
        marker_size = pd.to_numeric(chart_data["nominal_gdp_usd_bn"], errors="coerce").fillna(0).clip(lower=0)
        maximum = max(float(marker_size.max()), 1.0)
        colors: Any = "#38bdf8"
    else:
        growth = pd.to_numeric(chart_data["avg_real_gdp_growth_3y_pct"], errors="coerce")
        marker_size = growth.abs().fillna(0)
        maximum = max(float(marker_size.max()), 0.25)
        colors = ["#38bdf8" if pd.notna(value) and value >= 0 else "#f97316" if pd.notna(value) else "#64748b" for value in growth]
    text_positions = _avoid_nearby_label_collisions(chart_data)
    custom = chart_data[["country_name", "bis_observation_quarter", "nominal_gdp_usd_bn", "fiscal_balance_usd_bn", "avg_real_gdp_growth_3y_pct", "gdp_rank", "imf_vintage"]].to_numpy()
    fig = go.Figure(go.Scatter(
        x=chart_data["total_debt_gdp_pct"], y=chart_data["fiscal_balance_pct"], mode="markers+text",
        text=chart_data["iso3"], textposition=text_positions, textfont={"size": 10, "color": "#f8fafc"},
        name="Countries", showlegend=False,
        marker={"size": marker_size, "sizemode": "area", "sizeref": 2.0 * maximum / (58.0 ** 2),
                "sizemin": 9, "color": colors, "opacity": 0.78,
                "line": {"color": "#0f172a", "width": 1}}, customdata=custom,
        hovertemplate=(
            "<b>%{customdata[0]} (%{text})</b><br>Debt/GDP: %{x:.2f}% (%{customdata[1]})"
            f"<br>Fiscal balance ({year}): %{{y:.2f}}% GDP"
            "<br>Nominal GDP: $%{customdata[2]:,.1f}bn"
            "<br>Fiscal balance: $%{customdata[3]:,.1f}bn"
            "<br>Average real GDP growth (3Y): %{customdata[4]:.2f}%"
            "<br>Nominal GDP rank: %{customdata[5]}<br>WEO vintage: %{customdata[6]}<extra></extra>"
        ),
    ))
    if bubble_mode == "Real GDP Growth 3Y":
        for name, color in (("Positive growth", "#38bdf8"), ("Negative growth", "#f97316"), ("N/A", "#64748b")):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker={"size": 10, "color": color}, name=name))
    fig.update_layout(height=560, margin={"l": 55, "r": 20, "t": 24, "b": 45},
                      showlegend=bubble_mode == "Real GDP Growth 3Y",
                      xaxis_title="Latest BIS Total Debt / GDP (%)",
                      yaxis={"title": f"General Government Fiscal Balance ({year}, % GDP)", "autorange": "reversed",
                             "zeroline": True, "zerolinecolor": "#94a3b8", "ticksuffix": "%"})
    if bubble_mode == "Real GDP Growth 3Y":
        st.caption("Bubble area shows absolute 3Y average real GDP growth; blue = positive, orange = negative, gray = N/A.")
    else:
        st.caption("Bubble area is proportional to nominal GDP (USD billions).")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def _avoid_nearby_label_collisions(frame: pd.DataFrame) -> list[str]:
    """Choose different nearby label anchors where observations are geographically close."""
    x = pd.to_numeric(frame["total_debt_gdp_pct"], errors="coerce")
    y = pd.to_numeric(frame["fiscal_balance_pct"], errors="coerce")
    x_span = max(float(x.max() - x.min()), 1e-9)
    y_span = max(float(y.max() - y.min()), 1e-9)
    candidates = [
        ("top center", 0.0, 0.045), ("bottom center", 0.0, -0.045),
        ("middle right", 0.032, 0.0), ("middle left", -0.032, 0.0),
        ("top right", 0.028, 0.04), ("top left", -0.028, 0.04),
        ("bottom right", 0.028, -0.04), ("bottom left", -0.028, -0.04),
        ("top center", 0.0, 0.075), ("bottom center", 0.0, -0.075),
    ]
    placed: list[tuple[float, float]] = []
    selected: list[str] = []
    for xv, yv in zip(x, y):
        nx = (float(xv) - float(x.min())) / x_span
        ny = (float(yv) - float(y.min())) / y_span
        scored = []
        for order, (position, dx, dy) in enumerate(candidates):
            label_x, label_y = nx + dx, ny + dy
            overlap = sum(1 for px, py in placed if abs(px - label_x) < 0.035 and abs(py - label_y) < 0.06)
            scored.append((overlap, order, position, label_x, label_y))
        _, _, position, label_x, label_y = min(scored)
        selected.append(position)
        placed.append((label_x, label_y))
    return selected


def _render_country_trajectory(history: pd.DataFrame, table: pd.DataFrame, reference_year: int) -> None:
    st.subheader("Country Macro Trajectory")
    countries = table[["iso3", "country_name"]].drop_duplicates().dropna().sort_values("country_name")
    options = {f"{row.country_name} ({row.iso3})": row.iso3 for row in countries.itertuples(index=False)}
    default = next((label for label, code in options.items() if code == "USA"), next(iter(options), None))
    col_country, col_provisional = st.columns([3, 1])
    with col_country:
        selected_label = st.selectbox("Select Country", list(options), index=list(options).index(default) if default else 0, key="we_history_country") if options else None
    with col_provisional:
        show_provisional = st.toggle("Show Provisional Current Year", value=True, key="we_provisional_toggle")
    if not selected_label:
        st.info("No mapped countries are available for the historical selector.")
        return
    iso3 = options[selected_label]
    start_year = reference_year - 10
    selected = history.loc[history["iso3"].eq(iso3) & history["year"].between(start_year, reference_year)].copy()
    if not show_provisional:
        selected = selected.loc[~selected["is_provisional"].fillna(False)]
    selected["is_provisional"] = selected["is_provisional"].fillna(False).astype(bool)
    selected = selected.sort_values(["year", "is_provisional"])
    trajectory = _country_trajectory_table(selected, reference_year)
    trajectory_height = min(570, max(400, 42 + 35 * len(trajectory)))
    macro_chart_data = _trajectory_chart_data(selected, reference_year)

    table_col, macro_col = st.columns(2, gap="medium")
    with table_col:
        st.dataframe(trajectory, use_container_width=True, hide_index=True, height=trajectory_height)
        st.caption("Nominal GDP 5Y/10Y columns show CAGR; other summary columns show 5Y/10Y change. Monthly and quarterly series use calendar-year averages; the current year may be partial.")
    with macro_col:
        _render_growth_fiscal_current_account_plot(macro_chart_data, height=trajectory_height)

    debt_col, scatter_col = st.columns(2, gap="medium")
    with debt_col:
        _render_debt_interest_plot(macro_chart_data, height=trajectory_height)
    with scatter_col:
        _render_debt_fiscal_plot(selected, reference_year, show_provisional, height=trajectory_height)


def _trajectory_chart_data(selected: pd.DataFrame, reference_year: int) -> pd.DataFrame:
    annual = (
        selected.loc[~selected["is_provisional"].fillna(False)]
        .sort_values("year")
        .drop_duplicates("year", keep="last")
        .set_index("year")
    )
    years = list(range(reference_year - 10, reference_year + 1))
    periods = [str(year) for year in years] + ["5Y Avg", "10Y Avg"]
    sources = (
        "real_gdp_growth_pct", "fiscal_balance_pct", "current_account_pct",
        "total_debt_gdp_annual_avg_pct", "long_term_interest_rate_annual_avg_pct",
        "interest_on_total_debt_pct_gdp",
    )
    result = pd.DataFrame({"Period": periods})
    for source in sources:
        values = pd.to_numeric(annual[source], errors="coerce") if source in annual.columns else pd.Series(dtype=float)
        recent_5y = values.reindex(range(reference_year - 4, reference_year + 1))
        recent_10y = values.reindex(range(reference_year - 9, reference_year + 1))
        result[source] = [values.get(year, np.nan) for year in years] + [recent_5y.mean(), recent_10y.mean()]
    return result


def _render_growth_fiscal_current_account_plot(data: pd.DataFrame, *, height: int) -> None:
    fig = go.Figure()
    series = (
        ("Real GDP Growth", "real_gdp_growth_pct", "#176b8e"),
        ("Fiscal Balance", "fiscal_balance_pct", "#f37032"),
        ("Current Account Balance / GDP", "current_account_pct", "#45b6e8"),
    )
    for name, column, color in series:
        fig.add_trace(go.Bar(
            x=data["Period"], y=data[column], name=name, marker_color=color,
            text=data[column], texttemplate="%{text:.2f}", textposition="outside",
            textfont={"size": 9}, cliponaxis=False,
            hovertemplate=f"{name}: %{{y:.2f}}%<extra></extra>",
        ))
    fig.update_layout(
        title={"text": "GDP Growth, Fiscal Balance, Current Account", "x": 0.5},
        height=height, barmode="group", margin={"l": 45, "r": 15, "t": 48, "b": 65},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "x": 0.5, "xanchor": "center"},
        yaxis={"title": "% of GDP / growth", "zeroline": True, "gridcolor": "#293241"},
        xaxis={
            "title": None, "gridcolor": "#293241", "type": "category",
            "categoryorder": "array", "categoryarray": data["Period"].tolist(),
            "tickmode": "array", "tickvals": data["Period"].tolist(),
            "ticktext": data["Period"].tolist(), "automargin": True,
        },
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def _render_debt_interest_plot(data: pd.DataFrame, *, height: int) -> None:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=data["Period"], y=data["long_term_interest_rate_annual_avg_pct"],
        name="LT Interest Rate", marker_color="#f37032",
        hovertemplate="LT Interest Rate: %{y:.2f}%<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=data["Period"], y=data["interest_on_total_debt_pct_gdp"],
        name="Interest on Total Debt", marker_color="#45b6e8",
        hovertemplate="Interest on Total Debt: %{y:.2f}% GDP<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=data["Period"], y=data["total_debt_gdp_annual_avg_pct"],
        name="Total Debt / GDP", mode="lines+markers", line={"color": "#176b8e", "width": 3},
        marker={"size": 6}, hovertemplate="Total Debt / GDP: %{y:.2f}%<extra></extra>",
    ), secondary_y=True)
    fig.update_layout(
        title={"text": "Total Debt / GDP, LT Interest Rate, Interest on Total Debt", "x": 0.5},
        height=height, barmode="group", margin={"l": 48, "r": 48, "t": 48, "b": 65},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "x": 0.5, "xanchor": "center"},
        xaxis={
            "gridcolor": "#293241", "type": "category",
            "categoryorder": "array", "categoryarray": data["Period"].tolist(),
            "tickmode": "array", "tickvals": data["Period"].tolist(),
            "ticktext": data["Period"].tolist(), "automargin": True,
        },
    )
    fig.update_yaxes(title_text="Interest (% / % GDP)", gridcolor="#293241", secondary_y=False)
    fig.update_yaxes(title_text="Total Debt / GDP (%)", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def _country_trajectory_table(selected: pd.DataFrame, reference_year: int) -> pd.DataFrame:
    start_year = reference_year - 10
    annual = selected.loc[~selected["is_provisional"].fillna(False)].sort_values("year").drop_duplicates("year", keep="last")
    by_year = annual.set_index("year")
    series = [
        ("Nominal GDP (USD bn)", "nominal_gdp_usd_bn", "level"),
        ("Nominal GDP Growth", "nominal_gdp_growth_pct", "pct"),
        ("Real GDP Growth", "real_gdp_growth_pct", "pct"),
        ("Total Debt / GDP", "total_debt_gdp_annual_avg_pct", "pct"),
        ("Interest Rate", "long_term_interest_rate_annual_avg_pct", "pct"),
        ("Interest on total Debt (% GDP)", "interest_on_total_debt_pct_gdp", "pct"),
        ("DSR Private non-financial sector", "dsr_annual_avg_pct", "pct"),
        ("Fiscal Balance", "fiscal_balance_pct", "pct"),
        ("Government share in total Expenditure", "government_expenditure_pct", "pct"),
        ("Current Account Balance / GDP", "current_account_pct", "pct"),
    ]
    year_columns = [str(year) for year in range(start_year, reference_year + 1)]
    result: list[dict[str, Any]] = []
    for label, source, kind in series:
        row: dict[str, Any] = {"Indicator": label}
        values: dict[int, float] = {}
        for year in range(start_year, reference_year + 1):
            value = pd.to_numeric(by_year.loc[year, source], errors="coerce") if year in by_year.index and source in by_year.columns else np.nan
            if isinstance(value, pd.Series):
                value = value.iloc[-1]
            values[year] = float(value) if pd.notna(value) else np.nan
            row[str(year)] = _format_trajectory_value(values[year], kind)
        for horizon in (5, 10):
            first, last = values.get(reference_year - horizon, np.nan), values.get(reference_year, np.nan)
            if kind == "level":
                summary = ((last / first) ** (1 / horizon) - 1) * 100 if np.isfinite(first) and np.isfinite(last) and first > 0 and last > 0 else np.nan
            else:
                summary = last - first if np.isfinite(first) and np.isfinite(last) else np.nan
            row[f"{horizon}Y CAGR / Δ"] = _format_trajectory_value(summary, "pct")
        result.append(row)
    return pd.DataFrame(result, columns=["Indicator", *year_columns, "5Y CAGR / Δ", "10Y CAGR / Δ"])


def _format_trajectory_value(value: float, kind: str) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:,.0f}" if kind == "level" else f"{value:.2f}"


def _render_debt_fiscal_plot(
    selected: pd.DataFrame,
    reference_year: int,
    show_provisional: bool,
    *,
    height: int = 570,
) -> None:
    fig = go.Figure()
    # Separate each discontinuous valid run; missing annual points are never bridged.
    completed = selected.loc[~selected["is_provisional"]].sort_values("year")
    valid = completed.loc[completed["total_debt_gdp_pct"].notna() & completed["fiscal_balance_pct"].notna()]
    if not valid.empty:
        runs: list[list[Any]] = []
        run: list[Any] = []
        previous_year: int | None = None
        for row in valid.itertuples(index=False):
            current_year = int(row.year)
            if previous_year is not None and current_year != previous_year + 1:
                if run:
                    runs.append(run)
                run = []
            run.append(row)
            previous_year = current_year
        if run:
            runs.append(run)
        for run_index, run_rows in enumerate(runs):
            run_frame = pd.DataFrame([row._asdict() for row in run_rows])
            fig.add_trace(go.Scatter(
                x=run_frame["total_debt_gdp_pct"], y=run_frame["fiscal_balance_pct"], mode="lines+markers+text",
                text=run_frame["year"].astype(str), textposition="top center",
                name="Completed years" if run_index == 0 else None, showlegend=run_index == 0,
                line={"color": "#38bdf8", "width": 2}, marker={"size": [8 if y == reference_year - 1 else 6 for y in run_frame["year"]], "color": "#38bdf8"},
                customdata=run_frame[["country_name", "year", "bis_observation_quarter", "nominal_gdp_usd_bn", "real_gdp_growth_pct", "debt_yoy_delta_pp", "fiscal_yoy_delta_pp", "imf_vintage"]],
                hovertemplate="%{customdata[0]} · %{customdata[1]}<br>BIS: %{customdata[2]}<br>Debt/GDP: %{x:.2f}%<br>Fiscal: %{y:.2f}% GDP<br>GDP: $%{customdata[3]:,.1f}bn<br>Real growth: %{customdata[4]:.2f}%<br>Debt YoY Δ: %{customdata[5]:.2f}pp<br>Fiscal YoY Δ: %{customdata[6]:.2f}pp<br>WEO vintage: %{customdata[7]}<extra></extra>",
            ))
        for first, second in zip(valid.itertuples(index=False), list(valid.itertuples(index=False))[1:]):
            if int(second.year) - int(first.year) == 1:
                fig.add_annotation(x=second.total_debt_gdp_pct, y=second.fiscal_balance_pct,
                                   ax=first.total_debt_gdp_pct, ay=first.fiscal_balance_pct,
                                   xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                                   arrowhead=3, arrowsize=1, arrowwidth=1.3, arrowcolor="#7dd3fc")
    provisional = selected.loc[selected["is_provisional"] & selected["total_debt_gdp_pct"].notna() & selected["fiscal_balance_pct"].notna()]
    if not provisional.empty:
        point = provisional.iloc[-1]
        previous = valid.iloc[-1] if not valid.empty and int(valid.iloc[-1]["year"]) == reference_year - 1 else None
        if previous is not None:
            fig.add_trace(go.Scatter(x=[previous["total_debt_gdp_pct"], point["total_debt_gdp_pct"]],
                                     y=[previous["fiscal_balance_pct"], point["fiscal_balance_pct"]],
                                     mode="lines", line={"color": "#f8fafc", "dash": "dash"}, name="Provisional connection", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[point["total_debt_gdp_pct"]], y=[point["fiscal_balance_pct"]], mode="markers+text",
                                 text=[str(reference_year) + "*"], textposition="top center", name="Provisional current year",
                                 marker={"symbol": "circle-open", "size": 13, "color": "#facc15", "line": {"width": 2}},
                                 customdata=[[point["country_name"], point["bis_observation_quarter"], point["fiscal_status"], point["nominal_gdp_usd_bn"], point["real_gdp_growth_pct"], point["imf_vintage"]]],
                                 hovertemplate=f"%{{customdata[0]}} · {reference_year} provisional<br>BIS quarter: %{{customdata[1]}}<br>IMF status: %{{customdata[2]}}<br>Debt/GDP: %{{x:.2f}}%<br>Fiscal: %{{y:.2f}}% GDP<br>GDP: $%{{customdata[3]:,.1f}}bn<br>Real growth: %{{customdata[4]:.2f}}%<br>WEO vintage: %{{customdata[5]}}<extra></extra>"))
    if show_provisional and provisional.empty:
        st.info(f"No BIS observation in {reference_year}; no provisional endpoint is shown.")
    fig.update_layout(height=height, showlegend=False,
                      title={"text": "Country Debt & Fiscal Trajectory — 10 Years", "x": 0.5},
                      margin={"l": 55, "r": 20, "t": 48, "b": 45},
                      xaxis_title="BIS Total Debt / GDP (%)",
                      yaxis={"title": f"General Government Fiscal Balance (% GDP) — {reference_year}", "autorange": "reversed", "zeroline": True})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
    st.caption("Upper right means rising debt with a deteriorating fiscal balance; lower left means falling debt with fiscal improvement. Gaps are not interpolated.")


def _render_comparison_table(table: pd.DataFrame, year: int, source_metadata: dict[str, Any]) -> None:
    st.subheader("Global Economy Comparison")
    query = st.text_input("Search country", key="we_table_search").strip().casefold()
    show_top = st.session_state.get("we_countries_filter", "All Countries") == "Top 10 GDP"
    visible = table.nsmallest(10, "gdp_rank").copy() if show_top else table.copy()
    if query:
        visible = visible.loc[visible["country_name"].str.casefold().str.contains(query, na=False) | visible["iso3"].str.casefold().str.contains(query, na=False)]
    display = visible.rename(columns={
        "gdp_rank": "GDP Rank", "country_name": "Country", "iso3": "ISO3",
        "nominal_gdp_usd_bn": f"GDP {year}F, USD bn", "gdp_cagr_5y_pct": "Nominal USD GDP CAGR 5Y (%)",
        "gdp_cagr_10y_pct": "Nominal USD GDP CAGR 10Y (%)", "total_debt_gdp_pct": "Latest BIS Total Debt/GDP (%)",
        "bis_observation_quarter": "BIS Quarter", "debt_delta_5y_pp": "Debt/GDP Δ5Y (pp)",
        "debt_delta_10y_pp": "Debt/GDP Δ10Y (pp)", "fiscal_balance_pct": f"Fiscal Balance {year}F (% GDP)",
        "government_expenditure_pct": f"General Government Total Expenditure {year}F (% GDP)",
        "current_account_pct": f"Current Account Balance {year}F (% GDP)",
        "current_account_delta_5y_pp": "Current Account Δ5Y (pp)",
        "current_account_delta_10y_pp": "Current Account Δ10Y (pp)",
        "dsr_pct": "Private Non-Financial Sector DSR (%)",
        "dsr_observation_quarter": "DSR Quarter",
        "dsr_delta_5y_pp": "DSR Δ5Y (pp)", "dsr_delta_10y_pp": "DSR Δ10Y (pp)",
        "long_term_interest_rate_pct": "Long-term Interest Rate (%)",
        "long_term_interest_rate_observation_month": "Interest Rate Month",
        "long_term_interest_rate_delta_5y_pp": "Interest Rate Δ5Y (pp)",
        "long_term_interest_rate_delta_10y_pp": "Interest Rate Δ10Y (pp)",
        "cds_5y_bps": "5Y CDS (bps)", "cds_var_6m_pct": "CDS Var 6m (%)",
        "cds_pd_pct": "CDS Implied PD (%)",
        "fiscal_delta_5y_pp": "Fiscal Balance Δ5Y (pp)", "fiscal_delta_10y_pp": "Fiscal Balance Δ10Y (pp)",
        "fiscal_balance_usd_bn": f"Signed Fiscal Balance {year}F (USD bn)",
        f"real_gdp_growth_{year - 3}_pct": f"Real GDP Growth {year - 3} (%)",
        f"real_gdp_growth_{year - 2}_pct": f"Real GDP Growth {year - 2} (%)",
        f"real_gdp_growth_{year - 1}_pct": f"Real GDP Growth {year - 1} (%)",
        "avg_real_gdp_growth_3y_pct": f"Average Real GDP Growth {year - 3}–{year - 1} (%)",
    })
    gdp_label = f"GDP {year}F, USD bn"
    avg_growth_label = f"Average Real GDP Growth {year - 3}–{year - 1} (%)"
    display = display[[
        "GDP Rank", "Country", gdp_label, "Nominal USD GDP CAGR 5Y (%)", "Nominal USD GDP CAGR 10Y (%)",
        avg_growth_label, "Latest BIS Total Debt/GDP (%)", "BIS Quarter", "Debt/GDP Δ5Y (pp)",
        "Debt/GDP Δ10Y (pp)", f"Signed Fiscal Balance {year}F (USD bn)",
        f"Fiscal Balance {year}F (% GDP)", "Fiscal Balance Δ5Y (pp)", "Fiscal Balance Δ10Y (pp)",
        f"General Government Total Expenditure {year}F (% GDP)",
        f"Current Account Balance {year}F (% GDP)", "Current Account Δ5Y (pp)", "Current Account Δ10Y (pp)",
        "Private Non-Financial Sector DSR (%)", "DSR Δ10Y (pp)",
        "Long-term Interest Rate (%)", "Interest Rate Δ5Y (pp)", "Interest Rate Δ10Y (pp)",
        "5Y CDS (bps)", "CDS Var 6m (%)", "CDS Implied PD (%)",
    ]]
    # Use the same sortable grid data for both exports so the user gets the active ordering/filter.
    gb = GridOptionsBuilder.from_dataframe(display)
    numeric_formatter = JsCode("""function(params) {
        if (params.value == null || params.value === '') return 'N/A';
        const n = Number(params.value);
        if (!Number.isFinite(n)) return params.value;
        const field = params.colDef.field || '';
        const precision = field.includes('%') || field.includes('(pp)') ? 2 : 0;
        return n.toLocaleString(undefined, {minimumFractionDigits: precision, maximumFractionDigits: precision});
    }""")
    gb.configure_default_column(sortable=True, filter=False, resizable=True, wrapHeaderText=True,
                                autoHeaderHeight=True)
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            gb.configure_column(column, valueFormatter=numeric_formatter)
    gb.configure_grid_options(ensureDomOrder=True)
    gb.configure_column("GDP Rank", sort="asc", sortIndex=0)
    debt_style = JsCode("""function(params) { if (params.value == null || isNaN(params.value)) return {}; return Number(params.value) > 0 ? {backgroundColor:'#7f1d1d', color:'#f8fafc'} : Number(params.value) < 0 ? {backgroundColor:'#166534', color:'#f8fafc'} : {}; }""")
    for column in ["Debt/GDP Δ5Y (pp)", "Debt/GDP Δ10Y (pp)"]:
        gb.configure_column(column, cellStyle=debt_style)
    fiscal_style = JsCode("""function(params) { if (params.value == null || isNaN(params.value)) return {}; return Number(params.value) > 0 ? {backgroundColor:'#166534', color:'#f8fafc'} : Number(params.value) < 0 ? {backgroundColor:'#7f1d1d', color:'#f8fafc'} : {}; }""")
    for column in ["Fiscal Balance Δ5Y (pp)", "Fiscal Balance Δ10Y (pp)"]:
        gb.configure_column(column, cellStyle=fiscal_style)
    growth_style = JsCode("""function(params) { if (params.value == null || isNaN(params.value)) return {}; return Number(params.value) < 0 ? {backgroundColor:'#7f1d1d', color:'#f8fafc'} : {}; }""")
    gb.configure_column(f"Average Real GDP Growth {year - 3}–{year - 1} (%)", cellStyle=growth_style)
    gb.configure_grid_options(domLayout="normal", rowHeight=30, suppressMovableColumns=False)
    st.caption("DSR Δ10Y compares the same quarter 10 years earlier; OECD long-term interest rate changes compare the same month 5/10 years earlier. Rate and balance changes are percentage points.")
    response = AgGrid(display, gridOptions=gb.build(), data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                      update_mode=GridUpdateMode.SORTING_CHANGED | GridUpdateMode.MODEL_CHANGED,
                      fit_columns_on_grid_load=False, allow_unsafe_jscode=True,
                      height=min(720, max(280, 42 + 31 * len(display))),
                      key="world_economy_comparison_grid")
    sorted_filtered = getattr(response, "data", None)
    if not isinstance(sorted_filtered, pd.DataFrame):
        sorted_filtered = display
    left, right, _ = st.columns([1, 1, 4])
    with left:
        csv_metadata = (
            f"# Reference year: {year}\n# BIS source: {source_metadata.get('bis_topic_url', '')}\n"
            f"# BIS retrieved at UTC: {source_metadata.get('bis_retrieved_at_utc', '')}\n"
            f"# BIS DSR source: {source_metadata.get('bis_dsr_topic_url', '')}\n"
            f"# BIS DSR retrieved at UTC: {source_metadata.get('bis_dsr_retrieved_at_utc', '')}\n"
            f"# IMF source: {source_metadata.get('weo_url', '')}\n# IMF WEO vintage: {source_metadata.get('weo_vintage', '')}\n"
            f"# IMF retrieved at UTC: {source_metadata.get('weo_retrieved_at_utc', '')}\n"
            f"# OECD long-term rates source: {source_metadata.get('oecd_long_term_rates_url', '')}\n"
            f"# OECD long-term rates retrieved at UTC: {source_metadata.get('oecd_long_term_rates_retrieved_at_utc', '')}\n"
            f"# Sovereign CDS source: {source_metadata.get('sovereign_cds_url', '')}\n"
            f"# Sovereign CDS retrieved at UTC: {source_metadata.get('sovereign_cds_retrieved_at_utc', '')}\n"
            "# Units: GDP USD billions; ratios in percent; 5Y/10Y changes in percentage points\n"
        )
        st.download_button("Download CSV", (csv_metadata + sorted_filtered.to_csv(index=False, na_rep="N/A")).encode("utf-8-sig"),
                           file_name=f"world_economy_{year}.csv", mime="text/csv", key="we_csv_download")
    with right:
        st.download_button("Download XLSX", _xlsx_export(sorted_filtered, year, source_metadata), file_name=f"world_economy_{year}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="we_xlsx_download")

    top_gdp_cds = table.nsmallest(10, "gdp_rank").dropna(subset=["cds_5y_bps"]).sort_values(
        "gdp_rank", ascending=False
    )
    if not top_gdp_cds.empty:
        st.subheader("Sovereign 5Y CDS — Top 10 GDP Countries")
        fig = go.Figure(go.Bar(
            x=top_gdp_cds["cds_5y_bps"],
            y=top_gdp_cds["country_name"],
            orientation="h",
            marker_color="#5b9cf5",
            customdata=top_gdp_cds[["cds_var_6m_pct", "cds_pd_pct"]],
            hovertemplate=(
                "%{y}<br>5Y CDS: %{x:.2f} bps<br>6M variation: %{customdata[0]:.2f}%"
                "<br>Implied PD: %{customdata[1]:.2f}%<extra></extra>"
            ),
        ))
        fig.update_layout(
            height=max(300, 38 * len(top_gdp_cds) + 90),
            margin={"l": 30, "r": 20, "t": 20, "b": 45},
            xaxis_title="Basis points (bps)",
            yaxis={"title": "", "categoryorder": "array", "categoryarray": top_gdp_cds["country_name"].tolist()},
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        st.caption(
            f"Source: {source_metadata.get('sovereign_cds_url', 'World Government Bonds Sovereign CDS')} · "
            "PD is the source's implied probability of default using a 40% recovery assumption."
        )


def _xlsx_export(table: pd.DataFrame, year: int, source_metadata: dict[str, Any]) -> bytes:
    stream = BytesIO()
    metadata = pd.DataFrame([
        ("Reference year", year), ("BIS source", "BIS WS_TC F1.1 — total credit to non-financial sector, % GDP"),
        ("BIS DSR source", "BIS Debt Service Ratios — private non-financial sector"),
        ("OECD source", "Financial market — monthly long-term interest rates, percent per annum (IRLT/PA)"),
        ("Sovereign CDS source", "World Government Bonds — 5Y CDS, six-month variation, implied probability of default (40% recovery)"),
        ("IMF source", "IMF World Economic Outlook — NGDPD, GGXCNL_NGDP, GGX_NGDP, BCA_NGDPD, NGDP_RPCH"),
        ("GDP unit", "USD billions"), ("5Y/10Y changes", "Percentage points; DSR compares same quarter"),
        ("BIS source URL", source_metadata.get("bis_topic_url", "")),
        ("BIS retrieved at UTC", source_metadata.get("bis_retrieved_at_utc", "")),
        ("BIS DSR source URL", source_metadata.get("bis_dsr_topic_url", "")),
        ("BIS DSR retrieved at UTC", source_metadata.get("bis_dsr_retrieved_at_utc", "")),
        ("OECD long-term rates source URL", source_metadata.get("oecd_long_term_rates_url", "")),
        ("OECD long-term rates API URL", source_metadata.get("oecd_long_term_rates_api_url", "")),
        ("OECD long-term rates retrieved at UTC", source_metadata.get("oecd_long_term_rates_retrieved_at_utc", "")),
        ("Sovereign CDS source URL", source_metadata.get("sovereign_cds_url", "")),
        ("Sovereign CDS retrieved at UTC", source_metadata.get("sovereign_cds_retrieved_at_utc", "")),
        ("IMF source URL", source_metadata.get("weo_url", "")),
        ("IMF WEO vintage", source_metadata.get("weo_vintage", "")),
        ("IMF retrieved at UTC", source_metadata.get("weo_retrieved_at_utc", "")),
    ], columns=["Field", "Value"])
    with pd.ExcelWriter(stream, engine="xlsxwriter") as writer:
        table.to_excel(writer, index=False, sheet_name="World Economy")
        metadata.to_excel(writer, index=False, sheet_name="Metadata")
        writer.sheets["World Economy"].freeze_panes(1, 2)
        writer.sheets["World Economy"].autofilter(0, 0, len(table), max(0, len(table.columns) - 1))
        writer.sheets["Metadata"].set_column("A:A", 28)
        writer.sheets["Metadata"].set_column("B:B", 90)
    return stream.getvalue()
