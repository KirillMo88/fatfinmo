from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .gromen import ADAPTIVE_THETAS, STATIC_TARGET_SHARES
from .models import LukeGromenGoldSnapshot


def render_luke_gromen_gold_models(snapshot: LukeGromenGoldSnapshot | None) -> None:
    st.markdown("#### Luke Gromen Gold Price Models")
    st.caption("Two independent valuation lenses: U.S. official-gold coverage and global monetary-flow absorption.")
    if snapshot is None:
        st.info("Luke Gromen model data is unavailable. Existing Gold Regime calculations are unaffected.")
        return

    render_us_coverage_model(snapshot)
    render_global_reset_model(snapshot)
    render_convergence(snapshot)
    if snapshot.warnings:
        with st.expander("Data warnings"):
            for warning in snapshot.warnings:
                st.warning(warning)
    render_methodology()


def render_us_coverage_model(snapshot: LukeGromenGoldSnapshot) -> None:
    st.markdown("##### Model 1 — U.S. Gold Coverage")
    current = snapshot.current or {}
    cols = st.columns(4)
    cols[0].metric("Gold Price", _money(current.get("gold_price")))
    cols[1].metric("Gold Coverage", _percent(current.get("coverage_ratio")))
    cols[2].metric("Foreign-held Treasury Debt", _compact_usd(current.get("foreign_debt_usd")))
    cols[3].metric("U.S. Official Gold", _compact_oz(current.get("us_gold_oz")))

    history = snapshot.us_coverage_history.dropna(subset=["date", "coverage_ratio"]).copy()
    if not history.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["coverage_ratio"] * 100.0,
                name="Coverage",
                line={"color": "#fbbf24", "width": 2.1},
                hovertemplate="%{x|%Y-%m}<br>Coverage: %{y:.1f}%<extra></extra>",
            )
        )
        for level in (10, 20, 30, 40, 50, 60, 100):
            fig.add_hline(
                y=level,
                line_dash="dot",
                line_color="#475569" if level not in (40, 50, 60) else "#94a3b8",
                opacity=0.55,
            )
        fig.add_hrect(y0=40, y1=60, fillcolor="#22c55e", opacity=0.06, line_width=0)
        fig.update_layout(
            height=330,
            margin={"l": 45, "r": 20, "t": 20, "b": 35},
            template="plotly_dark",
            paper_bgcolor="#0b0e14",
            plot_bgcolor="#11161f",
            yaxis_title="Official gold value / foreign-held Treasury debt",
            yaxis_ticksuffix="%",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        st.caption("The 40–60% band is a historical hard-money reference range, not an official target.")

    target_rows = []
    for target in (20, 30, 40, 50):
        target_rows.append(
            {
                "Coverage Target": f"{target}%",
                "Required Gold Price": _money(current.get(f"required_price_{target}pct")),
            }
        )
    st.dataframe(pd.DataFrame(target_rows), hide_index=True, use_container_width=True)
    st.caption(
        f"As of {_date(current.get('date'))} · debt source: {current.get('foreign_debt_source', 'n/a')} · "
        f"gold reserve source: {current.get('us_gold_source', 'n/a')} · price source: TradingView TVC:GOLD"
    )


def render_global_reset_model(snapshot: LukeGromenGoldSnapshot) -> None:
    st.markdown("##### Model 2 — Global Gold Monetary Reset")
    current = snapshot.current or {}
    cols = st.columns(4)
    cols[0].metric("Calibration Year", _integer(current.get("calibration_year")))
    cols[1].metric("Global Positive CA", _compact_usd(current.get("global_positive_ca_usd")))
    cols[2].metric("Broad MDS", _percent(current.get("broad_mds")))
    cols[3].metric("Broad GMAR", _percent(current.get("broad_gmar")))

    calibration = current.get("calibration_year")
    latest_wgc = current.get("wgc_latest_completed_year")
    st.caption(
        f"WGC latest completed year: {_integer(latest_wgc)} · active common calibration year: {_integer(calibration)}. "
        "A WGC year is never paired with current-account data from another year."
    )
    st.caption(
        f"GlobalPositiveCA source: {current.get('ca_source') or 'IMF WEO'} · status: {current.get('ca_data_status') or 'n/a'} · "
        f"economies: {_integer(current.get('ca_available_economy_count'))}/{_integer(current.get('ca_total_eligible_economy_count'))} · "
        f"GDP coverage: {_percent(current.get('ca_gdp_coverage'))} · prior-year surplus coverage: {_percent(current.get('ca_prior_surplus_coverage'))}."
    )
    st.caption(
        f"WGC dataset as of {current.get('wgc_data_as_of') or 'n/a'} · "
        f"latest published quarter: {current.get('wgc_latest_published_quarter') or 'n/a'}."
    )

    annual = snapshot.wgc_annual.copy()
    if not annual.empty:
        charts = st.columns(2)
        with charts[0]:
            st.plotly_chart(_mds_figure(annual), use_container_width=True, config={"displayModeBar": False, "responsive": True})
        with charts[1]:
            st.plotly_chart(_gmar_figure(annual), use_container_width=True, config={"displayModeBar": False, "responsive": True})

    balance_ok = current.get("balance_valid")
    balance_text = "PASS" if balance_ok is True else "FAIL" if balance_ok is False else "n/a"
    st.caption(
        f"WGC reconciliation: {balance_text}; gap {_tonnes(current.get('balance_gap_tonnes'))}, "
        f"tolerance {_tonnes(current.get('balance_tolerance_tonnes'))}."
    )

    if not snapshot.wgc_ytd.empty:
        row = snapshot.wgc_ytd.iloc[-1]
        st.markdown("###### WGC Annualized YTD — display only")
        st.dataframe(
            pd.DataFrame(
                [
                    ["Through", row.get("through_period")],
                    ["Published Quarters", _integer(row.get("published_quarters"))],
                    ["Annualized Total Supply", _tonnes(row.get("total_supply_tonnes"))],
                    ["Annualized Investment", _tonnes(row.get("investment_tonnes"))],
                    ["Annualized Central Banks", _tonnes(row.get("central_banks_tonnes"))],
                    ["YTD Average LBMA Price", _money(row.get("lbma_gold_price_usd_oz"))],
                ],
                columns=["Metric", "Value"],
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Flow annualization = YTD × 4 / published quarters. This display-only row is never a calibration year.")

    if not snapshot.static_scenarios.empty:
        st.markdown("###### Static Monetary-Density Scenarios")
        view = snapshot.static_scenarios.copy()
        view["Target Share"] = view["target_share"].map(_percent)
        view["Implied Gold Price"] = view["implied_gold_price"].map(_money)
        st.dataframe(view[["scenario", "Target Share", "Implied Gold Price"]].rename(columns={"scenario": "Scenario"}), hide_index=True, use_container_width=True)

    if not snapshot.adaptive_matrix.empty:
        st.markdown("###### Adaptive Reset Matrix")
        matrix = snapshot.adaptive_matrix.set_index("theta")[[*STATIC_TARGET_SHARES]].copy()
        matrix.index = [f"θ {value:.0%}" for value in matrix.index]
        matrix.columns = [f"s {value:.0%}" for value in matrix.columns]
        styled = matrix.style.format(lambda value: "n/a" if not np.isfinite(value) else f"${value:,.0f}")
        st.dataframe(styled, use_container_width=True)
        st.caption("Rows are the monetizable-gold-flow share θ; columns are the target share s of GlobalPositiveCA.")

    if not snapshot.global_ca_history.empty:
        ca = snapshot.global_ca_history.sort_values("year").tail(8).copy()
        ca["Economies"] = ca.apply(
            lambda row: f"{_integer(row.get('available_economy_count'))}/{_integer(row.get('total_eligible_economy_count'))}",
            axis=1,
        )
        ca["GDP Coverage"] = ca["gdp_coverage"].map(_percent)
        ca["Prior Surplus Coverage"] = ca["prior_year_surplus_coverage"].map(_percent)
        ca["GlobalPositiveCA"] = ca["global_positive_ca_usd"].map(_compact_usd)
        ca["Valid"] = ca["is_valid"].map(lambda value: "YES" if value else "NO")
        columns = ["year", "GlobalPositiveCA", "Economies", "GDP Coverage", "Prior Surplus Coverage", "data_status", "Valid"]
        if "world_bank_positive_ca_usd" in ca.columns:
            ca["World Bank Cross-check"] = ca["world_bank_positive_ca_usd"].map(_compact_usd)
            ca["WB vs IMF"] = ca["world_bank_difference_pct"].map(_signed_percent)
            columns.extend(["World Bank Cross-check", "WB vs IMF"])
        with st.expander("GlobalPositiveCA coverage audit"):
            st.dataframe(ca[columns], hide_index=True, use_container_width=True)


def render_convergence(snapshot: LukeGromenGoldSnapshot) -> None:
    st.markdown("##### Convergence Panel")
    if snapshot.convergence.empty:
        st.info("Convergence cannot be calculated until both model inputs are available.")
        return
    view = snapshot.convergence.copy()
    view["Implied Gold Price"] = view["implied_gold_price"].map(_money)
    view["vs Current"] = view["vs_current_pct"].map(_signed_percent)
    st.dataframe(view[["valuation", "Implied Gold Price", "vs Current"]].rename(columns={"valuation": "Valuation"}), hide_index=True, use_container_width=True)


def render_methodology() -> None:
    with st.expander("Methodology and source rules"):
        st.markdown(
            """
- Gold price: the existing long weekly TradingView `TVC:GOLD` series, USD per troy ounce.
- Foreign-held U.S. Treasury debt: `FDHBFIN` through January 2003 (billions × 1e9), then `FORTREASPOS69995` from February 2003 (millions × 1e6); forward-filled, never interpolated.
- U.S. official gold: 261,499,000 fine troy oz before 2012; from January 2012, the sum of all eight Treasury/FRED components only when every component is present.
- GlobalPositiveCA primary source: IMF WEO `BCA` on the Countries sheet, summed as `MAX(BCA, 0)` across individual economies. Missing values remain missing; Country Groups are never loaded.
- IMF WEO coverage gate: GDP-weighted coverage ≥95% and prior-year positive-surplus coverage ≥95%. Only completed `ACTUAL` or `ESTIMATE` years may calibrate Model 2; `FORECAST` is excluded.
- World Bank `BN.CAB.XOKA.CD` is retained only as a historical cross-check and never controls the calibration gate.
- WGC physical reconciliation uses Jewellery Fabrication. Tolerance is max(1 tonne, 0.02% of Total Supply).
- Adaptive demand elasticities: jewellery −0.77, recycling +0.44; jewellery floor 400 t and recycling cap 3,000 t.
            """
        )


def _mds_figure(annual: pd.DataFrame) -> go.Figure:
    data = annual.loc[pd.to_numeric(annual["year"], errors="coerce") >= 2010]
    fig = go.Figure()
    for column, label, color in (
        ("core_mds", "Core MDS", "#60a5fa"),
        ("broad_mds", "Broad MDS", "#fbbf24"),
    ):
        fig.add_trace(go.Scatter(x=data["year"], y=data[column] * 100.0, name=label, mode="lines+markers", line={"color": color}))
    fig.update_layout(title="Monetary Demand Share", height=300, margin={"l": 40, "r": 15, "t": 45, "b": 30}, template="plotly_dark", paper_bgcolor="#0b0e14", plot_bgcolor="#11161f", yaxis_ticksuffix="%")
    return fig


def _gmar_figure(annual: pd.DataFrame) -> go.Figure:
    data = annual.dropna(subset=["broad_gmar"]).copy() if "broad_gmar" in annual.columns else pd.DataFrame()
    fig = go.Figure()
    if not data.empty:
        for column, label, color in (
            ("core_gmar", "Core GMAR", "#a78bfa"),
            ("broad_gmar", "Broad GMAR", "#34d399"),
        ):
            fig.add_trace(go.Scatter(x=data["year"], y=data[column] * 100.0, name=label, mode="lines+markers", line={"color": color}))
    fig.update_layout(title="Gold Monetary Absorption Ratio", height=300, margin={"l": 40, "r": 15, "t": 45, "b": 30}, template="plotly_dark", paper_bgcolor="#0b0e14", plot_bgcolor="#11161f", yaxis_ticksuffix="%")
    return fig


def _number(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except Exception:
        return np.nan


def _money(value: Any) -> str:
    number = _number(value)
    return "n/a" if not np.isfinite(number) else f"${number:,.0f}"


def _compact_usd(value: Any) -> str:
    number = _number(value)
    if not np.isfinite(number):
        return "n/a"
    if abs(number) >= 1e12:
        return f"${number / 1e12:,.2f}T"
    if abs(number) >= 1e9:
        return f"${number / 1e9:,.1f}B"
    return f"${number:,.0f}"


def _compact_oz(value: Any) -> str:
    number = _number(value)
    return "n/a" if not np.isfinite(number) else f"{number / 1e6:,.1f}M oz"


def _tonnes(value: Any) -> str:
    number = _number(value)
    return "n/a" if not np.isfinite(number) else f"{number:,.1f} t"


def _percent(value: Any) -> str:
    number = _number(value)
    return "n/a" if not np.isfinite(number) else f"{number * 100.0:.1f}%"


def _signed_percent(value: Any) -> str:
    number = _number(value)
    return "n/a" if not np.isfinite(number) else f"{number * 100.0:+.1f}%"


def _integer(value: Any) -> str:
    number = _number(value)
    return "n/a" if not np.isfinite(number) else f"{number:.0f}"


def _date(value: Any) -> str:
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return "n/a"
