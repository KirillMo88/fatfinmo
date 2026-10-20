from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

from finance_core import drop_incomplete_daily_bar
from market_cycle import extract_ohlcv
from market_cycle_tab import (
    MARKET_CYCLE_PLOTLY_CONFIG,
    MARKET_CYCLE_RANGE_OPTIONS,
    MARKET_CYCLE_TTL_SECONDS,
    apply_common_x_range,
    build_momentum_cycle_fig,
    build_sma200m_extension_fig,
    build_sma200w_extension_fig,
    build_structural_momentum_cycle_fig,
    filter_range_domain,
    normalize_range_state,
    range_domain,
)
from qqq_cycle import build_ndx_cycle_history


NDX_TICKER = "^NDX"
NDX_LABEL = "NDX"


@st.cache_data(show_spinner=True, ttl=MARKET_CYCLE_TTL_SECONDS)
def load_ndx_prices_cached(refresh_nonce: int = 0) -> pd.DataFrame:
    _ = refresh_nonce
    raw = yf.download(
        NDX_TICKER,
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    prices = drop_incomplete_daily_bar(extract_ohlcv(raw, NDX_TICKER))
    if prices.empty:
        raise RuntimeError("yfinance did not return Nasdaq-100 history for ^NDX")
    return prices


def render_qqq_tab() -> None:
    st.subheader("QQQ")
    control_col, info_col, _ = st.columns([1.0, 2.2, 6.8])
    with control_col:
        refresh = st.button("Refresh QQQ", use_container_width=True, key="qqq_refresh_button")
    with info_col:
        st.caption("Base index: Nasdaq-100 (NDX, yfinance ^NDX)")

    if refresh:
        st.session_state["qqq_refresh_nonce"] = int(st.session_state.get("qqq_refresh_nonce", 0)) + 1
        st.rerun()

    try:
        prices = load_ndx_prices_cached(int(st.session_state.get("qqq_refresh_nonce", 0)))
        history = build_ndx_cycle_history(prices)
    except Exception as exc:
        st.error(f"QQQ / NDX cycle data is unavailable: {exc}")
        return

    if history.empty:
        st.info("QQQ / NDX cycle data is unavailable.")
        return

    normalize_range_state("qqq_cycle_range")
    selected_range = st.radio(
        "QQQ range",
        MARKET_CYCLE_RANGE_OPTIONS,
        index=3,
        horizontal=True,
        key="qqq_cycle_range",
    )
    range_start, range_end = range_domain(history, selected_range)
    display_history = filter_range_domain(history, range_start, range_end)

    st.markdown("### Structural")
    structural_col, structural_extension_col = st.columns(2)
    with structural_col:
        st.plotly_chart(
            apply_common_x_range(
                build_structural_momentum_cycle_fig(display_history, asset_label=NDX_LABEL),
                range_start,
                range_end,
            ),
            use_container_width=True,
            config=MARKET_CYCLE_PLOTLY_CONFIG,
        )
    with structural_extension_col:
        st.plotly_chart(
            apply_common_x_range(
                build_sma200m_extension_fig(display_history, asset_label=NDX_LABEL),
                range_start,
                range_end,
            ),
            use_container_width=True,
            config=MARKET_CYCLE_PLOTLY_CONFIG,
        )

    st.markdown("### Medium Term")
    momentum_col, momentum_extension_col = st.columns(2)
    with momentum_col:
        st.plotly_chart(
            apply_common_x_range(
                build_momentum_cycle_fig(display_history, asset_label=NDX_LABEL),
                range_start,
                range_end,
            ),
            use_container_width=True,
            config=MARKET_CYCLE_PLOTLY_CONFIG,
        )
    with momentum_extension_col:
        st.plotly_chart(
            apply_common_x_range(
                build_sma200w_extension_fig(display_history, asset_label=NDX_LABEL),
                range_start,
                range_end,
            ),
            use_container_width=True,
            config=MARKET_CYCLE_PLOTLY_CONFIG,
        )

    last_date = pd.to_datetime(history["Date"], errors="coerce").max()
    if pd.notna(last_date):
        st.caption(
            f"NDX data through {last_date:%Y-%m-%d}. Calculations use the same methodology and thresholds as the SPX Market Cycle charts."
        )
