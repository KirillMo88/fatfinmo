from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from finance_core import download_completed_ohlcv
from fred_client import download_fred_series
from global_liquidity import read_global_liquidity
from market_cycle import load_tradingview_breadth_series


TIMEFRAME_OPTIONS = {"Daily": "1D", "Weekly": "1W", "Monthly": "1M"}

ASSETS = (
    ("ES1!", "ES=F", "Yahoo Finance"),
    ("NQ1!", "NQ=F", "Yahoo Finance"),
    ("GOLD", "GC=F", "Yahoo Finance"),
    ("BTC", "BTC-USD", "Yahoo Finance"),
    ("DXY", "DX-Y.NYB", "Yahoo Finance"),
    ("WTI", "CL=F", "Yahoo Finance"),
)

RISK_MARKET = (
    ("VIX", "^VIX"),
    ("MOVE", "^MOVE"),
)

MAP_UNIVERSES: dict[str, dict[str, str]] = {
    "Assets Map": {
        "SPY": "SPY", "QQQ": "QQQ", "GOLD": "GC=F", "WTI": "CL=F", "HYG": "HYG",
        "TLT": "TLT", "BTC": "BTC-USD", "IEUR": "IEUR", "MCHI": "MCHI", "INDA": "INDA",
        "EWJ": "EWJ", "EWZ": "EWZ", "ASEA": "ASEA", "AI": "BOTZ",
    },
    "Equity Map": {
        "SPY": "SPY", "QQQ": "QQQ", "IWM": "IWM", "EFA": "EFA", "EEM": "EEM",
        "IEUR": "IEUR", "MCHI": "MCHI", "INDA": "INDA", "EWJ": "EWJ", "EWZ": "EWZ",
        "ASEA": "ASEA", "AI": "BOTZ",
    },
    "US Sectors Map": {
        symbol: symbol for symbol in ("XLK", "XLV", "XLF", "XLI", "XLB", "XLE", "XLU", "XLC", "XLY", "XLP")
    },
    "Commodity Map": {
        "Gold": "GC=F", "HG1": "HG=F", "WTI": "CL=F", "RBOB": "RB=F", "NatGas": "NG=F", "DBA": "DBA", "IAU": "IAU",
    },
    "Crypto Map": {
        "BTC": "BTC-USD", "ETH/BTC": "ETH-USD::BTC-USD", "SOL/BTC": "SOL-USD::BTC-USD", "XRP/BTC": "XRP-USD::BTC-USD",
        "TRX/BTC": "TRX-USD::BTC-USD", "SUI/BTC": "SUI-USD::BTC-USD", "ETH": "ETH-USD",
    },
    "Currency Map": {
        "USDX": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "JPY/USD": "JPY=X", "CHF/USD": "CHF=X",
        "CAD/USD": "CAD=X", "AUD/USD": "AUD=X",
    },
    "Yield Map": {
        "US10Y": "FRED:DGS10", "JP10Y": "FRED:IRLTLT01JPM156N", "CN10Y": "FRED:IRLTLT01CNM156N",
        "DE10Y": "FRED:IRLTLT01DEM156N", "FR10Y": "FRED:IRLTLT01FRM156N",
    },
}


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


@st.cache_data(ttl=3600, show_spinner=False)
def _load_yahoo(ticker: str) -> pd.DataFrame:
    try:
        frame = download_completed_ohlcv(ticker, period="max", auto_adjust=False, allow_archive_fallback=True)
    except Exception:
        return _empty_ohlcv()
    return frame.copy() if isinstance(frame, pd.DataFrame) else _empty_ohlcv()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_fred(series_id: str, api_key: str | None) -> pd.DataFrame:
    try:
        return download_fred_series(series_id, api_key=api_key, observation_start="2000-01-01").copy()
    except Exception:
        return pd.DataFrame(columns=["Series_ID", "Date", "Value"])


@st.cache_data(ttl=3600, show_spinner=False)
def _load_tv_economic(symbol: str) -> pd.DataFrame:
    try:
        from tradingview_mcp import get_economic_data

        result = get_economic_data(symbol, date_from="2000-01-01")
        if not result.frame.empty:
            return result.frame.copy()
    except Exception:
        pass
    # The archive is intentionally usable even when the live MCP endpoint is
    # unavailable or its short cache has expired.  This preserves the full
    # history needed by Dashboard charts without pretending the latest point
    # is live.
    try:
        from tradingview_mcp import ECONOMIC_CACHE_PATH

        path = ECONOMIC_CACHE_PATH
        if path.exists():
            cached = pd.read_csv(path)
            cached = cached.loc[cached["symbol"].astype(str).eq(symbol)].copy()
            if not cached.empty:
                cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
                cached["value"] = pd.to_numeric(cached["value"], errors="coerce")
                return cached.dropna(subset=["date", "value"])[["date", "value"]].sort_values("date")
    except Exception:
        pass
    return pd.DataFrame(columns=["date", "value"])


def _price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_ohlcv()
    out = frame.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index()
    out = out.rename(columns={column: column.title() for column in out.columns})
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])


def _series_frame(frame: pd.DataFrame, date_column: str = "Date", value_column: str = "Value") -> pd.DataFrame:
    if frame is None or frame.empty or date_column not in frame.columns or value_column not in frame.columns:
        return pd.DataFrame(columns=["Close"])
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    out = pd.DataFrame({"Close": values.to_numpy()}, index=dates)
    return out.loc[out.index.notna()].dropna().sort_index()


def _aggregate_price(frame: pd.DataFrame, timeframe: str, ticker: str) -> pd.DataFrame:
    daily = _price_frame(frame)
    if daily.empty or timeframe == "1D":
        return daily
    rule = "W-SUN" if ticker == "BTC-USD" else "W-FRI"
    rule = rule if timeframe == "1W" else "M"
    grouped = daily.resample(rule)
    return grouped.agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])


def _aggregate_series(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    values = _series_frame(frame)
    if values.empty:
        return values
    if timeframe == "1D":
        return values
    rule = "W-FRI" if timeframe == "1W" else "M"
    return values.resample(rule).last().dropna()


def _indicator_frame(series: pd.Series, timeframe: str) -> pd.DataFrame:
    close = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    out = pd.DataFrame({"close": close})
    for window in (50, 100, 200):
        out[f"sma{window}"] = close.rolling(window, min_periods=window).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    out["rsi14"] = rsi.mask(loss.eq(0.0) & gain.gt(0.0), 100.0).mask(loss.eq(0.0) & gain.eq(0.0), 50.0)
    return out


def _percentile_frame(indicators: pd.DataFrame, timeframe: str, *, versus_sma200: bool) -> pd.Series:
    close = indicators["close"]
    window = {"1D": 1260, "1W": 260, "1M": 60}[timeframe]
    if versus_sma200:
        distance = (close / indicators["sma200"] - 1.0) * 100.0
    else:
        distance = close
    return distance.rolling(window, min_periods=max(20, window // 2)).rank(pct=True).mul(100.0)


def _change_block(series: pd.Series, timeframe: str) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if values.empty:
        return "1D change: n/a<br>1W change: n/a<br>1M change: n/a"
    frequency = values.index.to_series().diff().median()
    if pd.isna(frequency) or frequency <= pd.Timedelta(days=2):
        periods = {"1D": 1, "1W": 5, "1M": 21}
    elif frequency <= pd.Timedelta(days=8):
        periods = {"1D": 1, "1W": 1, "1M": 4}
    else:
        periods = {"1D": 1, "1W": 1, "1M": 1}
    lines = []
    for label, period in periods.items():
        change = values.pct_change(periods=period, fill_method=None).iloc[-1] * 100.0
        lines.append(f"{label} change: {change:+.2f}%" if np.isfinite(change) else f"{label} change: n/a")
    return " · ".join(lines)


def _figure(
    frame: pd.DataFrame,
    title: str,
    timeframe: str,
    *,
    title_font_size: float = 14,
    percentile: pd.Series | None = None,
    percentile_title: str = "5Y Percentile",
    extra_series: dict[str, pd.Series] | None = None,
    candlesticks: bool = True,
) -> go.Figure:
    rows = 3 if percentile is not None else 2
    heights = [0.62, 0.20, 0.18] if rows == 3 else [0.78, 0.22]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=heights)
    x = frame.index
    if candlesticks:
        fig.add_trace(go.Candlestick(x=x, open=frame.get("Open"), high=frame.get("High"), low=frame.get("Low"), close=frame["close"], name=title, increasing_line_color="#22c55e", decreasing_line_color="#ef4444", showlegend=False), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=x, y=frame["close"], mode="lines", name=title, line={"color": "#e5e7eb", "width": 1.7}), row=1, col=1)
    colors = {50: "#38bdf8", 100: "#f59e0b", 200: "#ef4444"}
    for window in (50, 100, 200):
        fig.add_trace(go.Scatter(x=x, y=frame[f"sma{window}"], mode="lines", name=f"SMA{window}", line={"color": colors[window], "width": 1.2}), row=1, col=1)
    for label, values in (extra_series or {}).items():
        fig.add_trace(go.Scatter(x=values.index, y=values, mode="lines", name=label, line={"width": 1.4}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=frame["rsi14"], mode="lines", name="RSI14", line={"color": "#a78bfa", "width": 1.2}), row=2, col=1)
    fig.add_hline(y=70, line={"color": "#ef4444", "dash": "dot", "width": 0.8}, row=2, col=1)
    fig.add_hline(y=30, line={"color": "#22c55e", "dash": "dot", "width": 0.8}, row=2, col=1)
    if percentile is not None:
        fig.add_trace(go.Scatter(x=percentile.index, y=percentile, mode="lines", name=percentile_title, line={"color": "#fbbf24", "width": 1.4}), row=3, col=1)
        fig.add_hline(y=50, line={"color": "#94a3b8", "dash": "dot", "width": 0.8}, row=3, col=1)
        fig.update_yaxes(title_text="5Y pctl", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    if rows > 2:
        fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=True, title_text="Date", tickformat="%Y", nticks=6, row=rows, col=1)
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": title_font_size}},
        height=470 if rows == 3 else 390,
        margin={"l": 38, "r": 10, "t": 52, "b": 34},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": -0.14, "x": 0, "font": {"size": 9}},
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        annotations=[{"text": _change_block(frame["close"], timeframe), "xref": "paper", "yref": "paper", "x": 1, "y": 1.06, "xanchor": "right", "yanchor": "bottom", "showarrow": False, "align": "right", "font": {"size": 16.5, "color": "#cbd5e1"}}],
    )
    fig.update_xaxes(gridcolor="#263041")
    fig.update_yaxes(gridcolor="#263041", zeroline=False)
    return fig


def _render_grid(items: list[dict[str, Any]], timeframe: str, *, columns: int = 3) -> None:
    for start in range(0, len(items), columns):
        row = items[start:start + columns]
        cols = st.columns(columns)
        for column, item in zip(cols, row):
            with column:
                if item.get("frame") is None or item["frame"].empty:
                    st.warning(f"{item['title']}: data unavailable")
                    continue
                plot_item = dict(item)
                frame = item["frame"].copy()
                display_frame = frame.tail(250)
                display_index = display_frame.index
                plot_item["frame"] = display_frame
                if item.get("percentile") is not None:
                    plot_item["percentile"] = item["percentile"].reindex(display_index)
                if item.get("extra_series"):
                    plot_item["extra_series"] = {
                        label: values.reindex(display_index)
                        for label, values in item["extra_series"].items()
                    }
                fig = _figure(
                    plot_item["frame"], plot_item["title"], timeframe,
                    title_font_size=plot_item.get("title_font_size", 14),
                    percentile=plot_item.get("percentile"),
                    percentile_title=plot_item.get("percentile_title", "5Y Percentile"),
                    extra_series=plot_item.get("extra_series"),
                    candlesticks=plot_item.get("candlesticks", True),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"charts_dashboard_{item['key']}")


def _market_item(title: str, ticker: str, timeframe: str, *, percentile_mode: str | None = None, key: str | None = None) -> dict[str, Any]:
    daily = _load_yahoo(ticker)
    bars = _aggregate_price(daily, timeframe, ticker)
    if bars.empty:
        return {"title": title, "frame": pd.DataFrame(), "key": key or title}
    indicators = _indicator_frame(bars["Close"], timeframe)
    frame = indicators.rename(columns={"close": "close"}).copy()
    for source, target in (("Open", "Open"), ("High", "High"), ("Low", "Low"), ("Close", "Close"), ("Volume", "Volume")):
        if source in bars:
            frame[target] = bars[source].reindex(frame.index)
    percentile = None
    if percentile_mode:
        percentile = _percentile_frame(indicators, timeframe, versus_sma200=percentile_mode == "asset")
    return {"title": title, "frame": frame, "percentile": percentile, "percentile_title": "5Y Percentile vs SMA200" if percentile_mode == "asset" else "5Y Percentile", "key": key or ticker}


def _fred_item(title: str, series_id: str, api_key: str | None, timeframe: str, *, percentile_mode: str | None = None, key: str | None = None) -> dict[str, Any]:
    raw = _load_fred(series_id, api_key)
    values = _aggregate_series(raw, timeframe)
    if values.empty:
        return {"title": title, "frame": pd.DataFrame(), "key": key or series_id}
    indicators = _indicator_frame(values["Close"], timeframe)
    percentile = _percentile_frame(indicators, timeframe, versus_sma200=percentile_mode == "asset") if percentile_mode else None
    return {"title": title, "frame": indicators, "percentile": percentile, "percentile_title": "5Y Percentile vs SMA200" if percentile_mode == "asset" else "5Y Percentile", "candlesticks": False, "key": key or series_id}


def _tv_item(title: str, symbol: str, timeframe: str, *, percentile_mode: str | None = None, key: str | None = None) -> dict[str, Any]:
    raw = _load_tv_economic(symbol)
    values = _aggregate_series(raw.rename(columns={"date": "Date", "value": "Value"}), timeframe)
    if values.empty:
        return {"title": title, "frame": pd.DataFrame(), "key": key or symbol}
    indicators = _indicator_frame(values["Close"], timeframe)
    percentile = _percentile_frame(indicators, timeframe, versus_sma200=percentile_mode == "asset") if percentile_mode else None
    return {"title": title, "frame": indicators, "percentile": percentile, "percentile_title": "5Y Percentile vs SMA200" if percentile_mode == "asset" else "5Y Percentile", "candlesticks": False, "key": key or symbol}


def _breadth_item(title: str, symbol: str, timeframe: str, *, key: str) -> dict[str, Any]:
    try:
        raw = load_tradingview_breadth_series(symbol, pd.Timestamp.now(tz="UTC").tz_localize(None))
    except Exception:
        raw = _empty_ohlcv()
    if raw.empty:
        try:
            archived = _load_tv_economic(symbol)
            if not archived.empty:
                values = pd.to_numeric(archived["value"], errors="coerce").to_numpy()
                dates = pd.to_datetime(archived["date"], errors="coerce").to_numpy()
                raw = pd.DataFrame(
                    {"Open": values, "High": values, "Low": values, "Close": values, "Volume": np.nan},
                    index=dates,
                ).dropna(subset=["Close"])
        except Exception:
            raw = _empty_ohlcv()
    if raw.empty:
        try:
            from market_cycle_tab import load_market_cycle_snapshot_cached

            column = {"S5FI": "SPXAboveSMA50D", "S5TH": "SPXAboveSMA200D"}[symbol]
            snapshot = load_market_cycle_snapshot_cached(int(st.session_state.get("market_cycle_refresh_nonce", 0)))
            history = snapshot.history
            date_column = "Date" if "Date" in history.columns else "date"
            dates = pd.to_datetime(history[date_column], errors="coerce").to_numpy()
            values = pd.to_numeric(history[column], errors="coerce").to_numpy()
            raw = pd.DataFrame(
                {"Open": values, "High": values, "Low": values, "Close": values, "Volume": np.nan},
                index=dates,
            ).dropna(subset=["Close"])
        except Exception:
            raw = _empty_ohlcv()
    bars = _aggregate_price(raw, timeframe, "INDEX")
    if bars.empty:
        return {"title": title, "frame": pd.DataFrame(), "key": key}
    indicators = _indicator_frame(bars["Close"], timeframe)
    return {"title": title, "frame": indicators, "candlesticks": False, "key": key}


@st.cache_data(ttl=3600, show_spinner=False)
def _load_ism_services_series() -> pd.DataFrame:
    try:
        from business_cycle import load_ism_services_history, load_investing_ism_services_releases

        frames = [load_ism_services_history("1990-01-01"), load_investing_ism_services_releases("1990-01-01")]
        frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
        if not frames:
            return pd.DataFrame(columns=["Date", "Value"])
        out = pd.concat(frames, ignore_index=True)
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
        return out.dropna(subset=["Date", "Value"]).sort_values("Date").drop_duplicates("Date", keep="last")[["Date", "Value"]]
    except Exception:
        return pd.DataFrame(columns=["Date", "Value"])


def _ism_services_item(timeframe: str, *, key: str) -> dict[str, Any]:
    raw = _load_ism_services_series()
    values = _aggregate_series(raw, timeframe)
    if values.empty:
        return _tv_item("U.S. ISM Services PMI", "ECONOMICS:USNMI", timeframe, key=key)
    return {"title": "U.S. ISM Services PMI", "frame": _indicator_frame(values["Close"], timeframe), "candlesticks": False, "key": key}


def _spread_item(title: str, left: pd.Series, right: pd.Series, timeframe: str, *, key: str) -> dict[str, Any]:
    spread = pd.concat(
        [pd.to_numeric(left, errors="coerce").rename("left"), pd.to_numeric(right, errors="coerce").rename("right")],
        axis=1,
    ).sort_index().ffill().dropna()
    if spread.empty:
        return {"title": title, "frame": pd.DataFrame(), "key": key}
    values = _aggregate_series(pd.DataFrame({"Date": spread.index, "Value": spread["left"] - spread["right"]}), timeframe)
    if values.empty:
        return {"title": title, "frame": pd.DataFrame(), "key": key}
    return {"title": title, "frame": _indicator_frame(values["Close"], timeframe), "candlesticks": False, "key": key}


def _liquidity_item(title: str, series: pd.Series, timeframe: str, key: str) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if values.empty:
        return {"title": title, "frame": pd.DataFrame(), "key": key}
    if timeframe != "1W":
        values = values.resample("D" if timeframe == "1D" else "M").last().ffill()
    indicators = _indicator_frame(values, timeframe)
    return {"title": title, "frame": indicators, "candlesticks": False, "key": key}


def _load_map_data(mapping: dict[str, str], api_key: str | None) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}

    def load(item: tuple[str, str]) -> tuple[str, pd.Series]:
        label, ticker = item
        if ticker.startswith("FRED:"):
            raw = _load_fred(ticker.split(":", 1)[1], api_key)
            return label, _series_frame(raw)["Close"] if not raw.empty else pd.Series(dtype="float64")
        if "::" in ticker:
            numerator_ticker, denominator_ticker = ticker.split("::", 1)
            numerator = _price_frame(_load_yahoo(numerator_ticker))
            denominator = _price_frame(_load_yahoo(denominator_ticker))
            if numerator.empty or denominator.empty:
                return label, pd.Series(dtype="float64")
            ratio = numerator["Close"].div(denominator["Close"]).dropna()
            return label, ratio
        frame = _price_frame(_load_yahoo(ticker))
        return label, frame["Close"] if not frame.empty else pd.Series(dtype="float64")

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(mapping)))) as executor:
        futures = [executor.submit(load, item) for item in mapping.items()]
        for future in as_completed(futures):
            label, values = future.result()
            if not values.empty:
                result[label] = values
    return result


def _render_map(mapping_name: str, horizons: str, api_key: str | None) -> None:
    mapping = MAP_UNIVERSES[mapping_name]
    data = _load_map_data(mapping, api_key)
    if not data:
        st.warning(f"{mapping_name}: map data unavailable")
        return
    short_period, long_period = {"1W / 1M": (7, 30), "1M / 3M": (30, 91), "1M / 6M": (30, 182), "1M / 12M": (30, 365)}[horizons]
    rows = []
    for label, series in data.items():
        series = series.dropna().sort_index()
        if series.empty:
            continue
        end = series.index[-1]
        latest = float(series.iloc[-1])
        short_base = series.loc[: end - pd.Timedelta(days=short_period)]
        long_base = series.loc[: end - pd.Timedelta(days=long_period)]
        if short_base.empty or long_base.empty:
            continue
        rows.append({"label": label, "short": (latest / float(short_base.iloc[-1]) - 1.0) * 100.0, "long": (latest / float(long_base.iloc[-1]) - 1.0) * 100.0})
    points = pd.DataFrame(rows)
    if points.empty:
        st.info(f"{mapping_name}: insufficient history")
        return
    fig = go.Figure()
    colors = np.where(points["short"].ge(0), "#22c55e", "#ef4444")
    fig.add_trace(go.Scatter(x=points["long"], y=points["short"], mode="markers+text", text=points["label"], textposition="top center", marker={"size": 10, "color": colors}, hovertemplate="%{text}<br>Long: %{x:.2f}%<br>Short: %{y:.2f}%<extra></extra>"))
    fig.add_hline(y=0, line={"color": "#64748b", "dash": "dot"})
    fig.add_vline(x=0, line={"color": "#64748b", "dash": "dot"})
    fig.update_layout(title=f"{mapping_name} — {horizons}", height=520, template="plotly_dark", margin={"l": 42, "r": 18, "t": 52, "b": 50}, showlegend=False)
    fig.update_xaxes(title=f"Performance {horizons.split(' / ')[1]} (%)", autorange=True)
    fig.update_yaxes(title=f"Performance {horizons.split(' / ')[0]} (%)", autorange=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"charts_dashboard_map_{mapping_name}")


def _liquidity_series() -> dict[str, pd.Series]:
    try:
        _, monthly, weekly = read_global_liquidity()
    except Exception:
        return {}
    def as_indexed(frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame) or frame.empty or "date" not in frame.columns:
            return pd.DataFrame()
        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out.dropna(subset=["date"]).set_index("date").sort_index()

    monthly_frame = as_indexed(monthly)
    weekly_frame = as_indexed(weekly)
    if monthly_frame.empty and weekly_frame.empty:
        return {}
    output: dict[str, pd.Series] = {}
    for column, label in (("global_m2_usd_bn", "Global M2 52W change − 200W SMA"), ("global_cb_assets_usd_bn", "Global CB Assets 52W change − 200W SMA"), ("us_net_liquidity_usd_bn", "US Net Liquidity 52W change − 200W SMA")):
        if column in weekly_frame:
            value = pd.to_numeric(weekly_frame[column], errors="coerce").dropna()
        else:
            value = pd.Series(dtype="float64")
        if value.empty and column in monthly_frame:
            value = pd.to_numeric(monthly_frame[column], errors="coerce").dropna().resample("W-FRI").last().ffill()
        if value.empty:
            continue
        change = value.pct_change(52, fill_method=None).mul(100.0)
        gap = change - change.rolling(200, min_periods=200).mean()
        output[label] = gap.dropna()
    for column, label in (("WTREGEN", "WTREGEN"), ("RRPONTSYD", "RRPONTSYD")):
        if column in weekly_frame:
            output[label] = pd.to_numeric(weekly_frame[column], errors="coerce").dropna()
        elif column in monthly_frame:
            output[label] = pd.to_numeric(monthly_frame[column], errors="coerce").dropna()
    return output


def render_charts_dashboard(fred_api_key: str | None = None) -> None:
    st.markdown("## Charts Dashboard")
    timeframe_label = st.radio("Candles", list(TIMEFRAME_OPTIONS), index=0, horizontal=True, key="charts_dashboard_timeframe")
    timeframe = TIMEFRAME_OPTIONS[timeframe_label]
    st.caption("Yahoo Finance market history, FRED macro series and existing Liquidity Cycle calculations. Each chart displays the latest 250 bars; indicators use the full available history.")

    st.markdown("### Assets")
    asset_items = [_market_item(label, ticker, timeframe, percentile_mode="asset", key=f"asset_{ticker}") for label, ticker, _ in ASSETS]
    for item in asset_items:
        item["title_font_size"] = 21
    _render_grid(asset_items, timeframe)

    st.markdown("### Risk")
    risk_items = [_market_item(label, ticker, timeframe, percentile_mode="risk", key=f"risk_{ticker}") for label, ticker in RISK_MARKET]
    risk_items.append(_fred_item("BAMLH0A0HYM2", "BAMLH0A0HYM2", fred_api_key, timeframe, percentile_mode="risk", key="risk_hy_oas"))
    risk_items.append(_breadth_item("S5FI — SPX above SMA50", "S5FI", timeframe, key="risk_s5fi"))
    risk_items.append(_breadth_item("S5TH — SPX above SMA200", "S5TH", timeframe, key="risk_s5th"))
    funding_items: list[dict[str, Any]] = []
    try:
        _, _, weekly = read_global_liquidity()
        if isinstance(weekly, pd.DataFrame) and not weekly.empty and {"date", "SOFR", "EFFR"}.issubset(weekly.columns):
            weekly = weekly.copy()
            weekly["date"] = pd.to_datetime(weekly["date"], errors="coerce")
            weekly = weekly.dropna(subset=["date"]).set_index("date")
            sofr = pd.to_numeric(weekly["SOFR"], errors="coerce").dropna()
            effr = pd.to_numeric(weekly["EFFR"], errors="coerce").dropna()
            funding_items.append(_spread_item("SOFR − EFFR", sofr, effr, timeframe, key="risk_sofr_effr"))
            if funding_items[-1]["frame"].empty:
                funding_items.clear()
    except Exception:
        pass
    if not funding_items:
        sofr_item = _fred_item("SOFR", "SOFR", fred_api_key, timeframe, key="risk_sofr")
        effr_item = _fred_item("EFFR", "EFFR", fred_api_key, timeframe, key="risk_effr_overlay")
        if not sofr_item["frame"].empty and not effr_item["frame"].empty:
            funding_items.append(_spread_item("SOFR − EFFR", sofr_item["frame"]["close"], effr_item["frame"]["close"], timeframe, key="risk_sofr_effr"))
        else:
            funding_items.append({"title": "SOFR − EFFR", "frame": pd.DataFrame(), "key": "risk_sofr_effr"})
    risk_items.extend(funding_items)
    _render_grid(risk_items, timeframe)

    st.markdown("### Liquidity")
    liquidity_items = [_liquidity_item(label, values, timeframe, f"liquidity_{index}") for index, (label, values) in enumerate(_liquidity_series().items())]
    _render_grid(liquidity_items, timeframe)

    st.markdown("### Economy")
    economy_items: list[dict[str, Any]] = []
    economy_items.append(_tv_item("USBCOI", "ECONOMICS:USBCOI", timeframe, key="economy_usbcoi"))
    economy_items.append(_ism_services_item(timeframe, key="economy_ism_services"))
    economy_items.append(_fred_item("US10Y", "DGS10", fred_api_key, timeframe, key="economy_us10y"))
    us02y = _fred_item("US02Y", "DGS2", fred_api_key, timeframe, key="economy_us02y")
    fedfunds = _fred_item("FEDFUNDS", "FEDFUNDS", fred_api_key, timeframe, key="economy_fedfunds")
    if not us02y["frame"].empty:
        extra = {"FEDFUNDS": fedfunds["frame"]["close"]} if not fedfunds["frame"].empty else {}
        us02y["title"] = "US02Y + FEDFUNDS"
        us02y["extra_series"] = extra
        us02y["key"] = "economy_us02y_fedfunds"
    economy_items.append(us02y)
    economy_items.append(_fred_item("EXPINF1YR", "EXPINF1YR", fred_api_key, timeframe, key="economy_expinf1yr"))
    economy_items.append(_fred_item("EXPINF5YR", "EXPINF5YR", fred_api_key, timeframe, key="economy_expinf5yr"))
    economy_items.append(_tv_item("USIRYY", "ECONOMICS:USIRYY", timeframe, key="economy_usiryy"))
    economy_items.append(_fred_item("PPIACO", "PPIACO", fred_api_key, timeframe, key="economy_ppiaco"))
    _render_grid(economy_items, timeframe)

    st.markdown("### Maps")
    map_col, horizon_col = st.columns([1.4, 1])
    with map_col:
        map_name = st.selectbox("Map", list(MAP_UNIVERSES), index=0, key="charts_dashboard_map_name")
    with horizon_col:
        horizon = st.radio("Performance horizons", ["1W / 1M", "1M / 3M", "1M / 6M", "1M / 12M"], index=0, horizontal=True, key="charts_dashboard_map_horizon")
    _render_map(map_name, horizon, fred_api_key)
