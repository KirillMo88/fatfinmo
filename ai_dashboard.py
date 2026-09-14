from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from alpha_engine import calculate_sma200d_robust_z_36m
from finance_core import drop_incomplete_daily_bar, extract_ohlcv_frame
from screener_metrics import historical_momentum_52w_metrics, sma200w_distance_percentile


AI_UNIVERSE: dict[str, list[str]] = {
    "Hyperscalers": ["MSFT", "AMZN", "GOOGL", "META", "ORCL", "PLTR"],
    "AI Compute": ["NVDA", "AMD", "AVGO", "MRVL", "ARM"],
    "Networking": ["ANET", "CSCO", "CRDO", "ALAB", "COHR", "LITE"],
    "Memory": ["MU", "000660.KS", "005930.KS"],
    "Manufacturing": ["TSM", "INTC"],
    "Semiconductor Equipment": ["ASML", "AMAT", "LRCX", "KLAC", "CDNS", "SNPS"],
    "AI Servers": ["CLS", "DELL", "HPE"],
    "Power & Cooling": ["VRT", "ETN", "JCI", "NVT"],
    "Grid": ["GEV", "PWR", "HUBB", "POWL"],
    "Power Producers": ["CEG", "TLN", "VST"],
    "Data Center Owners": ["EQIX", "DLR"],
}

AI_COMPANY_NAMES: dict[str, str] = {
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "ORCL": "Oracle",
    "PLTR": "Palantir",
    "NVDA": "NVIDIA",
    "AMD": "AMD",
    "AVGO": "Broadcom",
    "MRVL": "Marvell Technology",
    "ARM": "Arm Holdings",
    "ANET": "Arista Networks",
    "CSCO": "Cisco",
    "CRDO": "Credo Technology",
    "ALAB": "Astera Labs",
    "COHR": "Coherent",
    "LITE": "Lumentum",
    "MU": "Micron",
    "000660.KS": "SK hynix",
    "005930.KS": "Samsung Electronics",
    "TSM": "Taiwan Semiconductor",
    "INTC": "Intel",
    "ASML": "ASML",
    "AMAT": "Applied Materials",
    "LRCX": "Lam Research",
    "KLAC": "KLA",
    "CDNS": "Cadence Design Systems",
    "SNPS": "Synopsys",
    "CLS": "Celestica",
    "DELL": "Dell Technologies",
    "HPE": "Hewlett Packard Enterprise",
    "VRT": "Vertiv",
    "ETN": "Eaton",
    "JCI": "Johnson Controls",
    "NVT": "nVent Electric",
    "GEV": "GE Vernova",
    "PWR": "Quanta Services",
    "HUBB": "Hubbell",
    "POWL": "Powell Industries",
    "CEG": "Constellation Energy",
    "TLN": "Talen Energy",
    "VST": "Vistra",
    "EQIX": "Equinix",
    "DLR": "Digital Realty",
}

PERFORMANCE_WINDOWS: dict[str, int] = {
    "Perf 1D": 1,
    "Perf 1W": 7,
    "Perf 1M": 30,
    "Perf 3M": 90,
    "Perf 6M": 182,
    "Perf 12M": 365,
    "Perf 3Y": 365 * 3,
    "Perf 5Y": 365 * 5,
    "Perf 10Y": 365 * 10,
}

BENCHMARK_TICKERS = ["SPY", "QQQ"]
USD_KRW_TICKER = "KRW=X"
KRW_QUOTED_TICKERS = {"000660.KS", "005930.KS", "006930.KS"}

COMPANY_COLUMNS = [
    "Group",
    "Company",
    "Ticker",
    "Market Cap",
    *PERFORMANCE_WINDOWS.keys(),
    "Price vs SMA200D",
    "SMA200D Robust Z 36M",
    "SMA200W Percentile",
    "Perf 12M Percentile",
    "Relative Group 1M",
    "Relative Group 3M",
    "Relative Group 12M",
    "Relative Benchmark 1M",
    "Relative Benchmark 3M",
    "Relative Benchmark 6M",
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
    "Above SMA50",
    "Above SMA200",
    "Data Status",
]

GROUP_COLUMNS = [
    "Group",
    "Companies",
    "Total Market Cap",
    *PERFORMANCE_WINDOWS.keys(),
    "SMA200W Percentile",
    "Perf 12M Percentile",
    "SMA200D Robust Z 36M",
    "Price vs SMA200D",
    "% Above SMA50",
    "% Above SMA200",
    "% Positive 1M",
    "% Positive 3M",
    "% Positive 12M",
    "Median Trailing P/E",
    "Median Forward P/E",
    "Median PEG",
    "Median Price/Sales",
    "Median Quarterly Revenue Growth YoY",
    "Median Revenue Growth 3Y",
    "Median Operating Margin TTM",
    "Median Price vs ATH",
    "Relative Benchmark 1M",
    "Relative Benchmark 3M",
    "Relative Benchmark 6M",
    "Relative Benchmark 12M",
]


@dataclass(frozen=True)
class AIDashboardData:
    companies: pd.DataFrame
    groups_equal: pd.DataFrame
    groups_mcap: pd.DataFrame
    prices: dict[str, pd.DataFrame]
    benchmark_returns: dict[str, dict[str, float]]
    fundamentals_errors: dict[str, str]
    generated_at: str


def ai_tickers(include_benchmarks: bool = False) -> list[str]:
    tickers = [ticker for members in AI_UNIVERSE.values() for ticker in members]
    if include_benchmarks:
        tickers.extend(BENCHMARK_TICKERS)
    return list(dict.fromkeys(tickers))


def ticker_group_map() -> dict[str, str]:
    return {ticker: group for group, tickers in AI_UNIVERSE.items() for ticker in tickers}


def load_ai_price_history(tickers: list[str]) -> dict[str, pd.DataFrame]:
    unique = list(dict.fromkeys(tickers))
    download_tickers = unique.copy()
    needs_krw_fx = any(is_krw_quoted_ticker(ticker) for ticker in unique)
    if needs_krw_fx and USD_KRW_TICKER not in download_tickers:
        download_tickers.append(USD_KRW_TICKER)
    try:
        raw = yf.download(
            download_tickers,
            period="max",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        raw = pd.DataFrame()

    prices: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    fx_frame = drop_incomplete_daily_bar(extract_ohlcv_frame(raw, USD_KRW_TICKER)) if needs_krw_fx else pd.DataFrame()
    for ticker in unique:
        frame = drop_incomplete_daily_bar(extract_ohlcv_frame(raw, ticker))
        prices[ticker] = frame
        if frame.empty:
            missing.append(ticker)

    for ticker in missing:
        try:
            single = yf.download(
                ticker,
                period="max",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception:
            single = pd.DataFrame()
        prices[ticker] = drop_incomplete_daily_bar(extract_ohlcv_frame(single, ticker))
    if needs_krw_fx and fx_frame.empty:
        fx_frame = load_usd_krw_history()
    if needs_krw_fx and not fx_frame.empty:
        for ticker in unique:
            if is_krw_quoted_ticker(ticker):
                prices[ticker] = convert_krw_ohlcv_to_usd(prices.get(ticker, pd.DataFrame()), fx_frame)
    return prices


def load_ai_fundamentals(tickers: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    data: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    usd_krw_rate = load_latest_usd_krw_rate() if any(is_krw_quoted_ticker(ticker) for ticker in tickers) else np.nan
    for ticker in tickers:
        try:
            data[ticker] = load_yahoo_fundamental_row(ticker, usd_krw_rate)
        except Exception as exc:
            data[ticker] = {}
            errors[ticker] = str(exc)
    return data, errors


def load_yahoo_fundamental_row(ticker: str, usd_krw_rate: float = np.nan) -> dict[str, Any]:
    yf_ticker = yf.Ticker(ticker)
    info = safe_info(yf_ticker)
    annual = safe_financials(yf_ticker, quarterly=False)
    quarterly = safe_financials(yf_ticker, quarterly=True)
    local_market_cap = numeric_or_nan(info.get("marketCap"))
    revenue_ttm = numeric_or_nan(info.get("totalRevenue"))
    operating_margin = numeric_or_nan(info.get("operatingMargins"))
    if not np.isfinite(operating_margin):
        operating_margin = operating_margin_ttm(quarterly)
    price_sales = numeric_or_nan(info.get("priceToSalesTrailing12Months"))
    if not np.isfinite(price_sales) and np.isfinite(local_market_cap) and np.isfinite(revenue_ttm) and revenue_ttm > 0:
        price_sales = local_market_cap / revenue_ttm
    market_cap = convert_krw_value_to_usd(ticker, local_market_cap, usd_krw_rate)

    trailing_pe = numeric_or_nan(info.get("trailingPE"))
    trailing_eps = numeric_or_nan(info.get("trailingEps"))
    if np.isfinite(trailing_eps) and trailing_eps <= 0:
        trailing_pe = np.nan
    if np.isfinite(trailing_pe) and trailing_pe < 0:
        trailing_pe = np.nan

    return {
        "Market Cap": market_cap,
        "Trailing P/E": trailing_pe,
        "Forward P/E": numeric_or_nan(info.get("forwardPE")),
        "PEG Ratio": first_finite(info.get("pegRatio"), info.get("trailingPegRatio")),
        "Price/Sales": price_sales,
        "Quarterly Revenue Growth YoY": quarterly_revenue_growth_yoy(info, quarterly),
        "Revenue Growth 3Y": revenue_growth_3y(revenue_ttm, annual, quarterly),
        "Operating Margin TTM": operating_margin,
    }


def is_krw_quoted_ticker(ticker: str) -> bool:
    return str(ticker).upper() in KRW_QUOTED_TICKERS


def load_usd_krw_history() -> pd.DataFrame:
    try:
        raw = yf.download(
            USD_KRW_TICKER,
            period="max",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        raw = pd.DataFrame()
    return drop_incomplete_daily_bar(extract_ohlcv_frame(raw, USD_KRW_TICKER))


def load_latest_usd_krw_rate() -> float:
    try:
        raw = yf.download(
            USD_KRW_TICKER,
            period="10d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        raw = pd.DataFrame()
    frame = drop_incomplete_daily_bar(extract_ohlcv_frame(raw, USD_KRW_TICKER))
    close = pd.to_numeric(frame.get("Close", pd.Series(dtype="float64")), errors="coerce").dropna()
    return float(close.iloc[-1]) if not close.empty else np.nan


def convert_krw_ohlcv_to_usd(frame: pd.DataFrame, usd_krw: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or usd_krw.empty:
        return frame
    out = frame.copy().sort_index()
    fx_close = pd.to_numeric(usd_krw.get("Close", pd.Series(dtype="float64")), errors="coerce").dropna().sort_index()
    if fx_close.empty:
        return out
    aligned_fx = fx_close.reindex(out.index, method="ffill")
    valid = aligned_fx.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return out
    for column in ["Open", "High", "Low", "Close"]:
        if column in out.columns:
            values = pd.to_numeric(out[column], errors="coerce")
            out[column] = values / aligned_fx
    return out


def convert_krw_value_to_usd(ticker: str, value: float, usd_krw_rate: float) -> float:
    if not is_krw_quoted_ticker(ticker):
        return value
    if not np.isfinite(value) or not np.isfinite(usd_krw_rate) or usd_krw_rate <= 0:
        return value
    return value / usd_krw_rate


def safe_info(ticker: yf.Ticker) -> dict[str, Any]:
    try:
        info = ticker.get_info()
        return info if isinstance(info, dict) else {}
    except Exception:
        try:
            return ticker.info if isinstance(ticker.info, dict) else {}
        except Exception:
            return {}


def safe_financials(ticker: yf.Ticker, quarterly: bool) -> pd.DataFrame:
    try:
        frame = ticker.quarterly_financials if quarterly else ticker.financials
    except Exception:
        return pd.DataFrame()
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def metric_row(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype="float64")
    lower_map = {str(index).strip().lower(): index for index in frame.index}
    for name in names:
        key = name.lower()
        if key in lower_map:
            return pd.to_numeric(frame.loc[lower_map[key]], errors="coerce").dropna()
    return pd.Series(dtype="float64")


def quarterly_revenue_growth_yoy(info: dict[str, Any], quarterly: pd.DataFrame) -> float:
    revenue = metric_row(quarterly, ("Total Revenue", "Revenue"))
    if len(revenue) >= 5:
        revenue = revenue.sort_index()
        latest = float(revenue.iloc[-1])
        prior = float(revenue.iloc[-5])
        if prior != 0.0 and np.isfinite(latest) and np.isfinite(prior):
            return latest / prior - 1.0
    ready = numeric_or_nan(info.get("revenueGrowth"))
    return ready if np.isfinite(ready) else np.nan


def operating_margin_ttm(quarterly: pd.DataFrame) -> float:
    revenue = metric_row(quarterly, ("Total Revenue", "Revenue")).sort_index().tail(4)
    operating_income = metric_row(quarterly, ("Operating Income", "Operating Income Or Loss")).sort_index().tail(4)
    if len(revenue) < 4 or len(operating_income) < 4:
        return np.nan
    rev = float(revenue.sum())
    op = float(operating_income.sum())
    if rev == 0.0 or not np.isfinite(rev) or not np.isfinite(op):
        return np.nan
    return op / rev


def revenue_growth_3y(revenue_ttm: float, annual: pd.DataFrame, quarterly: pd.DataFrame) -> float:
    annual_revenue = metric_row(annual, ("Total Revenue", "Revenue")).sort_index()
    if np.isfinite(revenue_ttm) and revenue_ttm > 0 and len(annual_revenue) >= 3:
        base = float(annual_revenue.iloc[-3])
        if np.isfinite(base) and base > 0:
            return (revenue_ttm / base) ** (1.0 / 3.0) - 1.0
    if len(annual_revenue) >= 4:
        latest = float(annual_revenue.iloc[-1])
        base = float(annual_revenue.iloc[-4])
        if np.isfinite(latest) and np.isfinite(base) and base > 0:
            return (latest / base) ** (1.0 / 3.0) - 1.0
    return np.nan


def calculate_ai_dashboard(
    prices: dict[str, pd.DataFrame] | None = None,
    fundamentals: dict[str, dict[str, Any]] | None = None,
    fundamentals_errors: dict[str, str] | None = None,
) -> AIDashboardData:
    tickers = ai_tickers(include_benchmarks=True)
    prices = prices if prices is not None else load_ai_price_history(tickers)
    fund, errors = (fundamentals or {}), (fundamentals_errors or {})
    if fundamentals is None:
        fund, errors = load_ai_fundamentals(ai_tickers())

    companies = build_company_table(prices, fund)
    benchmark_returns = {
        ticker: performance_map(prices.get(ticker, pd.DataFrame()).get("Close", pd.Series(dtype="float64")))
        for ticker in BENCHMARK_TICKERS
    }
    companies = add_relative_group_returns(companies)
    companies = add_relative_benchmark_returns(companies, benchmark_returns.get("QQQ", {}))
    groups_equal = build_group_table(companies, weighting="Equal Weighted", benchmark_returns=benchmark_returns.get("QQQ", {}))
    groups_mcap = build_group_table(companies, weighting="Market Cap Weighted", benchmark_returns=benchmark_returns.get("QQQ", {}))
    return AIDashboardData(
        companies=companies,
        groups_equal=groups_equal,
        groups_mcap=groups_mcap,
        prices=prices,
        benchmark_returns=benchmark_returns,
        fundamentals_errors=errors,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def build_company_table(prices: dict[str, pd.DataFrame], fundamentals: dict[str, dict[str, Any]]) -> pd.DataFrame:
    group_map = ticker_group_map()
    rows: list[dict[str, Any]] = []
    for ticker in ai_tickers():
        group = group_map[ticker]
        frame = prices.get(ticker, pd.DataFrame())
        close = pd.to_numeric(frame.get("Close", pd.Series(dtype="float64")), errors="coerce").dropna()
        row: dict[str, Any] = {
            "Group": group,
            "Company": AI_COMPANY_NAMES.get(ticker, ticker),
            "Ticker": ticker,
            "Data Status": "PRICE_MISSING" if close.empty else "CURRENT",
        }
        row.update(fundamentals.get(ticker, {}))
        row.update(company_price_metrics(frame))
        rows.append(row)
    return pd.DataFrame(rows, columns=COMPANY_COLUMNS)


def company_price_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    close = pd.to_numeric(frame.get("Close", pd.Series(dtype="float64")), errors="coerce").dropna()
    high = pd.to_numeric(frame.get("High", pd.Series(dtype="float64")), errors="coerce").dropna()
    metrics = {period: np.nan for period in PERFORMANCE_WINDOWS}
    if close.empty:
        return {
            **metrics,
            "Price vs SMA200D": np.nan,
            "SMA200D Robust Z 36M": np.nan,
            "SMA200W Percentile": np.nan,
            "Price vs ATH": np.nan,
            "ATH Date": None,
            "Above SMA50": np.nan,
            "Above SMA200": np.nan,
            "Perf 12M Percentile": np.nan,
        }

    metrics.update(performance_map(close))
    current = float(close.iloc[-1])
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    sma200_latest = last_finite(sma200)
    weekly_close = close.resample("W-FRI").last().dropna()
    ath_series = high if not high.empty else close
    ath = float(ath_series.max()) if not ath_series.empty else np.nan
    ath_date = pd.Timestamp(ath_series.idxmax()).date().isoformat() if np.isfinite(ath) else None
    perf_12m_percentile, _ = historical_momentum_52w_metrics(close)
    return {
        **metrics,
        "Price vs SMA200D": np.nan if not np.isfinite(sma200_latest) or sma200_latest == 0 else current / sma200_latest - 1.0,
        "SMA200D Robust Z 36M": calculate_sma200d_robust_z_36m(close),
        "SMA200W Percentile": sma200w_distance_percentile(weekly_close),
        "Price vs ATH": np.nan if not np.isfinite(ath) or ath == 0 else current / ath - 1.0,
        "ATH Date": ath_date,
        "Above SMA50": bool(current > last_finite(sma50)) if np.isfinite(last_finite(sma50)) else np.nan,
        "Above SMA200": bool(current > sma200_latest) if np.isfinite(sma200_latest) else np.nan,
        "Perf 12M Percentile": perf_12m_percentile,
    }


def performance_map(close: pd.Series) -> dict[str, float]:
    prices = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if prices.empty:
        return {period: np.nan for period in PERFORMANCE_WINDOWS}
    end = prices.index[-1]
    return {period: performance_on_or_before(prices, end, days) for period, days in PERFORMANCE_WINDOWS.items()}


def performance_on_or_before(close: pd.Series, end: pd.Timestamp, days: int) -> float:
    start = end - pd.DateOffset(days=int(days))
    before = close.loc[:start]
    if before.empty:
        return np.nan
    v0 = float(before.iloc[-1])
    v1 = float(close.loc[:end].iloc[-1])
    if v0 == 0.0 or not np.isfinite(v0) or not np.isfinite(v1):
        return np.nan
    return (v1 / v0 - 1.0) * 100.0


def add_relative_group_returns(companies: pd.DataFrame) -> pd.DataFrame:
    out = companies.copy()
    equal_groups = build_group_return_lookup(out)
    for period, label in [("Perf 1M", "Relative Group 1M"), ("Perf 3M", "Relative Group 3M"), ("Perf 12M", "Relative Group 12M")]:
        out[label] = out.apply(lambda row: relative_to_lookup(row, period, equal_groups), axis=1)
    return out


def add_relative_benchmark_returns(companies: pd.DataFrame, benchmark: dict[str, float]) -> pd.DataFrame:
    out = companies.copy()
    for period, label in [
        ("Perf 1M", "Relative Benchmark 1M"),
        ("Perf 3M", "Relative Benchmark 3M"),
        ("Perf 6M", "Relative Benchmark 6M"),
        ("Perf 12M", "Relative Benchmark 12M"),
    ]:
        base = benchmark.get(period, np.nan)
        out[label] = pd.to_numeric(out[period], errors="coerce") - base if np.isfinite(base) else np.nan
    return out


def build_group_return_lookup(companies: pd.DataFrame) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for group, block in companies.groupby("Group", dropna=False):
        for period in PERFORMANCE_WINDOWS:
            lookup[(str(group), period)] = float(pd.to_numeric(block[period], errors="coerce").mean(skipna=True))
    return lookup


def relative_to_lookup(row: pd.Series, period: str, lookup: dict[tuple[str, str], float]) -> float:
    value = numeric_or_nan(row.get(period))
    base = lookup.get((str(row.get("Group")), period), np.nan)
    return value - base if np.isfinite(value) and np.isfinite(base) else np.nan


def build_group_table(companies: pd.DataFrame, weighting: str = "Equal Weighted", benchmark_returns: dict[str, float] | None = None) -> pd.DataFrame:
    benchmark_returns = benchmark_returns or {}
    rows: list[dict[str, Any]] = []
    for group, block in companies.groupby("Group", sort=False):
        row: dict[str, Any] = {
            "Group": group,
            "Companies": int(block["Ticker"].nunique()),
            "Total Market Cap": pd.to_numeric(block["Market Cap"], errors="coerce").sum(min_count=1),
        }
        for period in PERFORMANCE_WINDOWS:
            row[period] = aggregate_return(block, period, weighting)
        row["SMA200W Percentile"] = median_numeric(block["SMA200W Percentile"])
        row["Perf 12M Percentile"] = median_numeric(block["Perf 12M Percentile"])
        row["SMA200D Robust Z 36M"] = median_numeric(block["SMA200D Robust Z 36M"])
        row["Price vs SMA200D"] = median_numeric(block["Price vs SMA200D"])
        row["% Above SMA50"] = breadth_percent(block["Above SMA50"])
        row["% Above SMA200"] = breadth_percent(block["Above SMA200"])
        row["% Positive 1M"] = positive_percent(block["Perf 1M"])
        row["% Positive 3M"] = positive_percent(block["Perf 3M"])
        row["% Positive 12M"] = positive_percent(block["Perf 12M"])
        row["Median Trailing P/E"] = median_numeric(block["Trailing P/E"])
        row["Median Forward P/E"] = median_numeric(block["Forward P/E"])
        row["Median PEG"] = median_numeric(block["PEG Ratio"])
        row["Median Price/Sales"] = median_numeric(block["Price/Sales"])
        row["Median Quarterly Revenue Growth YoY"] = median_numeric(block["Quarterly Revenue Growth YoY"])
        row["Median Revenue Growth 3Y"] = median_numeric(block["Revenue Growth 3Y"])
        row["Median Operating Margin TTM"] = median_numeric(block["Operating Margin TTM"])
        row["Median Price vs ATH"] = median_numeric(block["Price vs ATH"])
        for period, label in [
            ("Perf 1M", "Relative Benchmark 1M"),
            ("Perf 3M", "Relative Benchmark 3M"),
            ("Perf 6M", "Relative Benchmark 6M"),
            ("Perf 12M", "Relative Benchmark 12M"),
        ]:
            base = benchmark_returns.get(period, np.nan)
            row[label] = row[period] - base if np.isfinite(row[period]) and np.isfinite(base) else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=GROUP_COLUMNS)


def aggregate_return(block: pd.DataFrame, period: str, weighting: str) -> float:
    returns = pd.to_numeric(block[period], errors="coerce")
    if weighting == "Market Cap Weighted":
        caps = pd.to_numeric(block["Market Cap"], errors="coerce")
        valid = returns.notna() & caps.notna() & (caps > 0)
        if not valid.any():
            return np.nan
        weights = caps.loc[valid] / caps.loc[valid].sum()
        return float((returns.loc[valid] * weights).sum())
    return float(returns.mean(skipna=True))


def group_rows_for_benchmark(companies: pd.DataFrame, weighting: str, benchmark_returns: dict[str, float]) -> pd.DataFrame:
    groups = build_group_table(companies, weighting=weighting, benchmark_returns=benchmark_returns)
    return groups


def group_components_table(companies: pd.DataFrame, groups: pd.DataFrame, group: str) -> pd.DataFrame:
    group_row = groups[groups["Group"].eq(group)]
    block = companies[companies["Group"].eq(group)].copy()
    if group_row.empty:
        return block
    first = {
        "Group": group,
        "Company": "GROUP",
        "Ticker": "GROUP",
        "Market Cap": group_row.iloc[0].get("Total Market Cap"),
    }
    for column in [
        *PERFORMANCE_WINDOWS.keys(),
        "Price vs SMA200D",
        "SMA200D Robust Z 36M",
        "SMA200W Percentile",
        "Perf 12M Percentile",
        "Price vs ATH",
        "Relative Benchmark 1M",
        "Relative Benchmark 3M",
        "Relative Benchmark 6M",
        "Relative Benchmark 12M",
    ]:
        mapped = column
        if column == "Price vs ATH":
            mapped = "Median Price vs ATH"
        first[column] = group_row.iloc[0].get(mapped)
    return pd.concat([pd.DataFrame([first]), block], ignore_index=True)


def normalized_group_history(prices: dict[str, pd.DataFrame], group: str, start_date: pd.Timestamp) -> pd.Series:
    series_list = []
    for ticker in AI_UNIVERSE.get(group, []):
        close = pd.to_numeric(prices.get(ticker, pd.DataFrame()).get("Close", pd.Series(dtype="float64")), errors="coerce").dropna()
        close = close.loc[close.index >= start_date]
        if close.empty:
            continue
        series_list.append((close / close.iloc[0]) * 100.0)
    if not series_list:
        return pd.Series(dtype="float64")
    combined = pd.concat(series_list, axis=1).sort_index()
    return combined.mean(axis=1, skipna=True).rename(group)


def normalized_ticker_history(prices: dict[str, pd.DataFrame], ticker: str, start_date: pd.Timestamp, name: str | None = None) -> pd.Series:
    close = pd.to_numeric(prices.get(ticker, pd.DataFrame()).get("Close", pd.Series(dtype="float64")), errors="coerce").dropna()
    close = close.loc[close.index >= start_date]
    if close.empty:
        return pd.Series(dtype="float64")
    return ((close / close.iloc[0]) * 100.0).rename(name or ticker)


def ai_breadth_history(prices: dict[str, pd.DataFrame], group: str, start_date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    frames = []
    for ticker in AI_UNIVERSE.get(group, []):
        close = pd.to_numeric(prices.get(ticker, pd.DataFrame()).get("Close", pd.Series(dtype="float64")), errors="coerce").dropna()
        if close.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    f"{ticker}_above_50": close > close.rolling(50, min_periods=50).mean(),
                    f"{ticker}_above_200": close > close.rolling(200, min_periods=200).mean(),
                    f"{ticker}_close": close,
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["Date", "% Above SMA50", "% Above SMA200", "Group Performance"])
    combined = pd.concat(frames, axis=1).loc[lambda frame: frame.index >= start_date]
    above50_cols = [col for col in combined.columns if col.endswith("_above_50")]
    above200_cols = [col for col in combined.columns if col.endswith("_above_200")]
    close_cols = [col for col in combined.columns if col.endswith("_close")]
    rows.append(
        pd.DataFrame(
            {
                "Date": combined.index,
                "% Above SMA50": combined[above50_cols].mean(axis=1) * 100.0,
                "% Above SMA200": combined[above200_cols].mean(axis=1) * 100.0,
                "Group Performance": (combined[close_cols] / combined[close_cols].dropna(how="all").iloc[0]).mean(axis=1) * 100.0,
            }
        )
    )
    return rows[0].dropna(how="all")


def period_start_date(prices: dict[str, pd.DataFrame], period: str) -> pd.Timestamp:
    days = {"1M": 30, "3M": 90, "6M": 182, "12M": 365, "3Y": 365 * 3, "5Y": 365 * 5}.get(period, 365)
    latest_dates = [
        frame.index.max()
        for frame in prices.values()
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    end = max(latest_dates) if latest_dates else pd.Timestamp.today().normalize()
    return pd.Timestamp(end) - pd.DateOffset(days=days)


def median_numeric(values: pd.Series) -> float:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    return float(valid.median()) if not valid.empty else np.nan


def breadth_percent(values: pd.Series) -> float:
    valid = values.dropna()
    if valid.empty:
        return np.nan
    return float(valid.astype(bool).mean() * 100.0)


def positive_percent(values: pd.Series) -> float:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return np.nan
    return float((valid > 0.0).mean() * 100.0)


def last_finite(values: pd.Series) -> float:
    valid = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(valid.iloc[-1]) if not valid.empty else np.nan


def numeric_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def first_finite(*values: Any) -> float:
    for value in values:
        numeric = numeric_or_nan(value)
        if np.isfinite(numeric):
            return numeric
    return np.nan
