from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import StringIO
from urllib.request import Request, urlopen

import pandas as pd


@dataclass(frozen=True)
class MultipleMetric:
    key: str
    title: str
    url: str
    unit: str


MULTPL_METRICS = (
    MultipleMetric("sp500_ps", "S&P 500 Price / Sales", "https://www.multpl.com/s-p-500-price-to-sales/table/by-quarter", "x"),
    MultipleMetric("sp500_pe", "S&P 500 P/E", "https://www.multpl.com/s-p-500-pe-ratio/table/by-month", "x"),
    MultipleMetric("shiller_pe", "Shiller P/E", "https://www.multpl.com/shiller-pe/table/by-month", "x"),
    MultipleMetric("earnings", "S&P 500 Earnings", "https://www.multpl.com/s-p-500-earnings/table/by-month", "$"),
    MultipleMetric("earnings_yield", "Earnings Yield", "https://www.multpl.com/s-p-500-earnings-yield/table/by-month", "%"),
    MultipleMetric("gdp_growth", "US GDP Growth Rate", "https://www.multpl.com/us-gdp-growth-rate/table/by-quarter", "%"),
    MultipleMetric("real_gdp_growth", "US Real GDP Growth Rate", "https://www.multpl.com/us-real-gdp-growth-rate/table/by-quarter", "%"),
    MultipleMetric("inflation", "US Inflation Rate", "https://www.multpl.com/inflation/table/by-year", "%"),
)

EARNINGS_GROWTH_URL = "https://www.multpl.com/s-p-500-earnings/table/by-month"
DERIVED_MULTPL_METRICS = (
    MultipleMetric("earnings_growth_12m", "Earnings Growth", EARNINGS_GROWTH_URL, "%"),
    MultipleMetric("sp500_pe_15y_percentile", "S&P 500 P/E 15Y Percentile", MULTPL_METRICS[1].url, "%"),
    MultipleMetric("sp500_peg", "S&P 500 PEG", MULTPL_METRICS[1].url, "x"),
    MultipleMetric("sp500_peg_15y_percentile", "S&P 500 PEG 15Y Percentile", MULTPL_METRICS[1].url, "%"),
    MultipleMetric("earnings_growth_15y_percentile", "Earnings Growth 15Y Percentile", EARNINGS_GROWTH_URL, "%"),
)
MULTPL_DISPLAY_METRICS = MULTPL_METRICS + tuple(
    metric for metric in DERIVED_MULTPL_METRICS if metric.key != "earnings_growth_12m"
)

MULTPL_METRIC_GROUPS = (
    ("sp500_pe", "sp500_pe_15y_percentile", "sp500_peg", "sp500_peg_15y_percentile"),
    ("sp500_ps", "shiller_pe"),
    ("earnings", "earnings_growth_15y_percentile", "earnings_yield"),
    ("gdp_growth", "real_gdp_growth", "inflation"),
)


def parse_multpl_table(page_html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(page_html))
    for table in tables:
        normalized = {str(column).strip().lower(): column for column in table.columns}
        date_column = normalized.get("date")
        value_column = normalized.get("value")
        if date_column is None or value_column is None:
            continue

        source_values = table[value_column].astype(str)
        numeric_values = source_values.map(_parse_numeric_value)
        dates = pd.to_datetime(table[date_column], errors="coerce", format="mixed")
        result = pd.DataFrame(
            {
                "Date": dates,
                "Value": numeric_values,
                "Estimate": source_values.str.contains("estimate", case=False, regex=False),
            }
        )
        result = result.dropna(subset=["Date", "Value"])
        if not result.empty:
            return result.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    raise ValueError("Multpl Date/Value table was not found")


def _parse_numeric_value(value: str) -> float:
    match = re.search(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)", value)
    return float(match.group(0).replace(",", "")) if match else float("nan")


def calculate_earnings_growth_12m(earnings: pd.DataFrame) -> pd.DataFrame:
    if earnings.empty:
        return pd.DataFrame(columns=["Date", "Value", "Estimate"])

    frame = earnings.copy()
    frame["Month"] = pd.to_datetime(frame["Date"], errors="coerce").dt.to_period("M")
    prior = frame.set_index("Month")[["Value", "Estimate"]].copy()
    prior.index = prior.index + 12
    frame["PriorValue"] = frame["Month"].map(prior["Value"])
    frame["PriorEstimate"] = frame["Month"].map(prior["Estimate"]).eq(True)
    frame["Value"] = (pd.to_numeric(frame["Value"], errors="coerce") / pd.to_numeric(frame["PriorValue"], errors="coerce") - 1.0) * 100.0
    frame["Estimate"] = frame["Estimate"].fillna(False).astype(bool) | frame["PriorEstimate"]
    return frame.dropna(subset=["Date", "Value"])[["Date", "Value", "Estimate"]].reset_index(drop=True)


def clean_sp500_pe_history(pe: pd.DataFrame) -> pd.DataFrame:
    if pe.empty:
        return pe.copy()
    dates = pd.to_datetime(pe["Date"], errors="coerce")
    excluded = dates.ge(pd.Timestamp("2009-01-01")) & dates.lt(pd.Timestamp("2009-10-01"))
    return pe.loc[~excluded].reset_index(drop=True)


def calculate_trailing_percentile(data: pd.DataFrame, years: int = 15) -> pd.DataFrame:
    columns = ["Date", "Value", "Estimate"]
    if data.empty:
        return pd.DataFrame(columns=columns)

    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Value"]).sort_values("Date").reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    values = frame["Value"].to_numpy(dtype=float)
    percentiles = [float("nan")] * len(frame)

    for index, row in frame.iterrows():
        window_start = row["Date"] - pd.DateOffset(years=years)
        if frame.at[0, "Date"] > window_start:
            continue
        start = frame["Date"].searchsorted(window_start, side="left")
        window = values[start : index + 1]
        if len(window):
            percentiles[index] = float((window <= values[index]).mean() * 100.0)

    frame["Value"] = percentiles
    frame["Estimate"] = frame.get("Estimate", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    return frame.dropna(subset=["Value"])[columns].reset_index(drop=True)


def calculate_sp500_pe_15y_percentile(pe: pd.DataFrame) -> pd.DataFrame:
    return calculate_trailing_percentile(pe, years=15)


def calculate_sp500_peg(pe: pd.DataFrame, earnings_growth: pd.DataFrame) -> pd.DataFrame:
    columns = ["Date", "Value", "Estimate"]
    if pe.empty or earnings_growth.empty:
        return pd.DataFrame(columns=columns)

    pe_frame = pe.copy()
    growth_frame = earnings_growth.copy()
    for frame in (pe_frame, growth_frame):
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame["Month"] = frame["Date"].dt.to_period("M")
        frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    pe_frame = pe_frame.dropna(subset=["Month", "Value"])
    growth_frame = growth_frame.dropna(subset=["Month", "Value"])
    growth_by_month = growth_frame.drop_duplicates("Month", keep="last").set_index("Month")
    pe_frame = pe_frame.drop_duplicates("Month", keep="last").copy()
    pe_frame["EarningsGrowth"] = pe_frame["Month"].map(growth_by_month["Value"])
    pe_frame["GrowthEstimate"] = pe_frame["Month"].map(
        growth_by_month.get("Estimate", pd.Series(False, index=growth_by_month.index))
    ).fillna(False).astype(bool)
    pe_frame = pe_frame.loc[pe_frame["EarningsGrowth"].gt(0)].copy()
    pe_frame["Value"] = pe_frame["Value"] / pe_frame["EarningsGrowth"]
    pe_frame["Estimate"] = (
        pe_frame.get("Estimate", pd.Series(False, index=pe_frame.index)).fillna(False).astype(bool)
        | pe_frame["GrowthEstimate"]
    )
    return pe_frame.dropna(subset=["Value"])[columns].reset_index(drop=True)


def fetch_multpl_metric(metric: MultipleMetric, timeout: float = 15.0) -> pd.DataFrame:
    request = Request(
        metric.url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Finxmo/1.0; +https://finxmo.com)"},
    )
    with urlopen(request, timeout=timeout) as response:
        page_html = response.read().decode("utf-8", errors="replace")
    return parse_multpl_table(page_html)


def load_multpl_metrics() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    metrics: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    by_key = {metric.key: metric for metric in MULTPL_METRICS}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fetch_multpl_metric, metric): metric for metric in MULTPL_METRICS}
        for future in as_completed(futures):
            metric = futures[future]
            try:
                metrics[metric.key] = future.result()
            except Exception as exc:
                errors[metric.key] = str(exc)
    if "earnings" in metrics:
        metrics["earnings_growth_12m"] = calculate_earnings_growth_12m(metrics["earnings"])
    if "sp500_pe" in metrics:
        metrics["sp500_pe"] = clean_sp500_pe_history(metrics["sp500_pe"])
        metrics["sp500_pe_15y_percentile"] = calculate_sp500_pe_15y_percentile(metrics["sp500_pe"])
    if "earnings_growth_12m" in metrics:
        metrics["earnings_growth_15y_percentile"] = calculate_trailing_percentile(
            metrics["earnings_growth_12m"], years=15
        )
    if "sp500_pe" in metrics and "earnings_growth_12m" in metrics:
        metrics["sp500_peg"] = calculate_sp500_peg(metrics["sp500_pe"], metrics["earnings_growth_12m"])
        metrics["sp500_peg_15y_percentile"] = calculate_trailing_percentile(metrics["sp500_peg"], years=15)
    result_keys = (*by_key, *(metric.key for metric in DERIVED_MULTPL_METRICS))
    return {key: metrics[key] for key in result_keys if key in metrics}, errors


def multiples_range_bounds(
    metrics: dict[str, pd.DataFrame], selection: str
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    available_dates = [frame["Date"] for frame in metrics.values() if not frame.empty]
    if not available_dates:
        return None, None
    end = max(dates.max() for dates in available_dates)
    if selection == "MAX":
        return None, end
    years = {"10Y": 10, "20Y": 20}.get(selection)
    if years is None:
        raise ValueError(f"Unsupported multiples range: {selection}")
    return end - pd.DateOffset(years=years), end
