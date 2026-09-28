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
    MultipleMetric("real_earnings_growth", "Real Earnings Growth", "https://www.multpl.com/s-p-500-real-earnings-growth/table/by-quarter", "%"),
    MultipleMetric("earnings_yield", "Earnings Yield", "https://www.multpl.com/s-p-500-earnings-yield/table/by-month", "%"),
    MultipleMetric("dividend_yield", "Dividend Yield", "https://www.multpl.com/s-p-500-dividend-yield/table/by-month", "%"),
    MultipleMetric("gdp_growth", "US GDP Growth Rate", "https://www.multpl.com/us-gdp-growth-rate/table/by-quarter", "%"),
    MultipleMetric("real_gdp_growth", "US Real GDP Growth Rate", "https://www.multpl.com/us-real-gdp-growth-rate/table/by-quarter", "%"),
    MultipleMetric("inflation", "US Inflation Rate", "https://www.multpl.com/inflation/table/by-year", "%"),
)

MULTPL_METRIC_GROUPS = (
    ("sp500_ps", "sp500_pe", "shiller_pe"),
    ("real_earnings_growth", "earnings_yield", "dividend_yield"),
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
    return {key: metrics[key] for key in by_key if key in metrics}, errors


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
