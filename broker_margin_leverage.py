from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


BROKER_MARGIN_LOANS_SERIES = "BOGZ1FL663067003Q"
US_EQUITY_MARKET_CAP_SERIES = "BOGZ1LM883164115Q"
BROKER_MARGIN_LOOKBACKS = {
    "10Y": 40,
    "20Y": 80,
    "Full History": None,
}


def _quarterly_series(frame: pd.DataFrame | None, name: str) -> pd.Series:
    if frame is None or frame.empty or "Close" not in frame:
        return pd.Series(dtype="float64", name=name)
    dates = pd.to_datetime(frame.index, errors="coerce")
    values = pd.to_numeric(frame["Close"], errors="coerce")
    data = pd.DataFrame({"Date": dates, name: values.to_numpy()}).dropna(subset=["Date", name])
    if data.empty:
        return pd.Series(dtype="float64", name=name)
    data["Quarter"] = data["Date"].dt.to_period("Q-DEC")
    return data.sort_values("Date").groupby("Quarter", sort=True)[name].last()


def build_broker_margin_leverage_history(
    margin_loans: pd.DataFrame | None,
    equity_market_cap: pd.DataFrame | None,
) -> pd.DataFrame:
    """Align both FRED end-of-quarter level series by reporting quarter.

    Source observations are in millions of USD. The returned frame starts in
    1990 and keeps unmatched source quarters so loan percentiles do not depend
    on market-cap availability; normalized ratios require both observations.
    """
    loans = _quarterly_series(margin_loans, "MarginLoansMillions")
    market_cap = _quarterly_series(equity_market_cap, "EquityMarketCapMillions")
    quarterly = pd.concat([loans, market_cap], axis=1).sort_index()
    if quarterly.empty:
        return pd.DataFrame(
            columns=[
                "Date", "MarginLoansMillions", "EquityMarketCapMillions",
                "MarginLoansBillions", "EquityMarketCapTrillions", "MarginLoansYoY",
                "MarginToMarketCapPct",
            ]
        )
    quarterly = quarterly.loc[quarterly.index >= pd.Period("1990Q1", freq="Q-DEC")].copy()
    if quarterly.empty:
        return pd.DataFrame(
            columns=[
                "Date", "MarginLoansMillions", "EquityMarketCapMillions",
                "MarginLoansBillions", "EquityMarketCapTrillions", "MarginLoansYoY",
                "MarginToMarketCapPct",
            ]
        )
    full_quarters = pd.period_range(quarterly.index.min(), quarterly.index.max(), freq="Q-DEC")
    quarterly = quarterly.reindex(full_quarters)
    quarterly.index.name = "Quarter"
    result = quarterly.reset_index()
    result["Date"] = result["Quarter"].dt.to_timestamp(how="end").dt.normalize()
    result["MarginLoansBillions"] = result["MarginLoansMillions"] / 1_000.0
    result["EquityMarketCapTrillions"] = result["EquityMarketCapMillions"] / 1_000_000.0
    loans_millions = pd.to_numeric(result["MarginLoansMillions"], errors="coerce")
    equity_millions = pd.to_numeric(result["EquityMarketCapMillions"], errors="coerce")
    result["MarginLoansYoY"] = loans_millions.pct_change(4, fill_method=None) * 100.0
    result["MarginToMarketCapPct"] = np.where(
        loans_millions.gt(0) & equity_millions.gt(0),
        100.0 * loans_millions / equity_millions,
        np.nan,
    )
    return result.drop(columns="Quarter").sort_values("Date").reset_index(drop=True)


def historical_percentile(series: pd.Series, lookback: str) -> pd.Series:
    """Inclusive, past-only percentile rank; fixed windows require all quarters."""
    values = pd.to_numeric(series, errors="coerce").reset_index(drop=True)
    window = BROKER_MARGIN_LOOKBACKS.get(lookback)
    result = np.full(len(values), np.nan, dtype="float64")
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue
        start = 0 if window is None else max(0, idx - window + 1)
        sample = values.iloc[start : idx + 1]
        sample = sample.dropna()
        if window is not None and len(sample) != window:
            continue
        if not sample.empty:
            result[idx] = float(sample.le(value).mean() * 100.0)
    return pd.Series(result, index=series.index, dtype="float64")


def apply_broker_margin_percentiles(frame: pd.DataFrame, lookback: str) -> pd.DataFrame:
    result = frame.copy()
    result["MarginLoansLevelPct"] = historical_percentile(result["MarginLoansBillions"], lookback)
    result["MarginLoansGrowthPct"] = historical_percentile(result["MarginLoansYoY"], lookback)
    result["MarginToMarketCapPctile"] = historical_percentile(result["MarginToMarketCapPct"], lookback)
    return result


def weekly_log_spx(daily: pd.DataFrame) -> pd.DataFrame:
    if daily is None or daily.empty or "SPX_Close" not in daily:
        return pd.DataFrame(columns=["Date", "SPX_Close", "LogSPX"])
    frame = daily.copy()
    if "Date" in frame:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).set_index("Date")
    else:
        frame.index = pd.to_datetime(frame.index, errors="coerce")
    close = pd.to_numeric(frame["SPX_Close"], errors="coerce").dropna().sort_index()
    weekly = close.resample("W-FRI").last().dropna()
    if not weekly.empty and weekly.index[-1] > close.index[-1].normalize():
        weekly = weekly.iloc[:-1]
    out = pd.DataFrame({"Date": weekly.index, "SPX_Close": weekly.to_numpy()})
    out["LogSPX"] = np.log(out["SPX_Close"].where(out["SPX_Close"].gt(0)))
    return out.dropna(subset=["LogSPX"]).reset_index(drop=True)


def latest_common_observation(frame: pd.DataFrame) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    valid = frame.dropna(subset=["MarginLoansBillions", "EquityMarketCapTrillions"])
    return None if valid.empty else valid.iloc[-1]


def classify_credit_growth(percentile: Any) -> str:
    value = pd.to_numeric(pd.Series([percentile]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        return "Insufficient Data"
    if value <= 10:
        return "Extreme Contraction"
    if value < 25:
        return "Weak Credit Growth"
    if value <= 75:
        return "Normal Credit Growth"
    if value < 90:
        return "Accelerating Credit"
    return "Extreme Credit Expansion"


def classify_normalized_leverage(percentile: Any) -> str:
    value = pd.to_numeric(pd.Series([percentile]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        return "Insufficient Data"
    if value <= 10:
        return "Extremely Low"
    if value < 25:
        return "Low"
    if value <= 75:
        return "Normal"
    if value < 90:
        return "Elevated"
    return "Extreme"


def broker_leverage_interpretation(growth_percentile: Any, ratio_percentile: Any) -> tuple[str, str]:
    growth = pd.to_numeric(pd.Series([growth_percentile]), errors="coerce").iloc[0]
    leverage = pd.to_numeric(pd.Series([ratio_percentile]), errors="coerce").iloc[0]
    if not np.isfinite(growth) or not np.isfinite(leverage):
        return (
            "Insufficient Data",
            "One or more 10-year percentile measures are not yet available; no interpretation is inferred.",
        )

    if growth >= 90 and leverage >= 90:
        title = "Extreme Credit Expansion / High Leverage"
        detail = "Broker credit is expanding at an extreme historical rate while normalized leverage is also historically extreme."
    elif growth >= 90 and 25 <= leverage <= 75:
        title = "Rapid Credit Expansion / Normal Leverage"
        detail = "Broker customer receivables are expanding at a historically elevated rate, while their level relative to US equity market capitalization remains within its normal historical range."
    elif growth >= 90 and leverage <= 25:
        title = "Credit Recovery / Low Leverage"
        detail = "Broker credit growth is unusually strong, while normalized broker leverage remains historically low."
    elif 25 <= growth <= 75 and leverage >= 90:
        title = "Elevated Leverage / Moderate Credit Growth"
        detail = "Normalized broker leverage is extreme even though the pace of credit growth is within its normal range."
    elif growth <= 10 and leverage >= 90:
        title = "Credit Contraction / Elevated Leverage"
        detail = "Broker credit growth is at an extreme low while normalized leverage remains elevated relative to history."
    elif growth <= 10 and leverage <= 25:
        title = "Broad Credit Deleveraging"
        detail = "Broker credit is contracting sharply and normalized leverage is low relative to its historical range."
    elif 25 <= growth <= 75 and 25 <= leverage <= 75:
        title = "Normal Credit / Normal Leverage"
        detail = "Both broker credit growth and normalized broker leverage are within their normal historical ranges."
    else:
        title = f"{classify_credit_growth(growth)} / {classify_normalized_leverage(leverage)} Leverage"
        detail = (
            f"Credit growth is classified as {classify_credit_growth(growth).lower()}; "
            f"normalized broker leverage is {classify_normalized_leverage(leverage).lower()} relative to history."
        )
    return title, detail
