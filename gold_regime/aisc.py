from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd


NORMAL_AISC_MULTIPLE = 1.625
AISC_QOQ_GROWTH = 0.025
AISC_ZONE_THRESHOLDS = (1.25, 1.45, 1.80, 2.10, 2.40)

# Annual historical global mining-industry AISC observations in USD/oz.
# The annual values are intentionally held flat across all four quarters.
ANNUAL_AISC_HISTORY: dict[int, float] = {
    2000: 250.0,
    2001: 245.0,
    2002: 260.0,
    2003: 290.0,
    2004: 315.0,
    2005: 325.0,
    2006: 445.0,
    2007: 520.0,
    2008: 655.0,
    2009: 670.0,
    2010: 801.0,
    2011: 908.0,
    2012: 1112.0,
}
ANNUAL_AISC_STATUS: dict[int, str] = {
    **{year: "RECONSTRUCTED" for year in range(2000, 2010)},
    2010: "METALS_FOCUS_RETROSPECTIVE",
    2011: "METALS_FOCUS_RETROSPECTIVE",
    2012: "METALS_FOCUS",
}

HISTORICAL_AISC_QUARTERLY: dict[str, float] = {
    f"{year}Q{quarter}": value
    for year, value in ANNUAL_AISC_HISTORY.items()
    for quarter in range(1, 5)
}
HISTORICAL_AISC_QUARTERLY_STATUS: dict[pd.Period, str] = {
    pd.Period(f"{year}Q{quarter}", freq="Q"): ANNUAL_AISC_STATUS[year]
    for year in ANNUAL_AISC_HISTORY
    for quarter in range(1, 5)
}

# Quarterly global mining-industry AISC observations in USD/oz. Keep this
# isolated from the chart code so later actual observations can replace or
# extend the dataset without changing the valuation calculations.
ACTUAL_AISC_QUARTERLY: dict[str, float] = {
    **HISTORICAL_AISC_QUARTERLY,
    "2013Q1": 1120.0, "2013Q2": 1080.0, "2013Q3": 1010.0, "2013Q4": 980.0,
    "2014Q1": 950.0, "2014Q2": 950.0, "2014Q3": 940.0, "2014Q4": 920.0,
    "2015Q1": 900.0, "2015Q2": 890.0, "2015Q3": 880.0, "2015Q4": 860.0,
    "2016Q1": 830.0, "2016Q2": 810.0, "2016Q3": 820.0, "2016Q4": 810.0,
    "2017Q1": 820.0, "2017Q2": 830.0, "2017Q3": 840.0, "2017Q4": 840.0,
    "2018Q1": 870.0, "2018Q2": 900.0, "2018Q3": 880.0, "2018Q4": 990.0,
    "2019Q1": 1000.0, "2019Q2": 1000.0, "2019Q3": 1010.0, "2019Q4": 1020.0,
    "2020Q1": 980.0, "2020Q2": 972.0, "2020Q3": 953.0, "2020Q4": 998.0,
    "2021Q1": 1048.0, "2021Q2": 1080.0, "2021Q3": 1123.0, "2021Q4": 1129.0,
    "2022Q1": 1232.0, "2022Q2": 1289.0, "2022Q3": 1289.0, "2022Q4": 1294.0,
    "2023Q1": 1358.0, "2023Q2": 1315.0, "2023Q3": 1343.0, "2023Q4": 1342.0,
    "2024Q1": 1375.0, "2024Q2": 1388.0, "2024Q3": 1456.0, "2024Q4": 1438.0,
    "2025Q1": 1536.0, "2025Q2": 1590.0, "2025Q3": 1605.0, "2025Q4": 1706.0,
    "2026Q1": 1785.0,
}
AISC_QUARTERLY_STATUS: dict[pd.Period, str] = {
    **HISTORICAL_AISC_QUARTERLY_STATUS,
    **{
        pd.Period(quarter, freq="Q"): "ACTUAL"
        for quarter in ACTUAL_AISC_QUARTERLY
        if pd.Period(quarter, freq="Q").year >= 2013
    },
}

AISC_DEFAULT_CONFIG = {
    "qoq_growth": AISC_QOQ_GROWTH,
    "normal_multiple": NORMAL_AISC_MULTIPLE,
    "actual_quarterly": ACTUAL_AISC_QUARTERLY,
}


def aisc_config() -> dict[str, Any]:
    return deepcopy(AISC_DEFAULT_CONFIG)


def _actual_periods(actual_quarterly: dict[str | pd.Period, float]) -> dict[pd.Period, float]:
    result: dict[pd.Period, float] = {}
    for quarter, value in actual_quarterly.items():
        period = quarter if isinstance(quarter, pd.Period) else pd.Period(str(quarter), freq="Q")
        numeric = float(value)
        if np.isfinite(numeric) and numeric > 0:
            result[period] = numeric
    return dict(sorted(result.items(), key=lambda item: item[0].ordinal))


def build_quarterly_aisc(
    through_quarter: pd.Period | str,
    actual_quarterly: dict[str | pd.Period, float] | None = None,
    qoq_growth: float = AISC_QOQ_GROWTH,
) -> pd.DataFrame:
    """Build actual plus compounded quarterly AISC through a target quarter."""
    actual = _actual_periods(actual_quarterly or ACTUAL_AISC_QUARTERLY)
    if not actual:
        return pd.DataFrame(columns=["quarter_period", "quarter", "aisc", "aisc_source"])

    target = through_quarter if isinstance(through_quarter, pd.Period) else pd.Period(str(through_quarter), freq="Q")
    first_actual = min(actual, key=lambda period: period.ordinal)
    last_actual = max(actual, key=lambda period: period.ordinal)
    if target < first_actual:
        return pd.DataFrame(columns=["quarter_period", "quarter", "aisc", "aisc_source"])

    periods = pd.period_range(first_actual, max(target, last_actual), freq="Q")
    last_actual_value = actual[last_actual]
    rows: list[dict[str, Any]] = []
    for period in periods:
        if period in actual:
            value = actual[period]
            source = AISC_QUARTERLY_STATUS.get(period, "ACTUAL")
        elif period > last_actual:
            periods_after_last_actual = period.ordinal - last_actual.ordinal
            value = last_actual_value * ((1.0 + float(qoq_growth)) ** periods_after_last_actual)
            source = "ESTIMATED"
        else:
            # No interpolation or indefinite forward-fill is allowed for a
            # missing historical observation before the last actual quarter.
            continue
        rows.append(
            {
                "quarter_period": period,
                "quarter": f"{period.year} Q{period.quarter}",
                "aisc": float(value),
                "aisc_source": source,
            }
        )
    return pd.DataFrame(rows)


def _normalize_gold_daily(gold_daily: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(gold_daily, pd.Series):
        frame = pd.DataFrame({"date": pd.to_datetime(gold_daily.index, errors="coerce"), "gold_close": gold_daily.to_numpy()})
    else:
        source = gold_daily.copy()
        date_column = "date" if "date" in source.columns else "Date" if "Date" in source.columns else None
        price_column = "gold_close" if "gold_close" in source.columns else "close" if "close" in source.columns else "Close" if "Close" in source.columns else None
        if date_column is None:
            source = source.reset_index()
            date_column = "date" if "date" in source.columns else "Date" if "Date" in source.columns else source.columns[0]
        if price_column is None:
            return pd.DataFrame(columns=["date", "gold_close"])
        frame = source[[date_column, price_column]].rename(columns={date_column: "date", price_column: "gold_close"})

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["gold_close"] = pd.to_numeric(frame["gold_close"], errors="coerce")
    return frame.dropna(subset=["date", "gold_close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def build_gold_aisc_valuation(
    gold_daily: pd.Series | pd.DataFrame,
    actual_quarterly: dict[str | pd.Period, float] | None = None,
    qoq_growth: float = AISC_QOQ_GROWTH,
    normal_multiple: float = NORMAL_AISC_MULTIPLE,
    regime_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join daily Gold with a quarterly step-function AISC valuation series."""
    daily = _normalize_gold_daily(gold_daily)
    columns = [
        "date", "gold_close", "quarter", "aisc", "aisc_source", "gold_aisc_ratio",
        "normal_gold_value", "premium_discount_pct", "regime",
    ]
    if daily.empty:
        return pd.DataFrame(columns=columns)

    daily["quarter_period"] = daily["date"].dt.to_period("Q")
    quarterly = build_quarterly_aisc(daily["quarter_period"].max(), actual_quarterly, qoq_growth)
    lookup = quarterly.set_index("quarter_period") if not quarterly.empty else pd.DataFrame()
    if quarterly.empty:
        daily["quarter"] = daily["quarter_period"].map(lambda period: f"{period.year} Q{period.quarter}")
        daily["aisc"] = np.nan
        daily["aisc_source"] = pd.NA
    else:
        daily["quarter"] = daily["quarter_period"].map(lookup["quarter"])
        daily["aisc"] = daily["quarter_period"].map(lookup["aisc"])
        daily["aisc_source"] = daily["quarter_period"].map(lookup["aisc_source"])

    daily["gold_aisc_ratio"] = daily["gold_close"] / daily["aisc"]
    daily["normal_gold_value"] = daily["aisc"] * float(normal_multiple)
    daily["premium_discount_pct"] = (daily["gold_close"] / daily["normal_gold_value"] - 1.0) * 100.0

    if regime_history is not None and not regime_history.empty and "gold_regime" in regime_history.columns:
        regime = regime_history.copy()
        regime_date = "date" if "date" in regime.columns else "Date" if "Date" in regime.columns else None
        if regime_date is not None:
            regime[regime_date] = pd.to_datetime(regime[regime_date], errors="coerce")
            regime = regime.dropna(subset=[regime_date]).sort_values(regime_date)
            regime = regime[[regime_date, "gold_regime"]].rename(
                columns={regime_date: "date", "gold_regime": "regime"}
            )
            daily = pd.merge_asof(daily.sort_values("date"), regime, on="date", direction="backward")
        else:
            daily["regime"] = pd.NA
    else:
        daily["regime"] = pd.NA

    return daily.drop(columns=["quarter_period"]).reindex(columns=columns)


def aisc_valuation_state(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "DATA INCOMPLETE"
    if ratio < 1.25:
        return "Very compressed producer economics"
    if ratio < 1.45:
        return "Below-normal margin environment"
    if ratio < 1.80:
        return "Normal historical range"
    if ratio < 2.10:
        return "Strong producer-margin environment"
    if ratio < 2.40:
        return "Historically elevated"
    return "Extreme / unusual"
