from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1.0


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return np.nan
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    elapsed_days = (equity.index[-1] - equity.index[0]).days
    years = elapsed_days / 365.25
    if years <= 0 or start_val <= 0 or end_val <= 0:
        return np.nan
    return (end_val / start_val) ** (1 / years) - 1


def sharpe_annualized(
    equity: pd.Series,
    risk_free_annual_pct: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[float, str | None]:
    if len(equity) < 3:
        return np.nan, "Not enough data points to compute Sharpe ratio."
    returns = equity.pct_change().dropna()
    rf_period = (1 + risk_free_annual_pct / 100.0) ** (1 / periods_per_year) - 1
    excess = returns - rf_period
    std = float(excess.std(ddof=0))
    if std == 0:
        return np.nan, "Sharpe ratio is undefined because return standard deviation is 0."
    sh = np.sqrt(periods_per_year) * float(excess.mean()) / std
    return sh, None


def headline_metrics(
    equity: pd.Series,
    risk_free_annual_pct: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    dd = drawdown_series(equity)
    sh, sh_warn = sharpe_annualized(equity, risk_free_annual_pct, periods_per_year=periods_per_year)
    cagr_val = cagr(equity)
    if sh_warn:
        warnings.append(sh_warn)

    out = {
        "CAGR %": cagr_val * 100 if pd.notna(cagr_val) else np.nan,
        "Max Drawdown %": float(dd.min() * 100),
        "Average Drawdown %": float(dd.mean() * 100),
        "Sharpe Ratio": sh,
    }
    return out, warnings


def yearly_table(equity: pd.Series) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame(columns=["Year", "Performance %", "Max Drawdown %", "Cumulative CAGR %"])

    rows: list[dict[str, float | int]] = []
    start_dt = equity.index[0]
    start_val = float(equity.iloc[0])

    years = sorted(equity.index.year.unique())
    for year in years:
        ys = equity[equity.index.year == year]
        if ys.empty:
            continue

        performance = float(ys.iloc[-1] / ys.iloc[0] - 1) * 100
        year_dd = drawdown_series(ys)
        max_dd = float(year_dd.min()) * 100

        elapsed_days = (ys.index[-1] - start_dt).days
        if elapsed_days > 0 and start_val > 0 and ys.iloc[-1] > 0:
            years_elapsed = elapsed_days / 365.25
            cum_cagr = ((float(ys.iloc[-1]) / start_val) ** (1 / years_elapsed) - 1) * 100
        else:
            cum_cagr = np.nan

        rows.append(
            {
                "Year": int(year),
                "Performance %": performance,
                "Max Drawdown %": max_dd,
                "Cumulative CAGR %": cum_cagr,
            }
        )

    return pd.DataFrame(rows)
