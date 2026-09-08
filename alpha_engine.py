from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd


ALPHA_CONFIG = {
    "weights": {
        "momentum": 0.35,
        "trend": 0.30,
        "persistence": 0.35,
    },
    "momentum": {
        "perf_1m": 0.20,
        "perf_3m": 0.45,
        "perf_6m": 0.35,
    },
    "trend": {
        "adx_di": 0.55,
        "sma_regime": 0.30,
        "high_52w": 0.15,
    },
    "adx_di": {
        "adx_points": [(15.0, 0.0), (20.0, 40.0), (25.0, 70.0), (30.0, 90.0), (35.0, 100.0)],
        "di_balance_full_score": 0.25,
    },
    "sma_regime": {
        "absolute": 0.65,
        "relative": 0.35,
        "absolute_points": [(-5.0, 0.0), (0.0, 50.0), (5.0, 100.0)],
        "relative_window_months": 36,
        "relative_min_observations": 504,
        "relative_fallback_score": 50.0,
        "robust_sigma_epsilon": 1e-9,
    },
    "high_52w": {
        "points": [(-25.0, 0.0), (-15.0, 50.0), (-5.0, 90.0), (0.0, 100.0)],
    },
    "persistence": {
        "median_rank": 0.35,
        "min_core_rank": 0.25,
        "positive_breadth": 0.20,
        "historical_12m": 0.20,
        "historical_12m_full_score_percentile": 85.0,
    },
    "overextension": {
        "sma200w_max_penalty": 12.0,
        "perf12m_max_penalty": 10.0,
        "smaz_max_penalty": 5.0,
        "rsi_max_penalty": 5.0,
        "sma200w_start_percentile": 90.0,
        "perf12m_start_percentile": 90.0,
        "smaz_start": 1.5,
        "smaz_full_penalty": 3.0,
        "rsi_points": [(65.0, 0.0), (70.0, 1.0), (75.0, 3.0), (80.0, 5.0)],
    },
    "state": {
        "healthy_alpha": 70.0,
        "healthy_component": 65.0,
        "healthy_max_penalty": 10.0,
        "emerging_momentum": 75.0,
        "emerging_trend": 55.0,
        "emerging_persistence": 65.0,
        "extended_base_alpha": 70.0,
        "extended_min_penalty": 10.0,
        "deteriorating_persistence": 65.0,
        "deteriorating_momentum": 50.0,
        "deteriorating_adx_di": 35.0,
        "bear_momentum": 50.0,
    },
}

ALPHA_OUTPUT_COLUMNS = [
    "Alpha_Rank",
    "Alpha_Score",
    "Alpha_State",
    "Momentum_Score",
    "Trend_Quality_Score",
    "Persistence_Score",
    "Overextension_Penalty",
    "ADX_DI_Trend_Score",
    "DI_Balance",
    "SMA_Regime_Score",
    "Absolute_SMA_Score",
    "Relative_SMA_Score",
    "SMA200d_Robust_Z_36M",
    "SMA200W_Penalty",
    "Perf12M_Penalty",
    "SMA_Z_Penalty",
    "RSI_Penalty",
    "Base_Alpha",
    "Alpha_Data_Complete",
]


def alpha_config() -> dict:
    return deepcopy(ALPHA_CONFIG)


def calculate_sma200d_robust_z_36m(close: pd.Series, config: dict | None = None) -> float:
    cfg = (config or ALPHA_CONFIG)["sma_regime"]
    prices = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if prices.empty:
        return np.nan

    sma200d = prices.rolling(window=200, min_periods=200).mean()
    spread = ((prices / sma200d) - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    if spread.empty:
        return np.nan

    latest_date = spread.index[-1]
    start_date = latest_date - pd.DateOffset(months=int(cfg["relative_window_months"]))
    window = spread.loc[start_date:latest_date].dropna()
    if len(window) < int(cfg["relative_min_observations"]):
        return np.nan

    median = float(window.median())
    mad = float((window - median).abs().median())
    robust_sigma = 1.4826 * mad
    if not np.isfinite(robust_sigma) or robust_sigma <= float(cfg["robust_sigma_epsilon"]):
        return 0.0
    return float((window.iloc[-1] - median) / robust_sigma)


def calculate_momentum_score(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = (config or ALPHA_CONFIG)["momentum"]
    out = pd.DataFrame(index=df.index)
    out["Perf1M_Rank"] = cross_sectional_percentile_rank(df["Perf_1M_%"])
    out["Perf3M_Rank"] = cross_sectional_percentile_rank(df["Perf_3M_%"])
    out["Perf6M_Rank"] = cross_sectional_percentile_rank(df["Perf_6M_%"])
    out["Momentum_Score"] = (
        float(cfg["perf_1m"]) * out["Perf1M_Rank"]
        + float(cfg["perf_3m"]) * out["Perf3M_Rank"]
        + float(cfg["perf_6m"]) * out["Perf6M_Rank"]
    )
    return out


def calculate_adx_di_score(
    adx: pd.Series,
    plus_di: pd.Series,
    minus_di: pd.Series,
    config: dict | None = None,
) -> pd.DataFrame:
    cfg = (config or ALPHA_CONFIG)["adx_di"]
    plus = pd.to_numeric(plus_di, errors="coerce")
    minus = pd.to_numeric(minus_di, errors="coerce")
    denom = plus + minus
    di_balance = ((plus - minus) / denom.replace(0.0, np.nan)).fillna(0.0)
    di_bull_score = 100.0 * (di_balance / float(cfg["di_balance_full_score"])).clip(0.0, 1.0)
    adx_strength_score = piecewise_score(pd.to_numeric(adx, errors="coerce"), cfg["adx_points"])
    return pd.DataFrame(
        {
            "DI_Balance": di_balance,
            "ADX_Strength_Score": adx_strength_score,
            "ADX_DI_Trend_Score": adx_strength_score * di_bull_score / 100.0,
        },
        index=adx.index,
    )


def calculate_sma_regime_score(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = (config or ALPHA_CONFIG)["sma_regime"]
    out = pd.DataFrame(index=df.index)
    out["Absolute_SMA_Score"] = piecewise_score(
        pd.to_numeric(df["SMA50w_vs_SMA200w_Spread_%"], errors="coerce"),
        cfg["absolute_points"],
    )

    smaz = pd.to_numeric(df["SMA200d_Robust_Z_36M"], errors="coerce")
    out["Relative_SMA_Score"] = (50.0 + 25.0 * smaz).clip(0.0, 100.0)
    relative_missing = out["Relative_SMA_Score"].isna()
    out.loc[relative_missing, "Relative_SMA_Score"] = float(cfg["relative_fallback_score"])
    out["SMA_Relative_Fallback_Used"] = relative_missing
    out["SMA_Regime_Score"] = (
        float(cfg["absolute"]) * out["Absolute_SMA_Score"]
        + float(cfg["relative"]) * out["Relative_SMA_Score"]
    )
    return out


def calculate_persistence_score(df: pd.DataFrame, ranks: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = (config or ALPHA_CONFIG)["persistence"]
    out = pd.DataFrame(index=df.index)
    out["Perf12M_Rank"] = cross_sectional_percentile_rank(df["Perf_12M_%"])
    rank_cols = ["Perf1M_Rank", "Perf3M_Rank", "Perf6M_Rank", "Perf12M_Rank"]
    all_ranks = pd.concat([ranks[["Perf1M_Rank", "Perf3M_Rank", "Perf6M_Rank"]], out["Perf12M_Rank"]], axis=1)
    out["Median_Rank"] = all_ranks[rank_cols].median(axis=1, skipna=False)
    out["Min_Core_Rank"] = all_ranks[["Perf3M_Rank", "Perf6M_Rank", "Perf12M_Rank"]].min(axis=1, skipna=False)
    perf_cols = ["Perf_1M_%", "Perf_3M_%", "Perf_6M_%", "Perf_12M_%"]
    perf_values = df[perf_cols].apply(pd.to_numeric, errors="coerce")
    out["Positive_Breadth_Score"] = perf_values.gt(0.0).sum(axis=1) * 25.0
    missing_perf = perf_values.isna().any(axis=1)
    out.loc[missing_perf, "Positive_Breadth_Score"] = np.nan
    hist = pd.to_numeric(df["Perf_12M_Percentile"], errors="coerce")
    out["Historical_12M_Score"] = (
        100.0 * (hist / float(cfg["historical_12m_full_score_percentile"])).clip(0.0, 1.0)
    )
    out["Persistence_Score"] = (
        float(cfg["median_rank"]) * out["Median_Rank"]
        + float(cfg["min_core_rank"]) * out["Min_Core_Rank"]
        + float(cfg["positive_breadth"]) * out["Positive_Breadth_Score"]
        + float(cfg["historical_12m"]) * out["Historical_12M_Score"]
    )
    return out


def calculate_overextension_penalty(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = (config or ALPHA_CONFIG)["overextension"]
    out = pd.DataFrame(index=df.index)
    sma200w = pd.to_numeric(df["SMA200W_Distance_Percentile"], errors="coerce")
    perf12m = pd.to_numeric(df["Perf_12M_Percentile"], errors="coerce")
    smaz = pd.to_numeric(df["SMA200d_Robust_Z_36M"], errors="coerce")
    rsi = pd.to_numeric(df["RSI_14"], errors="coerce")

    out["SMA200W_Penalty"] = quadratic_penalty(
        sma200w,
        start=float(cfg["sma200w_start_percentile"]),
        end=100.0,
        max_penalty=float(cfg["sma200w_max_penalty"]),
    )
    out["Perf12M_Penalty"] = quadratic_penalty(
        perf12m,
        start=float(cfg["perf12m_start_percentile"]),
        end=100.0,
        max_penalty=float(cfg["perf12m_max_penalty"]),
    )
    out["SMA_Z_Penalty"] = quadratic_penalty(
        smaz,
        start=float(cfg["smaz_start"]),
        end=float(cfg["smaz_full_penalty"]),
        max_penalty=float(cfg["smaz_max_penalty"]),
    ).fillna(0.0)
    out["RSI_Penalty"] = piecewise_score(rsi, cfg["rsi_points"])
    out["Overextension_Penalty"] = out[
        ["SMA200W_Penalty", "Perf12M_Penalty", "SMA_Z_Penalty", "RSI_Penalty"]
    ].sum(axis=1, min_count=4)
    return out


def calculate_alpha_score(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = (config or ALPHA_CONFIG)["weights"]
    out = pd.DataFrame(index=df.index)
    out["Base_Alpha"] = (
        float(cfg["momentum"]) * df["Momentum_Score"]
        + float(cfg["trend"]) * df["Trend_Quality_Score"]
        + float(cfg["persistence"]) * df["Persistence_Score"]
    )
    out["Alpha_Score"] = (out["Base_Alpha"] - df["Overextension_Penalty"]).clip(0.0, 100.0)
    return out


def classify_alpha_state(row: pd.Series, config: dict | None = None) -> str:
    cfg = (config or ALPHA_CONFIG)["state"]
    required = [
        "Alpha_Score",
        "Base_Alpha",
        "Momentum_Score",
        "Trend_Quality_Score",
        "Persistence_Score",
        "Overextension_Penalty",
        "ADX_DI_Trend_Score",
        "DI_Balance",
        "SMA50w_vs_SMA200w_Spread_%",
    ]
    if any(not np.isfinite(row.get(col, np.nan)) for col in required):
        return "Missing Data"
    if (
        row["Base_Alpha"] >= cfg["extended_base_alpha"]
        and row["Overextension_Penalty"] >= cfg["extended_min_penalty"]
    ):
        return "Extended Trend"
    if (
        row["Alpha_Score"] >= cfg["healthy_alpha"]
        and row["Momentum_Score"] >= cfg["healthy_component"]
        and row["Trend_Quality_Score"] >= cfg["healthy_component"]
        and row["Persistence_Score"] >= cfg["healthy_component"]
        and row["Overextension_Penalty"] < cfg["healthy_max_penalty"]
    ):
        return "Healthy Trend"
    if (
        row["Momentum_Score"] >= cfg["emerging_momentum"]
        and row["Trend_Quality_Score"] >= cfg["emerging_trend"]
        and row["Persistence_Score"] < cfg["emerging_persistence"]
    ):
        return "Emerging Momentum"
    if (
        row["Persistence_Score"] >= cfg["deteriorating_persistence"]
        and (
            row["Momentum_Score"] < cfg["deteriorating_momentum"]
            or row["ADX_DI_Trend_Score"] < cfg["deteriorating_adx_di"]
        )
    ):
        return "Deteriorating"
    if (
        row["SMA50w_vs_SMA200w_Spread_%"] < 0.0
        and row["DI_Balance"] <= 0.0
        and row["Momentum_Score"] < cfg["bear_momentum"]
    ):
        return "Bear / Broken Trend"
    return "Neutral"


def calculate_alpha_engine(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    config = config or ALPHA_CONFIG
    out = df.copy()
    if out.empty:
        for col in ALPHA_OUTPUT_COLUMNS:
            out[col] = pd.Series(dtype="float64" if col != "Alpha_State" else "object")
        return out

    momentum = calculate_momentum_score(out, config)
    adx_di = calculate_adx_di_score(out["ADX_14"], out["DI_Plus_14"], out["DI_Minus_14"], config)
    sma = calculate_sma_regime_score(out, config)
    high52w = piecewise_score(out["Price_vs_52W_High_%"], config["high_52w"]["points"]).rename("High52W_Score")
    trend_quality = (
        float(config["trend"]["adx_di"]) * adx_di["ADX_DI_Trend_Score"]
        + float(config["trend"]["sma_regime"]) * sma["SMA_Regime_Score"]
        + float(config["trend"]["high_52w"]) * high52w
    ).rename("Trend_Quality_Score")
    persistence = calculate_persistence_score(out, momentum, config)
    overextension = calculate_overextension_penalty(out, config)

    parts = pd.concat(
        [
            momentum,
            adx_di,
            sma,
            high52w,
            trend_quality,
            persistence,
            overextension,
        ],
        axis=1,
    )
    score_inputs = pd.concat([out, parts], axis=1)
    scores = calculate_alpha_score(score_inputs, config)
    out = pd.concat([out, parts, scores], axis=1)

    critical_cols = [
        "Momentum_Score",
        "ADX_DI_Trend_Score",
        "Absolute_SMA_Score",
        "High52W_Score",
        "Persistence_Score",
        "SMA200W_Penalty",
        "Perf12M_Penalty",
        "RSI_Penalty",
        "Overextension_Penalty",
    ]
    out["Alpha_Data_Complete"] = out[critical_cols].notna().all(axis=1) & ~out["SMA_Relative_Fallback_Used"]
    incomplete = out[critical_cols].isna().any(axis=1)
    out.loc[incomplete, ["Base_Alpha", "Alpha_Score"]] = np.nan
    out["Alpha_State"] = out.apply(lambda row: classify_alpha_state(row, config), axis=1)
    out["Alpha_Rank"] = alpha_rank(out)
    return out


def sort_by_alpha(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=[
            "Alpha_Score",
            "Persistence_Score",
            "Trend_Quality_Score",
            "Momentum_Score",
            "Overextension_Penalty",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
    )


def alpha_rank(df: pd.DataFrame) -> pd.Series:
    sorted_valid = sort_by_alpha(df.dropna(subset=["Alpha_Score"]))
    ranks = pd.Series(np.nan, index=df.index, dtype="float64")
    ranks.loc[sorted_valid.index] = np.arange(1, len(sorted_valid) + 1, dtype=float)
    return ranks


def cross_sectional_percentile_rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.dropna()
    ranks = pd.Series(np.nan, index=numeric.index, dtype="float64")
    if valid.empty:
        return ranks
    if len(valid) == 1:
        ranks.loc[valid.index] = 100.0
        return ranks
    order = valid.rank(method="average", ascending=True)
    ranks.loc[valid.index] = ((order - 1.0) / (len(valid) - 1.0)) * 100.0
    return ranks.clip(0.0, 100.0)


def piecewise_score(values: pd.Series, points: list[tuple[float, float]] | list[list[float]]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    xp = np.array([float(point[0]) for point in points], dtype="float64")
    fp = np.array([float(point[1]) for point in points], dtype="float64")
    out = pd.Series(np.nan, index=numeric.index, dtype="float64")
    mask = numeric.notna()
    if mask.any():
        out.loc[mask] = np.interp(numeric.loc[mask].astype(float), xp, fp)
    return out.clip(0.0, 100.0)


def quadratic_penalty(values: pd.Series, start: float, end: float, max_penalty: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    scaled = ((numeric - start) / (end - start)).clip(0.0, 1.0)
    return max_penalty * (scaled**2)
