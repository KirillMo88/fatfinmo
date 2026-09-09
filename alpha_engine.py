from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from entry_risk import calculate_entry_risk


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
        "relative_points": [(-2.0, 0.0), (-1.0, 25.0), (0.0, 50.0), (1.0, 70.0), (2.0, 90.0), (3.0, 100.0)],
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
    "market_regime": {
        "spy_sma_weeks": 40,
        "drawdown_threshold": -0.10,
        "vol_window_weeks": 13,
        "high_vol_percentile": 75.0,
        "alpha_confidence": {
            "BULL": 100.0,
            "BULL_HIGH_VOL": 100.0,
            "CORRECTION": 50.0,
            "STRESS": 20.0,
        },
    },
    "entry_risk": {
        "perf12_soft_threshold": 98.0,
        "perf12_extreme_threshold": 99.0,
        "perf12_soft_risk": 5.0,
        "perf12_extreme_risk": 10.0,
        "stress_momentum_threshold": 70.0,
        "stress_smaz_extreme": 3.0,
        "stress_min_entry_risk": 80.0,
    },
    "state": {
        "strong_alpha": 80.0,
        "positive_alpha": 70.0,
        "moderate_alpha": 60.0,
        "neutral_alpha": 50.0,
    },
}

ALPHA_OUTPUT_COLUMNS = [
    "Alpha_Rank",
    "Alpha_Score",
    "Alpha_State",
    "Momentum_Score",
    "Trend_Quality_Score",
    "Persistence_Score",
    "Market_Regime",
    "Alpha_Confidence",
    "Fast_Transition_Risk",
    "Fast_Transition_State",
    "Macro_Transition_Risk",
    "Macro_Transition_State",
    "Overall_Transition_Status",
    "Entry_Risk_Score",
    "Entry_Risk",
    "Opportunity_State",
    "Opportunity_Score",
    "ADX_DI_Trend_Score",
    "DI_Balance",
    "SMA_Regime_Score",
    "Absolute_SMA_Score",
    "Relative_SMA_Score",
    "SMA200d_Robust_Z_36M",
    "Regime_Dependent_SMAZ_Risk",
    "Perf12M_Extreme_Risk",
    "Base_Alpha",
    "Alpha_Data_Complete",
]

TEXT_OUTPUT_COLUMNS = {
    "Alpha_State",
    "Market_Regime",
    "Entry_Risk",
    "Opportunity_State",
    "Fast_Transition_State",
    "Macro_Transition_State",
    "Overall_Transition_Status",
}


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
    out["Relative_SMA_Score"] = piecewise_score(smaz, cfg["relative_points"])
    relative_missing = out["Relative_SMA_Score"].isna()
    out.loc[relative_missing, "Relative_SMA_Score"] = float(cfg["relative_fallback_score"])
    out["SMA_Relative_Fallback_Used"] = relative_missing
    out["SMA_Regime_Score"] = (
        float(cfg["absolute"]) * out["Absolute_SMA_Score"]
        + float(cfg["relative"]) * out["Relative_SMA_Score"]
    )
    return out


def calculate_trend_quality_score(
    adx_di: pd.DataFrame,
    sma: pd.DataFrame,
    high52w: pd.Series,
    config: dict | None = None,
) -> pd.Series:
    cfg = (config or ALPHA_CONFIG)["trend"]
    return (
        float(cfg["adx_di"]) * adx_di["ADX_DI_Trend_Score"]
        + float(cfg["sma_regime"]) * sma["SMA_Regime_Score"]
        + float(cfg["high_52w"]) * high52w
    ).rename("Trend_Quality_Score")


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


def calculate_alpha_score(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = (config or ALPHA_CONFIG)["weights"]
    out = pd.DataFrame(index=df.index)
    out["Base_Alpha"] = (
        float(cfg["momentum"]) * df["Momentum_Score"]
        + float(cfg["trend"]) * df["Trend_Quality_Score"]
        + float(cfg["persistence"]) * df["Persistence_Score"]
    )
    out["Alpha_Score"] = out["Base_Alpha"].clip(0.0, 100.0)
    return out


def classify_alpha_state(row: pd.Series, config: dict | None = None) -> str:
    cfg = (config or ALPHA_CONFIG)["state"]
    alpha = row.get("Alpha_Score", np.nan)
    if not np.isfinite(alpha):
        return "Missing Data"
    if alpha >= float(cfg["strong_alpha"]):
        return "Strong Alpha"
    if alpha >= float(cfg["positive_alpha"]):
        return "Positive Alpha"
    if alpha >= float(cfg["moderate_alpha"]):
        return "Moderate Alpha"
    if alpha >= float(cfg["neutral_alpha"]):
        return "Neutral"
    return "Weak"


def calculate_alpha_engine(
    df: pd.DataFrame,
    market_regime: dict | pd.Series | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    config = config or ALPHA_CONFIG
    out = df.copy()
    if out.empty:
        for col in ALPHA_OUTPUT_COLUMNS:
            out[col] = pd.Series(dtype="object" if col in TEXT_OUTPUT_COLUMNS else "float64")
        return out

    momentum = calculate_momentum_score(out, config)
    adx_di = calculate_adx_di_score(out["ADX_14"], out["DI_Plus_14"], out["DI_Minus_14"], config)
    sma = calculate_sma_regime_score(out, config)
    high52w = piecewise_score(out["Price_vs_52W_High_%"], config["high_52w"]["points"]).rename("High52W_Score")
    trend_quality = calculate_trend_quality_score(adx_di, sma, high52w, config)
    persistence = calculate_persistence_score(out, momentum, config)

    parts = pd.concat(
        [
            momentum,
            adx_di,
            sma,
            high52w,
            trend_quality,
            persistence,
        ],
        axis=1,
    )
    score_inputs = pd.concat([out, parts], axis=1)
    scores = calculate_alpha_score(score_inputs, config)
    out = pd.concat([out, parts, scores], axis=1)

    market = normalize_market_regime(market_regime, config)
    for col, value in market.items():
        out[col] = value

    entry_risk = calculate_entry_risk(out, config)
    out = pd.concat([out, entry_risk], axis=1)

    core_cols = [
        "Momentum_Score",
        "ADX_DI_Trend_Score",
        "Absolute_SMA_Score",
        "High52W_Score",
        "Persistence_Score",
    ]
    out["Alpha_Data_Complete"] = out[core_cols].notna().all(axis=1) & ~out["SMA_Relative_Fallback_Used"]
    incomplete = out[core_cols].isna().any(axis=1)
    out.loc[incomplete, ["Base_Alpha", "Alpha_Score", "Opportunity_Score"]] = np.nan
    out["Alpha_State"] = out.apply(lambda row: classify_alpha_state(row, config), axis=1)
    out["Opportunity_State"] = out.apply(classify_opportunity_state, axis=1)
    out["Alpha_Rank"] = alpha_rank(out)
    return out


def normalize_market_regime(market_regime: dict | pd.Series | None, config: dict | None = None) -> dict:
    cfg = (config or ALPHA_CONFIG)["market_regime"]
    if market_regime is None:
        market_regime = {}
    if isinstance(market_regime, pd.Series):
        market_regime = market_regime.to_dict()

    regime = market_regime.get("Market_Regime", "UNKNOWN")
    confidence = market_regime.get("Alpha_Confidence")
    if confidence is None:
        confidence = cfg["alpha_confidence"].get(regime, np.nan)
    return {
        "Market_Regime": regime,
        "Alpha_Confidence": confidence,
        "Fast_Transition_Risk": market_regime.get("Fast_Transition_Risk", np.nan),
        "Fast_Transition_State": market_regime.get("Fast_Transition_State", "DATA_INCOMPLETE"),
        "Macro_Transition_Risk": market_regime.get("Macro_Transition_Risk", np.nan),
        "Macro_Transition_State": market_regime.get("Macro_Transition_State", "DATA_INCOMPLETE"),
        "Overall_Transition_Status": market_regime.get("Overall_Transition_Status", "DATA_INCOMPLETE"),
        "SPY_vs_SMA40W_%": market_regime.get("SPY_vs_SMA40W_%", np.nan),
        "SPY_Drawdown_52W_%": market_regime.get("SPY_Drawdown_52W_%", np.nan),
        "SPY_Volatility_13W_%": market_regime.get("SPY_Volatility_13W_%", np.nan),
        "SPY_Volatility_Percentile": market_regime.get("SPY_Volatility_Percentile", np.nan),
    }


def sort_by_alpha(df: pd.DataFrame, sort_by: str = "Alpha Score") -> pd.DataFrame:
    sort_key = sort_by or "Alpha Score"
    if sort_key == "Off":
        return df
    if sort_key == "Opportunity State":
        return sort_by_opportunity_state(df)
    if sort_key == "Entry Risk":
        return df.sort_values(
            by=["Entry_Risk_Score", "Alpha_Score", "Opportunity_Score"],
            ascending=[True, False, False],
            na_position="last",
        )
    if sort_key == "Alpha Confidence":
        return df.sort_values(
            by=["Alpha_Confidence", "Alpha_Score", "Entry_Risk_Score"],
            ascending=[False, False, True],
            na_position="last",
        )
    if sort_key == "Opportunity Score":
        return df.sort_values(
            by=["Opportunity_Score", "Alpha_Score", "Entry_Risk_Score"],
            ascending=[False, False, True],
            na_position="last",
        )
    return df.sort_values(
        by=[
            "Alpha_Score",
            "Persistence_Score",
            "Trend_Quality_Score",
            "Momentum_Score",
            "Entry_Risk_Score",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
    )


def sort_by_opportunity_state(df: pd.DataFrame) -> pd.DataFrame:
    order = {
        "HIGH_CONVICTION": 0,
        "ATTRACTIVE_BUT_MACRO_WATCH": 1,
        "FAST_TRANSITION_WARNING": 2,
        "STRONG_BUT_MACRO_RISK": 3,
        "ATTRACTIVE": 4,
        "STRONG_BUT_EXTENDED": 5,
        "LOW_CONFIDENCE": 6,
        "STRESS_AVOID_CHASING": 7,
        "NEUTRAL": 8,
        "WEAK": 9,
        "Missing Data": 10,
    }
    out = df.copy()
    out["_Opportunity_State_Order"] = out["Opportunity_State"].map(order).fillna(99)
    return out.sort_values(
        by=["_Opportunity_State_Order", "Opportunity_Score", "Alpha_Score", "Entry_Risk_Score"],
        ascending=[True, False, False, True],
        na_position="last",
    ).drop(columns=["_Opportunity_State_Order"], errors="ignore")


def alpha_rank(df: pd.DataFrame) -> pd.Series:
    sorted_valid = sort_by_alpha(df.dropna(subset=["Alpha_Score"]), sort_by="Alpha Score")
    ranks = pd.Series(np.nan, index=df.index, dtype="float64")
    ranks.loc[sorted_valid.index] = np.arange(1, len(sorted_valid) + 1, dtype=float)
    return ranks


def classify_opportunity_state(row: pd.Series) -> str:
    alpha = row.get("Alpha_Score", np.nan)
    entry = row.get("Entry_Risk_Score", np.nan)
    regime = row.get("Market_Regime", "UNKNOWN")
    fast_risk = row.get("Fast_Transition_Risk", np.nan)
    macro_risk = row.get("Macro_Transition_Risk", np.nan)
    if not np.isfinite(alpha) or not np.isfinite(entry) or regime == "UNKNOWN":
        return "Missing Data"
    if regime == "STRESS" and entry >= 60.0:
        return "STRESS_AVOID_CHASING"
    if alpha >= 70.0 and np.isfinite(fast_risk) and fast_risk > 60.0:
        return "FAST_TRANSITION_WARNING"
    if alpha >= 70.0 and np.isfinite(macro_risk) and macro_risk > 60.0:
        return "STRONG_BUT_MACRO_RISK"
    if alpha < 50.0:
        return "WEAK"
    if (
        alpha >= 75.0
        and entry <= 30.0
        and regime in {"BULL", "BULL_HIGH_VOL"}
        and (not np.isfinite(fast_risk) or fast_risk <= 40.0)
        and (not np.isfinite(macro_risk) or macro_risk <= 40.0)
    ):
        return "HIGH_CONVICTION"
    if alpha >= 70.0 and (not np.isfinite(fast_risk) or fast_risk <= 40.0) and np.isfinite(macro_risk) and macro_risk > 40.0:
        return "ATTRACTIVE_BUT_MACRO_WATCH"
    if alpha >= 65.0 and regime == "CORRECTION":
        return "LOW_CONFIDENCE"
    if alpha >= 70.0 and entry > 40.0:
        return "STRONG_BUT_EXTENDED"
    if alpha >= 65.0 and entry <= 40.0:
        return "ATTRACTIVE"
    return "NEUTRAL"


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
