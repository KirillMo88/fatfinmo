from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finance_core import download_completed_ohlcv
from global_liquidity import read_global_liquidity
from global_macro_tab import (
    _download_fred_macro_series,
    _tradingview_economic_series_or_fallback,
    _tradingview_rate_series_or_fallback,
)
from hy_oas import combine_hy_oas_sources, weekly_archive_available_frame


SPY_MACRO_HORIZONS = ("3M", "6M", "9M", "12M")
SPY_MACRO_WEEKS = {"3M": 13, "6M": 26, "9M": 39, "12M": 52}
SPY_MACRO_RANGE_OPTIONS = ["1Y", "5Y", "10Y", "20Y", "2015 -> Latest", "FULL"]
SPY_MACRO_RANGE_YEARS = {"1Y": 1, "5Y": 5, "10Y": 10, "20Y": 20, "2015 -> Latest": None, "FULL": None}
PERCENTILE_WINDOW_WEEKS = 156
PERCENTILE_MIN_PERIODS = 52
DATA_DIR = Path(__file__).with_name("data")


@dataclass
class SPYMacroOutlook:
    history: pd.DataFrame
    current: dict[str, Any]
    pre_drawdown_averages: dict[str, float]
    data_quality: pd.DataFrame


def _clean_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    out = pd.to_numeric(series, errors="coerce").dropna().copy()
    if out.empty:
        return out
    idx = pd.to_datetime(out.index, errors="coerce")
    valid = ~pd.isna(idx)
    out = out.loc[valid]
    out.index = pd.DatetimeIndex(idx[valid]).tz_localize(None).normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def _download_market_close(ticker: str) -> pd.Series:
    try:
        frame = download_completed_ohlcv(ticker, period="max")
        close = pd.to_numeric(frame.get("Close", pd.Series(dtype="float64")), errors="coerce").dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        return close[~close.index.duplicated(keep="last")].sort_index()
    except Exception:
        return pd.Series(dtype="float64")


def _align_weekly(series: pd.Series | None, target_dates: pd.Series) -> pd.Series:
    source = _clean_series(series)
    target = pd.DatetimeIndex(pd.to_datetime(target_dates, errors="coerce").dropna()).tz_localize(None).normalize()
    if source.empty or len(target) == 0:
        return pd.Series(index=target, dtype="float64")
    weekly = source.resample("W-FRI").last().dropna()
    target_frame = pd.DataFrame({"Date": target, "_order": np.arange(len(target))}).sort_values("Date")
    source_frame = pd.DataFrame({"Date": weekly.index, "Value": weekly.to_numpy()})
    aligned = pd.merge_asof(target_frame, source_frame.sort_values("Date"), on="Date", direction="backward")
    aligned = aligned.sort_values("_order")
    return pd.Series(aligned["Value"].to_numpy(), index=target)


def rolling_percentile(series: pd.Series, window: int = PERCENTILE_WINDOW_WEEKS, min_periods: int = PERCENTILE_MIN_PERIODS) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        clean = values[np.isfinite(values)]
        if len(clean) < min_periods or not np.isfinite(values[-1]):
            return np.nan
        return float((clean <= values[-1]).sum() / len(clean) * 100.0)

    return pd.to_numeric(series, errors="coerce").rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def weighted_score(frame: pd.DataFrame, columns: list[str], weights: list[float]) -> pd.Series:
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    valid = values.notna().all(axis=1)
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    result.loc[valid] = values.loc[valid].mul(weights, axis=1).sum(axis=1)
    return result.clip(0, 100)


def _prepare_source_frame(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    if frame is None or frame.empty or date_column not in frame:
        return pd.DataFrame()
    out = frame.copy()
    out["Date"] = pd.to_datetime(out[date_column], errors="coerce").dt.tz_localize(None).dt.normalize()
    return out.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")


def _hy_series(api_key: str | None, end_date: pd.Timestamp) -> pd.Series:
    archive_path = DATA_DIR / "BAMLH0A0HYM2_weekly.csv"
    archive = pd.DataFrame()
    if archive_path.exists():
        try:
            raw = pd.read_csv(archive_path, usecols=["time", "close"])
            archive = weekly_archive_available_frame(raw)
        except Exception:
            archive = pd.DataFrame()
    fred = _download_fred_macro_series(api_key).get("BAMLH0A0HYM2", pd.Series(dtype="float64"))
    fred_frame = pd.DataFrame(
        {
            "Date": _clean_series(fred).index,
            "HY_OAS": _clean_series(fred).values,
            "HYOASourceFrequency": "DAILY_FRED",
        }
    )
    combined = combine_hy_oas_sources(archive, pd.DataFrame(), fred_frame, end_date=end_date)
    if combined.empty:
        return pd.Series(dtype="float64")
    return pd.Series(combined["HY_OAS"].to_numpy(), index=pd.to_datetime(combined["Date"]))


def calculate_monetary_block(frame: pd.DataFrame) -> pd.DataFrame:
    for name, level_weight, direction_weight in (("US2Y", 0.50, 0.50), ("DXY", 0.30, 0.70), ("US10Y", 0.50, 0.50)):
        frame[f"{name}_LevelPct"] = rolling_percentile(frame[name])
        lookback = 13 if name == "US2Y" else 26
        change = frame[name].diff(lookback) if name != "DXY" else frame[name].pct_change(lookback, fill_method=None)
        frame[f"{name}_DirectionPct"] = rolling_percentile(change)
        frame[f"{name}_Score"] = weighted_score(
            pd.DataFrame({"level": 100 - frame[f"{name}_LevelPct"], "direction": 100 - frame[f"{name}_DirectionPct"]}),
            ["level", "direction"],
            [level_weight, direction_weight],
        )
    frame["Monetary_Score"] = weighted_score(
        frame.rename(columns={"US2Y_Score": "us2y", "DXY_Score": "dxy", "US10Y_Score": "us10y"}),
        ["us2y", "dxy", "us10y"],
        [0.50, 0.43, 0.07],
    )
    return frame


def _calculate_liquidity_factor(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    level = pd.to_numeric(frame[prefix], errors="coerce")
    growth = level / level.shift(52) - 1.0
    frame[f"{prefix}_GrowthPct"] = growth
    frame[f"{prefix}_GrowthLevelPct"] = rolling_percentile(growth)
    for label, weeks in (("Fast", 13), ("Medium", 26), ("Slow", 39)):
        impulse = growth - growth.shift(weeks)
        frame[f"{prefix}_{label}ImpulsePct"] = impulse
        frame[f"{prefix}_{label}ImpulsePercentile"] = rolling_percentile(impulse)
    for horizon, fast_weight, medium_weight, slow_weight in (
        ("3M", 0.67, 0.33, 0.00),
        ("6M", 0.33, 0.34, 0.33),
        ("9M", 0.00, 0.25, 0.75),
        ("12M", 0.00, 0.00, 1.00),
    ):
        impulse = (
            frame[f"{prefix}_FastImpulsePercentile"] * fast_weight
            + frame[f"{prefix}_MediumImpulsePercentile"] * medium_weight
            + frame[f"{prefix}_SlowImpulsePercentile"] * slow_weight
        )
        frame[f"{prefix}_{horizon}"] = frame[f"{prefix}_GrowthLevelPct"] * 0.40 + impulse * 0.60
    return frame


def calculate_liquidity_block(frame: pd.DataFrame) -> pd.DataFrame:
    for prefix in ("Global_M2", "Global_CB_Assets", "US_Net_Liquidity"):
        _calculate_liquidity_factor(frame, prefix)
    for horizon in SPY_MACRO_HORIZONS:
        frame[f"Liquidity_{horizon}"] = weighted_score(
            pd.DataFrame(
                {
                    "m2": frame[f"Global_M2_{horizon}"],
                    "cb": frame[f"Global_CB_Assets_{horizon}"],
                    "usnl": frame[f"US_Net_Liquidity_{horizon}"],
                }
            ),
            ["m2", "cb", "usnl"],
            [0.30, 0.10, 0.60],
        )
    return frame


def calculate_growth_block(frame: pd.DataFrame) -> pd.DataFrame:
    frame["ISM_LevelPct"] = rolling_percentile(frame["ISM"])
    frame["ISM_DirectionPct"] = rolling_percentile(frame["ISM"].diff(26))
    frame["Retail_YoY"] = frame["Retail_Sales"] / frame["Retail_Sales"].shift(52) - 1.0
    frame["Retail_LevelPct"] = rolling_percentile(frame["Retail_YoY"])
    frame["Retail_DirectionPct"] = rolling_percentile(frame["Retail_YoY"].diff(26))
    frame["Growth_3M"] = weighted_score(
        pd.DataFrame({"ism": frame["ISM_DirectionPct"], "retail": frame["Retail_DirectionPct"]}),
        ["ism", "retail"],
        [0.80, 0.20],
    )
    frame["Growth_6M"] = weighted_score(
        pd.DataFrame({"ism": frame["ISM_DirectionPct"], "retail": frame["Retail_DirectionPct"]}),
        ["ism", "retail"],
        [0.75, 0.25],
    )
    frame["Growth_9M"] = weighted_score(
        pd.DataFrame(
            {
                "ism_direction": frame["ISM_DirectionPct"],
                "ism_maturity": 100 - frame["ISM_LevelPct"],
                "retail": frame["Retail_DirectionPct"],
            }
        ),
        ["ism_direction", "ism_maturity", "retail"],
        [0.60, 0.20, 0.20],
    )
    frame["BusinessCycle_12M"] = weighted_score(
        pd.DataFrame(
            {
                "ism_direction": frame["ISM_DirectionPct"],
                "ism_maturity": 100 - frame["ISM_LevelPct"],
                "retail_maturity": 100 - frame["Retail_LevelPct"],
            }
        ),
        ["ism_direction", "ism_maturity", "retail_maturity"],
        [0.40, 0.30, 0.30],
    )
    return frame


def calculate_transmission_block(frame: pd.DataFrame) -> pd.DataFrame:
    frame["HY_LevelPct"] = rolling_percentile(frame["HY_OAS"])
    frame["HY_DirectionPct"] = (
        rolling_percentile(frame["HY_OAS"].diff(4)) * 0.35
        + rolling_percentile(frame["HY_OAS"].diff(13)) * 0.65
    )
    frame["HY_Health"] = 0.65 * (100 - frame["HY_DirectionPct"]) + 0.35 * (100 - frame["HY_LevelPct"])
    frame.loc[(frame["HY_LevelPct"] >= 60) & (frame["HY_DirectionPct"] <= 30), "HY_Health"] += 15
    frame.loc[(frame["HY_LevelPct"] <= 40) & (frame["HY_DirectionPct"] >= 70), "HY_Health"] -= 15
    frame["HY_Health"] = frame["HY_Health"].clip(0, 100)
    frame["ANFCI_LevelPct"] = rolling_percentile(frame["ANFCI"])
    frame["ANFCI_DirectionPct"] = rolling_percentile(frame["ANFCI"].diff(13))
    frame["ANFCI_Health"] = 0.50 * (100 - frame["ANFCI_LevelPct"]) + 0.50 * (100 - frame["ANFCI_DirectionPct"])
    frame["Transmission_Score"] = weighted_score(
        frame.rename(columns={"HY_Health": "hy", "ANFCI_Health": "anfci"}),
        ["hy", "anfci"],
        [0.60, 0.40],
    )
    return frame


def calculate_modifier_fields(frame: pd.DataFrame) -> pd.DataFrame:
    frame["T5YIE_DirectionPct"] = rolling_percentile(frame["T5YIE"].diff(26))
    frame["T5YIE_Modifier"] = np.select(
        [frame["T5YIE_DirectionPct"] < 20, frame["T5YIE_DirectionPct"] > 80],
        ["Disinflation relief", "Inflation repricing"],
        default="Neutral",
    )
    frame["WTI_LevelPct"] = rolling_percentile(frame["WTI"])
    frame["WTI_MomentumPct"] = rolling_percentile(frame["WTI"].pct_change(13, fill_method=None))
    frame["WTI_ShockRisk"] = np.where(
        (frame["WTI_LevelPct"] > 80) & (frame["WTI_MomentumPct"] > 80) & (frame["T5YIE_DirectionPct"] > 60),
        "Confirmed Inflation Shock",
        "Neutral",
    )
    frame["JP10Y_LevelPct"] = rolling_percentile(frame["JP10Y"])
    frame["JP10Y_4W_Pct"] = rolling_percentile(frame["JP10Y"].diff(4))
    frame["JP10Y_13W_Pct"] = rolling_percentile(frame["JP10Y"].diff(13))
    frame["JP10Y_ShockRisk"] = np.where(
        (frame["JP10Y_LevelPct"] > 80) & ((frame["JP10Y_4W_Pct"] > 90) | (frame["JP10Y_13W_Pct"] > 80)),
        "Confirmed BoJ Tightening Shock",
        "Neutral",
    )
    return frame


def _set_transition(frame: pd.DataFrame, name: str, conditions: list[pd.Series]) -> None:
    frame[name] = np.select(conditions, ["Negative", "Positive"], default="Neutral")


def calculate_transition_flags(frame: pd.DataFrame) -> pd.DataFrame:
    _set_transition(
        frame,
        "Transition_US2Y",
        [
            (frame["US2Y_LevelPct"] > 70) & (frame["US2Y_DirectionPct"] > 70),
            (frame["US2Y_LevelPct"] > 70) & (frame["US2Y_DirectionPct"] < 30),
        ],
    )
    _set_transition(frame, "Transition_USD", [frame["DXY_DirectionPct"] > 70, frame["DXY_DirectionPct"] < 30])
    _set_transition(
        frame,
        "Transition_Liquidity",
        [
            (frame["Global_M2_GrowthLevelPct"] < 50) & (frame["Global_M2_SlowImpulsePercentile"] < 30),
            frame["Global_M2_SlowImpulsePercentile"] > 70,
        ],
    )
    _set_transition(
        frame,
        "Transition_HY",
        [
            (frame["HY_LevelPct"] < 40) & (frame["HY_DirectionPct"] > 70),
            (frame["HY_LevelPct"] > 60) & (frame["HY_DirectionPct"] < 30),
        ],
    )
    _set_transition(frame, "Transition_ANFCI", [frame["ANFCI_DirectionPct"] > 70, frame["ANFCI_DirectionPct"] < 30])
    _set_transition(
        frame,
        "Transition_BusinessCycle",
        [
            (frame["ISM_LevelPct"] > 70) & (frame["ISM_DirectionPct"] < 30),
            (frame["ISM_LevelPct"] < 30) & (frame["ISM_DirectionPct"] > 70),
        ],
    )
    dependencies = {
        "Transition_US2Y": ["US2Y_LevelPct", "US2Y_DirectionPct"],
        "Transition_USD": ["DXY_DirectionPct"],
        "Transition_Liquidity": ["Global_M2_GrowthLevelPct", "Global_M2_SlowImpulsePercentile"],
        "Transition_HY": ["HY_LevelPct", "HY_DirectionPct"],
        "Transition_ANFCI": ["ANFCI_DirectionPct"],
        "Transition_BusinessCycle": ["ISM_LevelPct", "ISM_DirectionPct"],
    }
    for column, required in dependencies.items():
        frame.loc[frame[required].isna().any(axis=1), column] = "Watch"
    return frame


def calculate_drawdown_fields(frame: pd.DataFrame) -> pd.DataFrame:
    spx = pd.to_numeric(frame["SPX"], errors="coerce")
    frame["SPX_Drawdown"] = spx / spx.cummax() - 1.0
    events: list[int] = []
    active = False
    for value in frame["SPX_Drawdown"].to_numpy():
        if not np.isfinite(value):
            events.append(0)
            continue
        if value <= -0.10 and not active:
            events.append(1)
            active = True
        else:
            events.append(0)
        if value >= 0:
            active = False
    frame["Drawdown_10pct_Event"] = events
    return frame


def _score_state(value: Any) -> str:
    if pd.isna(value):
        return "INSUFFICIENT DATA"
    value = float(value)
    if value >= 75:
        return "SUPPORTIVE"
    if value >= 55:
        return "CONSTRUCTIVE"
    if value >= 45:
        return "NEUTRAL"
    if value >= 25:
        return "CAUTIOUS"
    return "DEFENSIVE"


def score_state(value: Any) -> str:
    return _score_state(value)


def score_direction(outlook: SPYMacroOutlook, horizon: str) -> str:
    column = f"Macro_{horizon}"
    if column not in outlook.history:
        return "neutral"
    score = pd.to_numeric(outlook.history[column], errors="coerce")
    direction = score.rolling(8, min_periods=8).mean().diff(13).iloc[-1]
    if pd.isna(direction):
        return "neutral"
    if direction > 1:
        return "accelerating"
    if direction < -1:
        return "decelerating"
    return "neutral"


def term_structure_state(scores: dict[str, Any]) -> str:
    values = pd.Series(scores, dtype="float64").reindex(SPY_MACRO_HORIZONS)
    if values.isna().any():
        return "Insufficient data"
    if values.is_monotonic_increasing and values.iloc[-1] - values.iloc[0] > 5:
        return "Improving"
    if values.is_monotonic_decreasing and values.iloc[0] - values.iloc[-1] > 5:
        return "Deteriorating"
    return "Flat / mixed"


def _add_data_status(frame: pd.DataFrame) -> pd.DataFrame:
    score_columns = [f"Macro_{horizon}" for horizon in SPY_MACRO_HORIZONS]
    frame["SPYMacroDataStatus"] = np.where(frame[score_columns].notna().all(axis=1), "FULL", "INSUFFICIENT DATA")
    return frame


def calculate_pre_drawdown_averages(frame: pd.DataFrame) -> dict[str, float]:
    events = frame.loc[frame["Drawdown_10pct_Event"].eq(1), "Date"]
    result: dict[str, float] = {}
    for horizon in SPY_MACRO_HORIZONS:
        values: list[float] = []
        for event_date in events:
            mask = frame["Date"].between(event_date - pd.Timedelta(days=30), event_date)
            values.extend(pd.to_numeric(frame.loc[mask, f"Macro_{horizon}"], errors="coerce").dropna().tolist())
        result[horizon] = float(np.mean(values)) if values else float("nan")
        frame[f"PreDrawdown_Avg_{horizon}"] = result[horizon]
    return result


def build_spy_macro_outlook(market_cycle_history: pd.DataFrame, api_key: str | None = None) -> SPYMacroOutlook:
    base = market_cycle_history[["Date", "SPX_Close"]].copy()
    base["Date"] = pd.to_datetime(base["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    base = base.rename(columns={"SPX_Close": "SPX"}).dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
    dates = base["Date"]
    fred = _download_fred_macro_series(api_key)
    ism, _, _ = _tradingview_economic_series_or_fallback("ECONOMICS:USBCOI", "FRED / NAPM", fred.get("NAPM"))
    jp10y, _, _ = _tradingview_rate_series_or_fallback("TVC:JP10Y", "JP10Y", "FRED / IRLTLT01JPM156N", fred.get("IRLTLT01JPM156N"))
    _, monthly, weekly = read_global_liquidity()
    monthly = _prepare_source_frame(monthly)
    weekly = _prepare_source_frame(weekly)
    source_map = {
        "US2Y": fred.get("DGS2"),
        "US10Y": fred.get("DGS10"),
        "DXY": _download_market_close("DX-Y.NYB"),
        "T5YIE": fred.get("T5YIE"),
        "WTI": _download_market_close("CL=F"),
        "ISM": ism,
        "Retail_Sales": fred.get("RSAFS"),
        "ANFCI": fred.get("ANFCI"),
        "HY_OAS": _hy_series(api_key, dates.max()),
        "JP10Y": jp10y,
    }
    for name, series in source_map.items():
        base[name] = _align_weekly(series, dates).to_numpy()
    if not monthly.empty:
        for source_name, column in (("Global_M2", "global_m2_usd_bn"), ("Global_CB_Assets", "global_cb_assets_usd_bn")):
            base[source_name] = _align_weekly(pd.Series(monthly[column].to_numpy(), index=monthly["Date"]), dates).to_numpy()
    else:
        base["Global_M2"] = np.nan
        base["Global_CB_Assets"] = np.nan
    if not weekly.empty:
        base["US_Net_Liquidity"] = _align_weekly(
            pd.Series(weekly["us_net_liquidity_usd_bn"].to_numpy(), index=weekly["Date"]),
            dates,
        ).to_numpy()
    else:
        base["US_Net_Liquidity"] = np.nan
    frame = base.reset_index(drop=True)
    calculate_monetary_block(frame)
    calculate_liquidity_block(frame)
    calculate_growth_block(frame)
    calculate_transmission_block(frame)
    calculate_modifier_fields(frame)
    calculate_transition_flags(frame)
    calculate_drawdown_fields(frame)
    block_columns = {
        "Monetary": "Monetary_Score",
        "Liquidity": "Liquidity_{}",
        "Growth": "Growth_{}",
    }
    for horizon in SPY_MACRO_HORIZONS:
        growth_column = "BusinessCycle_12M" if horizon == "12M" else f"Growth_{horizon}"
        frame[f"Macro_{horizon}"] = weighted_score(
            pd.DataFrame(
                {
                    "monetary": frame["Monetary_Score"],
                    "liquidity": frame[f"Liquidity_{horizon}"],
                    "growth": frame[growth_column],
                }
            ),
            ["monetary", "liquidity", "growth"],
            {"3M": [0.45, 0.35, 0.20], "6M": [0.40, 0.30, 0.30], "9M": [0.25, 0.55, 0.20], "12M": [0.20, 0.50, 0.30]}[horizon],
        )
    pre_drawdown = calculate_pre_drawdown_averages(frame)
    _add_data_status(frame)
    latest = frame.dropna(subset=["Date"]).iloc[-1].to_dict() if not frame.empty else {}
    current_scores = {horizon: latest.get(f"Macro_{horizon}") for horizon in SPY_MACRO_HORIZONS}
    latest["TermStructureState"] = term_structure_state(current_scores)
    latest["DataStatus"] = latest.get("SPYMacroDataStatus", "INSUFFICIENT DATA")
    quality_rows = []
    for name in ("SPX", "US2Y", "US10Y", "DXY", "Global_M2", "Global_CB_Assets", "US_Net_Liquidity", "ISM", "Retail_Sales", "HY_OAS", "ANFCI", "T5YIE", "WTI", "JP10Y"):
        series = pd.to_numeric(frame.get(name, pd.Series(dtype="float64")), errors="coerce")
        valid = series.dropna()
        quality_rows.append({"Input": name, "First Date": frame.loc[valid.index, "Date"].min() if not valid.empty else pd.NaT, "Last Date": frame.loc[valid.index, "Date"].max() if not valid.empty else pd.NaT, "Observations": int(valid.size), "Status": "OK" if not valid.empty else "MISSING"})
    return SPYMacroOutlook(frame, latest, pre_drawdown, pd.DataFrame(quality_rows))


def build_model_details(outlook: SPYMacroOutlook, horizon: str) -> pd.DataFrame:
    row = outlook.history.iloc[-1]
    rows: list[dict[str, Any]] = []
    factor_specs = [
        ("Monetary", "US2Y", "US2Y_Score", 0.50, "US2Y_LevelPct", "US2Y_DirectionPct"),
        ("Monetary", "DXY", "DXY_Score", 0.43, "DXY_LevelPct", "DXY_DirectionPct"),
        ("Monetary", "US10Y", "US10Y_Score", 0.07, "US10Y_LevelPct", "US10Y_DirectionPct"),
        ("Liquidity", "Global M2", f"Global_M2_{horizon}", 0.30, "Global_M2_GrowthLevelPct", f"Global_M2_{horizon}"),
        ("Liquidity", "Global CB Assets", f"Global_CB_Assets_{horizon}", 0.10, "Global_CB_Assets_GrowthLevelPct", f"Global_CB_Assets_{horizon}"),
        ("Liquidity", "US Net Liquidity", f"US_Net_Liquidity_{horizon}", 0.60, "US_Net_Liquidity_GrowthLevelPct", f"US_Net_Liquidity_{horizon}"),
    ]
    horizon_weights = {"3M": (0.45, 0.35, 0.20), "6M": (0.40, 0.30, 0.30), "9M": (0.25, 0.55, 0.20), "12M": (0.20, 0.50, 0.30)}[horizon]
    for block, factor, score_column, weight, level_column, direction_column in factor_specs[:6]:
        score = row.get(score_column)
        rows.append({"Block": block, "Factor": factor, "Raw Value": row.get(factor.replace(" ", "_")), "Level Percentile": row.get(level_column), "Fast / Medium / Slow Impulse": "", "Direction Percentile": row.get(direction_column), "Factor Score": score, "Weight": weight, "Contribution": score * weight if pd.notna(score) else np.nan})
    growth_scores = {"3M": row.get("Growth_3M"), "6M": row.get("Growth_6M"), "9M": row.get("Growth_9M"), "12M": row.get("BusinessCycle_12M")}
    for factor, weight, level_column, direction_column in (
        ("ISM Manufacturing", {"3M": 0.80, "6M": 0.75, "9M": 0.60, "12M": 0.40}[horizon], "ISM_LevelPct", "ISM_DirectionPct"),
        ("Retail Sales", {"3M": 0.20, "6M": 0.25, "9M": 0.20, "12M": 0.30}[horizon], "Retail_LevelPct", "Retail_DirectionPct"),
    ):
        score = growth_scores[horizon]
        rows.append({"Block": "Growth", "Factor": factor, "Raw Value": row.get("ISM" if factor.startswith("ISM") else "Retail_Sales"), "Level Percentile": row.get(level_column), "Fast / Medium / Slow Impulse": "", "Direction Percentile": row.get(direction_column), "Factor Score": score, "Weight": weight, "Contribution": score * weight if pd.notna(score) else np.nan})
    block_scores = {"Monetary": row.get("Monetary_Score"), "Liquidity": row.get(f"Liquidity_{horizon}"), "Growth": growth_scores[horizon]}
    for block, score, weight in zip(("Monetary", "Liquidity", "Growth"), block_scores.values(), horizon_weights):
        rows.append({"Block": "Macro Score", "Factor": block, "Raw Value": "", "Level Percentile": "", "Fast / Medium / Slow Impulse": "", "Direction Percentile": "", "Factor Score": score, "Weight": weight, "Contribution": score * weight if pd.notna(score) else np.nan})
    return pd.DataFrame(rows)


def build_spy_macro_workbook(market_cycle_history: pd.DataFrame, outlook: SPYMacroOutlook) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        market_cycle_history.to_excel(writer, index=False, sheet_name="Market Cycle")
        outlook.history.to_excel(writer, index=False, sheet_name="SPY Macro")
        outlook.data_quality.to_excel(writer, index=False, sheet_name="SPY Macro Quality")
        sheet_frames = {
            "Market Cycle": market_cycle_history,
            "SPY Macro": outlook.history,
            "SPY Macro Quality": outlook.data_quality,
        }
        for sheet_name, sheet_frame in sheet_frames.items():
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, max(1, len(sheet_frame)), max(0, len(sheet_frame.columns) - 1))
    return output.getvalue()
