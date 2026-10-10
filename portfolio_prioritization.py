"""Deterministic three-asset prioritization model used by the Portfolio tab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from btc_cycle import halving_cycle_position


@dataclass(frozen=True)
class PortfolioConfig:
    spx_gold_sma_months: int = 50
    spx_gold_hysteresis: float = 0.05
    spx_gold_confirmation_months: int = 3
    btc_gold_regression_weeks: int = 208
    btc_buy_percentile: float = 0.5
    btc_sell_percentile: float = 98.5
    btc_confirmation_weeks: int = 3


DEFAULT_CONFIG = PortfolioConfig()
PRIORITY_MATRIX = {
    ("Gold", "BTC"): "BTC > Gold > SPX",
    ("Gold", "Gold"): "Gold > BTC > SPX",
    ("SPX", "BTC"): "SPX > BTC > Gold",
    ("SPX", "Gold"): "SPX > Gold > BTC",
}


def _numeric_series(values: pd.Series) -> pd.Series:
    out = pd.Series(pd.to_numeric(values, errors="coerce").to_numpy(), index=pd.to_datetime(values.index, errors="coerce"))
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out.replace([np.inf, -np.inf], np.nan).dropna().sort_index().loc[lambda value: ~value.index.duplicated(keep="last")]


def completed_monthly_ratio(
    spx_close: pd.Series,
    gold_close: pd.Series,
    as_of: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Build ratio OHLC from synchronized daily observations, never independent extrema."""
    spx = _numeric_series(spx_close).rename("SPX")
    gold = _numeric_series(gold_close).rename("Gold")
    aligned = pd.concat([spx, gold], axis=1, join="inner").dropna()
    aligned = aligned.loc[(aligned["SPX"] > 0) & (aligned["Gold"] > 0)]
    if aligned.empty:
        return pd.DataFrame(columns=["Date", "open", "high", "low", "close"])
    ratio = (aligned["SPX"] / aligned["Gold"]).rename("ratio")
    monthly = ratio.resample("ME").agg(["first", "max", "min", "last"])
    monthly.columns = ["open", "high", "low", "close"]
    cutoff = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.now(tz="UTC")).tz_localize(None).normalize()
    current_month_end = cutoff.to_period("M").to_timestamp("M")
    monthly = monthly.loc[monthly.index < current_month_end].dropna(subset=["close"])
    return monthly.reset_index(names="Date")


def completed_weekly_ratio(
    btc_weekly: pd.DataFrame,
    gold_weekly: pd.Series | pd.DataFrame,
    as_of: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Align completed BTC and Gold weekly closes without using a future Gold observation."""
    if btc_weekly is None or btc_weekly.empty or gold_weekly is None or len(gold_weekly) == 0:
        return pd.DataFrame(columns=["Date", "BTC", "Gold", "ratio"])
    btc_date = "Date" if "Date" in btc_weekly.columns else "date"
    btc_price = "Close" if "Close" in btc_weekly.columns else "close"
    btc = pd.DataFrame({
        "Date": pd.to_datetime(btc_weekly[btc_date], errors="coerce", utc=True).dt.tz_localize(None),
        "BTC": pd.to_numeric(btc_weekly[btc_price], errors="coerce"),
    }).dropna()
    if isinstance(gold_weekly, pd.Series):
        gold = pd.DataFrame({"Date": gold_weekly.index, "Gold": gold_weekly.to_numpy()})
    else:
        gold_date = "Date" if "Date" in gold_weekly.columns else "date" if "date" in gold_weekly.columns else None
        gold_price = next((name for name in ("Gold", "gold_price", "Close", "close") if name in gold_weekly.columns), None)
        if gold_price is None:
            return pd.DataFrame(columns=["Date", "BTC", "Gold", "ratio"])
        gold = pd.DataFrame({"Date": gold_weekly[gold_date] if gold_date else gold_weekly.index, "Gold": gold_weekly[gold_price]})
    gold["Date"] = pd.to_datetime(gold["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    gold["Gold"] = pd.to_numeric(gold["Gold"], errors="coerce")
    btc = btc.loc[btc["BTC"] > 0].sort_values("Date").drop_duplicates("Date", keep="last")
    gold = gold.dropna().loc[lambda frame: frame["Gold"] > 0].sort_values("Date").drop_duplicates("Date", keep="last")
    aligned = pd.merge_asof(btc, gold, on="Date", direction="backward").dropna()
    cutoff = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.now(tz="UTC")).tz_localize(None).normalize()
    aligned = aligned.loc[aligned["Date"] < cutoff]
    aligned["ratio"] = aligned["BTC"] / aligned["Gold"]
    return aligned.reset_index(drop=True)


def priority_ranking(spx_gold_position: Any, btc_gold_position: Any) -> tuple[str | None, str | None]:
    ranking = PRIORITY_MATRIX.get((str(spx_gold_position), str(btc_gold_position)))
    return (ranking, ranking.split(" > ")[0]) if ranking else (None, None)


def _condition_runs(condition: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    flags = condition.fillna(False).astype(bool)
    groups = flags.ne(flags.shift(fill_value=False)).cumsum()
    runs = flags.groupby(groups).cumcount().add(1).where(flags, 0).astype(int)
    initial = flags & ~flags.shift(fill_value=False)
    return runs, initial, flags


def _apply_confirmation_health(
    frame: pd.DataFrame,
    position_col: str,
    conditions: dict[str, str],
    persistence: int,
) -> pd.DataFrame:
    out = frame.copy()
    for direction, condition_col in conditions.items():
        runs, initial, flags = _condition_runs(out[condition_col])
        out[f"{direction}_condition_run"] = runs
        out[f"{direction}_initial_event"] = initial
        out[f"{direction}_sustained_event"] = runs.eq(persistence)
        out[f"{direction}_sustained_active"] = runs.ge(persistence)

    health: list[str] = []
    initial_dates: list[pd.Timestamp | pd.NaT] = []
    sustained_dates: list[pd.Timestamp | pd.NaT] = []
    sustained_reached: list[bool] = []
    previous_episode_outcomes: list[str | None] = []
    episode_position: str | None = None
    episode_initial = pd.NaT
    episode_sustained = pd.NaT
    episode_confirmed = False
    invalidated = False
    last_initial: dict[str, pd.Timestamp | pd.NaT] = {key: pd.NaT for key in conditions}
    last_sustained: dict[str, pd.Timestamp | pd.NaT] = {key: pd.NaT for key in conditions}

    for _, row in out.iterrows():
        position = row.get(position_col)
        previous_episode_outcome: str | None = None
        for direction in conditions:
            if bool(row.get(f"{direction}_initial_event", False)):
                last_initial[direction] = pd.Timestamp(row["Date"])
            if bool(row.get(f"{direction}_sustained_event", False)):
                last_sustained[direction] = pd.Timestamp(row["Date"])
        if position not in conditions:
            episode_position = None
            health.append("Pending")
            initial_dates.append(pd.NaT)
            sustained_dates.append(pd.NaT)
            sustained_reached.append(False)
            previous_episode_outcomes.append(None)
            continue
        if position != episode_position:
            if episode_position is not None:
                previous_episode_outcome = "Confirmed at Exit" if episode_confirmed else "Unconfirmed at Exit"
            episode_position = str(position)
            episode_initial = pd.NaT
            episode_sustained = pd.NaT
            episode_confirmed = False
            invalidated = False
            if int(row.get(f"{episode_position}_condition_run", 0)) > 0:
                episode_initial = last_initial[episode_position]
            if bool(row.get(f"{episode_position}_sustained_active", False)):
                episode_sustained = last_sustained[episode_position]
                episode_confirmed = pd.notna(episode_sustained)
        if bool(row.get(f"{episode_position}_initial_event", False)):
            episode_initial = pd.Timestamp(row["Date"])
        if bool(row.get(f"{episode_position}_sustained_event", False)):
            episode_sustained = pd.Timestamp(row["Date"])
            episode_confirmed = True
        other = next((key for key in conditions if key != episode_position), None)
        if other and bool(row.get(f"{other}_sustained_event", False)):
            invalidated = True
        current_condition = bool(row.get(conditions[episode_position], False))
        current_run = int(row.get(f"{episode_position}_condition_run", 0))
        if invalidated:
            state = "Invalidated"
        elif episode_confirmed and current_condition:
            state = "Active"
        elif episode_confirmed:
            state = "Under Review"
        elif current_condition and current_run > 0:
            state = "Initial"
        else:
            state = "Pending"
        health.append(state)
        initial_dates.append(episode_initial)
        sustained_dates.append(episode_sustained)
        sustained_reached.append(episode_confirmed)
        previous_episode_outcomes.append(previous_episode_outcome)
    out["confirmation_health"] = health
    out["initial_confirmation_date"] = initial_dates
    out["sustained_confirmation_date"] = sustained_dates
    out["sustained_confirmation_reached"] = sustained_reached
    out["previous_episode_outcome"] = previous_episode_outcomes
    return out


def build_spx_gold_history(monthly_ratio: pd.DataFrame, config: PortfolioConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    if monthly_ratio is None or monthly_ratio.empty:
        return pd.DataFrame()
    out = monthly_ratio.copy().sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    for column in ("open", "high", "low", "close"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["sma50"] = out["close"].rolling(config.spx_gold_sma_months, min_periods=config.spx_gold_sma_months).mean()
    out["sma100"] = out["close"].rolling(100, min_periods=100).mean()
    out["sma200"] = out["close"].rolling(200, min_periods=200).mean()
    out["upper"] = out["sma50"] * (1.0 + config.spx_gold_hysteresis)
    out["lower"] = out["sma50"] * (1.0 - config.spx_gold_hysteresis)
    out["recent_high"] = out["high"].rolling(12, min_periods=12).max()
    out["previous_high"] = out["high"].shift(12).rolling(12, min_periods=12).max()
    out["recent_low"] = out["low"].rolling(12, min_periods=12).min()
    out["previous_low"] = out["low"].shift(12).rolling(12, min_periods=12).min()
    out["hh"] = out["recent_high"] > out["previous_high"]
    out["hl"] = out["recent_low"] > out["previous_low"]
    out["lh"] = out["recent_high"] < out["previous_high"]
    out["ll"] = out["recent_low"] < out["previous_low"]
    out["spx_structural_condition"] = out["hh"] & out["hl"] & (out["close"] > out["sma100"]) & (out["close"] > out["sma200"])
    out["gold_structural_condition"] = out["lh"] & out["ll"] & (out["close"] < out["sma100"]) & (out["close"] < out["sma200"])

    position: str | None = None
    positions: list[str | None] = []
    switches: list[bool] = []
    previous_positions: list[str | None] = []
    thresholds: list[float] = []
    for _, row in out.iterrows():
        previous = position
        threshold = np.nan
        if np.isfinite(row["sma50"]):
            if position == "SPX" and row["close"] < row["lower"]:
                position, threshold = "Gold", float(row["lower"])
            elif position == "Gold" and row["close"] > row["upper"]:
                position, threshold = "SPX", float(row["upper"])
            elif position is None and row["close"] > row["upper"]:
                position, threshold = "SPX", float(row["upper"])
            elif position is None and row["close"] < row["lower"]:
                position, threshold = "Gold", float(row["lower"])
        previous_positions.append(previous)
        positions.append(position)
        switches.append(previous != position and position is not None)
        thresholds.append(threshold)
    out["previous_position"] = previous_positions
    out["position"] = positions
    out["trading_switch"] = switches
    out["trading_threshold"] = thresholds
    out = _apply_confirmation_health(
        out,
        "position",
        {"SPX": "spx_structural_condition", "Gold": "gold_structural_condition"},
        config.spx_gold_confirmation_months,
    )
    gold_confirmed = out["Gold_sustained_event"].cummax()
    spx_confirmed = out["SPX_sustained_event"].cummax()
    spx_sustained = out["SPX_sustained_active"]
    gold_sustained = out["Gold_sustained_active"]
    out["recovery_phase"] = "None / Transition"
    out.loc[gold_confirmed & out["lh"] & out["ll"] & (out["close"] > out["sma200"]) & (out["close"] < out["sma100"]), "recovery_phase"] = "SPX Counter-Trend Rally"
    out.loc[gold_confirmed & (out["close"] > out["sma100"]) & out["hl"] & ~spx_sustained, "recovery_phase"] = "SPX Potential Recovery"
    out.loc[spx_confirmed & out["hh"] & out["hl"] & (out["close"] < out["sma200"]) & (out["close"] > out["sma100"]), "recovery_phase"] = "Gold Counter-Trend Rally"
    out.loc[spx_confirmed & (out["close"] < out["sma100"]) & out["lh"] & ~gold_sustained, "recovery_phase"] = "Gold Potential Recovery"
    return out


def _rolling_log_regression(values: pd.Series, weeks: int) -> pd.DataFrame:
    y = np.log(pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64"))
    trend = np.full(len(y), np.nan)
    residual = np.full(len(y), np.nan)
    percentile = np.full(len(y), np.nan)
    lower = np.full(len(y), np.nan)
    upper = np.full(len(y), np.nan)
    x = np.arange(weeks, dtype="float64")
    for index in range(weeks, len(y)):
        sample = y[index - weeks:index]
        if not np.isfinite(sample).all() or not np.isfinite(y[index]):
            continue
        slope, intercept = np.polyfit(x, sample, 1)
        fitted = slope * x + intercept
        historical_residuals = sample - fitted
        forecast = slope * weeks + intercept
        current_residual = y[index] - forecast
        trend[index] = forecast
        residual[index] = current_residual
        # Agreed point-in-time convention: weak empirical rank against the 208
        # in-sample residuals from the regression estimated before this week.
        percentile[index] = np.count_nonzero(historical_residuals <= current_residual) / weeks * 100.0
        lower[index] = forecast + float(np.quantile(historical_residuals, 0.005))
        upper[index] = forecast + float(np.quantile(historical_residuals, 0.985))
    return pd.DataFrame({
        "regression_trend": np.exp(trend),
        "regression_residual": residual,
        "channel_percentile": percentile,
        "channel_lower": np.exp(lower),
        "channel_upper": np.exp(upper),
    }, index=values.index)


def build_btc_gold_history(weekly_ratio: pd.DataFrame, config: PortfolioConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    if weekly_ratio is None or weekly_ratio.empty:
        return pd.DataFrame()
    out = weekly_ratio.copy().sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    out["ratio"] = pd.to_numeric(out["ratio"], errors="coerce")
    regression = _rolling_log_regression(out["ratio"], config.btc_gold_regression_weeks)
    out = pd.concat([out, regression.reset_index(drop=True)], axis=1)
    monthly = out.set_index("Date")["ratio"].resample("ME").last().to_frame("monthly_close")
    monthly["sma50m"] = monthly["monthly_close"].rolling(50, min_periods=50).mean()
    # A month-end value is only available to a weekly row at or after that month end.
    out = pd.merge_asof(out.sort_values("Date"), monthly[["sma50m"]].reset_index(), on="Date", direction="backward")
    out["roc13w"] = out["ratio"].pct_change(13, fill_method=None)
    out["roc26w"] = out["ratio"].pct_change(26, fill_method=None)
    out["btc_structural_condition"] = (out["ratio"] > out["sma50m"]) & (out["roc13w"] > 0) & (out["roc26w"] > 0)
    out["gold_structural_condition"] = (out["ratio"] < out["sma50m"]) & (out["roc13w"] < 0) & (out["roc26w"] < 0)

    position: str | None = None
    positions: list[str | None] = []
    previous_positions: list[str | None] = []
    switches: list[bool] = []
    thresholds: list[float] = []
    for _, row in out.iterrows():
        previous = position
        threshold = np.nan
        percentile = row["channel_percentile"]
        if np.isfinite(percentile):
            if position == "Gold" and percentile <= config.btc_buy_percentile:
                position, threshold = "BTC", config.btc_buy_percentile
            elif position == "BTC" and percentile >= config.btc_sell_percentile:
                position, threshold = "Gold", config.btc_sell_percentile
            elif position is None and percentile <= config.btc_buy_percentile:
                position, threshold = "BTC", config.btc_buy_percentile
            elif position is None and percentile >= config.btc_sell_percentile:
                position, threshold = "Gold", config.btc_sell_percentile
        previous_positions.append(previous)
        positions.append(position)
        switches.append(previous != position and position is not None)
        thresholds.append(threshold)
    out["previous_position"] = previous_positions
    out["position"] = positions
    out["trading_switch"] = switches
    out["trading_threshold"] = thresholds
    out = _apply_confirmation_health(
        out,
        "position",
        {"BTC": "btc_structural_condition", "Gold": "gold_structural_condition"},
        config.btc_confirmation_weeks,
    )
    gold_confirmed = out["Gold_sustained_event"].cummax()
    btc_confirmed = out["BTC_sustained_event"].cummax()
    out["recovery_phase"] = "None / Transition"
    out.loc[gold_confirmed & (out["ratio"] < out["sma50m"]) & ((out["roc13w"] > 0) | (out["roc26w"] > 0)), "recovery_phase"] = "BTC Counter-Trend Rally"
    out.loc[gold_confirmed & (out["ratio"] > out["sma50m"]) & ((out["roc13w"] > 0) | (out["roc26w"] > 0)) & ~out["BTC_sustained_active"], "recovery_phase"] = "BTC Potential Recovery"
    out.loc[btc_confirmed & (out["ratio"] > out["sma50m"]) & ((out["roc13w"] < 0) | (out["roc26w"] < 0)), "recovery_phase"] = "Gold Counter-Trend Rally"
    out.loc[btc_confirmed & (out["ratio"] < out["sma50m"]) & ((out["roc13w"] < 0) | (out["roc26w"] < 0)) & ~out["Gold_sustained_active"], "recovery_phase"] = "Gold Potential Recovery"
    halvings = [halving_cycle_position(date) for date in out["Date"]]
    out["halving_phase"] = [item["phase"] for item in halvings]
    out["halving_progress"] = [item["progress"] for item in halvings]
    out["previous_halving"] = [item["previous_halving"] for item in halvings]
    out["next_halving"] = [item["next_halving"] for item in halvings]
    return out


def _asof_row(frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    available = frame.loc[pd.to_datetime(frame["Date"]) <= pd.Timestamp(date)]
    return None if available.empty else available.iloc[-1]


def build_signal_history(spx_gold: pd.DataFrame, btc_gold: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_priority: str | None = None
    events: list[tuple[pd.Timestamp, str, pd.Series]] = []
    for layer, frame in (("SPX / Gold", spx_gold), ("BTC / Gold", btc_gold)):
        if frame is None or frame.empty:
            continue
        for _, row in frame.loc[frame["trading_switch"]].iterrows():
            events.append((pd.Timestamp(row["Date"]), layer, row))
    for date, layer, row in sorted(events, key=lambda item: (item[0], item[1])):
        spx_row = _asof_row(spx_gold, date)
        btc_row = _asof_row(btc_gold, date)
        ranking, priority = priority_ranking(
            None if spx_row is None else spx_row.get("position"),
            None if btc_row is None else btc_row.get("position"),
        )
        current_priority = priority or previous_priority
        lag = np.nan
        initial = row.get("initial_confirmation_date")
        sustained = row.get("sustained_confirmation_date")
        if pd.notna(initial) and pd.notna(sustained):
            days = (pd.Timestamp(sustained) - pd.Timestamp(initial)).days
            lag = round(days / (7.0 if layer == "BTC / Gold" else 30.4375), 1)
        rows.append({
            "Signal Date": date,
            "Layer": layer,
            "Previous Position": row.get("previous_position") or "Uninitialized",
            "New Position": row.get("position"),
            "Ratio": row.get("close") if layer == "SPX / Gold" else row.get("ratio"),
            "Trading Threshold": row.get("trading_threshold"),
            "Structural Status": row.get("confirmation_health"),
            "Previous Episode Outcome": row.get("previous_episode_outcome"),
            "Initial Confirmation": initial,
            "Sustained Confirmation": sustained,
            "Confirmation Lag": lag,
            "Final Priority Ranking": ranking,
            "Final Priority Asset": current_priority,
            "Priority Asset Changed": bool(priority and priority != previous_priority),
        })
        if priority:
            previous_priority = priority
    return pd.DataFrame(rows)


def build_state_history(spx_gold: pd.DataFrame, btc_gold: pd.DataFrame) -> pd.DataFrame:
    """Create a shared as-of history without allowing a future monthly state into a weekly row."""
    if spx_gold is None or spx_gold.empty or btc_gold is None or btc_gold.empty:
        return pd.DataFrame()
    dates = pd.DataFrame({
        "date": pd.DatetimeIndex(
            sorted(set(pd.to_datetime(spx_gold["Date"]).tolist()) | set(pd.to_datetime(btc_gold["Date"]).tolist()))
        )
    })
    spx = spx_gold.rename(columns={
        "Date": "date",
        "close": "spx_gold_ratio",
        "sma50": "spx_gold_sma50",
        "upper": "spx_gold_upper",
        "lower": "spx_gold_lower",
        "position": "spx_gold_position",
        "sma100": "spx_gold_sma100",
        "sma200": "spx_gold_sma200",
        "hh": "spx_gold_hh",
        "hl": "spx_gold_hl",
        "lh": "spx_gold_lh",
        "ll": "spx_gold_ll",
        "confirmation_health": "spx_gold_confirmation_health",
        "recovery_phase": "spx_gold_recovery_phase",
    }).copy()
    spx["spx_gold_structural_condition"] = np.where(
        spx["spx_gold_position"].eq("SPX"),
        spx["spx_structural_condition"],
        spx["gold_structural_condition"],
    )
    spx_columns = [
        "date", "spx_gold_ratio", "spx_gold_sma50", "spx_gold_upper", "spx_gold_lower",
        "spx_gold_position", "spx_gold_sma100", "spx_gold_sma200", "spx_gold_hh",
        "spx_gold_hl", "spx_gold_lh", "spx_gold_ll", "spx_gold_structural_condition",
        "spx_gold_confirmation_health", "spx_gold_recovery_phase",
    ]
    btc = btc_gold.rename(columns={
        "Date": "date",
        "ratio": "btc_gold_ratio",
        "regression_trend": "btc_gold_regression_trend",
        "channel_percentile": "btc_gold_percentile",
        "position": "btc_gold_position",
        "sma50m": "btc_gold_sma50m",
        "roc13w": "btc_gold_roc13w",
        "roc26w": "btc_gold_roc26w",
        "confirmation_health": "btc_gold_confirmation_health",
        "recovery_phase": "btc_gold_recovery_phase",
        "halving_phase": "btc_halving_phase",
        "halving_progress": "btc_halving_progress",
    }).copy()
    btc["btc_gold_structural_condition"] = np.where(
        btc["btc_gold_position"].eq("BTC"),
        btc["btc_structural_condition"],
        btc["gold_structural_condition"],
    )
    btc_columns = [
        "date", "btc_gold_ratio", "btc_gold_regression_trend", "btc_gold_percentile",
        "btc_gold_position", "btc_gold_sma50m", "btc_gold_roc13w", "btc_gold_roc26w",
        "btc_gold_structural_condition", "btc_gold_confirmation_health",
        "btc_gold_recovery_phase", "btc_halving_phase", "btc_halving_progress",
    ]
    combined = pd.merge_asof(dates, spx[spx_columns].sort_values("date"), on="date", direction="backward")
    combined = pd.merge_asof(combined, btc[btc_columns].sort_values("date"), on="date", direction="backward")
    rankings = [priority_ranking(spx_position, btc_position) for spx_position, btc_position in zip(combined["spx_gold_position"], combined["btc_gold_position"])]
    combined["final_priority_ranking"] = [item[0] for item in rankings]
    combined["final_priority_asset"] = [item[1] for item in rankings]
    return combined


def build_portfolio_snapshot(
    spx_monthly_ratio: pd.DataFrame,
    btc_weekly_ratio: pd.DataFrame,
    config: PortfolioConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    spx_gold = build_spx_gold_history(spx_monthly_ratio, config)
    btc_gold = build_btc_gold_history(btc_weekly_ratio, config)
    if spx_gold.empty or btc_gold.empty:
        return {
            "spx_gold": spx_gold,
            "btc_gold": btc_gold,
            "state_history": pd.DataFrame(),
            "signals": pd.DataFrame(),
            "current": {},
        }
    spx_current = spx_gold.iloc[-1]
    btc_current = btc_gold.iloc[-1]
    ranking, priority = priority_ranking(spx_current.get("position"), btc_current.get("position"))
    current = {
        "ranking": ranking,
        "priority_asset": priority,
        "spx_gold": spx_current.to_dict(),
        "btc_gold": btc_current.to_dict(),
        "data_complete": bool(ranking),
        "config": config,
    }
    return {
        "spx_gold": spx_gold,
        "btc_gold": btc_gold,
        "state_history": build_state_history(spx_gold, btc_gold),
        "signals": build_signal_history(spx_gold, btc_gold),
        "current": current,
    }


def reconcile_btc_gold_signals(
    history: pd.DataFrame,
    checkpoints: tuple[tuple[str, str], ...] = (
        ("2018-11-12", "BTC"),
        ("2021-02-15", "Gold"),
        ("2022-06-13", "BTC"),
        ("2024-11-18", "Gold"),
        ("2026-01-19", "BTC"),
    ),
) -> pd.DataFrame:
    signals = history.loc[history.get("trading_switch", False)].copy() if not history.empty else pd.DataFrame()
    rows = []
    for week_start, expected in checkpoints:
        start = pd.Timestamp(week_start)
        end = start + pd.Timedelta(days=6)
        matched = signals.loc[pd.to_datetime(signals.get("Date"), errors="coerce").between(start, end)] if not signals.empty else pd.DataFrame()
        actual = None if matched.empty else str(matched.iloc[-1].get("position"))
        candidates = signals.loc[signals.get("position").astype(str).eq(expected)].copy() if not signals.empty else pd.DataFrame()
        if candidates.empty:
            nearest_date = pd.NaT
            offset = np.nan
        else:
            candidates["distance"] = (pd.to_datetime(candidates["Date"]) - start).abs()
            nearest = candidates.sort_values("distance").iloc[0]
            nearest_date = pd.Timestamp(nearest["Date"])
            offset = int((nearest_date - start).days)
        rows.append({
            "Week": start,
            "Expected Signal": expected,
            "Actual Signal": actual,
            "Matched": actual == expected,
            "Nearest Same-Direction Signal": nearest_date,
            "Day Offset": offset,
        })
    return pd.DataFrame(rows)
