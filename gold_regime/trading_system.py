from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


MODEL_START = pd.Period("1998-01", freq="M")
MIN_REGRESSION_MONTHS = 120
INITIAL_CAPITAL_USD = 1_000.0
MODE_INSTRUMENTS = {
    "1x": {"GOLD": "GOLD", "SILVER": "SILVER", "PLATINUM": "PLATINUM", "CASH": "CASH"},
    "3x": {"GOLD": "3GOL.L", "SILVER": "3SIL.L", "PLATINUM": "3SIL.L", "CASH": "CASH"},
}
OHLC_COLUMNS = ("open", "high", "low", "close")


def normalize_ohlcv(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize a market frame to date-indexed, lower-case OHLC columns."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=OHLC_COLUMNS, index=pd.DatetimeIndex([], name="date"))
    values = frame.copy()
    if isinstance(values.columns, pd.MultiIndex):
        return pd.DataFrame(columns=OHLC_COLUMNS, index=pd.DatetimeIndex([], name="date"))
    date_column = next((column for column in values.columns if str(column).lower() in {"date", "datetime", "timestamp"}), None)
    if date_column is not None:
        values = values.set_index(date_column)
    try:
        index = pd.to_datetime(values.index, errors="coerce", utc=True).tz_localize(None).normalize()
    except (AttributeError, TypeError):
        index = pd.DatetimeIndex(pd.to_datetime(values.index, errors="coerce"))
        if index.tz is not None:
            index = index.tz_convert("UTC").tz_localize(None)
        index = index.normalize()
    values.index = index
    values.index.name = "date"
    by_lower = {str(column).lower(): column for column in values.columns}
    out = pd.DataFrame(index=values.index)
    for column in OHLC_COLUMNS:
        source = by_lower.get(column)
        out[column] = pd.to_numeric(values[source], errors="coerce") if source is not None else np.nan
    out = out.loc[~out.index.isna()].sort_index()
    return out[~out.index.duplicated(keep="last")]


def ratio_close_series(
    numerator: pd.DataFrame | pd.Series,
    denominator: pd.DataFrame | pd.Series,
    name: str = "ratio",
) -> pd.Series:
    """Compute a ratio only on exact shared observation dates; never forward-fill."""
    numerator_close = _close_series(numerator)
    denominator_close = _close_series(denominator)
    paired = pd.concat(
        [numerator_close.rename("numerator"), denominator_close.rename("denominator")],
        axis=1,
        join="inner",
    ).dropna()
    paired = paired.loc[(paired["numerator"] > 0) & (paired["denominator"] > 0)]
    result = (paired["numerator"] / paired["denominator"]).astype(float)
    result.name = name
    return result


def monthly_ratio_series(
    numerator: pd.DataFrame | pd.Series,
    denominator: pd.DataFrame | pd.Series,
    as_of: pd.Timestamp | str | None = None,
    start: pd.Period | str = MODEL_START,
    name: str = "ratio",
) -> pd.Series:
    """Build completed month-end ratios from aligned monthly source closes."""
    numerator_close = _close_series(numerator)
    denominator_close = _close_series(denominator)
    left = _monthly_close_by_period(numerator_close)
    right = _monthly_close_by_period(denominator_close)
    paired = pd.concat([left.rename("numerator"), right.rename("denominator")], axis=1, join="inner").dropna()
    start_period = pd.Period(start, freq="M")
    if as_of is None:
        as_of_period = pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M")
    else:
        as_of_period = pd.Timestamp(as_of).to_period("M")
    paired = paired.loc[(paired.index >= start_period) & (paired.index < as_of_period)]
    paired = paired.loc[(paired["numerator"] > 0) & (paired["denominator"] > 0)]
    values = paired["numerator"] / paired["denominator"]
    values.index = pd.DatetimeIndex([period.end_time.normalize() for period in values.index], name="date")
    values.name = name
    return values.astype(float)


def expanding_regression(
    monthly_ratio: pd.Series,
    min_observations: int = MIN_REGRESSION_MONTHS,
    start: pd.Period | str = MODEL_START,
) -> pd.DataFrame:
    """Calculate a raw-ratio expanding OLS channel, preserving each historical fit."""
    columns = ["month_index", "ratio", "intercept", "slope", "sigma", "mean", "upper_1", "upper_2", "lower_1", "lower_2", "observations"]
    if monthly_ratio is None or monthly_ratio.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))
    values = pd.to_numeric(monthly_ratio, errors="coerce")
    dates = pd.to_datetime(values.index, errors="coerce")
    frame = pd.DataFrame({"date": dates, "ratio": values.to_numpy()}).dropna()
    frame = frame.loc[(frame["ratio"] > 0) & (frame["date"] >= pd.Period(start, freq="M").start_time)]
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if frame.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))
    frame["period"] = frame["date"].dt.to_period("M")
    frame = frame.drop_duplicates("period", keep="last").reset_index(drop=True)
    start_period = pd.Period(start, freq="M")
    x = np.asarray([period.ordinal - start_period.ordinal for period in frame["period"]], dtype=float)
    y = frame["ratio"].to_numpy(dtype=float)
    results: list[dict[str, float]] = []
    result_dates: list[pd.Timestamp] = []
    for last in range(max(0, int(min_observations) - 1), len(frame)):
        x_fit = x[: last + 1]
        y_fit = y[: last + 1]
        design = np.column_stack([np.ones(len(x_fit), dtype=float), x_fit])
        intercept, slope = np.linalg.lstsq(design, y_fit, rcond=None)[0]
        residual = y_fit - (intercept + slope * x_fit)
        observations = len(y_fit)
        sigma = float(np.sqrt(np.sum(np.square(residual)) / (observations - 2)))
        mean = float(intercept + slope * x[last])
        results.append(
            {
                "month_index": float(x[last]),
                "ratio": float(y[last]),
                "intercept": float(intercept),
                "slope": float(slope),
                "sigma": sigma,
                "mean": mean,
                "upper_1": mean + sigma,
                "upper_2": mean + 2.0 * sigma,
                "lower_1": mean - sigma,
                "lower_2": mean - 2.0 * sigma,
                "observations": float(observations),
            }
        )
        result_dates.append(pd.Timestamp(frame.loc[last, "date"]))
    result = pd.DataFrame(results, index=pd.DatetimeIndex(result_dates, name="date"), columns=columns)
    return result


def silver_platinum_deviation(
    silver: pd.DataFrame | pd.Series,
    platinum: pd.DataFrame | pd.Series,
    as_of: pd.Timestamp | str | None = None,
    window: int = 50,
) -> pd.Series:
    """Return completed-month Silver/Platinum deviation from its completed-month SMA."""
    monthly_ratio = monthly_ratio_series(silver, platinum, as_of=as_of, name="sp_ratio")
    if monthly_ratio.empty:
        return pd.Series(dtype=float, name="sp_deviation_pct")
    periods = monthly_ratio.index.to_period("M")
    observed = pd.Series(monthly_ratio.to_numpy(dtype=float), index=periods)
    complete_periods = pd.period_range(observed.index.min(), observed.index.max(), freq="M")
    observed = observed.reindex(complete_periods)
    average = observed.rolling(window=int(window), min_periods=int(window)).mean()
    deviation = 100.0 * (observed / average - 1.0)
    deviation.index = pd.DatetimeIndex([period.end_time.normalize() for period in deviation.index], name="date")
    deviation.name = "sp_deviation_pct"
    return deviation


def build_daily_observations(
    gold_daily: pd.DataFrame,
    silver_daily: pd.DataFrame,
    platinum_daily: pd.DataFrame,
    monthly_model: pd.DataFrame,
    monthly_sp_deviation: pd.Series,
    start: pd.Period | str = MODEL_START,
) -> pd.DataFrame:
    """Join synchronized daily ratio closes to the last model/SMA known on each date."""
    ratio = ratio_close_series(gold_daily, silver_daily, "gold_silver_ratio")
    if ratio.empty:
        return pd.DataFrame()
    left = ratio.rename_axis("date").reset_index().sort_values("date")
    model = monthly_model.reset_index().rename(columns={"date": "model_date"}).sort_values("model_date")
    if model.empty:
        for column in ("month_index", "mean", "intercept", "slope", "sigma", "upper_1", "upper_2", "lower_1", "lower_2", "observations"):
            left[column] = np.nan
    else:
        left = pd.merge_asof(left, model, left_on="date", right_on="model_date", direction="backward")
    sp_dev = monthly_sp_deviation.rename("sp_deviation_pct") if monthly_sp_deviation is not None else pd.Series(dtype=float)
    if not sp_dev.empty:
        sp_frame = sp_dev.rename_axis("sp_date").reset_index().sort_values("sp_date")
        left = pd.merge_asof(left.sort_values("date"), sp_frame, left_on="date", right_on="sp_date", direction="backward")
    else:
        left["sp_deviation_pct"] = np.nan
    start_period = pd.Period(start, freq="M")
    left["month_index"] = left["date"].dt.to_period("M").map(lambda period: float(period.ordinal - start_period.ordinal))
    # Coefficients are frozen at the last completed month, but the fitted line
    # continues along its time coordinate between monthly refits.
    left["mean"] = left["intercept"] + left["slope"] * left["month_index"]
    left["upper_1"] = left["mean"] + left["sigma"]
    left["upper_2"] = left["mean"] + 2.0 * left["sigma"]
    left["lower_1"] = left["mean"] - left["sigma"]
    left["lower_2"] = left["mean"] - 2.0 * left["sigma"]
    left["zscore"] = (left["gold_silver_ratio"] - left["mean"]) / left["sigma"].replace(0, np.nan)
    left["model_ready"] = left[["mean", "sigma", "upper_1", "upper_2", "lower_1", "lower_2"]].notna().all(axis=1) & left["sigma"].gt(0)
    return left.set_index("date").sort_index()


def run_state_machine(
    observations: pd.DataFrame,
    as_of: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process observed daily closes in order, confirming each signal at its week close."""
    history_columns = ["gold_silver_ratio", "mean", "sigma", "upper_1", "upper_2", "lower_1", "lower_2", "zscore", "sp_deviation_pct", "model_ready", "regime"]
    event_columns = ["signal_date", "confirmed_date", "from_regime", "to_regime", "threshold", "ratio", "zscore", "sp_deviation_pct", "sequence_ambiguous", "selection_note"]
    week_columns = ["week", "week_start", "week_confirmed_date", "ratio_high_observed", "ratio_low_observed", "ratio_close", "regime"]
    if observations is None or observations.empty:
        empty_history = pd.DataFrame(columns=history_columns, index=pd.DatetimeIndex([], name="date"))
        return empty_history, pd.DataFrame(columns=event_columns), pd.DataFrame(columns=week_columns)
    data = observations.copy().sort_index()
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data.loc[~data.index.isna()]
    if data.empty:
        return pd.DataFrame(columns=history_columns), pd.DataFrame(columns=event_columns), pd.DataFrame(columns=week_columns)
    as_of_date = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    current_week = as_of_date.to_period("W-FRI")
    data["week"] = data.index.to_period("W-FRI")
    data = data.loc[data["week"] < current_week]
    if data.empty:
        return pd.DataFrame(columns=history_columns), pd.DataFrame(columns=event_columns), pd.DataFrame(columns=week_columns)

    regime: str | None = None
    history_rows: list[dict[str, Any]] = []
    history_dates: list[pd.Timestamp] = []
    event_rows: list[dict[str, Any]] = []
    week_rows: list[dict[str, Any]] = []
    for week, group in data.groupby("week", sort=True):
        confirmed_date = pd.Timestamp(group.index[-1])
        for signal_date, row in group.iterrows():
            ready = bool(row.get("model_ready", False)) and _finite(row.get("upper_1")) and _finite(row.get("lower_2"))
            if not ready:
                history_dates.append(pd.Timestamp(signal_date))
                history_rows.append(_history_row(row, "NOT_READY"))
                continue
            if regime is None or regime == "NOT_READY":
                regime = "CASH"
            ratio = float(row["gold_silver_ratio"])
            zscore = _safe_float(row.get("zscore"))
            lower_2, lower_1 = float(row["lower_2"]), float(row["lower_1"])
            upper_1, upper_2 = float(row["upper_1"]), float(row["upper_2"])
            transitions: list[tuple[str, str, str, bool, str]] = []

            if regime == "CASH" and ratio >= upper_1:
                transitions.append(("CASH", "GOLD", "+1σ", False, ""))
                regime = "GOLD"
                # A daily close may jump across both upper levels. Apply the
                # ordered state transitions once, flagging the unresolved path.
                if ratio >= upper_2:
                    selected, note = _select_late_metal(row.get("sp_deviation_pct"))
                    transitions.append(("GOLD", selected, "+2σ", True, note))
                    regime = selected
            elif regime == "GOLD":
                if ratio <= lower_2:
                    transitions.append(("GOLD", "CASH", "−2σ", False, ""))
                    regime = "CASH"
                elif ratio >= upper_2:
                    selected, note = _select_late_metal(row.get("sp_deviation_pct"))
                    transitions.append(("GOLD", selected, "+2σ", False, note))
                    regime = selected
            elif regime in {"SILVER", "PLATINUM"}:
                if ratio <= lower_2:
                    transitions.append((regime, "CASH", "−2σ", False, ""))
                    regime = "CASH"
                elif ratio <= lower_1:
                    transitions.append((regime, "GOLD", "−1σ", False, ""))
                    regime = "GOLD"
                elif regime == "PLATINUM" and _finite(row.get("sp_deviation_pct")) and float(row["sp_deviation_pct"]) < 5.0:
                    transitions.append(("PLATINUM", "SILVER", "SP deviation < +5%", False, ""))
                    regime = "SILVER"

            for from_regime, to_regime, threshold, ambiguous, note in transitions:
                event_rows.append(
                    {
                        "signal_date": pd.Timestamp(signal_date),
                        "confirmed_date": confirmed_date,
                        "from_regime": from_regime,
                        "to_regime": to_regime,
                        "threshold": threshold,
                        "ratio": ratio,
                        "zscore": zscore,
                        "sp_deviation_pct": _safe_float(row.get("sp_deviation_pct")),
                        "sequence_ambiguous": ambiguous,
                        "selection_note": note,
                    }
                )
            history_dates.append(pd.Timestamp(signal_date))
            history_rows.append(_history_row(row, regime))

        ready_group = group.loc[group["model_ready"].fillna(False).astype(bool)]
        if not ready_group.empty:
            closes = pd.to_numeric(group["gold_silver_ratio"], errors="coerce").dropna()
            week_rows.append(
                {
                    "week": week,
                    "week_start": pd.Timestamp(group.index[0]),
                    "week_confirmed_date": confirmed_date,
                    "ratio_high_observed": float(closes.max()) if not closes.empty else np.nan,
                    "ratio_low_observed": float(closes.min()) if not closes.empty else np.nan,
                    "ratio_close": float(closes.iloc[-1]) if not closes.empty else np.nan,
                    "regime": regime or "CASH",
                }
            )
    history = pd.DataFrame(history_rows, index=pd.DatetimeIndex(history_dates, name="date"), columns=history_columns)
    events = pd.DataFrame(event_rows, columns=event_columns)
    weeks = pd.DataFrame(week_rows, columns=week_columns)
    return history, events, weeks


def instrument_for_regime(regime: str, mode: str) -> str:
    if mode not in MODE_INSTRUMENTS:
        raise ValueError(f"Unknown trading mode: {mode}")
    return MODE_INSTRUMENTS[mode].get(str(regime).upper(), "CASH")


def prepare_usd_ohlcv(
    frame: pd.DataFrame,
    currency: str,
    fx_close: pd.Series | None = None,
    fx_open: pd.Series | None = None,
) -> pd.DataFrame:
    """Convert adjusted OHLC to USD without filling missing asset or FX observations."""
    values = normalize_ohlcv(frame)
    source_currency = str(currency or "").strip().upper()
    if source_currency in {"USD", "US DOLLAR", "US DOLLARS"}:
        return values
    # Yahoo may report LSE pence as GBp (case-sensitive), while some endpoints
    # normalize it to GBX. GBP denotes pounds and must not receive this factor.
    pence_quoted = str(currency).strip() in {"GBp", "GBX", "GBpence"}
    if fx_close is None or fx_open is None:
        raise ValueError(f"USD conversion requires daily FX Open/Close for {currency}.")
    close_fx = _align_series(fx_close, values.index)
    open_fx = _align_series(fx_open, values.index)
    factor = 0.01 if pence_quoted else 1.0
    values["open"] = values["open"] * factor * open_fx
    values["high"] = values["high"] * factor * close_fx
    values["low"] = values["low"] * factor * close_fx
    values["close"] = values["close"] * factor * close_fx
    return values.dropna(subset=["open", "close"])


def run_backtest(
    prices: Mapping[str, pd.DataFrame],
    signal_history: pd.DataFrame,
    signal_events: pd.DataFrame,
    mode: str,
    as_of: pd.Timestamp | str | None = None,
    initial_capital: float = INITIAL_CAPITAL_USD,
) -> dict[str, Any]:
    """Run a long-only, fully invested open-execution backtest in USD."""
    if mode not in MODE_INSTRUMENTS:
        raise ValueError(f"Unknown trading mode: {mode}")
    required = ["GOLD", "SILVER", "PLATINUM"] if mode == "1x" else ["GOLD", "3GOL.L", "3SIL.L"]
    normalized = {ticker: normalize_ohlcv(prices.get(ticker)) for ticker in required}
    if any(frame.empty for frame in normalized.values()):
        return _empty_backtest(mode, "Required adjusted OHLC data is unavailable.")
    price_frame = _align_price_frames(normalized)
    if price_frame.empty:
        return _empty_backtest(mode, "No common dates across the required instruments; missing prices were not filled.")
    as_of_date = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    price_frame = price_frame.loc[
        (price_frame.index.to_period("W-FRI") < as_of_date.to_period("W-FRI"))
        & (price_frame.index <= as_of_date)
    ]
    if price_frame.empty:
        return _empty_backtest(mode, "No fully completed common trading weeks are available for backtesting.")
    if signal_history is None or signal_history.empty or "model_ready" not in signal_history:
        return _empty_backtest(mode, "Expanding regression has not reached its 120-month warm-up.")
    ready_dates = pd.to_datetime(signal_history.index[signal_history["model_ready"].fillna(False).astype(bool)], errors="coerce")
    ready_dates = ready_dates[~pd.isna(ready_dates)]
    if len(ready_dates) == 0:
        return _empty_backtest(mode, "Expanding regression has not reached its 120-month warm-up.")
    first_ready = pd.Timestamp(ready_dates.min())
    # The first capital point is the first common weekly open strictly after
    # the model's 120th completed month; do not start mid-week on a signal date.
    price_frame["week"] = price_frame.index.to_period("W-FRI")
    first_ready_week = first_ready.to_period("W-FRI")
    eligible = price_frame.index[(price_frame.index >= first_ready) & (price_frame["week"] > first_ready_week)]
    if eligible.empty:
        return _empty_backtest(mode, "No common instrument prices are available after regression warm-up.")
    start_date = pd.Timestamp(eligible[0])
    data = price_frame.loc[price_frame.index >= start_date].copy()
    data["week"] = data.index.to_period("W-FRI")
    weekly_open_dates = set(pd.to_datetime(data.groupby("week", sort=True).head(1).index))

    events = signal_events.copy() if signal_events is not None else pd.DataFrame()
    if not events.empty:
        events["confirmed_date"] = pd.to_datetime(events["confirmed_date"], errors="coerce")
        events = events.dropna(subset=["confirmed_date"]).sort_values(["confirmed_date", "signal_date"])
    schedule: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for week_date, group in events.groupby("confirmed_date", sort=True) if not events.empty else []:
        confirmed_date = pd.Timestamp(week_date)
        week = confirmed_date.to_period("W-FRI")
        if confirmed_date < start_date:
            continue
        candidate_opens = [value for value in weekly_open_dates if value.to_period("W-FRI") > week]
        if candidate_opens:
            schedule[min(candidate_opens)] = group.to_dict("records")

    prior_history = signal_history.loc[pd.to_datetime(signal_history.index) < start_date] if not signal_history.empty else pd.DataFrame()
    initial_regime = "CASH"
    if not prior_history.empty:
        latest_state = str(prior_history.iloc[-1].get("regime", "CASH"))
        if latest_state in {"GOLD", "SILVER", "PLATINUM", "CASH"}:
            initial_regime = latest_state

    regime = initial_regime
    instrument = instrument_for_regime(regime, mode)
    previous_close: dict[str, float] | None = None
    equity = float(initial_capital)
    gold_equity = float(initial_capital)
    rows: list[dict[str, Any]] = []
    trades = 0
    executed_events: list[dict[str, Any]] = []
    peak_strategy = float(initial_capital)
    peak_gold = float(initial_capital)

    for date_value, row in data.iterrows():
        date_value = pd.Timestamp(date_value)
        if previous_close is None:
            if date_value in schedule:
                scheduled_events = schedule[date_value]
                new_regime = str(scheduled_events[-1]["to_regime"])
                new_instrument = instrument_for_regime(new_regime, mode)
                if new_instrument != instrument:
                    trades += 1
                for event in scheduled_events:
                    executed_events.append(
                        {
                            "execution_date": date_value,
                            "from_regime": event.get("from_regime", regime),
                            "to_regime": event.get("to_regime", new_regime),
                            "instrument": instrument_for_regime(str(event.get("to_regime", new_regime)), mode),
                            "signal_date": event.get("signal_date", pd.NaT),
                            "confirmed_date": event.get("confirmed_date", pd.NaT),
                            "signal_events": [event],
                        }
                    )
                regime, instrument = new_regime, new_instrument
            rows.append(
                {
                    "date": date_value,
                    "equity": equity,
                    "gold_equity": gold_equity,
                    "regime": regime,
                    "instrument": instrument,
                    "phase": "Open",
                    "strategy_return": 0.0,
                    "gold_return": 0.0,
                    "drawdown": 0.0,
                    "gold_drawdown": 0.0,
                }
            )
        else:
            old_instrument = instrument
            if old_instrument != "CASH":
                equity *= float(row[f"{old_instrument}_open"]) / previous_close[old_instrument]
            gold_equity *= float(row["GOLD_open"]) / previous_close["GOLD"]
            if date_value in schedule:
                scheduled_events = schedule[date_value]
                new_regime = str(scheduled_events[-1]["to_regime"])
                new_instrument = instrument_for_regime(new_regime, mode)
                if new_instrument != instrument:
                    trades += 1
                for event in scheduled_events:
                    executed_events.append(
                        {
                            "execution_date": date_value,
                            "from_regime": event.get("from_regime", regime),
                            "to_regime": event.get("to_regime", new_regime),
                            "instrument": instrument_for_regime(str(event.get("to_regime", new_regime)), mode),
                            "signal_date": event.get("signal_date", pd.NaT),
                            "confirmed_date": event.get("confirmed_date", pd.NaT),
                            "signal_events": [event],
                        }
                    )
                regime, instrument = new_regime, new_instrument

        if instrument != "CASH":
            equity *= float(row[f"{instrument}_close"]) / float(row[f"{instrument}_open"])
        gold_equity *= float(row["GOLD_close"]) / float(row["GOLD_open"])
        previous_close = {key: float(row[f"{key}_close"]) for key in required}
        peak_strategy = max(peak_strategy, equity)
        peak_gold = max(peak_gold, gold_equity)
        rows.append(
            {
                "date": date_value,
                "equity": equity,
                "gold_equity": gold_equity,
                "regime": regime,
                "instrument": instrument,
                "phase": "Close",
                "strategy_return": np.nan,
                "gold_return": np.nan,
                "drawdown": 100.0 * (equity / peak_strategy - 1.0),
                "gold_drawdown": 100.0 * (gold_equity / peak_gold - 1.0),
            }
        )
    curve = pd.DataFrame(rows)
    if curve.empty:
        return _empty_backtest(mode, "No backtest observations are available.")
    curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
    close_curve = curve.loc[curve["phase"].eq("Close")].copy()
    if close_curve.empty:
        return _empty_backtest(mode, "No closing valuations are available.")
    strategy_return = float(equity / initial_capital - 1.0)
    gold_return = float(gold_equity / initial_capital - 1.0)
    end_date = pd.Timestamp(close_curve["date"].iloc[-1])
    elapsed_days = max(1, int((end_date - start_date).days))
    cagr = float((equity / initial_capital) ** (365.25 / elapsed_days) - 1.0) if equity > 0 else np.nan
    gold_cagr = float((gold_equity / initial_capital) ** (365.25 / elapsed_days) - 1.0) if gold_equity > 0 else np.nan
    strategy_mdd = float(close_curve["drawdown"].min())
    gold_mdd = float(close_curve["gold_drawdown"].min())
    calmar = cagr / abs(strategy_mdd / 100.0) if np.isfinite(cagr) and strategy_mdd < 0 else np.nan
    sessions = int(len(close_curve))
    time_in_state = {
        state: (100.0 * sum(1 for value in close_curve["regime"] if value == state) / sessions if sessions else np.nan)
        for state in MODE_INSTRUMENTS[mode]
    }
    annual = annual_performance(curve, start_date, as_of=as_of_date)
    return {
        "mode": mode,
        "status": "READY",
        "message": "",
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": float(initial_capital),
        "final_capital": float(equity),
        "total_return_pct": strategy_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "max_drawdown_pct": strategy_mdd,
        "calmar_ratio": calmar,
        "number_of_trades": trades,
        "time_in_state_pct": time_in_state,
        "gold_cagr_pct": gold_cagr * 100.0,
        "gold_max_drawdown_pct": gold_mdd,
        "curve": curve,
        "annual": annual,
        "executed_events": pd.DataFrame(executed_events),
        "latest_regime": str(close_curve["regime"].iloc[-1]),
        "latest_instrument": str(close_curve["instrument"].iloc[-1]),
        "latest_data_date": end_date,
    }


def annual_performance(
    curve: pd.DataFrame,
    start_date: pd.Timestamp | str,
    as_of: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    columns = ["Year", "Strategy Return", "Strategy MDD", "Gold Return", "Gold MDD", "Excess Return"]
    if curve is None or curve.empty:
        return pd.DataFrame(columns=columns)
    values = curve.copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce")
    values["phase_order"] = values["phase"].map({"Open": 0, "Close": 1}).fillna(2)
    values = values.loc[values["date"].notna()].sort_values(["date", "phase_order"])
    close_values = values.loc[values["phase"].eq("Close")].copy()
    if close_values.empty:
        return pd.DataFrame(columns=columns)
    initial = float(values.iloc[0]["equity"])
    initial_gold = float(values.iloc[0]["gold_equity"])
    start_year = pd.Timestamp(start_date).year
    start_value = pd.Timestamp(start_date)
    evaluation_date = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    rows: list[dict[str, Any]] = []
    previous_equity, previous_gold = initial, initial_gold
    for year, group in close_values.groupby(close_values["date"].dt.year, sort=True):
        strategy_open = previous_equity
        gold_open = previous_gold
        strategy_values = pd.concat([pd.Series([strategy_open]), group["equity"].reset_index(drop=True)], ignore_index=True)
        gold_values = pd.concat([pd.Series([gold_open]), group["gold_equity"].reset_index(drop=True)], ignore_index=True)
        strategy_peaks = strategy_values.cummax()
        gold_peaks = gold_values.cummax()
        strategy_dd = (strategy_values / strategy_peaks - 1.0) * 100.0
        gold_dd = (gold_values / gold_peaks - 1.0) * 100.0
        strategy_return = float(group["equity"].iloc[-1] / strategy_open - 1.0) * 100.0 if strategy_open else np.nan
        gold_return = float(group["gold_equity"].iloc[-1] / gold_open - 1.0) * 100.0 if gold_open else np.nan
        label = str(year)
        if int(year) == evaluation_date.year or (int(year) == start_year and start_value > pd.Timestamp(year=int(year), month=1, day=1)):
            label += " YTD"
        rows.append(
            {
                "Year": label,
                "Strategy Return": strategy_return,
                "Strategy MDD": float(strategy_dd.min()),
                "Gold Return": gold_return,
                "Gold MDD": float(gold_dd.min()),
                "Excess Return": strategy_return - gold_return,
            }
        )
        previous_equity = float(group["equity"].iloc[-1])
        previous_gold = float(group["gold_equity"].iloc[-1])
    return pd.DataFrame(rows, columns=columns)


def _history_row(row: pd.Series, regime: str) -> dict[str, Any]:
    output = {key: row.get(key, np.nan) for key in ("gold_silver_ratio", "mean", "sigma", "upper_1", "upper_2", "lower_1", "lower_2", "zscore", "sp_deviation_pct", "model_ready")}
    output["regime"] = regime
    return output


def _select_late_metal(sp_deviation: Any) -> tuple[str, str]:
    if not _finite(sp_deviation):
        return "SILVER", "SILVER used because the 50-month Silver/Platinum selector is unavailable."
    if float(sp_deviation) >= 30.0:
        return "PLATINUM", ""
    return "SILVER", ""


def _close_series(value: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(value, pd.Series):
        series = pd.to_numeric(value, errors="coerce").copy()
        index = pd.to_datetime(series.index, errors="coerce", utc=True).tz_localize(None).normalize()
        series.index = index
        return series.loc[~series.index.isna()].dropna().sort_index().groupby(level=0).last()
    normalized = normalize_ohlcv(value)
    return normalized["close"].dropna() if not normalized.empty else pd.Series(dtype=float)


def _monthly_close_by_period(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float, index=pd.PeriodIndex([], freq="M"))
    values = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    periods = pd.PeriodIndex(values.index, freq="M")
    frame = pd.DataFrame({"period": periods, "value": values.to_numpy()}).dropna()
    return frame.drop_duplicates("period", keep="last").set_index("period")["value"].sort_index()


def _align_price_frames(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for ticker, frame in frames.items():
        current = normalize_ohlcv(frame)[["open", "close"]].rename(columns={"open": f"{ticker}_open", "close": f"{ticker}_close"})
        current = current.dropna(subset=[f"{ticker}_open", f"{ticker}_close"])
        merged = current if merged is None else merged.join(current, how="inner")
    if merged is None:
        return pd.DataFrame()
    for column in merged:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged = merged.dropna().sort_index()
    merged.index.name = "date"
    return merged


def _align_series(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").copy()
    dates = pd.to_datetime(values.index, errors="coerce", utc=True).tz_localize(None).normalize()
    values.index = dates
    values = values.loc[~values.index.isna()].groupby(level=0).last()
    return values.reindex(index)


def _empty_backtest(mode: str, message: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "status": "DATA_INCOMPLETE",
        "message": message,
        "start_date": pd.NaT,
        "end_date": pd.NaT,
        "initial_capital": INITIAL_CAPITAL_USD,
        "final_capital": np.nan,
        "total_return_pct": np.nan,
        "cagr_pct": np.nan,
        "max_drawdown_pct": np.nan,
        "calmar_ratio": np.nan,
        "number_of_trades": 0,
        "time_in_state_pct": {},
        "gold_cagr_pct": np.nan,
        "gold_max_drawdown_pct": np.nan,
        "curve": pd.DataFrame(),
        "annual": pd.DataFrame(columns=["Year", "Strategy Return", "Strategy MDD", "Gold Return", "Gold MDD", "Excess Return"]),
        "executed_events": pd.DataFrame(),
        "latest_regime": "DATA_INCOMPLETE",
        "latest_instrument": "N/A",
        "latest_data_date": pd.NaT,
    }


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _finite(value: Any) -> bool:
    return np.isfinite(_safe_float(value))
