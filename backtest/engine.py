from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from backtest.divergence import DivergenceConfig, bear_divergence_signal, bull_divergence_signal

Condition = Literal["=", ">", "<", ">=", "<=", "bear divergence", "bull divergence"]


@dataclass(frozen=True)
class ConditionRow:
    row_id: int
    variable: str
    condition: Condition
    input_value: float | None
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    divergence_price_col: str | None = None


@dataclass(frozen=True)
class StrategyRules:
    enter_ticker: str
    exit_ticker: str
    enter_conditions: list[ConditionRow]
    exit_conditions: list[ConditionRow]


@dataclass(frozen=True)
class BacktestConfig:
    initial_deposit: float
    commission_pct: float = 0.0
    slippage_pct: float = 0.0
    divergence_pivot_window: int = 3
    divergence_lookback: int = 60


OP_MAP = {
    "=": lambda x, y: x == y,
    ">": lambda x, y: x > y,
    "<": lambda x, y: x < y,
    ">=": lambda x, y: x >= y,
    "<=": lambda x, y: x <= y,
}


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    warnings: list[str]


def _validate_inputs(df: pd.DataFrame, rules: StrategyRules, config: BacktestConfig) -> None:
    if df.empty:
        raise ValueError("Input data is empty.")
    if config.initial_deposit <= 0:
        raise ValueError("Initial Deposit must be greater than 0.")
    if config.commission_pct < 0 or config.slippage_pct < 0:
        raise ValueError("Commission and Slippage percentages must be non-negative.")

    missing = {rules.enter_ticker, rules.exit_ticker} - set(df.columns)
    cond_cols = {r.variable for r in rules.enter_conditions + rules.exit_conditions}
    missing |= cond_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")


def _precompute_divergence_signals(
    df: pd.DataFrame,
    conditions: list[ConditionRow],
    ticker_for_price: str,
    config: BacktestConfig,
) -> dict[tuple[str, str, str], pd.Series]:
    out: dict[tuple[str, str, str], pd.Series] = {}
    dcfg = DivergenceConfig(
        pivot_window=config.divergence_pivot_window,
        lookback=config.divergence_lookback,
    )

    for r in conditions:
        if r.condition not in {"bear divergence", "bull divergence"}:
            continue
        price_col = r.divergence_price_col or ticker_for_price
        key = (r.variable, price_col, r.condition)
        if key in out:
            continue
        indicator = df[r.variable]
        price = df[price_col]
        if r.condition == "bull divergence":
            out[key] = bull_divergence_signal(price=price, indicator=indicator, config=dcfg)
        else:
            out[key] = bear_divergence_signal(price=price, indicator=indicator, config=dcfg)
    return out


def _condition_matches(
    row: pd.Series,
    row_date: pd.Timestamp,
    condition_row: ConditionRow,
    ticker_for_divergence_price: str,
    divergence_cache: dict[tuple[str, str, str], pd.Series],
) -> bool:
    if not (condition_row.start_date <= row_date <= condition_row.end_date):
        return False

    if condition_row.condition in {"bull divergence", "bear divergence"}:
        price_col = condition_row.divergence_price_col or ticker_for_divergence_price
        key = (condition_row.variable, price_col, condition_row.condition)
        sig = divergence_cache[key]
        return bool(sig.loc[row_date])

    val = row[condition_row.variable]
    if pd.isna(val) or condition_row.input_value is None:
        return False
    return bool(OP_MAP[condition_row.condition](val, condition_row.input_value))


def _all_conditions_match(
    row: pd.Series,
    row_date: pd.Timestamp,
    conditions: list[ConditionRow],
    ticker_for_divergence_price: str,
    divergence_cache: dict[tuple[str, str, str], pd.Series],
) -> bool:
    if not conditions:
        return False
    return all(
        _condition_matches(
            row=row,
            row_date=row_date,
            condition_row=cond,
            ticker_for_divergence_price=ticker_for_divergence_price,
            divergence_cache=divergence_cache,
        )
        for cond in conditions
    )


def _condition_text(cond: ConditionRow) -> str:
    if cond.condition in {"bull divergence", "bear divergence"}:
        return f"r{cond.row_id}: {cond.variable} {cond.condition}"
    if cond.input_value is None:
        return f"r{cond.row_id}: {cond.variable} {cond.condition}"
    return f"r{cond.row_id}: {cond.variable} {cond.condition} {cond.input_value:.2f}"


def _build_reason(prefix: str, conditions: list[ConditionRow]) -> str:
    details = "; ".join(_condition_text(c) for c in sorted(conditions, key=lambda x: x.row_id))
    return f"{prefix} ({details})"


def run_backtest(
    df: pd.DataFrame,
    strategy_rules: StrategyRules,
    config: BacktestConfig,
    global_start: pd.Timestamp | None = None,
    global_end: pd.Timestamp | None = None,
) -> BacktestResult:
    """
    Deterministic semantics:
    - Enter Strategy signal on bar t when ALL enter conditions are true
    - Exit Strategy signal on bar t when ALL exit conditions are true
    - Signals execute on t+1 close (no look-ahead)
    - Long-only, single holding at a time
    """
    if df.index.name is None or not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex before running backtest.")

    _validate_inputs(df, strategy_rules, config)

    start = global_start or df.index.min()
    end = global_end or df.index.max()
    if start > end:
        raise ValueError("Global start date must be <= global end date.")

    window = df.loc[(df.index >= start) & (df.index <= end)].copy()
    if len(window) < 2:
        raise ValueError("Backtest window must include at least 2 rows for next-bar execution.")

    enter_div = _precompute_divergence_signals(
        window,
        strategy_rules.enter_conditions,
        strategy_rules.enter_ticker,
        config,
    )
    exit_div = _precompute_divergence_signals(
        window,
        strategy_rules.exit_conditions,
        strategy_rules.exit_ticker,
        config,
    )

    cash = float(config.initial_deposit)
    shares = 0.0
    holding_ticker: str | None = None

    pending_action: tuple[str, pd.Timestamp, str] | None = None
    warnings: list[str] = []
    trades: list[dict[str, object]] = []
    equity_values: list[float] = []

    for i, dt in enumerate(window.index):
        row = window.loc[dt]

        if pending_action is not None:
            action, signal_dt, signal_reason = pending_action
            pending_action = None

            if action == "Buy":
                target_ticker = strategy_rules.enter_ticker
                if holding_ticker != target_ticker:
                    if holding_ticker is not None and shares > 0:
                        sell_px = window.loc[dt, holding_ticker]
                        if pd.isna(sell_px) or sell_px <= 0:
                            warnings.append(f"Skipped sell on {dt.date()} for {holding_ticker}: invalid close price.")
                        else:
                            fill = float(sell_px) * (1 - config.slippage_pct / 100.0)
                            gross = shares * fill
                            commission = gross * (config.commission_pct / 100.0)
                            cash += gross - commission
                            trades.append(
                                {
                                    "date": dt,
                                    "signal_date": signal_dt,
                                    "action": "Sell",
                                    "ticker": holding_ticker,
                                    "price": fill,
                                    "shares": shares,
                                    "commission": commission,
                                    "reason": f"Pre-buy liquidation: {signal_reason}",
                                }
                            )
                            shares = 0.0
                            holding_ticker = None

                    buy_px = window.loc[dt, target_ticker]
                    if pd.isna(buy_px) or buy_px <= 0:
                        warnings.append(f"Skipped buy on {dt.date()} for {target_ticker}: invalid close price.")
                    elif cash > 0:
                        fill = float(buy_px) * (1 + config.slippage_pct / 100.0)
                        gross_to_invest = cash / (1 + config.commission_pct / 100.0)
                        new_shares = gross_to_invest / fill
                        commission = gross_to_invest * (config.commission_pct / 100.0)
                        cash -= gross_to_invest + commission
                        shares = new_shares
                        holding_ticker = target_ticker
                        trades.append(
                            {
                                "date": dt,
                                "signal_date": signal_dt,
                                "action": "Buy",
                                "ticker": target_ticker,
                                "price": fill,
                                "shares": new_shares,
                                "commission": commission,
                                "reason": signal_reason,
                            }
                        )

            elif action == "Sell":
                target_ticker = strategy_rules.exit_ticker
                if holding_ticker == target_ticker and shares > 0:
                    sell_px = window.loc[dt, holding_ticker]
                    if pd.isna(sell_px) or sell_px <= 0:
                        warnings.append(f"Skipped sell on {dt.date()} for {holding_ticker}: invalid close price.")
                    else:
                        fill = float(sell_px) * (1 - config.slippage_pct / 100.0)
                        gross = shares * fill
                        commission = gross * (config.commission_pct / 100.0)
                        cash += gross - commission
                        trades.append(
                            {
                                "date": dt,
                                "signal_date": signal_dt,
                                "action": "Sell",
                                "ticker": holding_ticker,
                                "price": fill,
                                "shares": shares,
                                "commission": commission,
                                "reason": signal_reason,
                            }
                        )
                        shares = 0.0
                        holding_ticker = None

        if holding_ticker is None or shares <= 0:
            equity = cash
        else:
            px = row[holding_ticker]
            if pd.isna(px) or px <= 0:
                warnings.append(f"Invalid mark-to-market price on {dt.date()} for {holding_ticker}; carrying prior equity.")
                equity = equity_values[-1] if equity_values else cash
            else:
                equity = shares * float(px) + cash
        equity_values.append(float(equity))

        if i == len(window) - 1:
            continue

        if holding_ticker is None:
            enter_now = _all_conditions_match(
                row=row,
                row_date=dt,
                conditions=strategy_rules.enter_conditions,
                ticker_for_divergence_price=strategy_rules.enter_ticker,
                divergence_cache=enter_div,
            )
            if enter_now:
                pending_action = (
                    "Buy",
                    dt,
                    _build_reason("Enter signal: all active enter conditions matched", strategy_rules.enter_conditions),
                )
        else:
            exit_now = _all_conditions_match(
                row=row,
                row_date=dt,
                conditions=strategy_rules.exit_conditions,
                ticker_for_divergence_price=strategy_rules.exit_ticker,
                divergence_cache=exit_div,
            )
            if exit_now:
                pending_action = (
                    "Sell",
                    dt,
                    _build_reason("Exit signal: all active exit conditions matched", strategy_rules.exit_conditions),
                )

    equity_series = pd.Series(equity_values, index=window.index, name="strategy_equity")
    trades_df = pd.DataFrame(trades)

    if pending_action is not None:
        warnings.append("Last signal ignored because there is no next bar for execution.")

    return BacktestResult(equity=equity_series, trades=trades_df, warnings=warnings)
