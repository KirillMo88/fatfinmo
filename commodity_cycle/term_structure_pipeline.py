from __future__ import annotations

import calendar
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from commodity_cycle.term_structure import (
    AGRICULTURE, ENERGY, METALS, MONTH_CODES, TERM_DB_PATH,
    _contract_expiry, _tradingview_quote, contract_symbol, quote_is_fresh,
)


CONTRACT_SPECS = {
    **{asset: {"root": root, "suffix": suffix, "months": months, "horizon_months": 18, "type": "energy"}
       for asset, (root, suffix, months) in ENERGY.items()},
    **{asset: {"root": root, "suffix": suffix, "months": months, "horizon_months": 30, "type": "agriculture"}
       for asset, (root, suffix, months, _, _) in AGRICULTURE.items()},
}
HARD_ROLL_BUSINESS_DAYS = 5
MAX_STALE_TRADING_DAYS = 3


@dataclass(frozen=True)
class ExpectedContract:
    asset: str
    symbol: str
    delivery_month: int
    delivery_year: int
    expiry: pd.Timestamp
    first_notice: pd.Timestamp | None
    exchange: str


class TermStructureContractProvider:
    """Provider-neutral normalized contract-history interface."""

    name = "provider"

    def fetch_contracts(self, contracts: Iterable[ExpectedContract]) -> dict[str, pd.DataFrame]:
        raise NotImplementedError


class YahooFuturesProvider(TermStructureContractProvider):
    name = "Yahoo Finance"

    def fetch_contracts(self, contracts: Iterable[ExpectedContract]) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        contracts = list(contracts)
        symbols = [contract.symbol for contract in contracts]
        if not symbols:
            return {}
        # A batch includes enough daily bars for crossover confirmation and MTD reconstruction.
        raw = yf.download(symbols, period="3mo", interval="1d", auto_adjust=False,
                          actions=False, group_by="ticker", threads=True, progress=False)
        frames: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if symbol in set(raw.columns.get_level_values(0)):
                        block = raw[symbol]
                    elif symbol in set(raw.columns.get_level_values(1)):
                        block = raw.xs(symbol, axis=1, level=1)
                    else:
                        continue
                else:
                    if len(symbols) != 1:
                        continue
                    block = raw
                if "Close" not in block:
                    continue
                out = pd.DataFrame(index=pd.to_datetime(block.index, errors="coerce"))
                out["price"] = pd.to_numeric(block["Close"], errors="coerce")
                out["volume"] = pd.to_numeric(block.get("Volume"), errors="coerce")
                out["open_interest"] = np.nan  # Yahoo historical daily OI is not exposed by yfinance.
                out["source"] = self.name
                out = out.loc[~out.index.isna()].dropna(subset=["price"]).sort_index()
                if not out.empty:
                    frames[symbol] = out
            except Exception:
                continue
        # Current OI is a secondary signal only. Capture it only if the batch lacks usable volume
        # for the first two calendar expiries; never fabricate historical OI observations.
        by_asset: dict[str, list[ExpectedContract]] = {}
        for contract in contracts:
            by_asset.setdefault(contract.asset, []).append(contract)
        for group in by_asset.values():
            first_two = sorted(group, key=lambda c: c.expiry)[:2]
            if len(first_two) < 2:
                continue
            needs_oi = any(
                contract.symbol not in frames
                or pd.to_numeric(frames[contract.symbol]["volume"], errors="coerce").dropna().empty
                for contract in first_two
            )
            if not needs_oi:
                continue
            for contract in first_two:
                try:
                    info = yf.Ticker(contract.symbol).get_info()
                    oi = pd.to_numeric(pd.Series([info.get("openInterest")]), errors="coerce").iloc[0]
                    if pd.notna(oi) and contract.symbol in frames:
                        completed = frames[contract.symbol].index[frames[contract.symbol].index.date < date.today()]
                        if len(completed):
                            frames[contract.symbol].loc[completed[-1], "open_interest"] = float(oi)
                except Exception:
                    continue
        return frames


class TradingViewFuturesProvider(TermStructureContractProvider):
    """Approved MCP fallback; does not scrape TradingView webpages."""

    name = "TradingView MCP"

    def fetch_contracts(self, contracts: Iterable[ExpectedContract]) -> dict[str, pd.DataFrame]:
        frames = {}
        for contract in contracts:
            try:
                quote = _tradingview_quote(contract.symbol)
                if quote is None:
                    continue
                price, observed, source = quote
                frames[contract.symbol] = pd.DataFrame(
                    {"price": [price], "volume": [np.nan], "open_interest": [np.nan], "source": [source]},
                    index=[pd.Timestamp(observed)],
                )
            except Exception:
                continue
        return frames


class WestmetallLMEProvider:
    name = "Westmetall LME Cash/3M"

    def fetch(self, asset: str) -> pd.DataFrame:
        import requests

        field = METALS[asset][0]
        response = requests.get(
            f"https://www.westmetall.com/en/markdaten.php?action=table&field={field}", timeout=20
        )
        response.raise_for_status()
        for table in pd.read_html(response.text):
            table.columns = [str(col).strip() for col in table.columns]
            date_col = next((col for col in table.columns if col.lower() == "date"), None)
            cash_col = next((col for col in table.columns if "cash-settlement" in col.lower()), None)
            three_col = next((col for col in table.columns if "3-month" in col.lower()), None)
            if date_col is None or cash_col is None or three_col is None:
                continue
            result = pd.DataFrame(index=pd.to_datetime(table[date_col], errors="coerce", dayfirst=True))
            for col, source_col in (("leg1_price", cash_col), ("leg2_price", three_col)):
                result[col] = pd.to_numeric(table[source_col].astype(str).str.replace(",", "", regex=False), errors="coerce").to_numpy()
            result["price"] = np.nan
            result["volume"] = np.nan
            result["open_interest"] = np.nan
            result["source"] = self.name
            result = result.loc[~result.index.isna()].dropna(subset=["leg1_price", "leg2_price"]).sort_index()
            return result
        raise ValueError(f"Westmetall daily LME Cash/3M table not found for {asset}")


class TermStructureStore:
    """Persistent contract observations, curve history, diagnostics, and roll state."""

    def __init__(self, path: Path | str = TERM_DB_PATH):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
          CREATE TABLE IF NOT EXISTS contract_observations (
            asset TEXT NOT NULL, symbol TEXT NOT NULL, as_of TEXT NOT NULL,
            delivery_month INTEGER, delivery_year INTEGER, expiry TEXT,
            price REAL, volume REAL, open_interest REAL, source TEXT NOT NULL,
            observed_at TEXT NOT NULL, PRIMARY KEY(symbol, as_of)
          );
          CREATE TABLE IF NOT EXISTS daily_curve_observations (
            asset TEXT NOT NULL, pair_key TEXT NOT NULL, as_of TEXT NOT NULL,
            leg1_symbol TEXT, leg2_symbol TEXT, leg1_month INTEGER, leg2_month INTEGER,
            leg1_price REAL, leg2_price REAL, spread REAL, quality TEXT NOT NULL,
            source TEXT NOT NULL, observed_at TEXT NOT NULL,
            PRIMARY KEY(asset, pair_key, as_of)
          );
          CREATE TABLE IF NOT EXISTS finalized_monthly_curves (
            asset TEXT NOT NULL, month TEXT NOT NULL, pair_key TEXT NOT NULL,
            leg1_symbol TEXT, leg2_symbol TEXT, leg1_month INTEGER, leg2_month INTEGER,
            avg_leg1_price REAL, avg_leg2_price REAL, monthly_spread REAL,
            observation_count INTEGER NOT NULL, source TEXT NOT NULL, finalized_at TEXT NOT NULL,
            PRIMARY KEY(asset, month, pair_key)
          );
          CREATE TABLE IF NOT EXISTS contract_diagnostics (
            refresh_id TEXT NOT NULL, asset TEXT NOT NULL, symbol TEXT NOT NULL,
            payload TEXT NOT NULL, PRIMARY KEY(refresh_id, symbol)
          );
          CREATE TABLE IF NOT EXISTS term_roll_state (
            asset TEXT PRIMARY KEY, active_symbol TEXT, previous_symbol TEXT,
            rollover_date TEXT, method TEXT, days_to_expiry INTEGER
          );
        """)
        return conn

    def quote_history(self, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
        symbols = list(symbols)
        if not symbols:
            return {}
        placeholders = ",".join("?" for _ in symbols)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT symbol, as_of, price, volume, open_interest, source FROM contract_observations WHERE symbol IN ({placeholders}) ORDER BY as_of",
                symbols,
            ).fetchall()
        frame = pd.DataFrame([dict(row) for row in rows])
        if frame.empty:
            return {}
        result = {}
        for symbol, part in frame.groupby("symbol"):
            part = part.copy()
            part.index = pd.to_datetime(part.pop("as_of"))
            result[symbol] = part.drop(columns="symbol").sort_index()
        return result

    def save_quotes(self, contracts: Iterable[ExpectedContract], frames: dict[str, pd.DataFrame], observed_at: str) -> None:
        with self._connect() as conn:
            for contract in contracts:
                frame = frames.get(contract.symbol)
                if frame is None or frame.empty:
                    continue
                source_values = frame["source"].dropna() if "source" in frame else pd.Series(dtype=object)
                source = str(source_values.iloc[0]) if not source_values.empty else "Yahoo Finance"
                for dt, row in frame.iterrows():
                    if pd.isna(row.get("price")):
                        continue
                    conn.execute("""INSERT INTO contract_observations
                      (asset,symbol,as_of,delivery_month,delivery_year,expiry,price,volume,open_interest,source,observed_at)
                      VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol,as_of) DO UPDATE SET
                      price=COALESCE(excluded.price,contract_observations.price),
                      volume=COALESCE(excluded.volume,contract_observations.volume),
                      open_interest=COALESCE(excluded.open_interest,contract_observations.open_interest),
                      source=excluded.source, observed_at=excluded.observed_at""", (
                        contract.asset, contract.symbol, pd.Timestamp(dt).date().isoformat(), contract.delivery_month,
                        contract.delivery_year, contract.expiry.date().isoformat(), _number(row.get("price")),
                        _number(row.get("volume")), _number(row.get("open_interest")), source, observed_at,
                    ))

    def save_daily_curves(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        with self._connect() as conn:
            for _, row in frame.iterrows():
                conn.execute("""INSERT OR REPLACE INTO daily_curve_observations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    row["asset"], row["pair_key"], pd.Timestamp(row["as_of"]).date().isoformat(),
                    row.get("leg1_symbol"), row.get("leg2_symbol"), row.get("leg1_month"), row.get("leg2_month"),
                    _number(row.get("leg1_price")), _number(row.get("leg2_price")), _number(row.get("spread")),
                    row.get("quality", "INVALID"), row.get("source", "N/A"), row.get("observed_at", ""),
                ))

    def save_diagnostics(self, frame: pd.DataFrame, refresh_id: str) -> None:
        if frame.empty:
            return
        with self._connect() as conn:
            for _, row in frame.iterrows():
                payload = {key: _json_safe(value) for key, value in row.to_dict().items()}
                conn.execute("INSERT OR REPLACE INTO contract_diagnostics VALUES(?,?,?,?)", (
                    refresh_id, str(row.get("Asset", "")), str(row.get("Yahoo Symbol", "")), json.dumps(payload),
                ))

    def diagnostics(self) -> pd.DataFrame:
        with self._connect() as conn:
            row = conn.execute("SELECT refresh_id FROM contract_diagnostics ORDER BY refresh_id DESC LIMIT 1").fetchone()
            if not row:
                return pd.DataFrame()
            rows = conn.execute("SELECT payload FROM contract_diagnostics WHERE refresh_id=?", (row[0],)).fetchall()
        return pd.DataFrame([json.loads(item[0]) for item in rows])

    def save_roll(self, asset: str, active: str | None, previous: str | None, method: str, as_of: date, days_to_expiry: int | None) -> dict[str, Any]:
        with self._connect() as conn:
            old = conn.execute("SELECT active_symbol,previous_symbol,rollover_date,method,days_to_expiry FROM term_roll_state WHERE asset=?", (asset,)).fetchone()
            old_active = old[0] if old else None
            rolled = bool(active and old_active and active != old_active)
            rollover_date = as_of.isoformat() if rolled or (old_active is None and previous) else (old[2] if old else None)
            rollover_method = method if rolled or not old else old[3]
            previous_symbol = old_active if rolled else (old[1] if old else previous)
            conn.execute("INSERT OR REPLACE INTO term_roll_state VALUES(?,?,?,?,?,?)", (
                asset, active, previous_symbol, rollover_date, rollover_method, days_to_expiry,
            ))
        return {"active_symbol": active, "previous_symbol": previous_symbol, "rollover_date": rollover_date,
                "rollover_method": rollover_method, "days_to_expiry": days_to_expiry}

    def roll_state(self, asset: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT active_symbol,previous_symbol,rollover_date,method,days_to_expiry FROM term_roll_state WHERE asset=?", (asset,)).fetchone()
        if not row:
            return {}
        return dict(zip(("active_symbol", "previous_symbol", "rollover_date", "rollover_method", "days_to_expiry"), row))

    def daily_curves(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM daily_curve_observations ORDER BY as_of").fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def finalize_closed_months(self, current_month: pd.Period, finalized_at: str) -> None:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM daily_curve_observations ORDER BY as_of").fetchall()
            frame = pd.DataFrame([dict(row) for row in rows])
            if frame.empty:
                return
            frame["as_of"] = pd.to_datetime(frame["as_of"], errors="coerce")
            frame["month"] = frame["as_of"].dt.to_period("M").astype(str)
            frame = frame.loc[frame["month"] < str(current_month)]
            for (asset, month, key), part in frame.groupby(["asset", "month", "pair_key"]):
                if conn.execute("SELECT 1 FROM finalized_monthly_curves WHERE asset=? AND month=? AND pair_key=?", (asset, month, key)).fetchone():
                    continue
                valid = part.loc[part["quality"].isin(["HIGH", "LOW_LIQUIDITY"])].copy()
                # Treat a month as a seasonal observation only when it has the same
                # minimum synchronized daily sample required for a current official percentile.
                if len(valid) < 5:
                    continue
                conn.execute("INSERT INTO finalized_monthly_curves VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    asset, month, key, valid.iloc[-1]["leg1_symbol"], valid.iloc[-1]["leg2_symbol"],
                    valid.iloc[-1]["leg1_month"], valid.iloc[-1]["leg2_month"],
                    float(pd.to_numeric(valid["leg1_price"], errors="coerce").mean()),
                    float(pd.to_numeric(valid["leg2_price"], errors="coerce").mean()),
                    float(pd.to_numeric(valid["spread"], errors="coerce").mean()), len(valid),
                    " + ".join(sorted(set(valid["source"].dropna().astype(str)))), finalized_at,
                ))

    def monthly_curves(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM finalized_monthly_curves ORDER BY month").fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def last_finalized_for(self, asset: str, pair_key: str | None = None) -> dict[str, Any]:
        with self._connect() as conn:
            if pair_key:
                row = conn.execute("SELECT * FROM finalized_monthly_curves WHERE asset=? AND pair_key=? ORDER BY month DESC LIMIT 1", (asset, pair_key)).fetchone()
            else:
                row = conn.execute("SELECT * FROM finalized_monthly_curves WHERE asset=? ORDER BY month DESC LIMIT 1", (asset,)).fetchone()
        return dict(row) if row else {}


def _completed_eod_frame(frame: pd.DataFrame, today: date) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out = out.loc[~out.index.isna()]
    out = out.loc[out.index.date < today]
    return out.sort_index().groupby(level=0).last()


def _current_month_daily(rows: list[dict[str, Any]], today: date) -> list[dict[str, Any]]:
    month = pd.Timestamp(today).to_period("M")
    return [row for row in rows if pd.Timestamp(row["as_of"]).to_period("M") == month]


def _after_current_month_roll(rows: list[dict[str, Any]], roll: dict[str, Any], today: date) -> list[dict[str, Any]]:
    """Keep pre-roll daily spreads already captured under the former active contract."""
    rollover_date = roll.get("rollover_date")
    if not rollover_date:
        return rows
    try:
        cutoff = pd.Timestamp(rollover_date)
    except (TypeError, ValueError):
        return rows
    if cutoff.to_period("M") != pd.Timestamp(today).to_period("M"):
        return rows
    return [row for row in rows if pd.Timestamp(row["as_of"]).normalize() >= cutoff.normalize()]


def expected_contracts(asset: str, today: date | None = None) -> list[ExpectedContract]:
    today = today or date.today()
    spec = CONTRACT_SPECS[asset]
    contracts = []
    for offset in range(spec["horizon_months"]):
        period = pd.Timestamp(today.year, today.month, 1) + pd.DateOffset(months=offset)
        month = int(period.month)
        if month not in spec["months"]:
            continue
        expiry = _contract_expiry(asset, int(period.year), month)
        first_notice = _first_notice(int(period.year), month) if spec["type"] == "agriculture" else None
        contracts.append(ExpectedContract(
            asset, contract_symbol(spec["root"], month, int(period.year), spec["suffix"]),
            month, int(period.year), expiry, first_notice, spec["suffix"],
        ))
    return sorted(contracts, key=lambda item: item.expiry)


def _first_notice(year: int, month: int) -> pd.Timestamp:
    # CBOT grain first notice starts on the last business day before delivery month.
    first = pd.Timestamp(year, month, 1) - pd.Timedelta(days=1)
    if np.is_busday(first.date()):
        return first
    return pd.Timestamp(np.busday_offset(first.date(), 0, roll="backward"))


def _days_to_roll_date(contract: ExpectedContract) -> int:
    limit = contract.first_notice if contract.first_notice is not None else contract.expiry
    roll_day = pd.Timestamp(np.busday_offset(limit.date(), -HARD_ROLL_BUSINESS_DAYS, roll="backward"))
    return int(np.busday_count(date.today(), roll_day.date()))


def _expected_contract_frame(asset: str, contracts: list[ExpectedContract], histories: dict[str, pd.DataFrame], today: date,
                             source: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    diagnostics = []
    metrics = {}
    for contract in contracts:
        frame = histories.get(contract.symbol, pd.DataFrame()).copy()
        if not frame.empty:
            frame.index = pd.to_datetime(frame.index, errors="coerce")
            if frame.index.tz is not None:
                frame.index = frame.index.tz_convert("UTC").tz_localize(None)
            frame = frame.loc[~frame.index.isna()].sort_index()
            # A Yahoo bar dated today may still be an incomplete session. Only completed prior dates count.
            frame = frame.loc[frame.index.date < today]
        last_date = pd.Timestamp(frame.index[-1]).normalize() if not frame.empty else pd.NaT
        last = frame.iloc[-1] if not frame.empty else pd.Series(dtype=object)
        price = _number(last.get("price"))
        volume = _number(last.get("volume"))
        open_interest = _number(last.get("open_interest"))
        volume5 = _number(pd.to_numeric(frame.get("volume", pd.Series(dtype=float)), errors="coerce").tail(5).sum(min_count=1))
        if contract.expiry.date() < today:
            status, reason = "EXPIRED", "CONTRACT_EXPIRED"
        elif pd.isna(last_date) or not quote_is_fresh(last_date, today=today, max_business_days=MAX_STALE_TRADING_DAYS):
            status, reason = "STALE", "NO_FRESH_COMPLETED_EOD"
        elif not np.isfinite(price) or price <= 0:
            status, reason = "INVALID", "INVALID_PRICE"
        elif pd.notna(volume) and volume > 0:
            status, reason = "LIQUID", "LATEST_SESSION_VOLUME_POSITIVE"
        elif (pd.notna(volume5) and volume5 > 0) or (pd.notna(open_interest) and open_interest > 0):
            status, reason = "LOW_LIQUIDITY", "NO_LATEST_VOLUME_BUT_RECENT_VOLUME_OR_OI"
        else:
            status, reason = "INVALID", "NO_VOLUME_OR_OPEN_INTEREST"
        metrics[contract.symbol] = {
            "frame": frame, "price": price, "price_date": last_date, "volume": volume,
            "volume5": volume5, "open_interest": open_interest, "status": status,
        }
        diagnostics.append({
            "Asset": asset, "Expected Contract": contract.symbol, "Yahoo Symbol": contract.symbol,
            "Delivery Month": calendar.month_abbr[contract.delivery_month], "Delivery Year": contract.delivery_year,
            "Expiration Date": contract.expiry.date().isoformat(),
            "First Notice Date": contract.first_notice.date().isoformat() if contract.first_notice is not None else None,
            "Latest Price": price, "Price Date": last_date.date().isoformat() if pd.notna(last_date) else None,
            "Volume": volume, "5D Cumulative Volume": volume5, "Open Interest": open_interest,
            "Recent Volume History": json.dumps({pd.Timestamp(dt).date().isoformat(): _number(value) for dt, value in frame.get("volume", pd.Series(dtype=float)).tail(5).items()}),
            "Recent Open Interest History": json.dumps({pd.Timestamp(dt).date().isoformat(): _number(value) for dt, value in frame.get("open_interest", pd.Series(dtype=float)).dropna().tail(5).items()}),
            "Days To Expiry": int(np.busday_count(today, contract.expiry.date())) if contract.expiry.date() >= today else 0,
            "Contract Status": status, "Rejection Reason": reason, "Selected As": "Not Selected",
            "Roll Method": None, "Source": (str(last.get("source")) if pd.notna(last.get("source")) else source),
        })
    return diagnostics, metrics


def _two_day_crossover(current: ExpectedContract, next_contract: ExpectedContract,
                       histories: dict[str, pd.DataFrame], field: str, today: date) -> bool:
    a = histories.get(current.symbol, pd.DataFrame())
    b = histories.get(next_contract.symbol, pd.DataFrame())
    if a.empty or b.empty or field not in a or field not in b:
        return False
    common = a.index.intersection(b.index).sort_values()
    common = common[common < pd.Timestamp(today)]
    common = common[-2:]
    if len(common) < 2 or not quote_is_fresh(common[-1], today=today, max_business_days=MAX_STALE_TRADING_DAYS):
        return False
    av = pd.to_numeric(a.loc[common, field], errors="coerce")
    bv = pd.to_numeric(b.loc[common, field], errors="coerce")
    return bool(av.notna().all() and bv.notna().all() and (bv > av).all())


def _active_front(asset: str, contracts: list[ExpectedContract], metrics: dict[str, dict[str, Any]],
                  histories: dict[str, pd.DataFrame], store: TermStructureStore, today: date) -> tuple[int, str, dict[str, Any]]:
    old = store.roll_state(asset)
    old_index = next((i for i, contract in enumerate(contracts) if contract.symbol == old.get("active_symbol")), None)
    index = old_index if old_index is not None else 0
    forced_expiry_roll = bool(old.get("active_symbol") and old_index is None)
    if index + 1 >= len(contracts):
        selected = contracts[index]
        meta = store.save_roll(asset, selected.symbol, None, "CALENDAR_FALLBACK", today,
                               int(np.busday_count(today, selected.expiry.date())))
        return index, "CALENDAR_FALLBACK", meta
    current, nxt = contracts[index], contracts[index + 1]
    method = "HARD_EXPIRY_ROLL" if forced_expiry_roll else "CALENDAR_FALLBACK"
    trigger_date = metrics.get(current.symbol, {}).get("price_date", pd.NaT)
    trigger_date = pd.Timestamp(trigger_date).date() if pd.notna(trigger_date) else today
    if forced_expiry_roll:
        # The previously active symbol has expired or left the generated calendar;
        # the first currently valid listed expiry is now the new front by definition.
        index = 0
    else:
        vol_reliable = _two_day_data_available(current, nxt, histories, "volume", today=today)
        if vol_reliable and _two_day_crossover(current, nxt, histories, "volume", today):
            index += 1
            method = "VOLUME_CROSSOVER"
        elif not vol_reliable and _two_day_data_available(current, nxt, histories, "open_interest", today=today) and _two_day_crossover(current, nxt, histories, "open_interest", today):
            index += 1
            method = "OPEN_INTEREST_CROSSOVER"
        else:
            hard_limit = current.first_notice if current.first_notice is not None else current.expiry
            days_to_limit = int(np.busday_count(today, hard_limit.date()))
            if days_to_limit <= HARD_ROLL_BUSINESS_DAYS:
                index += 1
                method = "HARD_EXPIRY_ROLL"
            elif metrics.get(current.symbol, {}).get("status") in {"STALE", "EXPIRED", "INVALID"} and metrics.get(nxt.symbol, {}).get("status") in {"LIQUID", "LOW_LIQUIDITY"}:
                index += 1
                method = "CALENDAR_FALLBACK"
            elif (metrics.get(current.symbol, {}).get("status") == "LOW_LIQUIDITY"
                  and metrics.get(nxt.symbol, {}).get("status") == "LIQUID"):
                # Preserve adjacent maturity; only roll when next-session liquidity has actually crossed.
                method = "CALENDAR_FALLBACK"
    selected = contracts[min(index, len(contracts) - 1)]
    next_symbol = contracts[index - 1].symbol if index else None
    meta = store.save_roll(asset, selected.symbol, next_symbol, method, trigger_date,
                           int(np.busday_count(today, selected.expiry.date())))
    return index, method, meta


def _two_day_data_available(current: ExpectedContract, nxt: ExpectedContract,
                            histories: dict[str, pd.DataFrame], field: str, today: date | None = None) -> bool:
    a, b = histories.get(current.symbol, pd.DataFrame()), histories.get(nxt.symbol, pd.DataFrame())
    if a.empty or b.empty or field not in a or field not in b:
        return False
    common = a.index.intersection(b.index).sort_values()[-2:]
    if len(common) < 2 or (today is not None and not quote_is_fresh(common[-1], today=today, max_business_days=MAX_STALE_TRADING_DAYS)):
        return False
    return a.loc[common, field].notna().all() and b.loc[common, field].notna().all()


def _pair_row(asset: str, leg1: ExpectedContract, leg2: ExpectedContract,
              metrics: dict[str, dict[str, Any]], histories: dict[str, pd.DataFrame], source: str,
              today: date, quality: str | None = None, pair_key: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    a = metrics.get(leg1.symbol, {}).get("frame", pd.DataFrame())
    b = metrics.get(leg2.symbol, {}).get("frame", pd.DataFrame())
    daily = []
    common = a.index.intersection(b.index).sort_values() if not a.empty and not b.empty else pd.DatetimeIndex([])
    for dt in common:
        if pd.Timestamp(dt).date() >= today:
            continue
        p1, p2 = _number(a.loc[dt].get("price")), _number(b.loc[dt].get("price"))
        if not (np.isfinite(p1) and np.isfinite(p2) and p1 > 0 and p2 > 0):
            continue
        v1, v2 = _number(a.loc[dt].get("volume")), _number(b.loc[dt].get("volume"))
        vol5_1 = _number(pd.to_numeric(a.loc[:dt].get("volume"), errors="coerce").tail(5).sum(min_count=1))
        vol5_2 = _number(pd.to_numeric(b.loc[:dt].get("volume"), errors="coerce").tail(5).sum(min_count=1))
        oi1, oi2 = _number(a.loc[dt].get("open_interest")), _number(b.loc[dt].get("open_interest"))
        liquidity = [_liquidity(v1, vol5_1, oi1), _liquidity(v2, vol5_2, oi2)]
        if liquidity[0] != "LIQUID" or liquidity[1] == "INVALID":
            continue  # F1 must trade today; deferred legs still need recent-volume/OI evidence.
        q = "HIGH" if liquidity == ["LIQUID", "LIQUID"] else "LOW_LIQUIDITY"
        daily.append({
            "asset": asset, "pair_key": pair_key or _pair_key(asset, leg1, leg2), "as_of": pd.Timestamp(dt),
            "leg1_symbol": leg1.symbol, "leg2_symbol": leg2.symbol,
            "leg1_month": leg1.delivery_month, "leg2_month": leg2.delivery_month,
            "leg1_price": p1, "leg2_price": p2, "spread": p1 / p2 - 1,
            "quality": q, "source": " + ".join(sorted({str(a.loc[dt].get("source", source)), str(b.loc[dt].get("source", source))})),
            "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    valid = [row for row in daily if quote_is_fresh(row["as_of"], today=today, max_business_days=MAX_STALE_TRADING_DAYS)]
    if not valid:
        row = {
            "asset": asset, "structure": _display_structure(asset, leg1, leg2, pair_key), "pair_key": pair_key or _pair_key(asset, leg1, leg2),
            "as_of": pd.NaT, "leg1_contract": leg1.symbol, "leg2_contract": leg2.symbol,
            "leg1_price": np.nan, "leg2_price": np.nan, "spread": np.nan, "raw_state": "N/A",
            "quality": "STALE" if daily else "INVALID", "status_reason": "NO_COMMON_FRESH_EOD" if daily else "NO_COMMON_EOD_DATE",
            "source": source, "observed_at": None,
        }
        return row, daily
    last = valid[-1]
    quality = quality or last["quality"]
    row = {
        "asset": asset, "structure": _display_structure(asset, leg1, leg2, last["pair_key"]), "pair_key": last["pair_key"],
        "as_of": last["as_of"], "leg1_contract": leg1.symbol, "leg2_contract": leg2.symbol,
        "leg1_price": last["leg1_price"], "leg2_price": last["leg2_price"], "spread": last["spread"],
        "raw_state": "Backwardation" if last["spread"] > 0 else "Contango" if last["spread"] < 0 else "Flat",
        "quality": quality, "status_reason": "OK", "source": source,
        "common_eod_date": last["as_of"],
        "observed_at": last["observed_at"],
    }
    return row, daily


def _liquidity(volume: float, volume5: float, open_interest: float) -> str:
    if pd.notna(volume) and volume > 0:
        return "LIQUID"
    if (pd.notna(volume5) and volume5 > 0) or (pd.notna(open_interest) and open_interest > 0):
        return "LOW_LIQUIDITY"
    return "INVALID"


def _pair_key(asset: str, first: ExpectedContract, second: ExpectedContract) -> str:
    if asset in AGRICULTURE:
        return f"{asset.replace(' ', '')}_{MONTH_CODES[first.delivery_month]}_{MONTH_CODES[second.delivery_month]}"
    if asset in METALS:
        return f"{asset}_Cash_3M"
    return f"{asset}_F1_{first.delivery_month:02d}_{first.delivery_year}_F2_{second.delivery_month:02d}_{second.delivery_year}"


def _display_structure(asset: str, first: ExpectedContract, second: ExpectedContract, pair_key: str | None = None) -> str:
    if asset in ENERGY:
        return "F1/F6" if str(pair_key).endswith("F1/F6") else "F1/F3"
    if asset in AGRICULTURE:
        return f"{calendar.month_abbr[first.delivery_month]}/{calendar.month_abbr[second.delivery_month]}"
    return "Cash/3M"


def _energy_pair_key(asset: str, kind: str) -> str:
    return f"{asset}_{kind}"


def _number(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NaT or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def percentile_with_history(value: float, history: pd.DataFrame, asset: str, month: int, pair_key: str,
                            n: int, before: pd.Timestamp | None = None) -> dict[str, Any]:
    if history.empty:
        sample = pd.DataFrame()
    else:
        sample = history.loc[(history["Asset"] == asset) & (history["Month"] == month) & (history["PairKey"] == pair_key)].copy()
        sample["Date"] = pd.to_datetime(sample["Date"], errors="coerce")
        sample["Spread"] = pd.to_numeric(sample["Spread"], errors="coerce")
        if before is not None:
            sample = sample.loc[sample["Date"] < pd.Timestamp(before)]
        sample = sample.dropna(subset=["Date", "Spread"]).sort_values("Date").drop_duplicates(sample.columns.intersection(["Year"]).tolist(), keep="last") if "Year" in sample else sample.dropna(subset=["Date", "Spread"]).sort_values("Date")
        sample = sample.tail(n)
    count = len(sample)
    if count < n or not np.isfinite(value):
        percentile = np.nan
    else:
        from commodity_cycle.model import midrank_percentile
        percentile = midrank_percentile(float(value), sample["Spread"].to_numpy())
    years = pd.to_datetime(sample["Date"], errors="coerce").dt.year.tolist() if count else []
    expected_end = (pd.Timestamp(before).year - 1) if before is not None else (max(years) if years else None)
    contiguous = bool(years and len(years) == n and years == list(range(expected_end - n + 1, expected_end + 1)))
    status = (f"FULL_{n}_OBS" if contiguous else "NON_CONTIGUOUS_HISTORY") if count >= n else f"INSUFFICIENT_{n}Y_HISTORY"
    return {
        "percentile": percentile, "history_n": count,
        "history_start": sample["Date"].min() if count else pd.NaT,
        "history_end": sample["Date"].max() if count else pd.NaT,
        "history_status": status,
    }


def _baseline_pair_key(asset: str, structure: str) -> str:
    if asset in ENERGY:
        return f"{asset}_F1/F3"
    if asset in METALS:
        return f"{asset}_Cash_3M"
    normalized = str(structure).replace(" ", "").lower()
    if asset in {"Corn", "Wheat"} and normalized in {"dec/mar", "z/h"}:
        return f"{asset}_Z_H"
    if asset == "Soybeans" and normalized in {"nov/jan", "x/f"}:
        return "Soybeans_X_F"
    return structure


def seasonal_history_frame(baseline: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not baseline.empty:
        base = baseline.loc[:, ["Asset", "Date", "Month", "Spread %", "Structure"]].copy()
        base = base.rename(columns={"Spread %": "Spread"})
        base["PairKey"] = [_baseline_pair_key(row.Asset, row.Structure) for row in base.itertuples()]
        frames.append(base[["Asset", "Date", "Month", "Spread", "PairKey"]])
    if not monthly.empty:
        app = monthly.rename(columns={"asset": "Asset", "month": "Date", "monthly_spread": "Spread", "pair_key": "PairKey"}).copy()
        app["Date"] = pd.to_datetime(app["Date"], errors="coerce")
        app["Month"] = app["Date"].dt.month
        frames.append(app[["Asset", "Date", "Month", "Spread", "PairKey"]])
    if not frames:
        return pd.DataFrame(columns=["Asset", "Date", "Month", "Spread", "PairKey"])
    result = pd.concat(frames, ignore_index=True)
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
    result["Spread"] = pd.to_numeric(result["Spread"], errors="coerce")
    result["Month"] = pd.to_numeric(result["Month"], errors="coerce")
    result = result.dropna(subset=["Date", "Spread"]).sort_values("Date")
    result["_year"] = result["Date"].dt.year
    result = result.drop_duplicates(["Asset", "PairKey", "_year", "Month"], keep="last").drop(columns="_year")
    return result


def _calculate_curve_seasonality(row: dict[str, Any], baseline: pd.DataFrame, monthly: pd.DataFrame) -> dict[str, Any]:
    current_month = pd.Timestamp(row["as_of"]).to_period("M") if pd.notna(row.get("as_of")) else pd.Period.now("M")
    history = seasonal_history_frame(baseline, monthly)
    result = {}
    for n, label in ((5, "5Y"), (10, "10Y")):
        info = percentile_with_history(float(row.get("mtd_spread", np.nan)), history, row["asset"], current_month.month,
                                      row["pair_key"], n, before=current_month.to_timestamp())
        result[f"Seasonal Pctl {label}"] = info["percentile"] if int(row.get("mtd_observation_count", 0)) >= 5 else np.nan
        result[f"{label} HistoryN"] = info["history_n"]
        result[f"{label} HistoryStartDate"] = info["history_start"]
        result[f"{label} HistoryEndDate"] = info["history_end"]
        result[f"{label} HistoryStatus"] = info["history_status"] if int(row.get("mtd_observation_count", 0)) >= 5 else "PROVISIONAL_INSUFFICIENT_MTD_SAMPLE"
    if int(row.get("mtd_observation_count", 0)) < 5:
        result["current_seasonal_status"] = "PROVISIONAL / N/A"
    elif pd.notna(result.get("Seasonal Pctl 5Y")):
        result["current_seasonal_status"] = "OFFICIAL" if pd.notna(result.get("Seasonal Pctl 10Y")) else "OFFICIAL_5Y / INSUFFICIENT_10Y_HISTORY"
    else:
        result["current_seasonal_status"] = "INSUFFICIENT_SEASONAL_HISTORY"
    return result


def fetch_current_term_structure(store: TermStructureStore, baseline: pd.DataFrame, today: date | None = None,
                                 yahoo_provider: TermStructureContractProvider | None = None,
                                 fallback_provider: TermStructureContractProvider | None = None,
                                 lme_provider: WestmetallLMEProvider | None = None) -> dict[str, pd.DataFrame]:
    """Deterministic refresh: calendar → quote histories → validation → legs → daily/monthly curves."""
    today = today or date.today()
    refresh_id = datetime.now(timezone.utc).isoformat(timespec="seconds")
    yahoo = yahoo_provider or YahooFuturesProvider()
    fallback = fallback_provider or TradingViewFuturesProvider()
    lme = lme_provider or WestmetallLMEProvider()
    universes = {asset: expected_contracts(asset, today) if asset in CONTRACT_SPECS else []
                 for asset in (*CONTRACT_SPECS, *METALS)}
    contracts = [item for group in universes.values() for item in group]
    fetchable_contracts = [item for item in contracts if item.expiry.date() >= today]
    try:
        fresh_histories = yahoo.fetch_contracts(fetchable_contracts)
        source = yahoo.name
    except Exception:
        fresh_histories = {}
        source = yahoo.name
    if not fresh_histories:
        try:
            fresh_histories = fallback.fetch_contracts(fetchable_contracts)
            source = fallback.name
        except Exception:
            fresh_histories = {}
    histories = {symbol: _completed_eod_frame(frame, today) for symbol, frame in fresh_histories.items()}
    histories = {symbol: frame for symbol, frame in histories.items() if not frame.empty}
    fresh_histories = histories.copy()
    # Merge local observations so two-session crossovers can survive transient fetch gaps.
    old_history = store.quote_history(item.symbol for item in contracts)
    for symbol, old in old_history.items():
        new = histories.get(symbol, pd.DataFrame())
        if not new.empty:
            histories[symbol] = pd.concat([old, new]).sort_index().groupby(level=0).last()
        elif symbol not in histories:
            histories[symbol] = old
    store.save_quotes(fetchable_contracts, fresh_histories, refresh_id)
    current_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    for asset, group in universes.items():
        fetchable_group = [c for c in group if c.expiry.date() >= today]
        asset_source = source
        diag, metrics = _expected_contract_frame(asset, group, histories, today, asset_source)
        diagnostics_rows.extend(diag)
        active_group = [item for item in group if item.expiry.date() >= today]
        if not active_group and asset not in METALS:
            continue
        if asset in ENERGY or asset in AGRICULTURE:
            index, roll_method, roll = _active_front(asset, active_group, metrics, histories, store, today)
            active = active_group[index]
            required = [active]
            if asset in ENERGY:
                required.extend(active_group[i] for i in (index + 2, index + 5) if i < len(active_group))
            elif index + 1 < len(active_group):
                required.append(active_group[index + 1])
            failed_required = [item for item in required if metrics.get(item.symbol, {}).get("status") in {"STALE", "INVALID"}]
            if source == yahoo.name and failed_required:
                try:
                    tv = fallback.fetch_contracts(failed_required)
                    tv = {symbol: _completed_eod_frame(frame, today) for symbol, frame in tv.items()}
                    tv = {symbol: frame for symbol, frame in tv.items() if not frame.empty}
                    if tv:
                        histories.update(tv)
                        fresh_histories.update(tv)
                        store.save_quotes(failed_required, tv, refresh_id)
                        diagnostics_rows = [r for r in diagnostics_rows if r.get("Asset") != asset]
                        diag, metrics = _expected_contract_frame(asset, group, histories, today, fallback.name)
                        diagnostics_rows.extend(diag)
                        index, roll_method, roll = _active_front(asset, active_group, metrics, histories, store, today)
                        active = active_group[index]
                        required_sources = [metrics.get(item.symbol, {}).get("frame", pd.DataFrame()) for item in required]
                        asset_source = " + ".join(sorted({str(frame.iloc[-1].get("source", fallback.name)) for frame in required_sources if not frame.empty})) or fallback.name
                except Exception:
                    pass
            for item in diagnostics_rows:
                if item["Asset"] == asset:
                    if item["Yahoo Symbol"] == active.symbol:
                        item["Selected As"] = "F1" if asset in ENERGY else "Seasonal Leg1"
                    item["Active F1"] = active.symbol
                    item["Previous F1"] = roll.get("previous_symbol")
                    item["Rollover Date"] = roll.get("rollover_date")
                    item["Roll Method"] = roll_method
                    item["Active F1 Days To Expiry"] = roll.get("days_to_expiry")
            second_idx = index + 1
            if asset in ENERGY:
                for label, offset in (("F3", 2), ("F6", 5)):
                    selected_idx = index + offset
                    if selected_idx < len(active_group):
                        leg = active_group[selected_idx]
                        for item in diagnostics_rows:
                            if item["Asset"] == asset and item["Yahoo Symbol"] == leg.symbol:
                                item["Selected As"] = label
                        key = _energy_pair_key(asset, f"F1/{label}")
                        row, daily = _pair_row(asset, active, leg, metrics, histories, asset_source, today, pair_key=key)
                        daily_rows.extend(_after_current_month_roll(_current_month_daily(daily, today), roll, today))
                        current_rows.append(_add_roll_metadata(row, roll))
            else:
                if second_idx >= len(active_group):
                    continue
                leg2 = active_group[second_idx]
                for item in diagnostics_rows:
                    if item["Asset"] == asset and item["Yahoo Symbol"] == leg2.symbol:
                        item["Selected As"] = "Seasonal Leg2"
                pair_key = _pair_key(asset, active, leg2)
                row, daily = _pair_row(asset, active, leg2, metrics, histories, asset_source, today, pair_key=pair_key)
                daily_rows.extend(_after_current_month_roll(_current_month_daily(daily, today), roll, today))
                current_rows.append(_add_roll_metadata(row, roll))
        else:
            try:
                lme_frame = lme.fetch(asset)
                lme_frame = lme_frame.loc[lme_frame.index.date < today]
                daily = []
                for dt, item in lme_frame.iterrows():
                    p1, p2 = _number(item["leg1_price"]), _number(item["leg2_price"])
                    if p1 <= 0 or p2 <= 0:
                        continue
                    daily.append({"asset": asset, "pair_key": f"{asset}_Cash_3M", "as_of": pd.Timestamp(dt),
                                  "leg1_symbol": "LME Cash", "leg2_symbol": "LME 3M", "leg1_month": None,
                                  "leg2_month": None, "leg1_price": p1, "leg2_price": p2, "spread": p1 / p2 - 1,
                                  "quality": "HIGH", "source": lme.name,
                                  "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                daily_rows.extend(_current_month_daily(daily, today))
                valid = [item for item in daily if quote_is_fresh(item["as_of"], today, MAX_STALE_TRADING_DAYS)]
                if valid:
                    last = valid[-1]
                    current_rows.append({"asset": asset, "structure": "Cash/3M", "pair_key": last["pair_key"],
                                         "as_of": last["as_of"], "leg1_contract": "LME Cash", "leg2_contract": "LME 3M",
                                         "leg1_price": last["leg1_price"], "leg2_price": last["leg2_price"],
                                         "spread": last["spread"], "raw_state": "Backwardation" if last["spread"] > 0 else "Contango" if last["spread"] < 0 else "Flat",
                                         "quality": "HIGH", "status_reason": "OK", "source": lme.name,
                                         "observed_at": last["observed_at"]})
                else:
                    current_rows.append({"asset": asset, "structure": "Cash/3M", "pair_key": f"{asset}_Cash_3M",
                                         "as_of": pd.NaT, "spread": np.nan, "raw_state": "N/A", "quality": "STALE",
                                         "status_reason": "NO_COMMON_FRESH_EOD", "source": lme.name, "observed_at": None})
            except Exception as exc:
                diagnostics_rows.append({"Asset": asset, "Expected Contract": "LME Cash/3M", "Yahoo Symbol": "N/A",
                                         "Contract Status": "INVALID", "Rejection Reason": f"SOURCE_FAILED: {exc}",
                                         "Selected As": "Not Selected", "Source": lme.name})
                current_rows.append({"asset": asset, "structure": "Cash/3M", "pair_key": f"{asset}_Cash_3M",
                                     "as_of": pd.NaT, "spread": np.nan, "raw_state": "N/A", "quality": "INVALID",
                                     "status_reason": "SOURCE_FAILED", "source": lme.name, "observed_at": None})
    daily_frame = pd.DataFrame(daily_rows)
    if not daily_frame.empty:
        store.save_daily_curves(daily_frame)
    current_month = pd.Timestamp(today).to_period("M")
    store.finalize_closed_months(current_month, refresh_id)
    monthly = store.monthly_curves()
    daily_history = store.daily_curves()
    if not daily_history.empty:
        daily_history["as_of"] = pd.to_datetime(daily_history["as_of"], errors="coerce")
        daily_history = daily_history.loc[daily_history["as_of"].dt.to_period("M").eq(current_month)]
    for row in current_rows:
        if pd.notna(row.get("as_of")):
            match = daily_history.loc[(daily_history["asset"] == row["asset"])
                                     & (daily_history["pair_key"] == row["pair_key"])] if not daily_history.empty else pd.DataFrame()
            row["mtd_observation_count"] = len(match)
            row["mtd_spread"] = float(pd.to_numeric(match["spread"], errors="coerce").mean()) if not match.empty else np.nan
            row["mtd_avg_leg1"] = float(pd.to_numeric(match["leg1_price"], errors="coerce").mean()) if not match.empty else np.nan
            row["mtd_avg_leg2"] = float(pd.to_numeric(match["leg2_price"], errors="coerce").mean()) if not match.empty else np.nan
            row["current_seasonal_status"] = "OFFICIAL" if row["mtd_observation_count"] >= 5 else "PROVISIONAL / N/A"
            row.update(_calculate_curve_seasonality(row, baseline, monthly))
        else:
            row.update({"mtd_observation_count": 0, "mtd_spread": np.nan, "mtd_avg_leg1": np.nan,
                        "mtd_avg_leg2": np.nan, "current_seasonal_status": "N/A"})
    diag_frame = pd.DataFrame(diagnostics_rows)
    store.save_diagnostics(diag_frame, refresh_id)
    return {"current": pd.DataFrame(current_rows), "daily": daily_frame, "daily_history": daily_history,
            "monthly": monthly, "diagnostics": diag_frame, "source": source}


def _add_roll_metadata(row: dict[str, Any], roll: dict[str, Any]) -> dict[str, Any]:
    row.update({
        "active_f1": roll.get("active_symbol"), "previous_f1": roll.get("previous_symbol"),
        "rollover_date": roll.get("rollover_date"), "rollover_method": roll.get("rollover_method"),
        "days_to_expiry": roll.get("days_to_expiry"),
    })
    return row
