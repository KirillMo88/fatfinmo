from __future__ import annotations

import calendar
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M", 7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}
ENERGY = {
    "WTI": ("CL", "NYM", tuple(range(1, 13))),
    "Natural Gas": ("NG", "NYM", tuple(range(1, 13))),
    "RBOB": ("RB", "NYM", tuple(range(1, 13))),
}
AGRICULTURE = {
    "Corn": ("ZC", "CBT", (3, 5, 7, 9, 12), 12, 3),
    "Wheat": ("ZW", "CBT", (3, 5, 7, 9, 12), 12, 3),
    "Soybeans": ("ZS", "CBT", (1, 3, 5, 7, 8, 9, 11), 11, 1),
}
METALS = {
    "Copper": ("LME_Cu_cash", "Westmetall LME Cash/3M"),
    "Aluminum": ("LME_Al_cash", "Westmetall LME Cash/3M"),
}
ALL_ASSETS = (*ENERGY, *AGRICULTURE, *METALS)
TERM_DB_PATH = Path(__file__).resolve().parent.parent / "persistent" / "commodity_cycle" / "term_structure.sqlite3"


@dataclass(frozen=True)
class Quote:
    contract: str
    expiry: pd.Timestamp
    price: float
    observed_at: pd.Timestamp
    source: str = "Yahoo Finance individual futures"


def contract_symbol(root: str, month: int, year: int, exchange: str) -> str:
    """Yahoo individual futures symbol, e.g. CLX26.NYM."""
    return f"{root}{MONTH_CODES[month]}{year % 100:02d}.{exchange}"


def candidate_contracts(asset: str, today: date | None = None, years_ahead: int = 1) -> list[tuple[str, pd.Timestamp]]:
    """Generate listed delivery months dynamically; not tied to a fixed year."""
    today = today or date.today()
    current_month = today.month
    current_year = today.year
    out: list[tuple[str, pd.Timestamp]] = []
    if asset in ENERGY:
        root, exchange, months = ENERGY[asset]
    elif asset in AGRICULTURE:
        root, exchange, months, _, _ = AGRICULTURE[asset]
    else:
        return out
    for year in range(current_year, current_year + years_ahead + 1):
        for month in months:
            expiry = _contract_expiry(asset, year, month)
            if expiry.date() < today:
                continue
            out.append((contract_symbol(root, month, year, exchange), expiry))
    return out


def _previous_business_day(value: pd.Timestamp) -> pd.Timestamp:
    if np.is_busday(value.date()):
        return pd.Timestamp(np.busday_offset(value.date(), -1))
    return pd.Timestamp(np.busday_offset(value.date(), 0, roll="backward"))


def _third_business_day_before(value: pd.Timestamp) -> pd.Timestamp:
    # Exclude the reference date and count the immediately preceding business day as day one.
    previous = _previous_business_day(value)
    return pd.Timestamp(np.busday_offset(previous.date(), -2))


def _contract_expiry(asset: str, year: int, month: int) -> pd.Timestamp:
    """Approximate exchange last-trade dates to exclude already expired deliveries."""
    first_of_delivery = pd.Timestamp(year, month, 1)
    prior_month_end = first_of_delivery - pd.Timedelta(days=1)
    if asset == "WTI":
        # NYMEX WTI: third business day before the 25th calendar day of the prior month.
        reference = prior_month_end.replace(day=min(25, prior_month_end.day))
        return _third_business_day_before(reference)
    if asset == "Natural Gas":
        # NYMEX NG: third business day before the first calendar day of delivery month.
        return _third_business_day_before(first_of_delivery)
    if asset == "RBOB":
        if np.is_busday(prior_month_end.date()):
            return prior_month_end
        return pd.Timestamp(np.busday_offset(prior_month_end.date(), 0, roll="backward"))
    # CBOT grain last-trade conventions fall in the delivery month; the precise
    # holiday-adjusted date differs by product, but the third week is a safe expiry cutoff.
    fifteenth = pd.Timestamp(year, month, 15)
    return _previous_business_day(fifteenth)


def quote_is_fresh(observed_at: pd.Timestamp | date | str, today: date | None = None, max_business_days: int = 3) -> bool:
    observed = pd.Timestamp(observed_at).date()
    today = today or date.today()
    if observed > today:
        return False
    elapsed = int(np.busday_count(observed, today))
    return elapsed <= max_business_days


def _yahoo_quote(symbol: str) -> tuple[float, pd.Timestamp] | None:
    import yfinance as yf

    frame = yf.Ticker(symbol).history(period="10d", interval="1d", auto_adjust=False, actions=False)
    if frame is None or frame.empty or "Close" not in frame:
        return None
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    close = close.loc[close.gt(0)]
    if close.empty:
        return None
    observed = pd.Timestamp(close.index[-1])
    if observed.tzinfo is not None:
        observed = observed.tz_convert("UTC").tz_localize(None)
    return float(close.iloc[-1]), observed.normalize()


def _collect_yahoo_quotes(asset: str, today: date | None = None, quote_fetcher: Callable | None = None) -> list[Quote]:
    quote_fetcher = quote_fetcher or _yahoo_quote
    quotes: list[Quote] = []
    for symbol, expiry in candidate_contracts(asset, today=today):
        try:
            result = quote_fetcher(symbol)
            if result is None:
                continue
            if len(result) == 3:
                price, observed_at, source = result
            else:
                price, observed_at = result
                source = "Yahoo Finance individual futures"
            observed = pd.Timestamp(observed_at)
            if observed.tzinfo is not None:
                observed = observed.tz_convert("UTC").tz_localize(None)
            if price is None or not np.isfinite(float(price)) or float(price) <= 0:
                continue
            if not quote_is_fresh(observed, today=today):
                continue
            quotes.append(Quote(symbol, expiry, float(price), observed.normalize(), str(source)))
        except Exception:
            continue
    return sorted(quotes, key=lambda quote: quote.expiry)


def _parse_westmetall(asset: str, request_get: Callable | None = None) -> dict | None:
    import requests

    get = request_get or requests.get
    field, source = METALS[asset]
    response = get(f"https://www.westmetall.com/en/markdaten.php?action=table&field={field}", timeout=20)
    response.raise_for_status()
    tables = pd.read_html(response.text)
    for table in tables:
        table.columns = [str(col).strip() for col in table.columns]
        date_col = next((col for col in table.columns if col.lower() == "date"), None)
        cash_col = next((col for col in table.columns if "cash-settlement" in col.lower()), None)
        three_col = next((col for col in table.columns if "3-month" in col.lower()), None)
        if date_col is None or cash_col is None or three_col is None:
            continue
        frame = table[[date_col, cash_col, three_col]].copy()
        frame.columns = ["date", "cash", "three_month"]
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", dayfirst=True)
        for col in ("cash", "three_month"):
            frame[col] = pd.to_numeric(frame[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        frame = frame.dropna(subset=["date", "cash", "three_month"]).sort_values("date")
        if frame.empty:
            continue
        row = frame.iloc[-1]
        if not quote_is_fresh(row["date"], max_business_days=3):
            return {"status": "STALE", "as_of": row["date"], "source": source}
        return {
            "asset": asset, "as_of": row["date"], "leg1": float(row["cash"]), "leg2": float(row["three_month"]),
            "leg1_contract": "LME Cash", "leg2_contract": "LME 3M", "source": source, "status": "CURRENT",
            "structure": "Cash/3M",
        }
    return None


def make_rows(today: date | None = None, quote_fetcher: Callable | None = None, request_get: Callable | None = None) -> pd.DataFrame:
    """Fetch current spreads using individual Yahoo listed-contract symbols and Westmetall LME tables."""
    today = today or date.today()
    rows: list[dict] = []
    def fetch_for(asset: str) -> Callable:
        if quote_fetcher:
            return quote_fetcher
        quotes = _download_contract_quotes(asset, today)
        return lambda symbol: quotes.get(symbol) if symbol in quotes else _yahoo_quote(symbol)
    for asset in ENERGY:
        quotes = _collect_yahoo_quotes(asset, today=today, quote_fetcher=fetch_for(asset))
        if not quotes:
            quotes = _collect_yahoo_quotes(asset, today=today, quote_fetcher=_tradingview_quote)
        for structure, leg_indices in (("F1/F3", (0, 2)), ("F1/F6", (0, 5))):
            if len(quotes) <= leg_indices[1]:
                continue
            first, second = (quotes[index] for index in leg_indices)
            rows.append(_spread_row(asset, first, second, structure, "Yahoo Finance individual futures"))
    for asset, (_, _, _, month1, month2) in AGRICULTURE.items():
        quotes = _collect_yahoo_quotes(asset, today=today, quote_fetcher=fetch_for(asset))
        if not quotes:
            quotes = _collect_yahoo_quotes(asset, today=today, quote_fetcher=_tradingview_quote)
        firsts = [q for q in quotes if q.expiry.month == month1]
        for first in firsts:
            second = next((q for q in quotes if q.expiry > first.expiry and q.expiry.month == month2), None)
            if second is not None:
                rows.append(_spread_row(asset, first, second, f"{calendar.month_abbr[month1]}/{calendar.month_abbr[month2]}", "Yahoo Finance individual futures"))
                break
    for asset in METALS:
        try:
            row = _parse_westmetall(asset, request_get=request_get)
            if row and row.get("status") == "CURRENT":
                row["spread"] = row["leg1"] / row["leg2"] - 1 if row["leg2"] else np.nan
                row["leg1_price"] = row["leg1"]
                row["leg2_price"] = row["leg2"]
                rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def _download_contract_quotes(asset: str, today: date) -> dict[str, tuple[float, pd.Timestamp]]:
    """Batch the dynamic individual-contract symbols through yfinance to avoid one HTTP call per month."""
    contracts = candidate_contracts(asset, today=today)
    symbols = [symbol for symbol, _ in contracts]
    if not symbols:
        return {}
    try:
        import yfinance as yf

        frame = yf.download(symbols, period="10d", interval="1d", auto_adjust=False, actions=False,
                            group_by="ticker", threads=True, progress=False)
        if frame is None or frame.empty or not isinstance(frame.columns, pd.MultiIndex):
            return {}
        result: dict[str, tuple[float, pd.Timestamp]] = {}
        top = set(frame.columns.get_level_values(0))
        for symbol in symbols:
            if symbol in top:
                close = frame[symbol]["Close"] if "Close" in frame[symbol] else pd.Series(dtype=float)
            elif "Close" in set(frame.columns.get_level_values(0)) and symbol in set(frame.columns.get_level_values(1)):
                close = frame["Close"][symbol]
            else:
                continue
            close = pd.to_numeric(close, errors="coerce").dropna()
            close = close.loc[close.gt(0)]
            if close.empty:
                continue
            observed = pd.Timestamp(close.index[-1])
            if observed.tzinfo is not None:
                observed = observed.tz_convert("UTC").tz_localize(None)
            result[symbol] = (float(close.iloc[-1]), observed.normalize())
        return result
    except Exception:
        return {}


def _tradingview_quote(yahoo_symbol: str) -> tuple[float, pd.Timestamp, str] | None:
    """Optional fallback through the app's authorized TradingView MCP client (never HTML scraping)."""
    import re
    from tradingview_mcp import get_ohlcv_data

    match = re.fullmatch(r"([A-Z]+)([FGHJKMNQUVXZ])(\d{2})\.(NYM|CBT)", yahoo_symbol)
    if not match:
        return None
    root, month_code, year_suffix, venue = match.groups()
    exchange = "NYMEX" if venue == "NYM" else "CBOT"
    frame = get_ohlcv_data(f"{exchange}:{root}{month_code}20{year_suffix}", interval="1D", count=20, force=True)
    if frame.empty:
        return None
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    row = frame.iloc[-1]
    return float(row["close"]), pd.Timestamp(row["date"]).normalize(), "TradingView MCP"


def _spread_row(asset: str, first: Quote, second: Quote, structure: str, source: str) -> dict:
    as_of = min(first.observed_at, second.observed_at)
    actual_sources = sorted({first.source, second.source})
    source_used = " + ".join(actual_sources)
    return {
        "asset": asset, "as_of": as_of, "leg1": first.price, "leg2": second.price,
        "leg1_price": first.price, "leg2_price": second.price,
        "leg1_contract": first.contract, "leg2_contract": second.contract,
        "spread": first.price / second.price - 1 if second.price else np.nan,
        "structure": structure, "source": source_used or source, "status": "CURRENT",
    }


class TermStructureStore:
    """Persist live snapshots separately from the immutable supplied workbook baseline."""

    def __init__(self, path: Path | str = TERM_DB_PATH):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=20)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS current_snapshots (
                asset TEXT NOT NULL, structure TEXT NOT NULL, as_of TEXT NOT NULL,
                leg1_contract TEXT, leg2_contract TEXT, leg1_price REAL, leg2_price REAL,
                raw_spread REAL, source TEXT NOT NULL, observed_at TEXT NOT NULL,
                status TEXT NOT NULL, PRIMARY KEY(asset, structure, as_of)
            );
            CREATE TABLE IF NOT EXISTS month_end_observations (
                asset TEXT NOT NULL, structure TEXT NOT NULL, as_of TEXT NOT NULL,
                leg1_contract TEXT, leg2_contract TEXT, leg1_price REAL, leg2_price REAL,
                raw_spread REAL, source TEXT NOT NULL, observed_at TEXT NOT NULL,
                PRIMARY KEY(asset, structure, as_of)
            );
        """)
        return conn

    def save_current(self, rows: pd.DataFrame, now: datetime | None = None) -> None:
        if rows.empty:
            return
        now = now or datetime.utcnow()
        with self._connect() as conn:
            for _, row in rows.iterrows():
                as_of = pd.Timestamp(row["as_of"]).date()
                source = str(row.get("source") or "UNKNOWN")
                values = (
                    str(row["asset"]), str(row.get("structure") or ""), as_of.isoformat(),
                    row.get("leg1_contract"), row.get("leg2_contract"), _safe_float(row.get("leg1_price", row.get("leg1"))),
                    _safe_float(row.get("leg2_price", row.get("leg2"))), _safe_float(row.get("spread")), source,
                    now.isoformat(timespec="seconds"), str(row.get("status") or "UNKNOWN"),
                )
                conn.execute("INSERT OR REPLACE INTO current_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
                # Only persist actual latest business/month-end quotes, never synthesize a calendar date.
                if pd.offsets.BMonthEnd().is_on_offset(pd.Timestamp(as_of)):
                    conn.execute("INSERT OR REPLACE INTO month_end_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values[:10])

    def observations(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM month_end_observations ORDER BY as_of").fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def latest_snapshots(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT c.* FROM current_snapshots c
                JOIN (SELECT asset, structure, MAX(as_of) AS max_date FROM current_snapshots GROUP BY asset, structure) x
                ON c.asset=x.asset AND c.structure=x.structure AND c.as_of=x.max_date
                ORDER BY c.asset, c.structure
            """).fetchall()
        return pd.DataFrame([dict(row) for row in rows])


def _safe_float(value) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def seasonal_percentile(asset: str, current_spread: float, as_of: pd.Timestamp, structure: str,
                        baseline: pd.DataFrame, stored: pd.DataFrame, years: int) -> float:
    """Midrank against same-month, comparable-structure observations only."""
    ref_frames: list[pd.DataFrame] = []
    if not baseline.empty:
        part = baseline.loc[baseline["Asset"].eq(asset), ["Date", "Spread %"]].copy()
        part["Structure"] = _baseline_structure(asset)
        ref_frames.append(part)
    if not stored.empty:
        part = stored.loc[stored["asset"].eq(asset), ["as_of", "raw_spread", "structure"]].copy()
        part = part.rename(columns={"as_of": "Date", "raw_spread": "Spread %", "structure": "Structure"})
        ref_frames.append(part)
    if not ref_frames or not np.isfinite(current_spread) or pd.isna(as_of):
        return np.nan
    refs = pd.concat(ref_frames, ignore_index=True)
    refs["Date"] = pd.to_datetime(refs["Date"], errors="coerce")
    refs["Spread %"] = pd.to_numeric(refs["Spread %"], errors="coerce")
    refs = refs.dropna(subset=["Date"]).sort_values("Date").copy()
    wanted = structure
    compatible = refs["Structure"].eq(wanted)
    # The workbook's energy baseline is documented as EIA contracts 1/3.
    if asset in ENERGY and wanted == "F1/F3":
        compatible |= refs["Structure"].eq("F1/F3")
    refs = refs.loc[compatible]
    refs["_year"] = refs["Date"].dt.year
    refs["_month"] = refs["Date"].dt.month
    refs = refs.drop_duplicates(["_year", "_month"], keep="last")
    asof = pd.Timestamp(as_of)
    refs = refs.loc[refs["_month"].eq(asof.month)
                    & refs["_year"].between(asof.year - years, asof.year - 1), "Spread %"].dropna()
    if len(refs) < years:
        return np.nan
    from commodity_cycle.model import midrank_percentile
    return midrank_percentile(float(current_spread), refs.to_numpy())


def _baseline_structure(asset: str) -> str:
    if asset in ENERGY:
        return "F1/F3"
    if asset in METALS:
        return "Cash/3M"
    if asset == "Corn" or asset == "Wheat":
        return "Dec/Mar"
    if asset == "Soybeans":
        return "Nov/Jan"
    return ""
