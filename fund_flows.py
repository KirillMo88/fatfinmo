from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
import json
import logging
import os
import re
import sqlite3
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ETF_COM_FUND_FLOWS_ENDPOINT = (
    "https://api-prod.etf.com/private/apps/fundflows/{ticker}/charts"
)
ETF_COM_SOURCE = "etf.com"
INITIAL_LOOKBACK_DAYS = 130
AUM_LOOK_FORWARD_DAYS = 7
RETRY_COOLDOWN_SECONDS = int(os.getenv("FUND_FLOW_RETRY_SECONDS", "21600"))

DATE_KEYS = ("date", "asOf", "asOfDate", "as_of_date", "flowDate", "flow_date")
FLOW_KEYS = (
    "net_flow",
    "netFlow",
    "netflows",
    "netFlows",
    "net_flow_mil",
    "netFlowMil",
    "netFlowMillions",
    "dailyNetFlows",
    "value",
)
AUM_KEYS = (
    "aum",
    "AUM",
    "assets",
    "assets_under_management",
    "assetsUnderManagement",
    "total_assets",
    "totalAssets",
    "net_assets",
    "netAssets",
)
UNSUPPORTED_TICKER_SUFFIXES = ("-USD", "-USDT", "-EUR", "-GBP", "=X")


@dataclass(frozen=True)
class FundFlowObservation:
    ticker: str
    date: date
    net_flow: float
    aum: float | None
    source: str = ETF_COM_SOURCE


@dataclass(frozen=True)
class FundFlowMetrics:
    flow_1m_pct: float
    flow_3m_pct: float
    latest_date: date | None
    source: str = ETF_COM_SOURCE
    method: str = "period_start_aum"


class EtfComRequestError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def default_fund_flow_cache_path() -> Path:
    raw = os.getenv("FUND_FLOW_CACHE_PATH")
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parent / "persistent" / "finance_cache" / "fund_flows.sqlite"


def normalize_ticker_for_etf_com(ticker: str) -> str | None:
    normalized = str(ticker or "").strip().upper()
    if not normalized:
        return None
    if normalized.startswith("^") or any(normalized.endswith(suffix) for suffix in UNSUPPORTED_TICKER_SUFFIXES):
        logger.info("ticker unsupported for ETF.com fund flows: %s", ticker)
        return None
    return normalized.replace(".", "-")


def fetch_etf_com_fund_flow_history(
    ticker: str,
    start_date: date,
    end_date: date,
    timeout: float = 12.0,
) -> list[FundFlowObservation]:
    start = start_date.strftime("%Y%m%d")
    end = end_date.strftime("%Y%m%d")
    url = ETF_COM_FUND_FLOWS_ENDPOINT.format(
        ticker=urllib.parse.quote(ticker)
    ) + f"?startDate={start}&endDate={end}"
    logger.info("ETF.com request started: %s %s %s", ticker, start, end)
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Referer": "https://www.etf.com/etfanalytics/etf-fund-flows-tool",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.warning("ETF.com request failed: %s status=%s", ticker, exc.code)
        raise EtfComRequestError(f"ETF.com HTTP {exc.code}", status_code=exc.code) from exc
    except Exception as exc:
        logger.warning("ETF.com request failed: %s error=%s", ticker, exc)
        raise EtfComRequestError(str(exc)) from exc

    observations = parse_etf_com_fund_flow_payload(ticker, payload)
    logger.info("ETF.com request successful: %s observations=%s", ticker, len(observations))
    return observations


def parse_etf_com_fund_flow_payload(ticker: str, payload: Any) -> list[FundFlowObservation]:
    rows = _extract_candidate_rows(payload)
    observations: list[FundFlowObservation] = []
    for row in rows:
        obs_date = _first_parsed_date(row, DATE_KEYS)
        net_flow = _first_money_value(row, FLOW_KEYS, default_unit="millions")
        if obs_date is None or net_flow is None:
            continue
        aum = _first_money_value(row, AUM_KEYS, default_unit="auto")
        observations.append(
            FundFlowObservation(
                ticker=ticker,
                date=obs_date,
                net_flow=float(net_flow),
                aum=None if aum is None else float(aum),
            )
        )
    observations.sort(key=lambda item: item.date)
    return observations


def calculate_fund_flow_metrics(
    observations: Iterable[FundFlowObservation],
    latest_date: date | None = None,
    fallback_aum: float | None = None,
) -> FundFlowMetrics | None:
    df = _observations_to_frame(observations)
    if df.empty:
        logger.info("Fund Flow calculation unavailable: no observations")
        return None

    latest = latest_date or df["date"].max().date()
    one_month_start = (pd.Timestamp(latest) - pd.DateOffset(months=1)).date()
    three_month_start = (pd.Timestamp(latest) - pd.DateOffset(months=3)).date()

    one_month = _period_flow_pct(df, one_month_start, latest, fallback_aum=fallback_aum)
    three_month = _period_flow_pct(df, three_month_start, latest, fallback_aum=fallback_aum)
    if one_month is None and three_month is None:
        logger.info("Fund Flow calculation unavailable: reference AUM unavailable")
        return None

    logger.info("FundFlows 1M %% calculated: %s", one_month)
    logger.info("FundFlows 3M %% calculated: %s", three_month)
    return FundFlowMetrics(
        flow_1m_pct=np.nan if one_month is None else float(one_month),
        flow_3m_pct=np.nan if three_month is None else float(three_month),
        latest_date=latest,
        method="period_start_aum" if df["aum"].notna().any() else "latest_aum_fallback",
    )


def get_fund_flow_metrics(
    ticker: str,
    cache_path: Path | None = None,
    today: date | None = None,
    fetcher: Callable[[str, date, date], list[FundFlowObservation]] | None = None,
    aum_fetcher: Callable[[str], float | None] | None = None,
) -> FundFlowMetrics | None:
    provider_ticker = normalize_ticker_for_etf_com(ticker)
    if provider_ticker is None:
        return None

    cache = FundFlowCache(cache_path or default_fund_flow_cache_path())
    today_date = today or datetime.now(timezone.utc).date()
    fetcher = fetcher or fetch_etf_com_fund_flow_history

    if cache.should_attempt_update(provider_ticker, today_date):
        latest_cached = cache.latest_observation_date(provider_ticker)
        start_date = (
            latest_cached + timedelta(days=1)
            if latest_cached is not None
            else today_date - timedelta(days=INITIAL_LOOKBACK_DAYS)
        )
        if start_date <= today_date:
            try:
                new_observations = fetcher(provider_ticker, start_date, today_date)
                if new_observations:
                    cache.upsert_observations(new_observations)
                    cache.mark_attempt(provider_ticker, success=True, new_count=len(new_observations))
                    logger.info("new observations stored: %s count=%s", provider_ticker, len(new_observations))
                else:
                    cache.mark_attempt(provider_ticker, success=True, new_count=0)
                    logger.info("no ETF.com fund flow observations returned: %s", provider_ticker)
            except EtfComRequestError as exc:
                cache.mark_attempt(provider_ticker, success=False, error=str(exc))
            except Exception as exc:
                cache.mark_attempt(provider_ticker, success=False, error=str(exc))
                logger.warning("ETF.com request failed: %s error=%s", provider_ticker, exc)
        else:
            cache.mark_attempt(provider_ticker, success=True, new_count=0)
            logger.info("no new observations: %s", provider_ticker)

    observations = cache.load_observations(
        provider_ticker,
        start_date=today_date - timedelta(days=INITIAL_LOOKBACK_DAYS + AUM_LOOK_FORWARD_DAYS),
    )
    if observations:
        logger.info("cached observations used: %s count=%s", provider_ticker, len(observations))
    fallback_aum = None
    if observations and all(obs.aum is None for obs in observations):
        fallback_aum = (aum_fetcher or fetch_yahoo_total_assets)(provider_ticker)
    return calculate_fund_flow_metrics(observations, fallback_aum=fallback_aum)


def fetch_yahoo_total_assets(ticker: str) -> float | None:
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        value = info.get("totalAssets") if isinstance(info, dict) else None
        if value is None:
            return None
        numeric = float(value)
        return numeric if np.isfinite(numeric) and numeric > 0 else None
    except Exception as exc:
        logger.info("Yahoo totalAssets unavailable: %s error=%s", ticker, exc)
        return None


class FundFlowCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    net_flow REAL NOT NULL,
                    aum REAL,
                    source TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (ticker, date, source)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ticker_status (
                    ticker TEXT PRIMARY KEY,
                    last_attempt_utc TEXT,
                    last_success_utc TEXT,
                    last_error TEXT,
                    unsupported INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def latest_observation_date(self, ticker: str) -> date | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(date) AS max_date FROM observations WHERE ticker = ? AND source = ?",
                (ticker, ETF_COM_SOURCE),
            ).fetchone()
        if not row or not row["max_date"]:
            return None
        return date.fromisoformat(row["max_date"])

    def should_attempt_update(self, ticker: str, today: date) -> bool:
        status = self.get_status(ticker)
        latest = self.latest_observation_date(ticker)
        if latest is not None and latest >= today:
            return False
        if status and status["last_attempt_utc"]:
            last_attempt = datetime.fromisoformat(status["last_attempt_utc"])
            elapsed = datetime.now(timezone.utc) - last_attempt
            if elapsed.total_seconds() < RETRY_COOLDOWN_SECONDS:
                return False
        return True

    def upsert_observations(self, observations: Iterable[FundFlowObservation]) -> int:
        rows = list(observations)
        if not rows:
            return 0
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO observations (ticker, date, net_flow, aum, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, date, source) DO UPDATE SET
                    net_flow = excluded.net_flow,
                    aum = COALESCE(excluded.aum, observations.aum),
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        obs.ticker,
                        obs.date.isoformat(),
                        obs.net_flow,
                        obs.aum,
                        obs.source,
                        updated_at,
                    )
                    for obs in rows
                ],
            )
        return len(rows)

    def load_observations(self, ticker: str, start_date: date | None = None) -> list[FundFlowObservation]:
        params: list[Any] = [ticker, ETF_COM_SOURCE]
        where = "ticker = ? AND source = ?"
        if start_date is not None:
            where += " AND date >= ?"
            params.append(start_date.isoformat())
        with self._connect() as conn:
            db_rows = conn.execute(
                f"""
                SELECT ticker, date, net_flow, aum, source
                FROM observations
                WHERE {where}
                ORDER BY date
                """,
                params,
            ).fetchall()
        return [
            FundFlowObservation(
                ticker=row["ticker"],
                date=date.fromisoformat(row["date"]),
                net_flow=float(row["net_flow"]),
                aum=None if row["aum"] is None else float(row["aum"]),
                source=row["source"],
            )
            for row in db_rows
        ]

    def mark_attempt(
        self,
        ticker: str,
        success: bool,
        new_count: int = 0,
        error: str | None = None,
        unsupported: bool = False,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ticker_status (
                    ticker, last_attempt_utc, last_success_utc, last_error, unsupported
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    last_attempt_utc = excluded.last_attempt_utc,
                    last_success_utc = COALESCE(excluded.last_success_utc, ticker_status.last_success_utc),
                    last_error = excluded.last_error,
                    unsupported = excluded.unsupported
                """,
                (
                    ticker,
                    now,
                    now if success else None,
                    None if success else error,
                    1 if unsupported else 0,
                ),
            )
        if success and new_count == 0 and not unsupported:
            logger.info("no new observations: %s", ticker)

    def get_status(self, ticker: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM ticker_status WHERE ticker = ?",
                (ticker,),
            ).fetchone()


def _period_flow_pct(df: pd.DataFrame, start: date, end: date, fallback_aum: float | None = None) -> float | None:
    reference_aum = _reference_aum(df, start)
    if reference_aum is None and fallback_aum is not None:
        reference_aum = fallback_aum
    if reference_aum is None or not np.isfinite(reference_aum) or reference_aum <= 0:
        logger.info("reference AUM unavailable: target=%s", start)
        return None
    logger.info("reference AUM found: target=%s aum=%s", start, reference_aum)
    mask = (df["date"].dt.date > start) & (df["date"].dt.date <= end)
    net_flow = float(df.loc[mask, "net_flow"].sum())
    return (net_flow / reference_aum) * 100.0


def _reference_aum(df: pd.DataFrame, target: date) -> float | None:
    available = df.dropna(subset=["aum"]).copy()
    if available.empty:
        return None
    before = available[available["date"].dt.date <= target]
    if not before.empty:
        return float(before.sort_values("date").iloc[-1]["aum"])
    after = available[available["date"].dt.date <= target + timedelta(days=AUM_LOOK_FORWARD_DAYS)]
    if after.empty:
        return None
    return float(after.sort_values("date").iloc[0]["aum"])


def _observations_to_frame(observations: Iterable[FundFlowObservation]) -> pd.DataFrame:
    rows = [
        {
            "date": pd.Timestamp(obs.date),
            "net_flow": obs.net_flow,
            "aum": obs.aum,
        }
        for obs in observations
    ]
    if not rows:
        return pd.DataFrame(columns=["date", "net_flow", "aum"])
    return pd.DataFrame(rows).sort_values("date")


def _extract_candidate_rows(payload: Any) -> list[dict[str, Any]]:
    direct = _dig(payload, ["data", "results", "data"])
    if _is_row_list(direct):
        return direct
    found: list[dict[str, Any]] = []
    stack = [payload]
    while stack:
        current = stack.pop()
        if _is_row_list(current):
            found.extend(current)
            continue
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return found


def _dig(payload: Any, keys: list[str]) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _is_row_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(row, dict) for row in value)
        and any(_has_any_key(row, DATE_KEYS) and _has_any_key(row, FLOW_KEYS) for row in value)
    )


def _has_any_key(row: dict[str, Any], keys: Iterable[str]) -> bool:
    lowered = {str(key).lower() for key in row}
    return any(key.lower() in lowered for key in keys)


def _first_parsed_date(row: dict[str, Any], keys: Iterable[str]) -> date | None:
    for key in keys:
        value = _case_insensitive_get(row, key)
        if value is None:
            continue
        parsed = _parse_date(value)
        if parsed is not None:
            return parsed
    return None


def _first_money_value(
    row: dict[str, Any],
    keys: Iterable[str],
    default_unit: str,
) -> float | None:
    for key in keys:
        value = _case_insensitive_get(row, key)
        if value is None:
            continue
        parsed = _parse_money(value)
        if parsed is None:
            continue
        return _normalize_money_unit(float(parsed), key, default_unit, value)
    return None


def _case_insensitive_get(row: dict[str, Any], target: str) -> Any:
    target_lower = target.lower()
    for key, value in row.items():
        if str(key).lower() == target_lower:
            return value
    return None


def _parse_date(value: Any) -> date | None:
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            text = str(int(value))
            if len(text) == 8:
                return datetime.strptime(text, "%Y%m%d").date()
        ts = pd.to_datetime(value, errors="coerce", utc=False)
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def _parse_money(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not np.isfinite(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "NULL", "NONE", "-"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = text.strip("()").replace("$", "").replace(",", "").replace("%", "").strip()
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*([KMBT]?)", cleaned, re.IGNORECASE)
    if not match:
        return None
    number = float(match.group(1))
    suffix = match.group(2).upper()
    multiplier = {"": 1.0, "K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0, "T": 1_000_000_000_000.0}[suffix]
    out = number * multiplier
    return -out if negative else out


def _normalize_money_unit(value: float, key: str, default_unit: str, raw_value: Any = None) -> float:
    if _has_explicit_money_suffix(raw_value):
        return value
    key_lower = key.lower()
    if "million" in key_lower or key_lower.endswith("_mil") or key_lower.endswith("mil"):
        return value * 1_000_000.0
    if default_unit == "millions":
        return value * 1_000_000.0
    if default_unit == "auto" and abs(value) < 10_000_000.0:
        return value * 1_000_000.0
    return value


def _has_explicit_money_suffix(value: Any) -> bool:
    if isinstance(value, str):
        return re.search(r"\d\s*[KMBT]\s*$", value.strip().strip("()"), re.IGNORECASE) is not None
    return False
