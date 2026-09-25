from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
from threading import Thread
import time
from typing import Any
from io import StringIO
from urllib.parse import urljoin

import httpx
import numpy as np
import pandas as pd
try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - production Docker installs beautifulsoup4.
    BeautifulSoup = None

from finance_core import download_completed_ohlcv, market_business_days_old
from fred_client import FredApiError, download_fred_series, download_fred_series_batch


CHINA_M2_SERIES_ID = "Money & Quasi-money (M2)"
FRED_CHINA_M2_LEGACY_SERIES = "MYAGM2CNM189N"
TRADINGVIEW_CNM2_YOY_SERIES = "ECONOMICS:CNM2_YOY"
TRADINGVIEW_PBOC_TOTAL_ASSETS_SERIES = "ECONOMICS:CNCBBS"
ECB_TOTAL_ASSETS_SERIES_ID = "ILM.W.U2.C.T000000.Z5.Z01"
BOJ_TOTAL_ASSETS_DB = "BS01"
BOJ_TOTAL_ASSETS_SERIES_ID = "MABJMTA"
PBOC_TOTAL_ASSETS_SERIES_ID = "PBOC_TOTAL_ASSETS"

GLOBAL_LIQUIDITY_STORAGE_DIR = Path(
    os.environ.get(
        "GLOBAL_LIQUIDITY_STORAGE_DIR",
        str(Path(__file__).with_name("persistent") / "global_liquidity"),
    )
)
RAW_STORAGE_PATH = GLOBAL_LIQUIDITY_STORAGE_DIR / "raw_global_liquidity.csv"
MONTHLY_STORAGE_PATH = GLOBAL_LIQUIDITY_STORAGE_DIR / "monthly_global_liquidity.csv"
WEEKLY_STORAGE_PATH = GLOBAL_LIQUIDITY_STORAGE_DIR / "weekly_global_liquidity.csv"
PBOC_TOTAL_ASSETS_STORAGE_PATH = GLOBAL_LIQUIDITY_STORAGE_DIR / "pboc_total_assets.csv"
PBOC_TOTAL_ASSETS_BUNDLED_PATH = Path(__file__).with_name("data") / "pboc_total_assets.csv"

GLOBAL_LIQUIDITY_CONFIG = {
    "start_date": "2010-01-01",
    "timeout": 20.0,
    "max_retries": 3,
    "ttl_seconds": 21600,
    "ecb_bsi_base_url": "https://data-api.ecb.europa.eu/service/data/BSI",
    "ecb_m2_key": "M.U2.Y.V.M20.X.1.U2.2300.Z01.E",
    "ecb_ilm_base_url": "https://data-api.ecb.europa.eu/service/data/ILM",
    "ecb_total_assets_key": "W.U2.C.T000000.Z5.Z01",
    "boj_base_url": "https://www.stat-search.boj.or.jp/api/v1/getDataCode",
    "boj_metadata_url": "https://www.stat-search.boj.or.jp/api/v1/getMetadata",
    "boj_m2_db": "MD02",
    "boj_m2_code": "MAM1NAM2M2MO",
    "boj_total_assets_db": BOJ_TOTAL_ASSETS_DB,
    "boj_total_assets_code": BOJ_TOTAL_ASSETS_SERIES_ID,
    "pboc_money_supply_url": "http://www.pbc.gov.cn/diaochatongjisi/116219/116319/index.html",
    "pboc_balance_sheet_url": "https://www.pbc.gov.cn/diaochatongjisi/116219/116319/index.html",
    "tradingview_calendar_url": "https://economic-calendar.tradingview.com/events",
    "tradingview_scanner_url": "https://scanner.tradingview.com/global/scan",
    "tradingview_cnm2_reports_url": "https://www.tradingview.com/symbols/ECONOMICS-CNM2/reports-history/",
    "tradingview_cncbbs_url": "https://www.tradingview.com/symbols/ECONOMICS-CNCBBS/",
    "investing_china_m2_url": "https://www.investing.com/economic-calendar/chinese-m2-money-stock-463",
    "tradingeconomics_cncbbs_url": "https://tradingeconomics.com/china/banks-balance-sheet",
}

FRED_GLOBAL_SERIES = (
    "M2SL",
    "WALCL",
    "WTREGEN",
    "RRPONTSYD",
    "DEXUSEU",
    "DEXJPUS",
    "DEXCHUS",
)
FRED_FORECAST_SERIES = ("THREEFYTP10", "SOFR", "EFFR", "WRESBAL")

RAW_COLUMNS = [
    "observation_date",
    "release_date",
    "source",
    "source_name",
    "source_mode",
    "source_url",
    "series_id",
    "region",
    "metric",
    "frequency",
    "currency",
    "unit",
    "raw_value",
    "download_timestamp",
    "data_status",
    "notes",
]

MONTHLY_COLUMNS = [
    "date",
    "us_m2_usd_bn",
    "ea_m2_usd_bn",
    "china_m2_usd_bn",
    "japan_m2_usd_bn",
    "global_m2_usd_bn",
    "global_m2_partial_usd_bn",
    "global_m2_fx_neutral_bn",
    "global_m2_fx_neutral_partial_bn",
    "global_m2_fx_effect_bn",
    "global_m2_1m_pct",
    "global_m2_3m_pct",
    "global_m2_6m_pct",
    "global_m2_12m_pct",
    "global_m2_yoy_pct",
    "global_m2_3m_annualized",
    "global_m2_acceleration",
    "us_m2_share",
    "ea_m2_share",
    "china_m2_share",
    "japan_m2_share",
    "us_contribution_12m",
    "ea_contribution_12m",
    "china_contribution_12m",
    "japan_contribution_12m",
    "global_m2_impulse",
    "global_m2_impulse_3y_percentile",
    "global_m2_impulse_5y_percentile",
    "global_m2_impulse_zscore",
    "global_m2_impulse_state",
    "fx_base_date",
    "fx_base_method",
    "data_status",
    "last_updated",
    "fed_assets_usd_bn",
    "ecb_assets_usd_bn",
    "boj_assets_usd_bn",
    "pboc_assets_usd_bn",
    "global_cb_assets_usd_bn",
    "global_cb_assets_partial_usd_bn",
    "global_cb_assets_1m_pct",
    "global_cb_assets_3m_pct",
    "global_cb_assets_6m_pct",
    "global_cb_assets_12m_pct",
    "global_cb_assets_change_1m_bn",
    "global_cb_assets_change_3m_bn",
    "global_cb_assets_change_6m_bn",
    "fed_cb_share",
    "ecb_cb_share",
    "boj_cb_share",
    "pboc_cb_share",
    "fed_cb_contribution_3m",
    "ecb_cb_contribution_3m",
    "boj_cb_contribution_3m",
    "pboc_cb_contribution_3m",
]

WEEKLY_COLUMNS = [
    "date",
    "fed_assets_usd_bn",
    "rrp_usd_bn",
    "tga_usd_bn",
    "us_net_liquidity_usd_bn",
    "us_net_liquidity_1w_change_bn",
    "us_net_liquidity_4w_change_bn",
    "us_net_liquidity_13w_change_bn",
    "us_net_liquidity_26w_change_bn",
    "us_net_liquidity_1w_pct",
    "us_net_liquidity_4w_pct",
    "us_net_liquidity_13w_pct",
    "us_net_liquidity_26w_pct",
    "us_net_liquidity_13w_percentile",
    "us_net_liquidity_26w_percentile",
    "ecb_assets_usd_bn",
    "boj_assets_usd_bn",
    "pboc_assets_usd_bn",
    "global_cb_assets_usd_bn",
    "global_cb_assets_partial_usd_bn",
    "global_cb_assets_4w_pct",
    "global_cb_assets_13w_pct",
    "global_cb_assets_26w_pct",
    "global_cb_assets_52w_pct",
    "global_cb_assets_change_4w_bn",
    "global_cb_assets_change_13w_bn",
    "global_cb_assets_change_26w_bn",
    "global_cb_impulse",
    "global_cb_impulse_3y_percentile",
    "global_cb_impulse_5y_percentile",
    "global_cb_impulse_zscore",
    "global_cb_impulse_state",
    "us_net_liquidity_impulse",
    "us_net_liquidity_impulse_3y_percentile",
    "us_net_liquidity_impulse_5y_percentile",
    "us_net_liquidity_impulse_zscore",
    "us_net_liquidity_impulse_state",
    "dxy",
    "dxy_4w_pct",
    "dxy_13w_pct",
    "dxy_26w_pct",
    "US10Y_TermPremium",
    "US10Y_TermPremium_4W_Change",
    "US10Y_TermPremium_13W_Change",
    "US10Y_TermPremium_26W_Change",
    "SOFR",
    "EFFR",
    "SOFR_EFFR_Spread",
    "SOFR_EFFR_4W_Change",
    "SOFR_EFFR_13W_Change",
    "US_BankReserves",
    "US_BankReserves_4W_Change",
    "US_BankReserves_13W_Change",
    "US_BankReserves_26W_Change",
    "US_BankReserves_4W_PctChange",
    "US_BankReserves_13W_PctChange",
    "US_BankReserves_26W_PctChange",
    "data_status",
    "last_updated",
]

_UPDATE_THREAD: Thread | None = None


@dataclass(frozen=True)
class LiquidityFreshness:
    block: str
    last_observation_date: str
    last_release_date: str
    last_updated: str
    data_status: str


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).tz_convert(None)


def get_with_retries(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: float | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    timeout = timeout or float(GLOBAL_LIQUIDITY_CONFIG["timeout"])
    last_error: Exception | None = None
    for attempt in range(int(GLOBAL_LIQUIDITY_CONFIG["max_retries"])):
        try:
            response = httpx.get(url, params=params, timeout=timeout, headers=headers, follow_redirects=True)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            time.sleep(min(0.8 * (2**attempt), 6.0))
    raise RuntimeError(str(last_error) if last_error else "request failed")


def post_json_with_retries(
    url: str,
    payload: dict[str, Any],
    timeout: float | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    timeout = timeout or float(GLOBAL_LIQUIDITY_CONFIG["timeout"])
    last_error: Exception | None = None
    for attempt in range(int(GLOBAL_LIQUIDITY_CONFIG["max_retries"])):
        try:
            response = httpx.post(url, json=payload, timeout=timeout, headers=headers, follow_redirects=True)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            time.sleep(min(0.8 * (2**attempt), 6.0))
    raise RuntimeError(str(last_error) if last_error else "request failed")


def empty_raw_row(source: str, source_name: str, source_url: str, series_id: str, region: str, metric: str, frequency: str, currency: str, unit: str, status: str, notes: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "observation_date": pd.NaT,
                "release_date": pd.NaT,
                "source": source,
                "source_name": source_name,
                "source_mode": "FALLBACK_SOURCE",
                "source_url": source_url,
                "series_id": series_id,
                "region": region,
                "metric": metric,
                "frequency": frequency,
                "currency": currency,
                "unit": unit,
                "raw_value": np.nan,
                "download_timestamp": fmt_datetime(now_utc()),
                "data_status": status,
                "notes": notes,
            }
        ],
        columns=RAW_COLUMNS,
    )


def fred_raw(api_key: str | None = None) -> pd.DataFrame:
    try:
        frame = download_fred_series_batch(FRED_GLOBAL_SERIES, api_key=api_key, observation_start=GLOBAL_LIQUIDITY_CONFIG["start_date"])
    except FredApiError as exc:
        return empty_raw_row("FRED", "Federal Reserve Economic Data", "https://fred.stlouisfed.org/", "FRED_GLOBAL_SERIES", "US", "FRED_BLOCK", "mixed", "mixed", "native", "ERROR", str(exc))
    meta = {
        "M2SL": ("US", "M2", "monthly", "USD", "USD billions"),
        "WALCL": ("US", "Fed Total Assets", "weekly", "USD", "USD millions"),
        "WTREGEN": ("US", "Treasury General Account", "weekly", "USD", "USD millions"),
        "RRPONTSYD": ("US", "Overnight Reverse Repo", "daily", "USD", "USD billions"),
        "DEXUSEU": ("FX", "EURUSD", "daily", "USD per EUR", "rate"),
        "DEXJPUS": ("FX", "USDJPY", "daily", "JPY per USD", "rate"),
        "DEXCHUS": ("FX", "USDCNY", "daily", "CNY per USD", "rate"),
        "THREEFYTP10": ("US", "10Y Term Premium", "daily", "USD", "percentage points"),
        "SOFR": ("US", "SOFR", "daily", "USD", "percent"),
        "EFFR": ("US", "EFFR", "daily", "USD", "percent"),
        "WRESBAL": ("US", "Bank Reserves", "weekly", "USD", "USD millions"),
    }
    optional = []
    optional_errors = []
    for series_id in FRED_FORECAST_SERIES:
        try:
            optional.append(download_fred_series(series_id, api_key=api_key, observation_start=GLOBAL_LIQUIDITY_CONFIG["start_date"]))
        except FredApiError as exc:
            optional_errors.append(empty_raw_row("FRED", "Federal Reserve Economic Data", f"https://fred.stlouisfed.org/series/{series_id}", series_id, "US", series_id, "mixed", "USD", "native", "ERROR", f"Refresh failed: {type(exc).__name__}"))
    if optional:
        frame = pd.concat([frame, *optional], ignore_index=True)
    rows = []
    downloaded = fmt_datetime(now_utc())
    for _, row in frame.iterrows():
        series_id = str(row["Series_ID"]).upper()
        region, metric, frequency, currency, unit = meta.get(series_id, ("UNKNOWN", series_id, "unknown", "unknown", "native"))
        rows.append(
            {
                "observation_date": pd.to_datetime(row["Date"], errors="coerce"),
                "release_date": pd.NaT,
                "source": "FRED",
                "source_name": "Federal Reserve Economic Data",
                "source_url": f"https://fred.stlouisfed.org/series/{series_id}",
                "series_id": series_id,
                "region": region,
                "metric": metric,
                "frequency": frequency,
                "currency": currency,
                "unit": unit,
                "raw_value": pd.to_numeric(row["Value"], errors="coerce"),
                "download_timestamp": downloaded,
                "data_status": "CURRENT",
                "notes": "RELEASE_DATE_UNAVAILABLE",
            }
        )
    result = pd.DataFrame(rows, columns=RAW_COLUMNS)
    return pd.concat([result, *optional_errors], ignore_index=True) if optional_errors else result


def ecb_m2_raw() -> pd.DataFrame:
    key = str(GLOBAL_LIQUIDITY_CONFIG["ecb_m2_key"])
    url = f"{GLOBAL_LIQUIDITY_CONFIG['ecb_bsi_base_url']}/{key}"
    try:
        response = get_with_retries(
            url,
            params={"format": "csvdata", "startPeriod": GLOBAL_LIQUIDITY_CONFIG["start_date"][:7]},
        )
        csv = pd.read_csv(StringIO(response.text))
        if csv.empty or "TIME_PERIOD" not in csv.columns or "OBS_VALUE" not in csv.columns:
            raise RuntimeError("ECB response does not contain TIME_PERIOD/OBS_VALUE")
        rows = []
        downloaded = fmt_datetime(now_utc())
        for _, row in csv.iterrows():
            rows.append(
                {
                    "observation_date": pd.to_datetime(str(row["TIME_PERIOD"]) + "-01", errors="coerce"),
                    "release_date": pd.NaT,
                    "source": "ECB",
                    "source_name": "ECB Data Portal",
                    "source_url": url,
                    "series_id": key,
                    "region": "Euro Area",
                    "metric": "M2",
                    "frequency": "monthly",
                    "currency": "EUR",
                    "unit": "EUR millions",
                    "raw_value": pd.to_numeric(row["OBS_VALUE"], errors="coerce"),
                    "download_timestamp": downloaded,
                    "data_status": "CURRENT",
                    "notes": "RELEASE_DATE_UNAVAILABLE",
                }
            )
        return pd.DataFrame(rows, columns=RAW_COLUMNS).dropna(subset=["observation_date", "raw_value"])
    except Exception as exc:
        return empty_raw_row("ECB", "ECB Data Portal", url, key, "Euro Area", "M2", "monthly", "EUR", "EUR millions", "ERROR", str(exc))


def ecb_total_assets_raw() -> pd.DataFrame:
    key = str(GLOBAL_LIQUIDITY_CONFIG["ecb_total_assets_key"])
    url = f"{GLOBAL_LIQUIDITY_CONFIG['ecb_ilm_base_url']}/{key}"
    try:
        response = get_with_retries(
            url,
            params={"format": "csvdata", "startPeriod": GLOBAL_LIQUIDITY_CONFIG["start_date"][:4]},
        )
        csv = pd.read_csv(StringIO(response.text))
        if csv.empty or "TIME_PERIOD" not in csv.columns or "OBS_VALUE" not in csv.columns:
            raise RuntimeError("ECB ILM response does not contain TIME_PERIOD/OBS_VALUE")
        rows = []
        downloaded = fmt_datetime(now_utc())
        for _, row in csv.iterrows():
            observation_date = parse_ecb_week_period(str(row["TIME_PERIOD"]))
            rows.append(
                {
                    "observation_date": observation_date,
                    "release_date": pd.NaT,
                    "source": "ECB",
                    "source_name": "ECB Data Portal",
                    "source_url": url,
                    "series_id": ECB_TOTAL_ASSETS_SERIES_ID,
                    "region": "Euro Area",
                    "metric": "ECB Total Assets",
                    "frequency": "weekly",
                    "currency": "EUR",
                    "unit": "EUR millions",
                    "raw_value": pd.to_numeric(row["OBS_VALUE"], errors="coerce"),
                    "download_timestamp": downloaded,
                    "data_status": "CURRENT",
                    "notes": "RELEASE_DATE_UNAVAILABLE; source unit EUR millions",
                }
            )
        return pd.DataFrame(rows, columns=RAW_COLUMNS).dropna(subset=["observation_date", "raw_value"])
    except Exception as exc:
        return empty_raw_row("ECB", "ECB Data Portal", url, ECB_TOTAL_ASSETS_SERIES_ID, "Euro Area", "ECB Total Assets", "weekly", "EUR", "EUR millions", "ERROR", str(exc))


def parse_ecb_week_period(value: str) -> pd.Timestamp | pd.NaT:
    match = re.search(r"(\d{4})-W(\d{2})", str(value))
    if not match:
        return pd.NaT
    try:
        monday = pd.to_datetime(f"{match.group(1)}-W{match.group(2)}-1", format="%G-W%V-%u")
        return monday + pd.Timedelta(days=4)
    except Exception:
        return pd.NaT


def boj_m2_raw() -> pd.DataFrame:
    url = str(GLOBAL_LIQUIDITY_CONFIG["boj_base_url"])
    try:
        response = get_with_retries(
            url,
            params={
                "format": "json",
                "lang": "en",
                "db": GLOBAL_LIQUIDITY_CONFIG["boj_m2_db"],
                "code": GLOBAL_LIQUIDITY_CONFIG["boj_m2_code"],
                "startDate": GLOBAL_LIQUIDITY_CONFIG["start_date"][:7].replace("-", ""),
            },
        )
        payload = response.json()
        if int(payload.get("STATUS", 0)) != 200:
            raise RuntimeError(str(payload.get("MESSAGE") or payload))
        resultset = payload.get("RESULTSET", []) or []
        if not resultset:
            raise RuntimeError("BOJ response has no RESULTSET")
        rows = []
        downloaded = fmt_datetime(now_utc())
        for series in resultset:
            values = series.get("VALUES", {}) or {}
            survey_dates = values.get("SURVEY_DATES", []) or []
            observations = values.get("VALUES", []) or []
            for period, value in zip(survey_dates, observations):
                period_str = str(period)
                rows.append(
                    {
                        "observation_date": pd.to_datetime(f"{period_str[:4]}-{period_str[4:6]}-01", errors="coerce"),
                        "release_date": pd.NaT,
                        "source": "BOJ",
                        "source_name": "Bank of Japan Time-Series Data Search",
                        "source_url": url,
                        "series_id": f"{GLOBAL_LIQUIDITY_CONFIG['boj_m2_db']}'{GLOBAL_LIQUIDITY_CONFIG['boj_m2_code']}",
                        "region": "Japan",
                        "metric": "M2",
                        "frequency": "monthly",
                        "currency": "JPY",
                        "unit": "JPY 100 million",
                        "raw_value": pd.to_numeric(value, errors="coerce"),
                        "download_timestamp": downloaded,
                        "data_status": "CURRENT",
                        "notes": "RELEASE_DATE_UNAVAILABLE",
                    }
                )
        return pd.DataFrame(rows, columns=RAW_COLUMNS).dropna(subset=["observation_date", "raw_value"])
    except Exception as exc:
        return empty_raw_row(
            "BOJ",
            "Bank of Japan Time-Series Data Search",
            url,
            f"{GLOBAL_LIQUIDITY_CONFIG['boj_m2_db']}'{GLOBAL_LIQUIDITY_CONFIG['boj_m2_code']}",
            "Japan",
            "M2",
            "monthly",
            "JPY",
            "JPY 100 million",
            "ERROR",
            str(exc),
        )


def boj_total_assets_raw() -> pd.DataFrame:
    db = str(GLOBAL_LIQUIDITY_CONFIG["boj_total_assets_db"])
    code = validate_boj_total_assets_code()
    url = str(GLOBAL_LIQUIDITY_CONFIG["boj_base_url"])
    try:
        rows = []
        start_position = None
        downloaded = fmt_datetime(now_utc())
        while True:
            params = {
                "format": "json",
                "lang": "en",
                "db": db,
                "code": code,
                "startDate": GLOBAL_LIQUIDITY_CONFIG["start_date"][:7].replace("-", ""),
            }
            if start_position:
                params["startPosition"] = str(start_position)
            response = get_with_retries(url, params=params)
            payload = response.json()
            if int(payload.get("STATUS", 0)) != 200:
                raise RuntimeError(str(payload.get("MESSAGE") or payload))
            for series in payload.get("RESULTSET", []) or []:
                name = str(series.get("NAME_OF_TIME_SERIES") or "")
                unit = str(series.get("UNIT") or "")
                if "Bank of Japan Accounts/Assets/Total" not in name:
                    raise RuntimeError(f"Unexpected BOJ total-assets series: {name}")
                values = series.get("VALUES", {}) or {}
                survey_dates = values.get("SURVEY_DATES", []) or []
                observations = values.get("VALUES", []) or []
                last_update = pd.to_datetime(str(series.get("LAST_UPDATE") or ""), format="%Y%m%d", errors="coerce")
                for period, value in zip(survey_dates, observations):
                    period_str = str(period)
                    rows.append(
                        {
                            "observation_date": pd.to_datetime(f"{period_str[:4]}-{period_str[4:6]}-01", errors="coerce"),
                            "release_date": last_update,
                            "source": "BOJ",
                            "source_name": "Bank of Japan Time-Series Data Search",
                            "source_url": url,
                            "series_id": f"{db}'{code}",
                            "region": "Japan",
                            "metric": "BoJ Total Assets",
                            "frequency": "monthly",
                            "currency": "JPY",
                            "unit": unit or "100 million yen",
                            "raw_value": pd.to_numeric(value, errors="coerce"),
                            "download_timestamp": downloaded,
                            "data_status": "CURRENT",
                            "notes": "Validated via BOJ metadata; source_unit=100 million yen; unit_multiplier_to_jpy_bn=0.1",
                        }
                    )
            start_position = payload.get("NEXTPOSITION")
            if not start_position:
                break
        return pd.DataFrame(rows, columns=RAW_COLUMNS).dropna(subset=["observation_date", "raw_value"])
    except Exception as exc:
        return empty_raw_row("BOJ", "Bank of Japan Time-Series Data Search", url, f"{db}'{code}", "Japan", "BoJ Total Assets", "monthly", "JPY", "100 million yen", "ERROR", str(exc))


def validate_boj_total_assets_code() -> str:
    db = str(GLOBAL_LIQUIDITY_CONFIG["boj_total_assets_db"])
    code = str(GLOBAL_LIQUIDITY_CONFIG["boj_total_assets_code"])
    response = get_with_retries(
        str(GLOBAL_LIQUIDITY_CONFIG["boj_metadata_url"]),
        params={"format": "json", "lang": "en", "db": db},
    )
    payload = response.json()
    for series in payload.get("RESULTSET", []) or []:
        if series.get("SERIES_CODE") != code:
            continue
        name = str(series.get("NAME_OF_TIME_SERIES") or "")
        unit = str(series.get("UNIT") or "")
        frequency = str(series.get("FREQUENCY") or "")
        if "Bank of Japan Accounts/Assets/Total" not in name:
            raise RuntimeError(f"BOJ code {code} failed name validation: {name}")
        if "financial institutions" in name.lower():
            raise RuntimeError(f"BOJ code {code} points to financial institutions, not BOJ balance sheet")
        if "100 million yen" not in unit.lower():
            raise RuntimeError(f"BOJ code {code} unexpected unit: {unit}")
        if frequency.upper() != "MONTHLY":
            raise RuntimeError(f"BOJ code {code} unexpected frequency: {frequency}")
        return code
    raise RuntimeError(f"BOJ code {code} not found in metadata")


def tradingview_mcp_economic_raw(
    symbol: str,
    series_id: str,
    region: str,
    metric: str,
    expected: str,
    source_url_key: str,
    min_observations: int = 24,
) -> pd.DataFrame:
    from tradingview_mcp import (
        get_economic_data,
        normalize_cny_to_100mn,
        validate_economic_result,
    )

    result = get_economic_data(
        symbol,
        date_from=str(GLOBAL_LIQUIDITY_CONFIG["start_date"]),
        date_to=now_utc().strftime("%Y-%m-%d"),
    )
    valid, status = validate_economic_result(result, min_observations=min_observations)
    if not valid:
        raise RuntimeError(f"{symbol} validation failed: {status}")

    rows = []
    downloaded = fmt_datetime(now_utc())
    unit_notes: list[str] = []
    for _, item in result.frame.iterrows():
        raw_value, unit_note = normalize_cny_to_100mn(
            float(item["value"]),
            unit=result.unit,
            scale=result.scale,
            expected=expected,
        )
        if not np.isfinite(raw_value):
            raise RuntimeError(f"{symbol} unit validation failed: {unit_note}")
        unit_notes.append(unit_note)
        release_date = item.get("release_date", pd.NaT) if isinstance(item, pd.Series) else pd.NaT
        rows.append(
            {
                "observation_date": pd.to_datetime(item["date"], errors="coerce"),
                "release_date": pd.to_datetime(release_date, errors="coerce"),
                "source": "TRADINGVIEW_MCP",
                "source_name": f"TradingView MCP / {symbol}",
                "source_mode": "MCP_PRIMARY",
                "source_url": str(GLOBAL_LIQUIDITY_CONFIG[source_url_key]),
                "series_id": series_id,
                "region": region,
                "metric": metric,
                "frequency": result.frequency or "monthly",
                "currency": "CNY",
                "unit": "CNY 100 million",
                "raw_value": raw_value,
                "download_timestamp": downloaded,
                "data_status": "OK",
                "notes": (
                    f"{result.description or symbol}; {unit_note}; "
                    f"{result.notes}; point_in_time="
                    f"{'PARTIAL' if pd.notna(pd.to_datetime(release_date, errors='coerce')) else 'RELEASE_DATE_UNKNOWN'}"
                ),
            }
        )

    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    frame["release_date"] = pd.to_datetime(frame["release_date"], errors="coerce")
    frame["raw_value"] = pd.to_numeric(frame["raw_value"], errors="coerce")
    frame = frame.dropna(subset=["observation_date", "raw_value"])
    if frame.empty:
        raise RuntimeError(f"{symbol} returned no normalized observations")
    return frame.sort_values(["observation_date", "download_timestamp"]).drop_duplicates(
        subset=["observation_date", "series_id"],
        keep="last",
    )[RAW_COLUMNS]


def tradingview_mcp_china_m2_raw() -> pd.DataFrame:
    return tradingview_mcp_economic_raw(
        "ECONOMICS:CNM2",
        CHINA_M2_SERIES_ID,
        "China",
        "M2",
        "cnm2",
        "tradingview_cnm2_reports_url",
        min_observations=36,
    )


def tradingview_mcp_pboc_total_assets_raw() -> pd.DataFrame:
    return tradingview_mcp_economic_raw(
        "ECONOMICS:CNCBBS",
        PBOC_TOTAL_ASSETS_SERIES_ID,
        "China",
        "PBoC Total Assets",
        "cncbbs",
        "tradingview_cncbbs_url",
        min_observations=36,
    )


def pboc_m2_raw(api_key: str | None = None) -> pd.DataFrame:
    source_url = str(GLOBAL_LIQUIDITY_CONFIG["pboc_money_supply_url"])
    mcp_error = ""
    try:
        investing_frame = china_m2_investing_update_raw(api_key=api_key)
        if not investing_frame.empty:
            investing_frame["source_mode"] = "FALLBACK_SOURCE"
            return investing_frame[RAW_COLUMNS]
    except Exception as exc:
        investing_error = str(exc)
    else:
        investing_error = ""
    try:
        mcp_frame = tradingview_mcp_china_m2_raw()
        if not mcp_frame.empty:
            return mcp_frame
    except Exception as exc:
        mcp_error = str(exc)
    try:
        links = discover_pboc_money_supply_links(source_url)
        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        for link in links[:80]:
            try:
                rows.extend(parse_pboc_money_supply_page(link))
            except Exception as exc:
                errors.append(f"{link}: {exc}")
        if not rows:
            details = "; ".join(errors[:3]) if errors else "no candidate money supply pages found"
            raise RuntimeError(details)
        frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
        frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
        frame["raw_value"] = pd.to_numeric(frame["raw_value"], errors="coerce")
        frame = frame.dropna(subset=["observation_date", "raw_value"])
        if frame.empty:
            raise RuntimeError("PBoC parser found pages but no numeric M2 observations")
        frame = frame.sort_values(["observation_date", "download_timestamp"]).drop_duplicates(
            subset=["observation_date", "series_id"],
            keep="last",
        )
        if len(frame) < 36:
            raise RuntimeError(f"PBoC parser returned partial M2 history: {len(frame)} observations")
        return frame[RAW_COLUMNS]
    except Exception as exc:
        fallback = china_m2_fred_tradingview_fallback_raw(api_key=api_key, official_error=str(exc))
        if not fallback.empty:
            if "notes" in fallback.columns and mcp_error:
                fallback["notes"] = fallback["notes"].astype(str) + f"; TradingView MCP primary unavailable: {mcp_error}; Investing fallback unavailable: {investing_error}"
            if "source_mode" in fallback.columns:
                fallback["source_mode"] = "FALLBACK_SOURCE"
            return fallback
        return empty_raw_row(
            "PBOC",
            "People's Bank of China",
            source_url,
            CHINA_M2_SERIES_ID,
            "China",
            "M2",
            "monthly",
            "CNY",
            "CNY 100 million",
            "ERROR",
            f"TradingView MCP primary unavailable: {mcp_error}; Investing fallback unavailable: {investing_error}; Official PBoC parser unavailable: {exc}. Missing data is not filled with zero.",
        )


def china_m2_fred_tradingview_fallback_raw(api_key: str | None = None, official_error: str = "") -> pd.DataFrame:
    try:
        fred_frame = download_fred_series_batch(
            (FRED_CHINA_M2_LEGACY_SERIES,),
            api_key=api_key,
            observation_start=GLOBAL_LIQUIDITY_CONFIG["start_date"],
        )
        fred_frame = fred_frame.dropna(subset=["Date", "Value"]).copy()
        fred_frame["Date"] = pd.to_datetime(fred_frame["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        fred_frame["Value"] = pd.to_numeric(fred_frame["Value"], errors="coerce") / 100_000_000.0
        fred_frame = fred_frame.dropna(subset=["Date", "Value"]).sort_values("Date")
        if fred_frame.empty:
            raise RuntimeError("FRED legacy China M2 returned no observations")

        yoy = tradingview_china_m2_yoy_raw(
            start_date=fred_frame["Date"].max() + pd.DateOffset(days=1),
            end_date=now_utc() + pd.DateOffset(days=45),
        )
        reconstructed = reconstruct_china_m2_from_yoy(fred_frame, yoy)
        downloaded = fmt_datetime(now_utc())
        rows: list[dict[str, Any]] = []
        for _, row in fred_frame.iterrows():
            rows.append(
                {
                    "observation_date": row["Date"],
                    "release_date": pd.NaT,
                    "source": "FRED_IMF",
                    "source_name": "FRED / IMF International Financial Statistics",
                    "source_url": f"https://fred.stlouisfed.org/series/{FRED_CHINA_M2_LEGACY_SERIES}",
                    "series_id": CHINA_M2_SERIES_ID,
                    "region": "China",
                    "metric": "M2",
                    "frequency": "monthly",
                    "currency": "CNY",
                    "unit": "CNY 100 million",
                    "raw_value": float(row["Value"]),
                    "download_timestamp": downloaded,
                    "data_status": "FALLBACK_SOURCE",
                    "notes": f"Legacy FRED IMF level series {FRED_CHINA_M2_LEGACY_SERIES}; PBoC parser unavailable: {official_error}",
                }
            )
        for _, row in reconstructed.iterrows():
            rows.append(
                {
                    "observation_date": row["observation_date"],
                    "release_date": row["release_date"],
                    "source": "FRED_IMF+TRADINGVIEW",
                    "source_name": "FRED IMF legacy level + TradingView CNM2 YoY",
                    "source_url": str(GLOBAL_LIQUIDITY_CONFIG["tradingview_cnm2_reports_url"]),
                    "series_id": CHINA_M2_SERIES_ID,
                    "region": "China",
                    "metric": "M2",
                    "frequency": "monthly",
                    "currency": "CNY",
                    "unit": "CNY 100 million",
                    "raw_value": float(row["raw_value"]),
                    "download_timestamp": downloaded,
                    "data_status": "FALLBACK_SOURCE",
                    "notes": (
                        f"Reconstructed from {FRED_CHINA_M2_LEGACY_SERIES} base and TradingView ECONOMICS:CNM2 YoY; "
                        f"YoY={row['yoy_pct']:.3f}%; PBoC parser unavailable: {official_error}"
                    ),
                }
            )
        frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
        return frame.sort_values("observation_date").drop_duplicates(["observation_date", "series_id"], keep="last")[RAW_COLUMNS]
    except Exception as exc:
        return empty_raw_row(
            "FRED_IMF+TRADINGVIEW",
            "FRED IMF legacy level + TradingView CNM2 YoY",
            str(GLOBAL_LIQUIDITY_CONFIG["tradingview_cnm2_reports_url"]),
            CHINA_M2_SERIES_ID,
            "China",
            "M2",
            "monthly",
            "CNY",
            "CNY 100 million",
            "ERROR",
            f"China M2 fallback failed: {exc}; PBoC parser unavailable: {official_error}",
        )


def tradingview_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0",
        "Origin": "https://www.tradingview.com",
        "Referer": "https://www.tradingview.com/economic-calendar/",
    }


def investing_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.investing.com/",
    }


def _parse_investing_release_html(html: str) -> pd.DataFrame:
    """Parse the visible Investing release table without requiring future rows."""
    month_names: dict[str, int] = {}
    for number in range(1, 13):
        month = pd.Timestamp(year=2000, month=number, day=1)
        month_names[month.month_name().lower()] = number
        month_names[month.strftime("%b").lower()] = number
    embedded_rows: list[dict[str, Any]] = []
    occurrences_start = html.find('"occurrences":[')
    occurrences_end = html.find('],', occurrences_start) if occurrences_start >= 0 else -1
    occurrence_payload = html[occurrences_start:occurrences_end] if occurrences_start >= 0 and occurrences_end > occurrences_start else html
    embedded_pattern = re.compile(
        r'\{"actual":(?P<actual>[-+]?\d+(?:\.\d+)?).*?"occurrence_time":"(?P<release>[^\"]+)".*?"reference_period":"(?P<reference>[A-Za-z]{3,9})"',
        flags=re.DOTALL,
    )
    for match in embedded_pattern.finditer(occurrence_payload):
        release_date = pd.to_datetime(match.group("release"), errors="coerce")
        reference_month = month_names.get(match.group("reference").lower())
        if pd.isna(release_date) or reference_month is None:
            continue
        reference_year = int(release_date.year) - int(reference_month > release_date.month)
        embedded_rows.append(
            {
                "observation_date": pd.Timestamp(reference_year, reference_month, 1),
                "release_date": pd.Timestamp(release_date).tz_localize(None).normalize(),
                "actual": float(match.group("actual")),
                "forecast": np.nan,
                "previous": np.nan,
            }
        )
    if embedded_rows:
        return pd.DataFrame(embedded_rows).sort_values(["observation_date", "release_date"]).drop_duplicates("observation_date", keep="last").reset_index(drop=True)
    try:
        tables = pd.read_html(StringIO(html), flavor="lxml")
    except Exception as exc:
        raise RuntimeError(f"Investing release table parse failed: {exc}") from exc
    matching_tables: list[pd.DataFrame] = []
    for candidate in tables:
        columns = [str(value).strip().lower() for value in candidate.columns]
        if all(any(token in column for column in columns) for token in ["release date", "actual", "previous"]):
            matching_tables.append(candidate)
    if not matching_tables:
        raise RuntimeError("Investing release table not found")
    table = max(matching_tables, key=len)
    data = table.iloc[:, :5].copy()
    data.columns = ["release_date", "release_time", "actual", "forecast", "previous"]
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        date_text = str(row.get("release_date") or "")
        match = re.search(r"([A-Za-z]{3,9}\s+\d{1,2},\s+20\d{2})", date_text)
        if not match:
            continue
        release_date = pd.to_datetime(match.group(1), errors="coerce")
        ref_match = re.search(r"\(([A-Za-z]{3,9})\)", date_text)
        if pd.isna(release_date) or not ref_match:
            continue
        reference_month = month_names.get(ref_match.group(1).lower())
        if reference_month is None:
            continue
        reference_year = int(release_date.year) - int(reference_month > release_date.month)
        actual_text = str(row.get("actual") or "").replace(",", "").strip()
        actual_match = re.search(r"[-+]?\d+(?:\.\d+)?", actual_text)
        actual = float(actual_match.group(0)) if actual_match else np.nan
        if not np.isfinite(actual):
            continue
        rows.append(
            {
                "observation_date": pd.Timestamp(reference_year, reference_month, 1),
                "release_date": pd.Timestamp(release_date).normalize(),
                "actual": actual,
                "forecast": pd.to_numeric(str(row.get("forecast") or "").replace(",", ""), errors="coerce"),
                "previous": pd.to_numeric(str(row.get("previous") or "").replace(",", ""), errors="coerce"),
            }
        )
    if not rows:
        raise RuntimeError("Investing release table contains no completed numeric observations")
    return pd.DataFrame(rows).sort_values(["observation_date", "release_date"]).drop_duplicates("observation_date", keep="last").reset_index(drop=True)


def investing_china_m2_yoy_raw(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    url = str(GLOBAL_LIQUIDITY_CONFIG["investing_china_m2_url"])
    response = get_with_retries(url, headers=investing_headers())
    parsed = _parse_investing_release_html(response.text)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    parsed = parsed.loc[parsed["observation_date"].between(start.to_period("M").to_timestamp(), end.to_period("M").to_timestamp())].copy()
    if parsed.empty:
        return pd.DataFrame(columns=["observation_date", "release_date", "series_id", "yoy_pct"])
    return parsed.rename(columns={"actual": "yoy_pct"})[["observation_date", "release_date", "yoy_pct"]].assign(series_id=TRADINGVIEW_CNM2_YOY_SERIES)[["observation_date", "release_date", "series_id", "yoy_pct"]]


def _existing_raw_series_frame(series_id: str) -> pd.DataFrame:
    if not RAW_STORAGE_PATH.exists():
        return pd.DataFrame(columns=RAW_COLUMNS)
    try:
        frame = pd.read_csv(RAW_STORAGE_PATH)
    except Exception:
        return pd.DataFrame(columns=RAW_COLUMNS)
    for column in RAW_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    frame["raw_value"] = pd.to_numeric(frame["raw_value"], errors="coerce")
    return frame.loc[frame["series_id"].astype(str).eq(series_id)].dropna(subset=["observation_date", "raw_value"])[RAW_COLUMNS].copy()


def china_m2_investing_update_raw(api_key: str | None = None) -> pd.DataFrame:
    """Keep the saved CNM2 levels and extend them from Investing YoY releases."""
    existing = _existing_raw_series_frame(CHINA_M2_SERIES_ID)
    if not existing.empty:
        base = existing[["observation_date", "raw_value"]].rename(columns={"observation_date": "Date", "raw_value": "Value"})
        base["Date"] = pd.to_datetime(base["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        base = base.sort_values("Date").drop_duplicates("Date", keep="last")
    else:
        fred = download_fred_series_batch(
            (FRED_CHINA_M2_LEGACY_SERIES,),
            api_key=api_key,
            observation_start=GLOBAL_LIQUIDITY_CONFIG["start_date"],
        )
        base = fred.dropna(subset=["Date", "Value"])[["Date", "Value"]].copy()
        base["Date"] = pd.to_datetime(base["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        base["Value"] = pd.to_numeric(base["Value"], errors="coerce") / 100_000_000.0
        base = base.dropna(subset=["Date", "Value"]).sort_values("Date").drop_duplicates("Date", keep="last")
    if base.empty:
        raise RuntimeError("No saved CNM2 level history or FRED base is available")
    yoy = investing_china_m2_yoy_raw(base["Date"].max() - pd.DateOffset(months=12), now_utc() + pd.DateOffset(days=45))
    reconstructed = reconstruct_china_m2_from_yoy(base, yoy, overwrite_existing=True)
    downloaded = fmt_datetime(now_utc())
    rows: list[dict[str, Any]] = []
    if existing.empty:
        for _, row in base.iterrows():
            rows.append(
                {
                    "observation_date": row["Date"],
                    "release_date": pd.NaT,
                    "source": "FRED_IMF",
                    "source_name": "FRED / IMF International Financial Statistics",
                    "source_mode": "FALLBACK_SOURCE",
                    "source_url": f"https://fred.stlouisfed.org/series/{FRED_CHINA_M2_LEGACY_SERIES}",
                    "series_id": CHINA_M2_SERIES_ID,
                    "region": "China",
                    "metric": "M2",
                    "frequency": "monthly",
                    "currency": "CNY",
                    "unit": "CNY 100 million",
                    "raw_value": float(row["Value"]),
                    "download_timestamp": downloaded,
                    "data_status": "FALLBACK_SOURCE",
                    "notes": "Preserved FRED IMF base; updated from Investing China M2 YoY releases.",
                }
            )
    for _, row in reconstructed.iterrows():
        rows.append(
            {
                "observation_date": row["observation_date"],
                "release_date": row["release_date"],
                "source": "INVESTING_COM",
                "source_name": "Investing.com / China M2 Money Stock YoY",
                "source_mode": "FALLBACK_SOURCE",
                "source_url": str(GLOBAL_LIQUIDITY_CONFIG["investing_china_m2_url"]),
                "series_id": CHINA_M2_SERIES_ID,
                "region": "China",
                "metric": "M2",
                "frequency": "monthly",
                "currency": "CNY",
                "unit": "CNY 100 million",
                "raw_value": float(row["raw_value"]),
                "download_timestamp": downloaded,
                "data_status": "CURRENT",
                "notes": f"Absolute CNY level reconstructed from prior-year level and Investing YoY={row['yoy_pct']:.3f}%; converted to USD downstream using USDCNY.",
            }
        )
    incoming = pd.DataFrame(rows, columns=RAW_COLUMNS)
    combined = pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming
    combined["observation_date"] = pd.to_datetime(combined["observation_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    combined["raw_value"] = pd.to_numeric(combined["raw_value"], errors="coerce")
    combined["_priority"] = np.where(combined["source"].eq("INVESTING_COM"), 2, 1)
    return combined.dropna(subset=["observation_date", "raw_value"]).sort_values(["observation_date", "_priority", "download_timestamp"]).drop_duplicates(["observation_date", "series_id"], keep="last")[RAW_COLUMNS].reset_index(drop=True)


def tradingview_china_m2_yoy_raw(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    url = str(GLOBAL_LIQUIDITY_CONFIG["tradingview_calendar_url"])
    rows: list[dict[str, Any]] = []
    current = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    while current <= end:
        window_end = min(current + pd.DateOffset(years=1) - pd.DateOffset(days=1), end)
        response = get_with_retries(
            url,
            params={
                "from": current.strftime("%Y-%m-%dT00:00:00.000Z"),
                "to": window_end.strftime("%Y-%m-%dT23:59:59.000Z"),
                "countries": "CN",
            },
            headers=tradingview_headers(),
        )
        payload = response.json()
        for event in payload.get("result", []) or []:
            if event.get("ticker") != "ECONOMICS:CNM2":
                continue
            reference_date = pd.to_datetime(event.get("referenceDate"), errors="coerce")
            release_date = pd.to_datetime(event.get("date"), errors="coerce")
            value = pd.to_numeric(event.get("actualRaw", event.get("actual")), errors="coerce")
            if pd.isna(reference_date) or pd.isna(value):
                continue
            rows.append(
                {
                    "observation_date": reference_date.tz_localize(None).to_period("M").to_timestamp(),
                    "release_date": release_date.tz_localize(None) if pd.notna(release_date) else pd.NaT,
                    "series_id": TRADINGVIEW_CNM2_YOY_SERIES,
                    "yoy_pct": float(value),
                }
            )
        current = window_end + pd.DateOffset(days=1)
    if not rows:
        return pd.DataFrame(columns=["observation_date", "release_date", "series_id", "yoy_pct"])
    return (
        pd.DataFrame(rows)
        .sort_values(["observation_date", "release_date"])
        .drop_duplicates("observation_date", keep="last")
        .reset_index(drop=True)
    )


def tradingview_pboc_total_assets_latest_raw() -> pd.DataFrame:
    page_url = str(GLOBAL_LIQUIDITY_CONFIG["tradingview_cncbbs_url"])
    scanner_url = str(GLOBAL_LIQUIDITY_CONFIG["tradingview_scanner_url"])
    page_response = get_with_retries(page_url, headers=tradingview_headers())
    observation_date = parse_tradingview_observation_month(page_response.text)
    if observation_date is None:
        raise RuntimeError("TradingView CNCBBS page does not contain Observation period")

    payload = {
        "symbols": {"tickers": [TRADINGVIEW_PBOC_TOTAL_ASSETS_SERIES], "query": {"types": []}},
        "columns": ["close", "currency"],
    }
    scanner_response = post_json_with_retries(scanner_url, payload, headers=tradingview_headers())
    scanner_payload = scanner_response.json()
    rows = scanner_payload.get("data", []) or []
    row = next((item for item in rows if item.get("s") == TRADINGVIEW_PBOC_TOTAL_ASSETS_SERIES), None)
    if not row:
        raise RuntimeError("TradingView scanner returned no CNCBBS row")
    data = row.get("d", []) or []
    if len(data) < 2:
        raise RuntimeError("TradingView CNCBBS row does not contain close/currency")
    close_cny = pd.to_numeric(data[0], errors="coerce")
    currency = str(data[1] or "")
    if not np.isfinite(close_cny):
        raise RuntimeError("TradingView CNCBBS close is not numeric")
    if currency.upper() != "CNY":
        raise RuntimeError(f"TradingView CNCBBS currency is {currency}, expected CNY")
    raw_value = float(close_cny) / 100_000_000.0
    return pd.DataFrame(
        [
            {
                "observation_date": observation_date,
                "release_date": pd.NaT,
                "source": "TRADINGVIEW",
                "source_name": "TradingView ECONOMICS:CNCBBS",
                "source_url": page_url,
                "series_id": PBOC_TOTAL_ASSETS_SERIES_ID,
                "region": "China",
                "metric": "PBoC Total Assets",
                "frequency": "monthly",
                "currency": "CNY",
                "unit": "CNY 100 million",
                "raw_value": raw_value,
                "download_timestamp": fmt_datetime(now_utc()),
                "data_status": "CURRENT",
                "notes": f"TradingView {TRADINGVIEW_PBOC_TOTAL_ASSETS_SERIES} close converted from CNY to CNY 100 million",
            }
        ],
        columns=RAW_COLUMNS,
    )


def tradingeconomics_pboc_total_assets_latest_raw() -> pd.DataFrame:
    """Read the latest Trading Economics value, already quoted in CNY Hundred Million."""
    page_url = str(GLOBAL_LIQUIDITY_CONFIG["tradingeconomics_cncbbs_url"])
    response = get_with_retries(page_url, headers=investing_headers())
    text = clean_text(BeautifulSoup(response.text, "html.parser").get_text(" ") if BeautifulSoup is not None else response.text)
    match = re.search(
        r"Banks Balance Sheet in China .*? to\s+([\d,.]+)\s+CNY Hundred Million in\s+([A-Za-z]+)\s+from .*?of\s+(20\d{2})",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        raise RuntimeError("Trading Economics CNCBBS page does not contain the latest value/reference month")
    source_value = pd.to_numeric(match.group(1).replace(",", ""), errors="coerce")
    observation_date = pd.to_datetime(f"1 {match.group(2)} {match.group(3)}", errors="coerce")
    if not np.isfinite(source_value) or pd.isna(observation_date):
        raise RuntimeError("Trading Economics CNCBBS latest value is not numeric")
    # Trading Economics labels the observation CNY Hundred Million.  The
    # published CNCBBS level is ten times larger than the historical series
    # basis used by the application, so normalize before storing the raw level.
    raw_value = float(source_value) / 10.0
    return pd.DataFrame(
        [
            {
                "observation_date": pd.Timestamp(observation_date).to_period("M").to_timestamp(),
                "release_date": pd.NaT,
                "source": "TRADINGECONOMICS",
                "source_name": "Trading Economics / China Banks Balance Sheet",
                "source_mode": "FALLBACK_SOURCE",
                "source_url": page_url,
                "series_id": PBOC_TOTAL_ASSETS_SERIES_ID,
                "region": "China",
                "metric": "PBoC Total Assets",
                "frequency": "monthly",
                "currency": "CNY",
                "unit": "CNY 100 million",
                "raw_value": raw_value,
                "download_timestamp": fmt_datetime(now_utc()),
                "data_status": "CURRENT",
                "notes": f"Trading Economics source unit is CNY Hundred Million; source value {float(source_value):.2f} normalized by /10 to the historical CNCBBS level basis.",
            }
        ],
        columns=RAW_COLUMNS,
    )


def parse_tradingview_observation_month(html: str) -> pd.Timestamp | None:
    text = clean_text(BeautifulSoup(html, "html.parser").get_text(" ") if BeautifulSoup is not None else html)
    match = re.search(r"Observation period\s+([A-Za-z]{3,9})\s+(\d{4})", text)
    if not match:
        return None
    parsed = pd.to_datetime(f"{match.group(1)} 1 {match.group(2)}", errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).to_period("M").to_timestamp()


def reconstruct_china_m2_from_yoy(fred_frame: pd.DataFrame, yoy: pd.DataFrame, overwrite_existing: bool = False) -> pd.DataFrame:
    if yoy.empty:
        return pd.DataFrame(columns=["observation_date", "release_date", "raw_value", "yoy_pct"])
    values = fred_frame.set_index("Date")["Value"].sort_index().astype(float).to_dict()
    latest_fred_date = max(values)
    rows = []
    for _, row in yoy.sort_values("observation_date").iterrows():
        observation_date = pd.Timestamp(row["observation_date"])
        if observation_date <= latest_fred_date and not overwrite_existing:
            continue
        base_date = observation_date - pd.DateOffset(years=1)
        base_value = values.get(base_date)
        yoy_pct = float(row["yoy_pct"])
        if base_value is None or not np.isfinite(base_value) or not np.isfinite(yoy_pct):
            continue
        raw_value = float(base_value * (1.0 + yoy_pct / 100.0))
        values[observation_date] = raw_value
        rows.append(
            {
                "observation_date": observation_date,
                "release_date": row.get("release_date", pd.NaT),
                "raw_value": raw_value,
                "yoy_pct": yoy_pct,
            }
        )
    return pd.DataFrame(rows, columns=["observation_date", "release_date", "raw_value", "yoy_pct"])


def discover_pboc_money_supply_links(index_url: str) -> list[str]:
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 is not installed")
    response = get_with_retries(index_url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = [index_url]
    for anchor in soup.find_all("a"):
        href = str(anchor.get("href") or "").strip()
        text = clean_text(anchor.get_text(" "))
        if not href:
            continue
        if "货币供应量" not in text and "Money Supply" not in text and "M2" not in text:
            continue
        links.append(urljoin(index_url, href))
    seen: set[str] = set()
    unique: list[str] = []
    for link in links:
        if link not in seen:
            unique.append(link)
            seen.add(link)
    return unique


def parse_pboc_money_supply_page(url: str) -> list[dict[str, Any]]:
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 is not installed")
    response = get_with_retries(url)
    soup = BeautifulSoup(response.text, "html.parser")
    release_date = extract_pboc_release_date(clean_text(soup.get_text(" ")))
    downloaded = fmt_datetime(now_utc())
    rows: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        parsed_rows = [
            [clean_text(cell.get_text(" ")) for cell in tr.find_all(["td", "th"])]
            for tr in table.find_all("tr")
        ]
        parsed_rows = [row for row in parsed_rows if any(row)]
        date_headers: list[pd.Timestamp] = []
        for row in parsed_rows:
            row_dates = [parse_pboc_period(cell) for cell in row]
            valid_dates = [date for date in row_dates if date is not None]
            if len(valid_dates) >= 2:
                date_headers = valid_dates
            row_text = " ".join(row)
            is_m2_row = "货币和准货币" in row_text or "Money & Quasi-money" in row_text
            if not is_m2_row or not date_headers:
                continue
            values = [
                parse_pboc_number(cell)
                for cell in row
                if "货币和准货币" not in cell and "Money & Quasi-money" not in cell and "M2" not in cell
            ]
            numeric_values = [value for value in values if np.isfinite(value)]
            for date, value in zip(date_headers, numeric_values):
                rows.append(
                    {
                        "observation_date": date,
                        "release_date": release_date,
                        "source": "PBOC",
                        "source_name": "People's Bank of China",
                        "source_url": url,
                        "series_id": "Money & Quasi-money (M2)",
                        "region": "China",
                        "metric": "M2",
                        "frequency": "monthly",
                        "currency": "CNY",
                        "unit": "CNY 100 million",
                        "raw_value": value,
                        "download_timestamp": downloaded,
                        "data_status": "CURRENT",
                        "notes": "Official PBoC Money Supply table",
                    }
                )
    return rows


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def parse_pboc_period(value: str) -> pd.Timestamp | None:
    text = clean_text(value)
    match = re.search(r"(20\d{2})\s*[.\-/年]\s*(\d{1,2})", text)
    if not match:
        return None
    month = int(match.group(2))
    if month < 1 or month > 12:
        return None
    return pd.Timestamp(year=int(match.group(1)), month=month, day=1)


def parse_pboc_number(value: str) -> float:
    text = clean_text(value).replace(",", "")
    if "%" in text:
        return math.nan
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else math.nan


def extract_pboc_release_date(text: str) -> pd.Timestamp | pd.NaT:
    match = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if not match:
        return pd.NaT
    try:
        return pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)), day=int(match.group(3)))
    except ValueError:
        return pd.NaT


def pboc_total_assets_table_paths() -> list[Path]:
    paths: list[Path] = []
    env_path = os.environ.get("PBOC_TOTAL_ASSETS_TABLE_PATH")
    if env_path:
        paths.append(Path(env_path))
    paths.extend([PBOC_TOTAL_ASSETS_STORAGE_PATH, PBOC_TOTAL_ASSETS_BUNDLED_PATH])
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def pboc_total_assets_table_raw(path: Path | None = None) -> pd.DataFrame:
    table_path = path
    if table_path is None:
        frames: list[pd.DataFrame] = []
        existing_paths = [candidate for candidate in reversed(pboc_total_assets_table_paths()) if candidate.exists()]
        for priority, candidate in enumerate(existing_paths):
            frame = pboc_total_assets_table_raw(candidate)
            if not frame.empty:
                frame = frame.copy()
                frame["_table_priority"] = priority
                frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=RAW_COLUMNS)
        combined = pd.concat(frames, ignore_index=True)
        combined["observation_date"] = pd.to_datetime(combined["observation_date"], errors="coerce")
        combined["raw_value"] = pd.to_numeric(combined["raw_value"], errors="coerce")
        return (
            combined.dropna(subset=["observation_date", "raw_value"])
            .sort_values(["observation_date", "_table_priority", "download_timestamp"])
            .drop_duplicates(subset=["observation_date", "series_id"], keep="last")[RAW_COLUMNS]
            .reset_index(drop=True)
        )
    if table_path is None or not table_path.exists():
        return pd.DataFrame(columns=RAW_COLUMNS)

    table = pd.read_csv(table_path)
    if "observation_date" not in table.columns:
        raise RuntimeError(f"{table_path} must contain observation_date")
    table["observation_date"] = pd.to_datetime(table["observation_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if "raw_value_cny_100mn" in table.columns:
        table["raw_value"] = pd.to_numeric(table["raw_value_cny_100mn"], errors="coerce")
    elif "pboc_total_assets_cny_trn" in table.columns:
        table["raw_value"] = pd.to_numeric(table["pboc_total_assets_cny_trn"], errors="coerce") * 10_000.0
    elif "raw_value" in table.columns:
        table["raw_value"] = pd.to_numeric(table["raw_value"], errors="coerce")
    else:
        raise RuntimeError(f"{table_path} must contain raw_value_cny_100mn or pboc_total_assets_cny_trn")

    table = table.dropna(subset=["observation_date", "raw_value"]).copy()
    if table.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)
    table = table.sort_values("observation_date").drop_duplicates(subset=["observation_date"], keep="last")
    downloaded = fmt_datetime(now_utc())

    def text_column(name: str, default: str) -> pd.Series:
        if name in table.columns:
            return table[name].fillna(default)
        return pd.Series(default, index=table.index)

    source_name = text_column("source_name", "PBoC Total Assets local table")
    source_url = text_column("source_url", str(table_path))
    source_note = text_column("source_note", "")
    notes = text_column("notes", "Monthly PBoC Total Assets table")
    rows: list[dict[str, Any]] = []
    for idx, row in table.iterrows():
        row_notes = "; ".join(part for part in [str(source_note.loc[idx]).strip(), str(notes.loc[idx]).strip()] if part)
        rows.append(
            {
                "observation_date": row["observation_date"],
                "release_date": pd.NaT,
                "source": "PBOC_LOCAL_TABLE",
                "source_name": str(source_name.loc[idx]) or "PBoC Total Assets local table",
                "source_url": str(source_url.loc[idx]) or str(table_path),
                "series_id": PBOC_TOTAL_ASSETS_SERIES_ID,
                "region": "China",
                "metric": "PBoC Total Assets",
                "frequency": "monthly",
                "currency": "CNY",
                "unit": "CNY 100 million",
                "raw_value": float(row["raw_value"]),
                "download_timestamp": downloaded,
                "data_status": "CURRENT",
                "notes": row_notes,
            }
        )
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame["data_status"] = np.where(
        pd.to_numeric(frame["raw_value"], errors="coerce").pct_change().abs().gt(0.25),
        "OUTLIER_CHECK",
        frame["data_status"],
    )
    return frame[RAW_COLUMNS]


def write_pboc_total_assets_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    try:
        table = frame.dropna(subset=["observation_date", "raw_value"]).copy()
        if table.empty:
            return
        table["observation_date"] = pd.to_datetime(table["observation_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        table["raw_value"] = pd.to_numeric(table["raw_value"], errors="coerce")
        table = table.dropna(subset=["observation_date", "raw_value"])
        if table.empty:
            return
        table = table.sort_values(["observation_date", "download_timestamp"]).drop_duplicates(
            subset=["observation_date", "series_id"],
            keep="last",
        )
        output = pd.DataFrame(
            {
                "observation_date": table["observation_date"].dt.strftime("%Y-%m-%d"),
                "pboc_total_assets_cny_trn": table["raw_value"] / 10_000.0,
                "raw_value_cny_100mn": table["raw_value"],
                "source_name": table["source_name"],
                "source_url": table["source_url"],
                "unit": "CNY trillion",
                "source_note": table["source"],
                "notes": table["notes"],
            }
        )
        PBOC_TOTAL_ASSETS_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(PBOC_TOTAL_ASSETS_STORAGE_PATH, index=False)
    except Exception:
        return


def pboc_total_assets_raw() -> pd.DataFrame:
    source_url = str(GLOBAL_LIQUIDITY_CONFIG["pboc_balance_sheet_url"])
    mcp_error = ""

    local_frames: list[pd.DataFrame] = []
    try:
        local_frames.append(pboc_total_assets_table_raw())
    except Exception:
        local_frames.append(pd.DataFrame(columns=RAW_COLUMNS))
    try:
        local_frames.append(tradingeconomics_pboc_total_assets_latest_raw())
    except Exception:
        local_frames.append(pd.DataFrame(columns=RAW_COLUMNS))
    try:
        local_frames.append(tradingview_pboc_total_assets_latest_raw())
    except Exception:
        local_frames.append(pd.DataFrame(columns=RAW_COLUMNS))

    local_and_tradingview = pd.concat(local_frames, ignore_index=True)
    if not local_and_tradingview.empty:
        local_and_tradingview["observation_date"] = pd.to_datetime(local_and_tradingview["observation_date"], errors="coerce")
        local_and_tradingview["raw_value"] = pd.to_numeric(local_and_tradingview["raw_value"], errors="coerce")
        local_and_tradingview = local_and_tradingview.dropna(subset=["observation_date", "raw_value"])
        if not local_and_tradingview.empty:
            source_priority = {"PBOC_LOCAL_TABLE": 0, "TRADINGVIEW": 1, "TRADINGECONOMICS": 2}
            local_and_tradingview["_source_priority"] = local_and_tradingview["source"].map(source_priority).fillna(0)
            frame = (
                local_and_tradingview.sort_values(["observation_date", "_source_priority", "download_timestamp"])
                .drop_duplicates(subset=["observation_date", "series_id"], keep="last")[RAW_COLUMNS]
                .reset_index(drop=True)
            )
            if "source_mode" in frame.columns:
                frame["source_mode"] = "FALLBACK_SOURCE"
            if "notes" in frame.columns and mcp_error:
                frame["notes"] = frame["notes"].astype(str) + f"; TradingView MCP primary unavailable: {mcp_error}"
            write_pboc_total_assets_table(frame)
            return frame

    try:
        mcp_frame = tradingview_mcp_pboc_total_assets_raw()
        if not mcp_frame.empty:
            write_pboc_total_assets_table(mcp_frame)
            return mcp_frame
    except Exception as exc:
        mcp_error = str(exc)

    try:
        links = discover_pboc_balance_sheet_links(source_url)
        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        for link in links[:100]:
            try:
                rows.extend(parse_pboc_balance_sheet_page(link))
            except Exception as exc:
                errors.append(f"{link}: {exc}")
        if not rows:
            details = "; ".join(errors[:3]) if errors else "no candidate balance-sheet pages found"
            raise RuntimeError(details)
        frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
        frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
        frame["raw_value"] = pd.to_numeric(frame["raw_value"], errors="coerce")
        frame = frame.dropna(subset=["observation_date", "raw_value"])
        if frame.empty:
            raise RuntimeError("PBoC parser found balance-sheet pages but no numeric Total Assets observations")
        frame = frame.sort_values(["observation_date", "download_timestamp"]).drop_duplicates(
            subset=["observation_date", "series_id"],
            keep="last",
        )
        frame["data_status"] = np.where(
            pd.to_numeric(frame["raw_value"], errors="coerce").pct_change().abs().gt(0.25),
            "OUTLIER_CHECK",
            frame["data_status"],
        )
        return frame[RAW_COLUMNS]
    except Exception as exc:
        return empty_raw_row(
            "PBOC",
            "People's Bank of China",
            source_url,
            PBOC_TOTAL_ASSETS_SERIES_ID,
            "China",
            "PBoC Total Assets",
            "monthly",
            "CNY",
            "CNY 100 million",
            "PARSER_ERROR",
            f"PBOC_TOTAL_ASSETS_UNAVAILABLE: {exc}. Missing data is not filled with zero.",
        )


def discover_pboc_balance_sheet_links(index_url: str) -> list[str]:
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 is not installed")
    response = get_with_retries(index_url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = [index_url]
    for anchor in soup.find_all("a"):
        href = str(anchor.get("href") or "").strip()
        text = clean_text(anchor.get_text(" "))
        if not href:
            continue
        if not any(token in text for token in ["货币当局资产负债表", "Balance Sheet of Monetary Authority"]):
            continue
        links.append(urljoin(index_url, href))
    seen: set[str] = set()
    unique: list[str] = []
    for link in links:
        if "pbc.gov.cn" not in link:
            continue
        if link not in seen:
            unique.append(link)
            seen.add(link)
    return unique


def parse_pboc_balance_sheet_page(url: str) -> list[dict[str, Any]]:
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 is not installed")
    response = get_with_retries(url)
    soup = BeautifulSoup(response.text, "html.parser")
    page_text = clean_text(soup.get_text(" "))
    if "货币当局资产负债表" not in page_text and "Balance Sheet of Monetary Authority" not in page_text:
        return []
    release_date = extract_pboc_release_date(page_text)
    unit = detect_pboc_unit(page_text)
    downloaded = fmt_datetime(now_utc())
    rows: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        table_text = clean_text(table.get_text(" "))
        if "资产" not in table_text and "Assets" not in table_text:
            continue
        parsed_rows = [
            [clean_text(cell.get_text(" ")) for cell in tr.find_all(["td", "th"])]
            for tr in table.find_all("tr")
        ]
        parsed_rows = [row for row in parsed_rows if any(row)]
        date_headers: list[pd.Timestamp] = []
        for row in parsed_rows:
            valid_dates = [date for date in [parse_pboc_period(cell) for cell in row] if date is not None]
            if len(valid_dates) >= 1:
                date_headers = valid_dates
            row_text = " ".join(row)
            if not is_pboc_total_assets_row(row_text):
                continue
            if not date_headers:
                continue
            values = [
                parse_pboc_number(cell)
                for cell in row
                if not is_pboc_total_assets_row(cell) and parse_pboc_period(cell) is None
            ]
            numeric_values = [value for value in values if np.isfinite(value)]
            for date, value in zip(date_headers, numeric_values):
                rows.append(
                    {
                        "observation_date": date,
                        "release_date": release_date,
                        "source": "PBOC",
                        "source_name": "People's Bank of China",
                        "source_url": url,
                        "series_id": PBOC_TOTAL_ASSETS_SERIES_ID,
                        "region": "China",
                        "metric": "PBoC Total Assets",
                        "frequency": "monthly",
                        "currency": "CNY",
                        "unit": unit,
                        "raw_value": value,
                        "download_timestamp": downloaded,
                        "data_status": "CURRENT",
                        "notes": "Official PBoC Balance Sheet of Monetary Authority; semantic Total Assets row parser",
                    }
                )
    return rows


def detect_pboc_unit(text: str) -> str:
    if "亿元" in text or "100 million" in text:
        return "CNY 100 million"
    return "UNKNOWN_UNIT"


def is_pboc_total_assets_row(text: str) -> bool:
    normalized = clean_text(text)
    if "金融机构" in normalized:
        return False
    return (
        normalized in {"资产合计", "总资产", "合计 资产", "Total Assets"}
        or normalized.startswith("资产合计 ")
        or normalized.startswith("总资产 ")
        or normalized.startswith("Total Assets ")
    )


def foreign_cb_asset_placeholders() -> pd.DataFrame:
    rows = [
        ("PBOC", "People's Bank of China", "PBOC_TOTAL_ASSETS", "China", "PBoC Total Assets", "CNY"),
    ]
    frames = [
        empty_raw_row(
            source,
            source_name,
            "",
            series_id,
            region,
            metric,
            "native",
            currency,
            "native",
            "NOT_CONFIGURED",
            "Official total-assets adapter is reserved for the next calibration step; values are not zero-filled.",
        )
        for source, source_name, series_id, region, metric, currency in rows
    ]
    return pd.concat(frames, ignore_index=True)


def fetch_raw_global_liquidity(api_key: str | None = None) -> pd.DataFrame:
    frames = [
        fred_raw(api_key),
        ecb_m2_raw(),
        boj_m2_raw(),
        pboc_m2_raw(api_key),
        ecb_total_assets_raw(),
        boj_total_assets_raw(),
        pboc_total_assets_raw(),
    ]
    raw = pd.concat(frames, ignore_index=True)
    raw["observation_date"] = pd.to_datetime(raw["observation_date"], errors="coerce")
    raw["release_date"] = pd.to_datetime(raw["release_date"], errors="coerce")
    raw["raw_value"] = pd.to_numeric(raw["raw_value"], errors="coerce")
    raw["source_mode"] = np.where(
        raw["source"].eq("TRADINGVIEW_MCP"),
        "MCP_PRIMARY",
        raw["source_mode"].fillna("FALLBACK_SOURCE"),
    )
    return raw[RAW_COLUMNS]


def raw_series(raw: pd.DataFrame, series_id: str) -> pd.Series:
    d = raw[raw["series_id"].eq(series_id)].dropna(subset=["observation_date", "raw_value"]).copy()
    if d.empty:
        return pd.Series(dtype="float64")
    return d.sort_values("observation_date").groupby("observation_date")["raw_value"].last()


def monthly_average(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    out = series.copy()
    out.index = pd.to_datetime(out.index)
    return out.resample("MS").mean().dropna()


def monthly_last(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    out = series.copy()
    out.index = pd.to_datetime(out.index)
    return out.resample("MS").last().dropna()


def weekly_average(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    out = series.copy()
    out.index = pd.to_datetime(out.index)
    return out.resample("W-FRI").mean().dropna()


def trailing_percentile(series: pd.Series, window: int, min_periods: int = 104) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    def rank_last(x: np.ndarray) -> float:
        clean = x[np.isfinite(x)]
        if len(clean) < min_periods or not np.isfinite(clean[-1]):
            return math.nan
        return float((clean <= clean[-1]).sum() / len(clean) * 100.0)

    return values.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def rolling_zscore(series: pd.Series, window: int = 156, min_periods: int = 104) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(window, min_periods=min_periods).mean()
    std = values.rolling(window, min_periods=min_periods).std(ddof=0)
    return (values - mean) / std.replace(0.0, np.nan)


def impulse_state(percentile: pd.Series) -> pd.Series:
    labels = []
    for value in pd.to_numeric(percentile, errors="coerce"):
        if not np.isfinite(value):
            labels.append("DATA_INCOMPLETE")
        elif value >= 80:
            labels.append("STRONG_EXPANSION")
        elif value >= 60:
            labels.append("EXPANSION")
        elif value >= 40:
            labels.append("NEUTRAL")
        elif value >= 20:
            labels.append("CONTRACTION")
        else:
            labels.append("STRONG_CONTRACTION")
    return pd.Series(labels, index=percentile.index)


def build_monthly_layer(raw: pd.DataFrame) -> pd.DataFrame:
    us_m2 = monthly_last(raw_series(raw, "M2SL"))
    ea_m2 = monthly_last(raw_series(raw, GLOBAL_LIQUIDITY_CONFIG["ecb_m2_key"]))
    japan_m2 = monthly_last(raw_series(raw, f"{GLOBAL_LIQUIDITY_CONFIG['boj_m2_db']}'{GLOBAL_LIQUIDITY_CONFIG['boj_m2_code']}"))
    china_m2 = monthly_last(raw_series(raw, "Money & Quasi-money (M2)"))
    eurusd = monthly_average(raw_series(raw, "DEXUSEU"))
    usdjpy = monthly_average(raw_series(raw, "DEXJPUS"))
    usdcny = monthly_average(raw_series(raw, "DEXCHUS"))
    fed_assets = monthly_last(raw_series(raw, "WALCL")) / 1000.0
    ecb_assets = monthly_last(raw_series(raw, ECB_TOTAL_ASSETS_SERIES_ID)) / 1000.0
    boj_assets = monthly_last(raw_series(raw, f"{BOJ_TOTAL_ASSETS_DB}'{BOJ_TOTAL_ASSETS_SERIES_ID}")) * 0.1
    pboc_assets = monthly_last(raw_series(raw, PBOC_TOTAL_ASSETS_SERIES_ID)) * 0.1
    m2_index = union_index([us_m2, ea_m2, japan_m2, china_m2])
    cb_index = union_index([fed_assets, ecb_assets, boj_assets, pboc_assets])
    index = m2_index.union(cb_index).sort_values()
    current_month = now_utc().to_period("M").to_timestamp()
    index = index[index < current_month]
    if index.empty:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)
    monthly = pd.DataFrame(index=index)
    monthly["us_m2_usd_bn"] = us_m2.reindex(index)
    monthly["ea_m2_usd_bn"] = ea_m2.reindex(index) * eurusd.reindex(index) / 1000.0
    monthly["japan_m2_usd_bn"] = japan_m2.reindex(index) * 0.1 / usdjpy.reindex(index)
    monthly["china_m2_usd_bn"] = china_m2.reindex(index) * 0.1 / usdcny.reindex(index)
    m2_components = ["us_m2_usd_bn", "ea_m2_usd_bn", "china_m2_usd_bn", "japan_m2_usd_bn"]
    monthly["global_m2_usd_bn"] = monthly[m2_components].sum(axis=1, min_count=4)
    monthly["global_m2_partial_usd_bn"] = monthly[m2_components].sum(axis=1, min_count=1)
    monthly["global_m2_fx_neutral_bn"] = build_fx_neutral_m2(us_m2, ea_m2, japan_m2, china_m2, eurusd, usdjpy, usdcny, index)
    monthly["global_m2_fx_neutral_partial_bn"] = build_fx_neutral_m2(
        us_m2,
        ea_m2,
        japan_m2,
        china_m2,
        eurusd,
        usdjpy,
        usdcny,
        index,
        require_all=False,
    )
    monthly["global_m2_fx_effect_bn"] = monthly["global_m2_usd_bn"] - monthly["global_m2_fx_neutral_bn"]
    for months in [1, 3, 6, 12]:
        monthly[f"global_m2_{months}m_pct"] = monthly["global_m2_usd_bn"].pct_change(months, fill_method=None)
    monthly["global_m2_yoy_pct"] = monthly["global_m2_12m_pct"]
    monthly["global_m2_3m_annualized"] = (1.0 + monthly["global_m2_3m_pct"]) ** 4 - 1.0
    monthly["global_m2_acceleration"] = monthly["global_m2_3m_annualized"] - monthly["global_m2_yoy_pct"]
    share_cols = {
        "us_m2_share": "us_m2_usd_bn",
        "ea_m2_share": "ea_m2_usd_bn",
        "china_m2_share": "china_m2_usd_bn",
        "japan_m2_share": "japan_m2_usd_bn",
    }
    for out_col, in_col in share_cols.items():
        monthly[out_col] = monthly[in_col] / monthly["global_m2_usd_bn"]
    for region, col in [("us", "us_m2_usd_bn"), ("ea", "ea_m2_usd_bn"), ("china", "china_m2_usd_bn"), ("japan", "japan_m2_usd_bn")]:
        monthly[f"{region}_contribution_12m"] = monthly[col].diff(12) / monthly["global_m2_usd_bn"].diff(12)
    monthly["global_m2_impulse"] = monthly["global_m2_3m_annualized"]
    monthly["global_m2_impulse_3y_percentile"] = trailing_percentile(monthly["global_m2_impulse"], 36, 24)
    monthly["global_m2_impulse_5y_percentile"] = trailing_percentile(monthly["global_m2_impulse"], 60, 36)
    monthly["global_m2_impulse_zscore"] = rolling_zscore(monthly["global_m2_impulse"], 60, 36)
    monthly["global_m2_impulse_state"] = impulse_state(monthly["global_m2_impulse_3y_percentile"])
    monthly["fx_base_date"] = monthly.index.year.map(lambda year: f"{year - 1}-12")
    monthly["fx_base_method"] = "December previous year monthly average"
    monthly["data_status"] = np.where(monthly["global_m2_usd_bn"].notna(), "CURRENT", "PARTIAL_DATA")
    monthly["last_updated"] = fmt_datetime(now_utc())
    monthly["fed_assets_usd_bn"] = fed_assets.reindex(index)
    monthly["ecb_assets_usd_bn"] = ecb_assets.reindex(index) * eurusd.reindex(index)
    monthly["boj_assets_usd_bn"] = boj_assets.reindex(index) / usdjpy.reindex(index)
    monthly["pboc_assets_usd_bn"] = pboc_assets.reindex(index) / usdcny.reindex(index)
    cb_components = ["fed_assets_usd_bn", "ecb_assets_usd_bn", "boj_assets_usd_bn", "pboc_assets_usd_bn"]
    monthly["global_cb_assets_usd_bn"] = monthly[cb_components].sum(axis=1, min_count=4)
    monthly["global_cb_assets_partial_usd_bn"] = monthly[cb_components].sum(axis=1, min_count=1)
    for months in [1, 3, 6, 12]:
        monthly[f"global_cb_assets_{months}m_pct"] = monthly["global_cb_assets_usd_bn"].pct_change(months, fill_method=None)
    for months in [1, 3, 6]:
        monthly[f"global_cb_assets_change_{months}m_bn"] = monthly["global_cb_assets_usd_bn"].diff(months)
    share_cols = {
        "fed_cb_share": "fed_assets_usd_bn",
        "ecb_cb_share": "ecb_assets_usd_bn",
        "boj_cb_share": "boj_assets_usd_bn",
        "pboc_cb_share": "pboc_assets_usd_bn",
    }
    for out_col, in_col in share_cols.items():
        monthly[out_col] = monthly[in_col] / monthly["global_cb_assets_usd_bn"]
    for region, col in [("fed", "fed_assets_usd_bn"), ("ecb", "ecb_assets_usd_bn"), ("boj", "boj_assets_usd_bn"), ("pboc", "pboc_assets_usd_bn")]:
        monthly[f"{region}_cb_contribution_3m"] = monthly[col].diff(3) / monthly["global_cb_assets_usd_bn"].diff(3)
    monthly["data_status"] = np.where(
        monthly["global_m2_usd_bn"].notna() | monthly["global_cb_assets_usd_bn"].notna(),
        "CURRENT",
        "PARTIAL_DATA",
    )
    monthly = monthly.reset_index()
    monthly = monthly.rename(columns={monthly.columns[0]: "date"})
    for column in MONTHLY_COLUMNS:
        if column not in monthly.columns:
            monthly[column] = np.nan
    return monthly[MONTHLY_COLUMNS]


def build_fx_neutral_m2(
    us_m2: pd.Series,
    ea_m2: pd.Series,
    japan_m2: pd.Series,
    china_m2: pd.Series,
    eurusd: pd.Series,
    usdjpy: pd.Series,
    usdcny: pd.Series,
    index: pd.DatetimeIndex,
    require_all: bool = True,
) -> pd.Series:
    out = pd.Series(np.nan, index=index, dtype="float64")
    us_values = us_m2.reindex(index)
    ea_values = ea_m2.reindex(index)
    japan_values = japan_m2.reindex(index)
    china_values = china_m2.reindex(index)
    for date in index:
        base_month = pd.Timestamp(year=int(date.year) - 1, month=12, day=1)
        components = [
            float(us_values.loc[date]) if pd.notna(us_values.loc[date]) else math.nan,
            float(ea_values.loc[date]) * value_at(eurusd, base_month) / 1000.0 if pd.notna(ea_values.loc[date]) else math.nan,
            float(japan_values.loc[date]) * 0.1 / value_at(usdjpy, base_month) if pd.notna(japan_values.loc[date]) else math.nan,
            float(china_values.loc[date]) * 0.1 / value_at(usdcny, base_month) if pd.notna(china_values.loc[date]) else math.nan,
        ]
        if require_all and all(np.isfinite(value) for value in components):
            out.loc[date] = float(sum(components))
        elif not require_all and any(np.isfinite(value) for value in components):
            out.loc[date] = float(sum(value for value in components if np.isfinite(value)))
    return out


def build_weekly_layer(raw: pd.DataFrame) -> pd.DataFrame:
    walcl = weekly_last(raw_series(raw, "WALCL")) / 1000.0
    tga = weekly_last(raw_series(raw, "WTREGEN")) / 1000.0
    rrp = weekly_last(raw_series(raw, "RRPONTSYD"))
    ecb_assets_eur_bn = weekly_last(raw_series(raw, ECB_TOTAL_ASSETS_SERIES_ID)) / 1000.0
    eurusd_weekly = weekly_average(raw_series(raw, "DEXUSEU"))
    dxy = weekly_dxy()
    index = union_index([walcl, tga, rrp, ecb_assets_eur_bn, dxy])
    if index.empty:
        return pd.DataFrame(columns=WEEKLY_COLUMNS)
    weekly = pd.DataFrame(index=index)
    weekly["fed_assets_usd_bn"] = walcl.reindex(index).ffill()
    weekly["tga_usd_bn"] = tga.reindex(index).ffill()
    weekly["rrp_usd_bn"] = rrp.reindex(index).ffill()
    weekly["us_net_liquidity_usd_bn"] = weekly["fed_assets_usd_bn"] - weekly["tga_usd_bn"] - weekly["rrp_usd_bn"]
    for weeks in [1, 4, 13, 26]:
        weekly[f"us_net_liquidity_{weeks}w_change_bn"] = weekly["us_net_liquidity_usd_bn"].diff(weeks)
        weekly[f"us_net_liquidity_{weeks}w_pct"] = weekly["us_net_liquidity_usd_bn"].pct_change(weeks, fill_method=None)
    weekly["us_net_liquidity_13w_percentile"] = trailing_percentile(weekly["us_net_liquidity_13w_change_bn"], 156, 104)
    weekly["us_net_liquidity_26w_percentile"] = trailing_percentile(weekly["us_net_liquidity_26w_change_bn"], 156, 104)
    weekly["ecb_assets_usd_bn"] = ecb_assets_eur_bn.reindex(index).ffill() * eurusd_weekly.reindex(index).ffill()
    weekly["boj_assets_usd_bn"] = np.nan
    weekly["pboc_assets_usd_bn"] = np.nan
    weekly["global_cb_assets_usd_bn"] = weekly[["fed_assets_usd_bn", "ecb_assets_usd_bn"]].sum(axis=1, min_count=2)
    weekly["global_cb_assets_partial_usd_bn"] = weekly[["fed_assets_usd_bn", "ecb_assets_usd_bn"]].sum(axis=1, min_count=1)
    for weeks in [4, 13, 26, 52]:
        weekly[f"global_cb_assets_{weeks}w_pct"] = weekly["global_cb_assets_usd_bn"].pct_change(weeks, fill_method=None)
    for weeks in [4, 13, 26]:
        weekly[f"global_cb_assets_change_{weeks}w_bn"] = weekly["global_cb_assets_usd_bn"].diff(weeks)
    weekly["global_cb_impulse"] = weekly["global_cb_assets_13w_pct"]
    weekly["global_cb_impulse_3y_percentile"] = trailing_percentile(weekly["global_cb_impulse"], 156, 104)
    weekly["global_cb_impulse_5y_percentile"] = trailing_percentile(weekly["global_cb_impulse"], 260, 156)
    weekly["global_cb_impulse_zscore"] = rolling_zscore(weekly["global_cb_impulse"], 156, 104)
    weekly["global_cb_impulse_state"] = impulse_state(weekly["global_cb_impulse_3y_percentile"])
    weekly["us_net_liquidity_impulse"] = weekly["us_net_liquidity_13w_change_bn"]
    weekly["us_net_liquidity_impulse_3y_percentile"] = trailing_percentile(weekly["us_net_liquidity_impulse"], 156, 104)
    weekly["us_net_liquidity_impulse_5y_percentile"] = trailing_percentile(weekly["us_net_liquidity_impulse"], 260, 156)
    weekly["us_net_liquidity_impulse_zscore"] = rolling_zscore(weekly["us_net_liquidity_impulse"], 156, 104)
    weekly["us_net_liquidity_impulse_state"] = impulse_state(weekly["us_net_liquidity_impulse_3y_percentile"])
    weekly["dxy"] = dxy.reindex(index).ffill()
    weekly["dxy_4w_pct"] = weekly["dxy"].pct_change(4, fill_method=None)
    weekly["dxy_13w_pct"] = weekly["dxy"].pct_change(13, fill_method=None)
    weekly["dxy_26w_pct"] = weekly["dxy"].pct_change(26, fill_method=None)
    forecast_sources = {
        "US10Y_TermPremium": ("THREEFYTP10", 1.0),
        "SOFR": ("SOFR", 1.0),
        "EFFR": ("EFFR", 1.0),
        "US_BankReserves": ("WRESBAL", 1.0 / 1000.0),
    }
    for column, (series_id, multiplier) in forecast_sources.items():
        lag = 5 if series_id == "THREEFYTP10" else 2 if series_id == "WRESBAL" else 1
        weekly[column] = weekly_available_last(raw_series(raw, series_id), lag).reindex(index).ffill() * multiplier
    for weeks in (4, 13, 26):
        weekly[f"US10Y_TermPremium_{weeks}W_Change"] = weekly["US10Y_TermPremium"].diff(weeks)
        weekly[f"US_BankReserves_{weeks}W_Change"] = weekly["US_BankReserves"].diff(weeks)
        weekly[f"US_BankReserves_{weeks}W_PctChange"] = weekly["US_BankReserves"].pct_change(weeks, fill_method=None) * 100.0
    weekly["SOFR_EFFR_Spread"] = weekly["SOFR"] - weekly["EFFR"]
    for weeks in (4, 13):
        weekly[f"SOFR_EFFR_{weeks}W_Change"] = weekly["SOFR_EFFR_Spread"].diff(weeks)
    weekly["data_status"] = np.where(
        weekly["us_net_liquidity_usd_bn"].notna() & weekly["global_cb_assets_usd_bn"].notna(),
        "CURRENT",
        "PARTIAL_DATA",
    )
    weekly["last_updated"] = fmt_datetime(now_utc())
    weekly = weekly.reset_index()
    weekly = weekly.rename(columns={weekly.columns[0]: "date"})
    for column in WEEKLY_COLUMNS:
        if column not in weekly.columns:
            weekly[column] = np.nan
    return weekly[WEEKLY_COLUMNS]


def weekly_last(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    out = series.copy()
    out.index = pd.to_datetime(out.index)
    return out.resample("W-FRI").last().dropna()


def weekly_available_last(series: pd.Series, business_day_lag: int) -> pd.Series:
    if series.empty:
        return series
    available = series.copy()
    available.index = pd.to_datetime(available.index) + pd.offsets.BDay(business_day_lag)
    return available.resample("W-FRI").last().dropna()


def weekly_dxy() -> pd.Series:
    try:
        ohlcv = download_completed_ohlcv("DX-Y.NYB", period="max")
    except Exception:
        return pd.Series(dtype="float64")
    if ohlcv.empty or "Close" not in ohlcv.columns:
        return pd.Series(dtype="float64")
    close = pd.to_numeric(ohlcv["Close"], errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    if market_business_days_old(close) is None or market_business_days_old(close) > 2:
        return pd.Series(dtype="float64")
    close = close.loc[pd.Timestamp(GLOBAL_LIQUIDITY_CONFIG["start_date"]) :]
    weekly = close.resample("W-FRI").last().dropna()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if not weekly.empty and weekly.index[-1] > today:
        weekly = weekly.iloc[:-1]
    return weekly


def union_index(series_list: list[pd.Series]) -> pd.DatetimeIndex:
    indexes = [pd.DatetimeIndex(series.dropna().index) for series in series_list if not series.empty]
    if not indexes:
        return pd.DatetimeIndex([])
    index = indexes[0]
    for other in indexes[1:]:
        index = index.union(other)
    return pd.DatetimeIndex(index).sort_values()


def value_at(series: pd.Series, date: pd.Timestamp) -> float:
    if series.empty:
        return math.nan
    values = series.sort_index().loc[:date].dropna()
    return float(values.iloc[-1]) if not values.empty else math.nan


def freshness(raw: pd.DataFrame, monthly: pd.DataFrame, weekly: pd.DataFrame) -> list[LiquidityFreshness]:
    rows = []
    for block, frame, date_col in [
        ("Global M2", monthly, "date"),
        ("US Net Liquidity", weekly, "date"),
        ("Global CB Assets", weekly, "date"),
        ("USD Conditions", weekly, "date"),
    ]:
        status = "ERROR" if frame.empty else str(frame["data_status"].dropna().iloc[-1])
        last_obs = "n/a" if frame.empty else fmt_date(pd.to_datetime(frame[date_col], errors="coerce").max())
        rows.append(LiquidityFreshness(block, last_obs, "n/a", fmt_datetime(now_utc()), status))
    source_groups = raw.groupby(["source", "series_id"], dropna=False)
    for (source, series_id), group in source_groups:
        status = str(group["data_status"].dropna().iloc[-1]) if group["data_status"].notna().any() else "ERROR"
        rows.append(
            LiquidityFreshness(
                f"{source} {series_id}",
                fmt_date(pd.to_datetime(group["observation_date"], errors="coerce").max()),
                "n/a",
                str(group["download_timestamp"].dropna().iloc[-1]) if group["download_timestamp"].notna().any() else "n/a",
                status,
            )
        )
    return rows


def update_global_liquidity(api_key: str | None = None, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not force and is_storage_fresh():
        return read_global_liquidity()
    raw = fetch_raw_global_liquidity(api_key)
    monthly = build_monthly_layer(raw)
    weekly = build_weekly_layer(raw)
    write_frame(raw, RAW_STORAGE_PATH)
    write_frame(monthly, MONTHLY_STORAGE_PATH)
    write_frame(weekly, WEEKLY_STORAGE_PATH)
    return raw, monthly, weekly


def read_global_liquidity() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return read_frame(RAW_STORAGE_PATH, RAW_COLUMNS), read_frame(MONTHLY_STORAGE_PATH, MONTHLY_COLUMNS), read_frame(WEEKLY_STORAGE_PATH, WEEKLY_COLUMNS)


def is_storage_fresh() -> bool:
    if not MONTHLY_STORAGE_PATH.exists() or not WEEKLY_STORAGE_PATH.exists():
        return False
    age = time.time() - min(MONTHLY_STORAGE_PATH.stat().st_mtime, WEEKLY_STORAGE_PATH.stat().st_mtime)
    return age < int(GLOBAL_LIQUIDITY_CONFIG["ttl_seconds"])


def start_background_update_if_stale(api_key: str | None = None, force: bool = False) -> bool:
    global _UPDATE_THREAD
    if _UPDATE_THREAD is not None and _UPDATE_THREAD.is_alive():
        return True
    if not force and is_storage_fresh():
        return False
    _UPDATE_THREAD = Thread(target=_safe_background_update, args=(api_key, force), daemon=True)
    _UPDATE_THREAD.start()
    return True


def global_liquidity_update_in_progress() -> bool:
    return _UPDATE_THREAD is not None and _UPDATE_THREAD.is_alive()


def _safe_background_update(api_key: str | None, force: bool) -> None:
    try:
        update_global_liquidity(api_key=api_key, force=force)
    except Exception:
        pass


def read_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    for column in ["date", "observation_date", "release_date"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame[columns]


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def fmt_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def fmt_datetime(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M UTC")
