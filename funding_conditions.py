from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from finance_core import download_completed_ohlcv, market_business_days_old
from fred_client import download_fred_series, get_fred_api_key


MODEL_VERSION = "FUNDING_CONDITIONS_V1"
STORAGE_DIR = Path("persistent") / "funding_conditions"
DAILY_PATH = STORAGE_DIR / "daily_history.parquet"
WEEKLY_PATH = STORAGE_DIR / "weekly_history.parquet"
STATUS_PATH = STORAGE_DIR / "status.json"
FRED_IDS = ("SOFR99", "DFF", "IORB", "WRESBAL", "GDP")
CORE_IDS = ("SOFR99", "DFF", "WRESBAL", "GDP")
MOVE_CACHE_MAX_AGE = pd.Timedelta(hours=20)
MOVE_MAX_BUSINESS_DAYS_OLD = 2
MOVE_FETCH_ATTEMPTS = 3
MOVE_FETCH_RETRY_SECONDS = 5
STATES = (
    "NORMAL",
    "TECHNICAL FUNDING PRESSURE",
    "PERSISTENT FUNDING PRESSURE",
    "TREASURY VOLATILITY",
    "SYSTEMIC FUNDING STRESS",
)
EXPORT_FIELDS = (
    "FundingSpreadRaw", "FundingSpreadSmooth", "FundingSpreadRobustZ", "MoneyMarketStress",
    "CalendarTechnicalFlag", "TechnicalFundingFlag", "PersistentFundingFlag",
    "UnconfirmedFundingPressureFlag",
    "ReserveGDP", "ReserveGDP_Z", "ReserveVulnerability", "ReserveChange13W", "ReserveDrain",
    "ReservePressure", "FundingCore", "MOVE_Z", "CollateralStress", "ReserveVulnerabilityWatch",
    "FundingState", "FundingDirection", "FundingStateStartDate", "WeeksInFundingState",
    "DataCoverage", "FundingConditionsModelVersion",
)


@dataclass
class FundingSnapshot:
    daily: pd.DataFrame
    weekly: pd.DataFrame
    status: dict


def expanding_z(values: pd.Series, minimum: int = 52) -> pd.Series:
    source = pd.to_numeric(values, errors="coerce")
    mean = source.expanding(min_periods=minimum).mean()
    std = source.expanding(min_periods=minimum).std(ddof=0)
    return ((source - mean) / std.replace(0, np.nan)).clip(-4, 4)


def expanding_robust_z(values: pd.Series, minimum: int = 52) -> pd.Series:
    source = pd.to_numeric(values, errors="coerce")
    median = source.expanding(min_periods=minimum).median()
    mad = source.expanding(min_periods=minimum).apply(
        lambda window: np.median(np.abs(window - np.median(window))), raw=True
    )
    robust_scale = 1.4826 * mad
    iqr_scale = (source.expanding(min_periods=minimum).quantile(0.75) -
                 source.expanding(min_periods=minimum).quantile(0.25)) / 1.349
    std = source.expanding(min_periods=minimum).std(ddof=0)
    scale = robust_scale.where(robust_scale > 0, iqr_scale).where(lambda x: x > 0, std)
    result = (source - median) / scale.replace(0, np.nan)
    return result.where(scale.notna(), np.nan).clip(-4, 4).fillna(
        pd.Series(np.where(scale.eq(0) & source.eq(median), 0.0, np.nan), index=source.index)
    )


def calendar_technical_flag(observation_dates: pd.DatetimeIndex) -> pd.Series:
    dates = pd.DatetimeIndex(observation_dates).normalize()
    ends = pd.date_range(dates.min() - pd.DateOffset(months=1), dates.max() + pd.DateOffset(months=1), freq="BME")
    flag = pd.Series(False, index=dates)
    for end in ends:
        flag |= (dates >= end - pd.offsets.BDay(2)) & (dates <= end + pd.offsets.BDay(3))
    return flag


def persistent_funding_flag(stress: pd.Series, calendar_flag: pd.Series) -> pd.Series:
    severe = stress.gt(2)
    moderate = stress.gt(1)
    off_calendar = ~calendar_flag.astype(bool)
    severe_persistent = severe.rolling(5, min_periods=5).sum().eq(5) & (
        (severe & off_calendar).rolling(5, min_periods=5).sum().ge(3)
    )
    moderate_persistent = moderate & moderate.rolling(15, min_periods=15).sum().ge(10) & (
        (moderate & off_calendar).rolling(15, min_periods=15).sum().ge(6)
    )
    return severe_persistent | moderate_persistent


def classify_funding_state(frame: pd.DataFrame) -> pd.Series:
    persistent = frame["PersistentFundingFlag"]
    money = frame["MoneyMarketStress"]
    systemic = persistent & money.ge(2) & (
        frame["ReservePressure"].ge(1) | frame["ReserveVulnerability"].ge(1.5) |
        frame["CollateralStress"].ge(1.5)
    )
    treasury = frame["CollateralStress"].ge(2) & money.lt(1) & ~persistent
    states = pd.Series(np.select(
        [systemic, persistent, frame["TechnicalFundingFlag"], treasury],
        ["SYSTEMIC FUNDING STRESS", "PERSISTENT FUNDING PRESSURE", "TECHNICAL FUNDING PRESSURE", "TREASURY VOLATILITY"],
        default="NORMAL",
    ), index=frame.index)
    states.loc[money.isna()] = "DATA INCOMPLETE"
    return states


def _asof(source: pd.DataFrame, dates: pd.DatetimeIndex, column: str = "Value") -> pd.Series:
    if source.empty:
        return pd.Series(np.nan, index=dates)
    available = source.sort_values(["AvailableDate", "ObservationDate"]).drop_duplicates("AvailableDate", keep="last")
    values = pd.Series(pd.to_numeric(available[column], errors="coerce").astype(float).to_numpy(),
                       index=pd.DatetimeIndex(available["AvailableDate"]))
    return values.reindex(values.index.union(dates.unique())).sort_index().ffill().reindex(dates)


def _reserve_history(reserves: pd.DataFrame, gdp: pd.DataFrame) -> pd.DataFrame:
    reserve = reserves.sort_values("ObservationDate").drop_duplicates("ObservationDate", keep="first").copy()
    raw_reserves = pd.to_numeric(reserve["Value"], errors="coerce")
    # Initial-release WRESBAL vintages can be in USD billions even though the
    # current FRED series is in USD millions. The 2010+ reserve series is well
    # above $100bn, so values below 100,000 are billion-denominated vintages.
    reserve["WRESBAL"] = raw_reserves.where(raw_reserves.ge(100_000), raw_reserves * 1000)
    reserve["GDP"] = _asof(gdp, pd.DatetimeIndex(reserve["AvailableDate"])).to_numpy()
    # WRESBAL is USD millions; nominal GDP is USD billions.
    reserve["ReserveGDP"] = reserve["WRESBAL"] / 1000.0 / reserve["GDP"]
    reserve["ReserveGDP_Z"] = expanding_z(reserve["ReserveGDP"], 52)
    reserve["ReserveVulnerability"] = (-reserve["ReserveGDP_Z"]).clip(lower=0, upper=4)
    reserve["ReserveChange13W"] = reserve["WRESBAL"].pct_change(13, fill_method=None)
    reserve["ReserveDrain"] = (-expanding_z(reserve["ReserveChange13W"], 52)).clip(lower=0, upper=4)
    reserve["ReservePressure"] = 0.65 * reserve["ReserveVulnerability"] + 0.35 * reserve["ReserveDrain"]
    reserve["ReserveVulnerabilityWatch"] = (
        reserve["ReserveVulnerability"].ge(1.5) | reserve["ReserveDrain"].ge(1.5)
    )
    return reserve


def build_history(sources: dict[str, pd.DataFrame], move: pd.Series | None = None) -> FundingSnapshot:
    sofr = sources["SOFR99"].rename(columns={"Value": "SOFR99", "AvailableDate": "SOFRAvailable"})
    dff = sources["DFF"].rename(columns={"Value": "DFF", "AvailableDate": "DFFAvailable"})
    daily = sofr[["ObservationDate", "SOFRAvailable", "SOFR99"]].merge(
        dff[["ObservationDate", "DFFAvailable", "DFF"]], on="ObservationDate", how="inner", validate="one_to_one"
    )
    daily = daily.dropna(subset=["SOFR99", "DFF"]).sort_values("ObservationDate").reset_index(drop=True)
    if daily.empty:
        raise RuntimeError("SOFR99 and DFF have no matching released observations")
    daily["Date"] = daily[["SOFRAvailable", "DFFAvailable"]].max(axis=1)
    daily["FundingSpreadRaw"] = daily["SOFR99"] - daily["DFF"]
    daily["FundingSpreadSmooth"] = daily["FundingSpreadRaw"].rolling(5, min_periods=5).median()
    daily["FundingSpreadRobustZ"] = expanding_robust_z(daily["FundingSpreadSmooth"])
    daily["MoneyMarketStress"] = daily["FundingSpreadRobustZ"].clip(lower=0, upper=4)
    daily["CalendarTechnicalFlag"] = calendar_technical_flag(pd.DatetimeIndex(daily["ObservationDate"])).to_numpy()
    moderate = daily["MoneyMarketStress"].gt(1)
    daily["PersistentFundingFlag"] = persistent_funding_flag(daily["MoneyMarketStress"], daily["CalendarTechnicalFlag"])
    daily["TechnicalFundingFlag"] = moderate & daily["CalendarTechnicalFlag"] & ~daily["PersistentFundingFlag"]
    daily["UnconfirmedFundingPressureFlag"] = moderate & ~daily["TechnicalFundingFlag"] & ~daily["PersistentFundingFlag"]

    reserve = _reserve_history(sources["WRESBAL"], sources["GDP"])
    dates = pd.DatetimeIndex(daily["Date"])
    for column in ("WRESBAL", "GDP", "ReserveGDP", "ReserveGDP_Z", "ReserveVulnerability",
                   "ReserveChange13W", "ReserveDrain", "ReservePressure", "ReserveVulnerabilityWatch"):
        daily[column] = _asof(reserve, dates, column).to_numpy()
    daily["ReserveVulnerabilityWatch"] = daily["ReserveVulnerabilityWatch"].fillna(False).astype(bool)
    daily["FundingCore"] = 0.60 * daily["MoneyMarketStress"] + 0.40 * daily["ReservePressure"]

    iorb = sources.get("IORB", pd.DataFrame())
    daily["IORB"] = _asof(iorb, dates).to_numpy()
    daily["SOFR99_IORB_Spread"] = daily["SOFR99"] - daily["IORB"]
    move_source = pd.Series(dtype="float64")
    move_frame = pd.DataFrame()
    if move is not None and not move.dropna().empty:
        move_source = pd.to_numeric(move, errors="coerce").dropna().sort_index()
        move_source.index = pd.to_datetime(move_source.index).tz_localize(None).normalize()
        move_source = move_source[~move_source.index.duplicated(keep="last")]
        move_z = expanding_z(move_source.loc[move_source.index >= "2010-01-01"], 156)
        move_frame = pd.DataFrame({"ObservationDate": move_z.index, "AvailableDate": move_z.index,
                                   "Value": move_source.reindex(move_z.index).to_numpy(), "MOVE_Z": move_z.to_numpy()})

        # FRED releases can lag market closes. Keep the latest released core
        # values as-of while allowing newer completed MOVE observations into
        # the report history.
        market_dates = pd.DatetimeIndex(move_frame["AvailableDate"]).unique()
        base = daily.sort_values("Date").drop_duplicates("Date", keep="last")
        extended_dates = pd.DatetimeIndex(base["Date"]).union(market_dates).sort_values()
        if len(extended_dates) > len(base):
            daily = pd.merge_asof(
                pd.DataFrame({"Date": extended_dates}),
                base,
                on="Date",
                direction="backward",
            )

    if move is None or move.dropna().empty:
        daily["MOVE"] = np.nan
        daily["MOVE_Z"] = np.nan
    else:
        report_dates = pd.DatetimeIndex(daily["Date"])
        daily["MOVE"] = _asof(move_frame, report_dates).to_numpy()
        daily["MOVE_Z"] = _asof(move_frame, report_dates, "MOVE_Z").to_numpy()
    daily["CollateralStress"] = daily["MOVE_Z"].clip(lower=0, upper=4)

    money = daily["MoneyMarketStress"]
    reserve_pressure = daily["ReservePressure"]
    daily["FundingState"] = classify_funding_state(daily)
    daily["DataCoverage"] = np.select(
        [money.isna() & reserve_pressure.isna(), money.isna() | reserve_pressure.isna(), daily["CollateralStress"].isna()],
        ["CORE INCOMPLETE", "PARTIAL DATA", "PARTIAL DATA"], default="FULL"
    )
    change_4w = daily["FundingCore"].diff(20)
    threshold = change_4w.abs().expanding(min_periods=100).quantile(0.60)
    daily["FundingDirection"] = np.select(
        [change_4w.gt(threshold), change_4w.lt(-threshold)], ["RISING", "FALLING"], default="STABLE"
    )
    daily.loc[threshold.isna() | change_4w.isna(), "FundingDirection"] = "DATA INCOMPLETE"

    drivers = daily[["MoneyMarketStress", "ReserveVulnerability", "ReserveDrain", "CollateralStress"]].fillna(-1)
    labels = {"MoneyMarketStress": "SOFR tail spread", "ReserveVulnerability": "Low reserve/GDP",
              "ReserveDrain": "Reserve depletion", "CollateralStress": "Treasury volatility"}
    daily["PrimaryDriver"] = drivers.idxmax(axis=1).map(labels)
    daily.loc[drivers.max(axis=1).lt(0), "PrimaryDriver"] = "DATA INCOMPLETE"
    daily["FundingConditionsModelVersion"] = MODEL_VERSION
    daily = daily.drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)
    weekly = daily.set_index("Date").resample("W-FRI").last().dropna(subset=["MoneyMarketStress"]).reset_index()
    changed = weekly["FundingState"].ne(weekly["FundingState"].shift())
    weekly["FundingStateStartDate"] = weekly["Date"].where(changed).ffill()
    weekly["WeeksInFundingState"] = ((weekly["Date"] - weekly["FundingStateStartDate"]).dt.days // 7 + 1).astype(int)
    daily["FundingStateStartDate"] = daily["Date"].where(daily["FundingState"].ne(daily["FundingState"].shift())).ffill()
    daily["WeeksInFundingState"] = ((daily["Date"] - daily["FundingStateStartDate"]).dt.days // 7 + 1).astype(int)
    return FundingSnapshot(daily, weekly, {})


def _cache_path(series_id: str) -> Path:
    return STORAGE_DIR / f"fred_{series_id.lower()}_initial.parquet"


def read_wresbal_history() -> pd.DataFrame:
    """Return the full cached point-in-time WRESBAL history independently of SOFR."""
    path = _cache_path("WRESBAL")
    columns = ["ObservationDate", "AvailableDate", "WRESBAL_USD_Millions", "WRESBAL_USD_Bn"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_parquet(path)
    required = {"ObservationDate", "AvailableDate", "Value"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    frame = frame[["ObservationDate", "AvailableDate", "Value"]].copy()
    frame["ObservationDate"] = pd.to_datetime(frame["ObservationDate"], errors="coerce")
    frame["AvailableDate"] = pd.to_datetime(frame["AvailableDate"], errors="coerce")
    raw = pd.to_numeric(frame["Value"], errors="coerce")
    frame["WRESBAL_USD_Millions"] = raw.where(raw.ge(100_000), raw * 1000)
    frame["WRESBAL_USD_Bn"] = frame["WRESBAL_USD_Millions"] / 1000.0
    return (
        frame[columns]
        .dropna(subset=["ObservationDate", "AvailableDate", "WRESBAL_USD_Millions"])
        .sort_values(["ObservationDate", "AvailableDate"])
        .drop_duplicates("ObservationDate", keep="first")
        .reset_index(drop=True)
    )


def _load_initial_release(
    series_id: str, api_key: str | None, refresh: bool = False, cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    cache_dir = cache_dir or STORAGE_DIR
    path = cache_dir / f"fred_{series_id.lower()}_initial.parquet"
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if not refresh and not cached.empty and now - pd.Timestamp(path.stat().st_mtime, unit="s") < pd.Timedelta(hours=20):
        return cached, "INITIAL_RELEASE_CACHE"
    try:
        key = get_fred_api_key(api_key)
        latest = pd.to_datetime(cached["AvailableDate"]).max() if not cached.empty else None
        window_start = max(pd.Timestamp("2010-01-01"), latest - pd.DateOffset(years=1)) if latest is not None else pd.Timestamp("2010-01-01")
        frames = [cached] if not cached.empty else []
        while window_start <= now.normalize():
            window_end = min(window_start + pd.DateOffset(years=4) - pd.Timedelta(days=1), now.normalize())
            params = {
                "series_id": series_id, "api_key": key, "file_type": "json", "output_type": 4,
                "realtime_start": window_start.strftime("%Y-%m-%d"),
                "realtime_end": window_end.strftime("%Y-%m-%d"),
                "observation_start": (window_start - pd.DateOffset(years=1)).strftime("%Y-%m-%d"),
            }
            try:
                response = httpx.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=90)
            except httpx.HTTPError as exc:
                raise RuntimeError(f"FRED connection failed: {type(exc).__name__}") from None
            payload = response.json()
            if response.status_code >= 400 or "error_code" in payload:
                message = str(payload.get("error_message", f"HTTP {response.status_code}")).replace(key, "[redacted]")
                raise RuntimeError(message)
            observations = pd.DataFrame(payload.get("observations", []))
            if not observations.empty:
                frames.append(pd.DataFrame({
                    "ObservationDate": pd.to_datetime(observations["date"], errors="coerce"),
                    "AvailableDate": pd.to_datetime(observations["realtime_start"], errors="coerce"),
                    "Value": pd.to_numeric(observations["value"].replace(".", np.nan), errors="coerce"),
                }).dropna())
            window_start = window_end + pd.Timedelta(days=1)
        if not frames:
            raise RuntimeError("No initial-release observations")
        result = pd.concat(frames, ignore_index=True).sort_values(
            ["ObservationDate", "AvailableDate"]
        ).drop_duplicates("ObservationDate", keep="first")
        if result.empty:
            raise RuntimeError("No numeric initial-release observations")
        cache_dir.mkdir(parents=True, exist_ok=True)
        result.to_parquet(path, index=False)
        return result, "FRED_INITIAL_RELEASE"
    except Exception as exc:
        if not cached.empty:
            return cached, "STALE_INITIAL_RELEASE_CACHE"
        raise RuntimeError(f"{series_id} initial-release history unavailable: {exc}") from exc


def _load_iorb_fallback(api_key: str | None, refresh: bool = False) -> tuple[pd.DataFrame, str]:
    path = STORAGE_DIR / "fred_iorb_lagged.parquet"
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if not refresh and not cached.empty and now - pd.Timestamp(path.stat().st_mtime, unit="s") < pd.Timedelta(hours=20):
        return cached, "FRED_LAGGED_CACHE"
    try:
        source = download_fred_series("IORB", api_key=api_key, observation_start="2021-07-29")
        source = source.loc[pd.to_datetime(source["Date"]).ge(pd.Timestamp("2021-07-29"))].dropna(subset=["Value"])
        result = pd.DataFrame({
            "ObservationDate": pd.to_datetime(source["Date"]),
            "AvailableDate": pd.to_datetime(source["Date"]) + pd.offsets.BDay(1),
            "Value": pd.to_numeric(source["Value"], errors="coerce"),
        }).dropna()
        if result.empty:
            raise RuntimeError("IORB has no valid post-2021 observations")
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        result.to_parquet(path, index=False)
        return result, "FRED_CURRENT_VINTAGE_1BD_LAG"
    except Exception:
        if not cached.empty:
            return cached, "STALE_FRED_LAGGED_CACHE"
        return pd.DataFrame(), "UNAVAILABLE"


def _move_business_days_old(move: pd.Series, now: pd.Timestamp | None = None) -> int | None:
    return market_business_days_old(move, now)


def _load_move_series(move_path: Path, refresh: bool = False) -> tuple[pd.Series, str]:
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    cached = pd.Series(dtype="float64")
    if move_path.exists():
        try:
            cached = pd.to_numeric(pd.read_parquet(move_path)["Close"], errors="coerce").dropna()
        except Exception:
            cached = pd.Series(dtype="float64")

    cache_is_fresh = move_path.exists() and now - pd.Timestamp(move_path.stat().st_mtime, unit="s") < MOVE_CACHE_MAX_AGE
    if cache_is_fresh and not refresh:
        return cached, "CACHE"

    fetched = pd.Series(dtype="float64")
    for attempt in range(MOVE_FETCH_ATTEMPTS):
        bars = download_completed_ohlcv("^MOVE", period="max")
        if not bars.empty and bars["Close"].notna().sum() >= 156:
            fetched = pd.to_numeric(bars["Close"], errors="coerce").dropna()
            if _move_business_days_old(fetched) is not None and _move_business_days_old(fetched) <= MOVE_MAX_BUSINESS_DAYS_OLD:
                break
        if attempt < MOVE_FETCH_ATTEMPTS - 1:
            time.sleep(MOVE_FETCH_RETRY_SECONDS)

    if not fetched.empty:
        fetched_latest = pd.Timestamp(fetched.index.max())
        cached_latest = pd.Timestamp(cached.index.max()) if not cached.empty else None
        source_is_stale = (_move_business_days_old(fetched) or 0) > MOVE_MAX_BUSINESS_DAYS_OLD
        if cached_latest is not None and fetched_latest < cached_latest:
            return cached, "MARKET_STALE_CACHE"
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        fetched.to_frame("Close").to_parquet(move_path)
        return fetched, "MARKET_STALE" if source_is_stale else "MARKET"
    if not cached.empty:
        return cached, "STALE_CACHE"
    return pd.Series(dtype="float64"), "UNAVAILABLE"


def load_sources(api_key: str | None, refresh: bool = False) -> tuple[dict[str, pd.DataFrame], pd.Series, dict]:
    sources: dict[str, pd.DataFrame] = {}
    status: dict[str, str] = {}
    for series_id in FRED_IDS:
        try:
            sources[series_id], status[series_id] = _load_initial_release(series_id, api_key, refresh)
        except Exception:
            if series_id in CORE_IDS:
                raise
            sources[series_id], status[series_id] = _load_iorb_fallback(api_key, refresh)
    move_path = STORAGE_DIR / "move_price.parquet"
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    move, status["MOVE"] = _load_move_series(move_path, refresh)
    if not move.empty and now.normalize() - pd.Timestamp(move.dropna().index.max()).tz_localize(None).normalize() > pd.Timedelta(days=10):
        move = pd.Series(dtype="float64")
        status["MOVE"] = "STALE_SOURCE"
    return sources, move, status


def refresh_snapshot(api_key: str | None = None, refresh: bool = False) -> FundingSnapshot:
    sources, move, source_status = load_sources(api_key, refresh)
    sofr = sources["SOFR99"]
    dff = sources["DFF"]
    core_releases = sofr[["ObservationDate", "AvailableDate"]].merge(
        dff[["ObservationDate", "AvailableDate"]], on="ObservationDate", how="inner"
    )
    core_data_asof = core_releases[["AvailableDate_x", "AvailableDate_y"]].max(axis=1).max()
    snapshot = build_history(sources, move)
    if snapshot.daily["MoneyMarketStress"].notna().sum() < 52:
        raise RuntimeError("Insufficient released SOFR99/DFF observations")
    calculated_at = pd.Timestamp.now(tz="UTC").isoformat()
    snapshot.daily["LastUpdated"] = calculated_at
    snapshot.weekly["LastUpdated"] = calculated_at
    status = {
        "ModelVersion": MODEL_VERSION,
        "DataAsOf": str(pd.Timestamp(core_data_asof).date()),
        "CalculatedAt": calculated_at,
        "MOVEDataAsOf": str(pd.Timestamp(move.dropna().index.max()).date()) if not move.dropna().empty else None,
        "MOVEBusinessDaysOld": _move_business_days_old(move),
        "SourceStatus": source_status,
        "TimingConvention": "SOFR99/DFF/WRESBAL/GDP use FRED initial releases; IORB uses one-business-day-lagged current vintage if ALFRED is unavailable; completed MOVE daily close",
    }
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    for frame, path in ((snapshot.daily, DAILY_PATH), (snapshot.weekly, WEEKLY_PATH)):
        temporary = path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(temporary, STATUS_PATH)
    snapshot.status = status
    return snapshot


def read_snapshot() -> FundingSnapshot:
    return FundingSnapshot(
        pd.read_parquet(DAILY_PATH) if DAILY_PATH.exists() else pd.DataFrame(),
        pd.read_parquet(WEEKLY_PATH) if WEEKLY_PATH.exists() else pd.DataFrame(),
        json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.exists() else {},
    )
