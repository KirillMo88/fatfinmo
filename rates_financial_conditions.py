from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from finance_core import download_completed_ohlcv_fresh, market_business_days_old
from fred_client import download_fred_series, get_fred_api_key


MODEL_VERSION = "RATES_FINANCIAL_CONDITIONS_V1"
STORAGE_DIR = Path("persistent") / "rates_financial_conditions"
FRED_CACHE = STORAGE_DIR / "fred_series.parquet"
FRED_INITIAL_CACHE = STORAGE_DIR / "fred_initial_release.parquet"
CREDIT_ARCHIVE = STORAGE_DIR / "credit_weekly_archive.parquet"
SNAPSHOT_PATH = STORAGE_DIR / "weekly_history.parquet"
STATUS_PATH = STORAGE_DIR / "status.json"
FRED_IDS = ("DGS2", "DGS10", "DFII10", "FEDFUNDS", "BAMLH0A0HYM2", "BAMLC0A0CM", "NFCI", "ANFCI")
INITIAL_RELEASE_IDS = ("FEDFUNDS", "NFCI", "ANFCI")
MARKET_TICKERS = {"DXY": "DX-Y.NYB", "MOVE": "^MOVE", "SPY": "SPY", "QQQ": "QQQ", "GLD": "GLD", "BTC": "BTC-USD"}
MARKET_MAX_BUSINESS_DAYS_OLD = 7
REGIMES = (
    "BROAD EASING",
    "RATES TIGHTENING / MARKET RESILIENT",
    "RATE RELIEF / FINANCIAL STRESS",
    "BROAD TIGHTENING",
)
HORIZONS = {"3M": 13, "6M": 26, "12M": 52}
EXPORT_FIELDS = (
    "US2YMomentum", "RealYieldMomentum", "RatesPressureScore", "RatesDirection", "FedFundsDirection",
    "HY_OAS_Z", "IG_OAS_Z", "CreditLevel", "HY_OAS_Momentum", "IG_OAS_Momentum",
    "CreditDirectionScore", "CreditDirection", "DXY_Level", "MOVE_Level", "DXYMomentum", "MOVEMomentum",
    "FinancialConditionsLevel", "FinancialConditionsDirectionScore", "FinancialConditionsDirection",
    "FinancialConditionsStressLevel", "NFCILevel", "NFCIMomentum", "NFCIDirection", "ANFCILevel",
    "ANFCIMomentum", "ANFCIDirection", "FCConfirmationCount", "FCConfirmationStatus",
    "FCConfirmationConfidence", "RatesFinancialConditionsRegime", "Delta2Y_13W", "Delta10Y_13W",
    "DeltaSpread_13W", "YieldCurveRegime_13W", "YieldCurveRegime_26W", "RatesFinancialConditionsModelVersion",
)


@dataclass
class RatesSnapshot:
    history: pd.DataFrame
    returns: pd.DataFrame
    transitions: pd.DataFrame
    quality: pd.DataFrame
    status: dict


def expanding_z(values: pd.Series, minimum: int = 156) -> pd.Series:
    source = pd.to_numeric(values, errors="coerce")
    mean = source.expanding(min_periods=minimum).mean()
    std = source.expanding(min_periods=minimum).std(ddof=0)
    return (source - mean) / std.replace(0, np.nan)


def momentum(values: pd.Series) -> pd.Series:
    source = pd.to_numeric(values, errors="coerce")
    return 0.70 * expanding_z(source.diff(26)) + 0.30 * expanding_z(source.diff(13))


def direction(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series("DATA INCOMPLETE", index=numeric.index)
    result.loc[numeric > 0] = "TIGHTENING"
    result.loc[numeric < 0] = "EASING"
    result.loc[numeric.eq(0)] = "UNCHANGED"
    return result


def _curve_regime(two_year: pd.Series, ten_year: pd.Series, weeks: int) -> pd.Series:
    two_move = two_year.diff(weeks)
    ten_move = ten_year.diff(weeks)
    average = (two_move + ten_move) / 2
    spread_move = ten_move - two_move
    result = pd.Series("DATA INCOMPLETE", index=two_year.index)
    for rate_sign, slope_sign, label in (
        (-1, 1, "BULL STEEPENING"), (-1, -1, "BULL FLATTENING"),
        (1, 1, "BEAR STEEPENING"), (1, -1, "BEAR FLATTENING"),
    ):
        result.loc[(average * rate_sign > 0) & (spread_move * slope_sign > 0)] = label
    result.loc[(average.eq(0) | spread_move.eq(0)) & average.notna() & spread_move.notna()] = "UNCHANGED"
    return result


def _confirmation(core: str, nfci: str, anfci: str) -> tuple[str, str, str]:
    states = (core, nfci, anfci)
    if any(value not in {"EASING", "TIGHTENING"} for value in states):
        return "DATA INCOMPLETE", "DATA INCOMPLETE", "LOW"
    agreement = states.count(core)
    if agreement == 3:
        return "3_OF_3", f"3_OF_3_CONFIRMED_{core}", "HIGH"
    if agreement == 2:
        return "2_OF_3", "PARTIAL_CONFIRMATION", "MEDIUM"
    return "1_OF_3", "DIVERGING", "LOW"


def _weekly_asof(series: pd.Series, calendar: pd.DatetimeIndex, kind: str) -> pd.Series:
    source = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if source.empty:
        return pd.Series(np.nan, index=calendar)
    source.index = pd.to_datetime(source.index).tz_localize(None)
    if kind == "chicago":
        # Published Wednesday/Thursday for the prior Friday: seven days is conservative.
        source.index += pd.Timedelta(days=7)
    elif kind == "monthly":
        source.index = source.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=7)
    elif kind == "fred_daily":
        source.index += pd.offsets.BDay(1)
    source = source[~source.index.duplicated(keep="last")]
    weekly = source.resample("W-FRI").last()
    return weekly.reindex(weekly.index.union(calendar)).sort_index().ffill().reindex(calendar)


def build_history(calendar: pd.DatetimeIndex, fred: dict[str, pd.Series], market: dict[str, pd.Series], released_series_ids: frozenset[str] = frozenset()) -> pd.DataFrame:
    frame = pd.DataFrame(index=calendar)
    for series_id in FRED_IDS:
        kind = "released" if series_id in released_series_ids else "chicago" if series_id in {"NFCI", "ANFCI"} else "monthly" if series_id == "FEDFUNDS" else "fred_daily"
        frame[series_id] = _weekly_asof(fred[series_id], calendar, kind)
    for name in MARKET_TICKERS:
        frame[name] = _weekly_asof(market[name], calendar, "market")

    frame["US2YMomentum"] = momentum(frame["DGS2"])
    frame["RealYieldMomentum"] = momentum(frame["DFII10"])
    frame["RatesPressureScore"] = 0.65 * frame["US2YMomentum"] + 0.35 * frame["RealYieldMomentum"]
    frame["RatesDirection"] = direction(frame["RatesPressureScore"])
    fed13, fed26 = frame["FEDFUNDS"].diff(13), frame["FEDFUNDS"].diff(26)
    frame["FedFundsDirection"] = direction(fed26.where(fed26.ne(0), fed13))

    for name, source in (("HY_OAS", "BAMLH0A0HYM2"), ("IG_OAS", "BAMLC0A0CM")):
        frame[f"{name}_Z"] = expanding_z(frame[source])
        frame[f"{name}_Momentum"] = momentum(frame[source])
    frame["CreditLevel"] = 0.70 * frame["HY_OAS_Z"] + 0.30 * frame["IG_OAS_Z"]
    frame["CreditDirectionScore"] = 0.70 * frame["HY_OAS_Momentum"] + 0.30 * frame["IG_OAS_Momentum"]
    frame["CreditDirection"] = direction(frame["CreditDirectionScore"])

    for name in ("DXY", "MOVE"):
        frame[f"{name}_Level"] = expanding_z(frame[name])
        frame[f"{name}Momentum"] = momentum(frame[name])
        frame[f"{name}Direction"] = direction(frame[f"{name}Momentum"])
    frame["FinancialConditionsLevel"] = 0.45 * frame["CreditLevel"] + 0.30 * frame["DXY_Level"] + 0.25 * frame["MOVE_Level"]
    frame["FinancialConditionsDirectionScore"] = 0.45 * frame["CreditDirectionScore"] + 0.30 * frame["DXYMomentum"] + 0.25 * frame["MOVEMomentum"]
    frame["FinancialConditionsDirection"] = direction(frame["FinancialConditionsDirectionScore"])
    level = frame["FinancialConditionsLevel"]
    frame["FinancialConditionsStressLevel"] = np.select(
        [level < 0.65, level < 1.0, level < 1.5, level >= 1.5],
        ["NORMAL", "ELEVATED", "HIGH", "ACUTE"], default="DATA INCOMPLETE",
    )

    for name in ("NFCI", "ANFCI"):
        frame[f"{name}Level"] = frame[name]
        frame[f"{name}Momentum"] = momentum(frame[name])
        frame[f"{name}Direction"] = direction(frame[f"{name}Momentum"])
    confirmations = [
        _confirmation(core, nfci, anfci)
        for core, nfci, anfci in zip(frame["FinancialConditionsDirection"], frame["NFCIDirection"], frame["ANFCIDirection"])
    ]
    frame[["FCConfirmationCount", "FCConfirmationStatus", "FCConfirmationConfidence"]] = confirmations
    regimes = {
        ("EASING", "EASING"): REGIMES[0],
        ("TIGHTENING", "EASING"): REGIMES[1],
        ("EASING", "TIGHTENING"): REGIMES[2],
        ("TIGHTENING", "TIGHTENING"): REGIMES[3],
    }
    frame["RatesFinancialConditionsRegime"] = [
        regimes.get((rates, conditions), "DATA INCOMPLETE")
        for rates, conditions in zip(frame["RatesDirection"], frame["FinancialConditionsDirection"])
    ]
    frame["US10Y_2Y_Spread"] = frame["DGS10"] - frame["DGS2"]
    for weeks in (13, 26):
        frame[f"Delta2Y_{weeks}W"] = frame["DGS2"].diff(weeks)
        frame[f"Delta10Y_{weeks}W"] = frame["DGS10"].diff(weeks)
        frame[f"DeltaSpread_{weeks}W"] = frame["US10Y_2Y_Spread"].diff(weeks)
        frame[f"YieldCurveRegime_{weeks}W"] = _curve_regime(frame["DGS2"], frame["DGS10"], weeks)
    frame["RatesFinancialConditionsModelVersion"] = MODEL_VERSION
    return frame.rename_axis("Date").reset_index()


def forward_return_stats(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset in ("SPY", "QQQ", "GLD", "BTC"):
        prices = pd.to_numeric(history[asset], errors="coerce")
        date = pd.to_datetime(history["Date"])
        eligible = date >= (pd.Timestamp("2020-04-01") if asset == "BTC" else pd.Timestamp("2010-01-01"))
        for horizon, weeks in HORIZONS.items():
            final = prices.shift(-weeks)
            future_path = pd.concat([prices.shift(-step) for step in range(1, weeks + 1)], axis=1)
            returns = (final / prices - 1) * 100
            adverse = ((future_path.min(axis=1) / prices - 1) * 100).clip(upper=0)
            valid = eligible & prices.notna() & final.notna() & future_path.notna().all(axis=1)
            for regime in REGIMES:
                selected = valid & history["RatesFinancialConditionsRegime"].eq(regime)
                r, a = returns.loc[selected], adverse.loc[selected]
                rows.append({
                    "Asset": asset, "Horizon": horizon, "Regime": regime, "N": int(len(r)),
                    "Average Return": r.mean(), "Median Return": r.median(), "Hit Rate": (r > 0).mean() * 100 if len(r) else np.nan,
                    "Average Adverse Excursion": a.mean(), "Median Adverse Excursion": a.median(),
                    "P10 Adverse Excursion": a.quantile(0.10),
                })
    return pd.DataFrame(rows)


def transition_stats(history: pd.DataFrame) -> pd.DataFrame:
    frame = history[["Date", "RatesFinancialConditionsRegime"]].copy()
    if frame.empty:
        return pd.DataFrame(columns=["Year", "Transitions", "Median Duration Weeks", "Episodes <=4W", "Episodes <=8W"])
    frame["Episode"] = frame["RatesFinancialConditionsRegime"].ne(frame["RatesFinancialConditionsRegime"].shift()).cumsum()
    episodes = frame.loc[frame["RatesFinancialConditionsRegime"].isin(REGIMES)].groupby("Episode", sort=True).agg(Start=("Date", "first"), Weeks=("Date", "size"))
    if episodes.empty:
        return pd.DataFrame(columns=["Year", "Transitions", "Median Duration Weeks", "Episodes <=4W", "Episodes <=8W"])
    episodes["Year"] = pd.to_datetime(episodes["Start"]).dt.year
    episodes["Transition"] = 1
    episodes.iloc[0, episodes.columns.get_loc("Transition")] = 0
    out = episodes.groupby("Year").agg(Transitions=("Transition", "sum"),
                                        **{"Median Duration Weeks": ("Weeks", "median"), "Episodes <=4W": ("Weeks", lambda values: int((values <= 4).sum())), "Episodes <=8W": ("Weeks", lambda values: int((values <= 8).sum()))}).reset_index()
    return out


def validation_episodes(history: pd.DataFrame) -> pd.DataFrame:
    episodes = (
        ("2011 sovereign stress", "2011-08-05"),
        ("2015-16 credit slowdown", "2016-02-12"),
        ("Q4 2018 tightening", "2018-12-21"),
        ("March 2020 COVID stress", "2020-03-20"),
        ("2021 reflation", "2021-06-18"),
        ("2022 tightening", "2022-10-14"),
        ("March 2023 banking stress", "2023-03-17"),
        ("2025-26 current cycle", None),
    )
    frame = history.loc[history["RatesFinancialConditionsRegime"].isin(REGIMES)].sort_values("Date")
    rows = []
    for label, date in episodes:
        eligible = frame.loc[pd.to_datetime(frame["Date"]).le(pd.Timestamp(date))] if date else frame
        if eligible.empty:
            continue
        row = eligible.iloc[-1]
        rows.append({"Episode": label, "Date": row["Date"], "Rates": row["RatesDirection"],
                     "Financial Conditions": row["FinancialConditionsDirection"],
                     "FC Level": row["FinancialConditionsLevel"], "Regime": row["RatesFinancialConditionsRegime"]})
    return pd.DataFrame(rows)


def _series_from_frame(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float64")
    values = pd.Series(pd.to_numeric(frame["Value"], errors="coerce").to_numpy(), index=pd.to_datetime(frame["Date"], errors="coerce"))
    return values.dropna().sort_index()


def _read_fred_cache() -> pd.DataFrame:
    return pd.read_parquet(FRED_CACHE) if FRED_CACHE.exists() else pd.DataFrame(columns=["Series_ID", "Date", "Value"])


def _load_initial_releases(api_key: str | None, refresh: bool = False) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    if FRED_INITIAL_CACHE.exists() and not refresh and now - pd.Timestamp(FRED_INITIAL_CACHE.stat().st_mtime, unit="s") < pd.Timedelta(days=1):
        return pd.read_parquet(FRED_INITIAL_CACHE)
    try:
        key = get_fred_api_key(api_key)
        cached = pd.read_parquet(FRED_INITIAL_CACHE) if FRED_INITIAL_CACHE.exists() else pd.DataFrame()
        frames = [cached] if not cached.empty else []
        for series_id in INITIAL_RELEASE_IDS:
            prior = cached.loc[cached["Series_ID"].eq(series_id)] if not cached.empty else pd.DataFrame()
            if prior.empty:
                windows = [(year, min(year + 9, now.year)) for year in (1990, 2000, 2010, 2020)]
            else:
                latest = pd.to_datetime(prior["ObservationDate"], errors="coerce").max()
                windows = [(max(1990, int(latest.year) - 1), now.year)]
            for year, end in windows:
                if end < year:
                    continue
                params = {
                    "series_id": series_id, "api_key": key, "file_type": "json", "output_type": 4,
                    "realtime_start": "1990-01-01", "realtime_end": now.strftime("%Y-%m-%d"),
                    "observation_start": f"{year}-01-01", "observation_end": f"{end}-12-31",
                }
                response = httpx.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=90)
                response.raise_for_status()
                payload = response.json()
                if "error_code" in payload:
                    raise RuntimeError(f"{series_id} initial-release request failed: {payload.get('error_message')}")
                observations = pd.DataFrame(payload.get("observations", []))
                if observations.empty:
                    continue
                frames.append(pd.DataFrame({
                    "Series_ID": series_id,
                    "Date": pd.to_datetime(observations["realtime_start"], errors="coerce"),
                    "ObservationDate": pd.to_datetime(observations["date"], errors="coerce"),
                    "Value": pd.to_numeric(observations["value"].replace(".", np.nan), errors="coerce"),
                }))
        initial = pd.concat(frames, ignore_index=True).dropna(subset=["Date", "Value"])
        initial = initial.sort_values(["Series_ID", "Date", "ObservationDate"]).drop_duplicates(["Series_ID", "ObservationDate"], keep="first")
        for series_id in INITIAL_RELEASE_IDS:
            if initial["Series_ID"].eq(series_id).sum() < 156:
                raise RuntimeError(f"{series_id}: insufficient initial-release history")
        initial.to_parquet(FRED_INITIAL_CACHE, index=False)
        return initial
    except Exception as exc:
        if FRED_INITIAL_CACHE.exists():
            return pd.read_parquet(FRED_INITIAL_CACHE)
        raise RuntimeError(f"FRED initial-release history unavailable: {exc}") from exc


def _load_credit_archive(refresh: bool = False) -> tuple[pd.DataFrame, str]:
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    if CREDIT_ARCHIVE.exists() and not refresh and now - pd.Timestamp(CREDIT_ARCHIVE.stat().st_mtime, unit="s") < pd.Timedelta(days=1):
        return pd.read_parquet(CREDIT_ARCHIVE), "TRADINGVIEW_WEEKLY_CACHE"
    try:
        from tradingview_mcp import call_tool

        parts = []
        for series_id in ("BAMLH0A0HYM2", "BAMLC0A0CM"):
            payload = call_tool("get_ohlcv", {"symbol": f"FRED:{series_id}", "interval": "1W", "count": 5000, "summary": False})
            bars = payload.get("bars") or payload.get("data") or payload.get("candles") or []
            frame = pd.DataFrame(bars)
            if not {"t", "c"}.issubset(frame.columns) or len(frame) < 500:
                raise RuntimeError(f"{series_id}: insufficient same-series weekly history")
            dates = pd.to_datetime(frame["t"], unit="s", errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
            values = pd.to_numeric(frame["c"], errors="coerce")
            # TradingView weekly bars are Monday-labelled; the value is not known until the week ends.
            parts.append(pd.DataFrame({"Series_ID": series_id, "Date": dates + pd.Timedelta(days=7), "Value": values}).dropna())
        archive = pd.concat(parts, ignore_index=True).sort_values(["Series_ID", "Date"])
        archive.to_parquet(CREDIT_ARCHIVE, index=False)
        return archive, "TRADINGVIEW_WEEKLY"
    except Exception as exc:
        if CREDIT_ARCHIVE.exists():
            return pd.read_parquet(CREDIT_ARCHIVE), "STALE_TRADINGVIEW_CACHE"
        raise RuntimeError(f"Required HY/IG OAS full history unavailable: {exc}") from exc


def load_sources(api_key: str | None, refresh: bool = False) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict]:
    cache = _read_fred_cache()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    cache_fresh = FRED_CACHE.exists() and now - pd.Timestamp(FRED_CACHE.stat().st_mtime, unit="s") < pd.Timedelta(days=1)
    frames = []
    status = {}
    for series_id in FRED_IDS:
        cached = cache.loc[cache["Series_ID"].eq(series_id)].copy()
        if cache_fresh and not refresh and not cached.empty:
            frame = cached
            status[series_id] = "CACHE"
        else:
            try:
                frame = download_fred_series(series_id, api_key=api_key, observation_start="1990-01-01")
                if frame.empty or frame["Value"].notna().sum() < 156:
                    raise ValueError(f"{series_id}: insufficient FRED history")
                status[series_id] = "FRED"
            except Exception as exc:
                if cached.empty:
                    raise RuntimeError(f"Required FRED series {series_id} unavailable: {exc}") from exc
                frame = cached
                status[series_id] = "STALE_CACHE"
        frames.append(frame[["Series_ID", "Date", "Value"]])
    all_fred = pd.concat(frames, ignore_index=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    all_fred.to_parquet(FRED_CACHE, index=False)
    initial = _load_initial_releases(api_key, refresh)
    initial_stale = now - pd.Timestamp(FRED_INITIAL_CACHE.stat().st_mtime, unit="s") >= pd.Timedelta(days=1)
    archive, archive_status = _load_credit_archive(refresh)
    fred = {}
    for series_id in FRED_IDS:
        source = all_fred.loc[all_fred["Series_ID"].eq(series_id)]
        if series_id in {"BAMLH0A0HYM2", "BAMLC0A0CM"}:
            older = archive.loc[archive["Series_ID"].eq(series_id)]
            if older.empty:
                raise RuntimeError(f"Required {series_id} historical archive is empty")
            source = (
                pd.concat([older.assign(_priority=1), source.assign(_priority=2)], ignore_index=True)
                .sort_values(["Date", "_priority"])
                .drop_duplicates("Date", keep="last")
            )
            status[series_id] += "+" + archive_status
        elif series_id in INITIAL_RELEASE_IDS:
            source = initial.loc[initial["Series_ID"].eq(series_id)]
            status[series_id] = "STALE_INITIAL_RELEASE_CACHE" if initial_stale else "FRED_INITIAL_RELEASE"
        fred[series_id] = _series_from_frame(source)

    market = {}
    for name, ticker in MARKET_TICKERS.items():
        path = STORAGE_DIR / f"{name.lower()}_price.parquet"
        fresh = path.exists() and now - pd.Timestamp(path.stat().st_mtime, unit="s") < pd.Timedelta(days=1)
        cached = pd.read_parquet(path)["Close"] if path.exists() else pd.Series(dtype="float64")
        if fresh and not refresh and market_business_days_old(cached) is not None and market_business_days_old(cached) <= MARKET_MAX_BUSINESS_DAYS_OLD:
            price = cached
            status[name] = "CACHE"
        else:
            bars, fetch_status = download_completed_ohlcv_fresh(
                ticker, period="max", max_business_days_old=MARKET_MAX_BUSINESS_DAYS_OLD,
            )
            if not bars.empty and len(bars) >= (100 if name == "BTC" else 156):
                price = pd.to_numeric(bars["Close"], errors="coerce").dropna()
                fetched_latest = pd.Timestamp(price.index.max()) if not price.empty else None
                cached_latest = pd.Timestamp(cached.index.max()) if not cached.empty else None
                if cached_latest is not None and fetched_latest is not None and fetched_latest < cached_latest:
                    price = cached
                    status[name] = "MARKET_STALE_CACHE"
                else:
                    price.to_frame("Close").to_parquet(path)
                    status[name] = "MARKET" if fetch_status == "CURRENT" else "MARKET_STALE"
            elif not cached.empty:
                price = cached
                status[name] = "STALE_CACHE"
            else:
                raise RuntimeError(f"Required market series {name} ({ticker}) unavailable")
        market[name] = price
    return fred, market, status


def refresh_snapshot(api_key: str | None = None, refresh: bool = False) -> RatesSnapshot:
    fred, market, sources = load_sources(api_key, refresh=refresh)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    last_friday = today - pd.Timedelta(days=(today.weekday() - 4) % 7)
    if today.weekday() == 4:
        last_friday -= pd.Timedelta(days=7)
    stale = [name for name, series in fred.items()
             if series.dropna().empty or pd.Timestamp(series.dropna().index.max()) < last_friday - pd.Timedelta(days=60 if name == "FEDFUNDS" else 21)]
    stale.extend(
        name for name, series in market.items()
        if market_business_days_old(series) is None or market_business_days_old(series) > MARKET_MAX_BUSINESS_DAYS_OLD
    )
    if stale:
        raise RuntimeError("Required source observations are stale: " + ", ".join(stale))
    calendar = pd.date_range("1990-01-05", last_friday, freq="W-FRI")
    history = build_history(calendar, fred, market, frozenset(INITIAL_RELEASE_IDS))
    quality = pd.DataFrame([{"Series": name, "Source": sources[name], "Last Available": str(series.dropna().index.max().date()) if not series.dropna().empty else "n/a"} for name, series in {**fred, **market}.items()])
    status = {"ModelVersion": MODEL_VERSION, "DataAsOf": str(last_friday.date()), "CalculatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
              "SourceStatus": sources, "TimingConvention": "Completed W-FRI; FRED daily +1 business day; NFCI/ANFCI/FEDFUNDS use initial-release dates; ICE weekly archive +7 days; other FRED/market series use current historical vintages"}
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    temp = SNAPSHOT_PATH.with_suffix(".parquet.tmp")
    history.to_parquet(temp, index=False)
    os.replace(temp, SNAPSHOT_PATH)
    temp_status = STATUS_PATH.with_suffix(".json.tmp")
    temp_status.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(temp_status, STATUS_PATH)
    return RatesSnapshot(history, forward_return_stats(history), transition_stats(history), quality, status)


def read_snapshot() -> RatesSnapshot:
    history = pd.read_parquet(SNAPSHOT_PATH) if SNAPSHOT_PATH.exists() else pd.DataFrame()
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.exists() else {}
    quality = pd.DataFrame([{"Series": name, "Source": source} for name, source in status.get("SourceStatus", {}).items()])
    if history.empty:
        return RatesSnapshot(history, pd.DataFrame(), pd.DataFrame(), quality, status)
    return RatesSnapshot(history, forward_return_stats(history), transition_stats(history), quality, status)
