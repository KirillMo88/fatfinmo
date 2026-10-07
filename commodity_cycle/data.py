from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from fred_client import download_fred_series
from positioning import (
    cftc_asset_config,
    cftc_asset_series,
    cftc_contract_status,
    cftc_latest_status,
    load_positioning_data,
)
from commodity_cycle.term_structure import AGRICULTURE, ENERGY, MONTH_CODES, _contract_expiry

ROOT = Path(__file__).resolve().parent
TERM_STRUCTURE_WORKBOOK = ROOT / "data" / "commodity_term_structure_seasonal_10y.xlsx"

FRED_SERIES = {
    "Petroleum / Energy": "R4247IM163SCEN",
    "Metals": "R4235IM163SCEN",
    "Agriculture": "R4245IM163SCEN",
    "Chemicals": "R4246IM163SCEN",
    "Lumber": "R4233IM163SCEN",
    "Hardware / Plumbing": "R4237IM163SCEN",
    "Machinery": "R4238IM163SCEN",
    "Electrical / Electronics": "R4236IM163SCEN",
    "PPIACO": "PPIACO",
    "CPIAUCSL": "CPIAUCSL",
    "CAPEX": "E318RC1Q027SBEA",
    "FPI": "FPI",
}
PRICE_TICKERS = {
    "WTI": "CL=F",
    "Natural Gas": "NG=F",
    "RBOB": "RB=F",
    "Copper": "HG=F",
    "Aluminum": "ALI=F",
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Soybeans": "ZS=F",
}
PERFORMANCE_TV_SYMBOLS = {
    "WTI": "NYMEX:CL1!",
    "Natural Gas": "NYMEX:NG1!",
    "RBOB": "NYMEX:RB1!",
    "Copper": "COMEX:HG1!",
    "Aluminum": "COMEX:ALI1!",
    "Corn": "CBOT:ZC1!",
    "Wheat": "CBOT:ZW1!",
    "Soybeans": "CBOT:ZS1!",
}
CFTC_ASSET_MAP = {
    "WTI": "WTI", "Natural Gas": "Natural Gas", "RBOB": "RBOB",
    "Copper": "Copper", "Aluminum": "Aluminum", "Corn": "Corn",
    "Wheat": "Wheat", "Soybeans": "Soybeans",
}


def _source_vendor_set(value: Any) -> set[str]:
    text = "" if value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)) else str(value).lower()
    vendors = set()
    for needle, label in (
        ("yahoo", "Yahoo Finance"),
        ("tradingview", "TradingView"),
        ("eia.gov", "EIA"),
        ("westmetall", "Westmetall"),
        ("barchart", "Barchart"),
        ("cme", "CME"),
    ):
        if needle in text:
            vendors.add(label)
    return vendors


def _safe_history_count(value: Any) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return int(numeric) if pd.notna(numeric) else 0


def _seasonal_history_sources(
    seasonal_history: pd.DataFrame,
    asset: str,
    pair_key: str,
    before: pd.Timestamp,
    *,
    month: int | None,
    count: int,
) -> tuple[str, set[str]]:
    if seasonal_history.empty or "Source" not in seasonal_history:
        return "N/A", set()
    dates = pd.to_datetime(seasonal_history["Date"], errors="coerce")
    mask = (
        seasonal_history["Asset"].astype(str).eq(asset)
        & seasonal_history["PairKey"].astype(str).eq(pair_key)
        & dates.lt(pd.Timestamp(before))
    )
    if month is not None:
        mask &= dates.dt.month.eq(month)
    sample = seasonal_history.loc[mask].copy()
    sample["Date"] = pd.to_datetime(sample["Date"], errors="coerce")
    sources = sample.sort_values("Date").tail(count)["Source"].dropna().astype(str).unique().tolist()
    vendors: set[str] = set()
    for source in sources:
        vendors.update(_source_vendor_set(source))
    return " + ".join(sources) if sources else "N/A", vendors


def _official_seasonal_result(
    row: pd.Series,
    seasonal_history: pd.DataFrame,
    percentile_with_history: Any,
) -> dict[str, Any]:
    """Rank the live spread against same-period history; never substitute a historical signal."""
    asset = str(row.get("asset", row.get("Asset", "")))
    pair_key = str(row.get("pair_key", row.get("PairKey", "")))
    as_of = pd.to_datetime(row.get("As Of", row.get("as_of")), errors="coerce")
    source = str(row.get("Source", row.get("source", "N/A")))
    current_vendors = _source_vendor_set(source)
    result: dict[str, Any] = {
        "p5": np.nan,
        "p10": np.nan,
        "as_of": None,
        "status": "N/A",
        "history_5y_n": 0,
        "history_5y_status": "INSUFFICIENT_5Y_HISTORY",
        "history_10y_n": 0,
        "history_10y_status": "INSUFFICIENT_10Y_HISTORY",
        "history_source": "N/A",
        "history_vendor": "N/A",
        "current_curve_vendor": " + ".join(sorted(current_vendors)) if current_vendors else "UNKNOWN",
        "vendor_consistency": "UNKNOWN",
        "10y_explanation": "N/A: no comparable seasonal observation",
    }

    def finish(
        value5: float,
        value10: float,
        observation_date: pd.Timestamp,
        status: str,
        info5: dict[str, Any],
        info10: dict[str, Any],
        *,
        month_specific: bool,
        as_of_format: str,
    ) -> dict[str, Any]:
        source_text, history_vendors = _seasonal_history_sources(
            seasonal_history,
            asset,
            pair_key,
            observation_date,
            month=observation_date.month if month_specific else None,
            count=10 if _safe_history_count(info10.get("history_n", 0)) else 5,
        )
        consistency = "UNKNOWN"
        if current_vendors and history_vendors:
            consistency = "SAME_VENDOR" if current_vendors == history_vendors else "CROSS_VENDOR"
        history_10y_n = _safe_history_count(info10.get("history_n", 0))
        explanation = (
            "AVAILABLE"
            if pd.notna(value10)
            else f"N/A: requires 10 prior comparable observations; {history_10y_n}/10 available"
        )
        return {
            **result,
            "p5": value5,
            "p10": value10,
            "as_of": observation_date.strftime(as_of_format),
            "status": status,
            "history_5y_n": _safe_history_count(info5.get("history_n", 0)),
            "history_5y_status": info5.get("history_status", "INSUFFICIENT_5Y_HISTORY"),
            "history_10y_n": history_10y_n,
            "history_10y_status": info10.get("history_status", "INSUFFICIENT_10Y_HISTORY"),
            "history_source": source_text,
            "history_vendor": " + ".join(sorted(history_vendors)) if history_vendors else "UNKNOWN",
            "vendor_consistency": consistency,
            "10y_explanation": explanation,
        }

    if asset in {"Corn", "Wheat", "Soybeans"}:
        # Agriculture percentiles are calculated from the current seasonal
        # contract spread against prior comparable seasons upstream.
        p5 = row.get("Seasonal Pctl 5Y", np.nan)
        p10 = row.get("Seasonal Pctl 10Y", np.nan)
        current_spread = pd.to_numeric(pd.Series([row.get("Spread %", row.get("spread"))]), errors="coerce").iloc[0]
        if pd.isna(current_spread) or pd.isna(as_of):
            return result
        info5 = {
            "history_n": row.get("5Y HistoryN", 0),
            "history_status": row.get("5Y HistoryStatus", "INSUFFICIENT_5Y_HISTORY"),
        }
        info10 = {
            "history_n": row.get("10Y HistoryN", 0),
            "history_status": row.get("10Y HistoryStatus", "INSUFFICIENT_10Y_HISTORY"),
        }
        status = "CURRENT_SPREAD" if pd.notna(p5) or pd.notna(p10) else "N/A"
        return finish(p5, p10, pd.Timestamp(as_of), status, info5, info10,
                      month_specific=False, as_of_format="%Y-%m-%d")
    current_spread = pd.to_numeric(pd.Series([row.get("Spread %", row.get("spread"))]), errors="coerce").iloc[0]
    if pd.isna(as_of) or not np.isfinite(current_spread):
        return result
    current_period = pd.Timestamp(as_of).to_period("M")
    info5 = percentile_with_history(
        float(current_spread), seasonal_history, asset, current_period.month, pair_key, 5,
        before=current_period.to_timestamp()
    )
    info10 = percentile_with_history(
        float(current_spread), seasonal_history, asset, current_period.month, pair_key, 10,
        before=current_period.to_timestamp()
    )
    status = "CURRENT_SPREAD" if pd.notna(info5["percentile"]) or pd.notna(info10["percentile"]) else "N/A"
    return finish(info5["percentile"], info10["percentile"], pd.Timestamp(as_of), status,
                  info5, info10, month_specific=True, as_of_format="%Y-%m-%d")


def _official_seasonal_percentiles(
    row: pd.Series,
    seasonal_history: pd.DataFrame,
    percentile_with_history: Any,
) -> tuple[float, float, str | None, str]:
    """Compatibility wrapper for the four primary seasonal outputs."""
    result = _official_seasonal_result(row, seasonal_history, percentile_with_history)
    return result["p5"], result["p10"], result["as_of"], result["status"]


def _contract_selection_quality(asset: str, rollover_method: Any) -> str:
    if asset in {"Copper", "Aluminum"}:
        return "NOT_APPLICABLE_CASH_3M"
    return {
        "VOLUME_CROSSOVER": "HIGH",
        "OPEN_INTEREST_CROSSOVER": "MEDIUM",
        "HARD_EXPIRY_ROLL": "MEDIUM",
        "CALENDAR_FALLBACK": "LOW",
    }.get(str(rollover_method or ""), "UNKNOWN")


def _combined_curve_quality(price_quality: Any, selection_quality: Any) -> str:
    price = str(price_quality or "INVALID")
    selection = str(selection_quality or "UNKNOWN")
    if price not in {"HIGH", "LOW_LIQUIDITY"}:
        return price
    if selection == "LOW":
        return "LOW_CONTRACT_SELECTION"
    if selection == "MEDIUM":
        return "MEDIUM_CONTRACT_SELECTION"
    return price


def load_commodity_term_structure() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load immutable monthly baseline and refresh normalized current/live curve data."""
    if not TERM_STRUCTURE_WORKBOOK.exists():
        raise FileNotFoundError(f"Commodity seasonal workbook not found: {TERM_STRUCTURE_WORKBOOK}")
    history = pd.read_excel(TERM_STRUCTURE_WORKBOOK, sheet_name="App_Export", engine="openpyxl")
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    history["Spread %"] = pd.to_numeric(history["Spread %"], errors="coerce")
    history = history.dropna(subset=["Date"])

    from commodity_cycle.term_structure_pipeline import (
        TermStructureStore, fetch_current_term_structure, percentile_with_history, seasonal_history_frame,
    )

    store = TermStructureStore()
    bundle = fetch_current_term_structure(store, history)
    current = bundle["current"].copy()
    monthly = bundle["monthly"].copy()
    seasonal_history = seasonal_history_frame(history, monthly)
    agriculture_seasonal = bundle.get("agriculture_seasonal", pd.DataFrame()).copy()
    if not agriculture_seasonal.empty:
        valid_agriculture = agriculture_seasonal.loc[
            agriculture_seasonal["Data Quality"].isin(["OK", "ACCEPTABLE"])
            & pd.to_numeric(agriculture_seasonal["Median Seasonal Spread"], errors="coerce").notna()
        ].copy()
        if not valid_agriculture.empty:
            ag_history = pd.DataFrame({
                "Asset": valid_agriculture["Asset"].astype(str),
                "Date": pd.to_datetime(valid_agriculture["Matched Date End"], errors="coerce"),
                "Month": pd.to_datetime(valid_agriculture["Matched Date End"], errors="coerce").dt.month,
                "Spread": pd.to_numeric(valid_agriculture["Median Seasonal Spread"], errors="coerce"),
                "PairKey": valid_agriculture["PairKey"].astype(str),
                "Source": valid_agriculture.get("Source", "TradingView MCP daily close"),
                "History Data Quality": valid_agriculture.get("Data Quality", "N/A"),
            })
            seasonal_history = pd.concat([seasonal_history, ag_history], ignore_index=True).dropna(
                subset=["Asset", "Date", "Spread", "PairKey"]
            )
            seasonal_history = seasonal_history.sort_values("Date").drop_duplicates(
                ["Asset", "PairKey", "Date"], keep="last"
            )
    if current.empty:
        current = pd.DataFrame(columns=[
            "Asset", "As Of", "Leg1", "Leg2", "Spread %", "Raw Curve State", "Seasonal Pctl 10Y",
            "Seasonal Pctl 5Y", "Structure", "Source", "Status", "CurveDataQuality",
        ])
    else:
        current["Asset"] = current["asset"]
        current["As Of"] = pd.to_datetime(current["as_of"], errors="coerce")
        current["Leg1"] = pd.to_numeric(current.get("leg1_price"), errors="coerce")
        current["Leg2"] = pd.to_numeric(current.get("leg2_price"), errors="coerce")
        current["Spread %"] = pd.to_numeric(current.get("spread"), errors="coerce")
        current["Leg1 Contract"] = current.get("leg1_contract")
        current["Leg2 Contract"] = current.get("leg2_contract")
        current["Structure"] = current.get("structure", current.get("pair_key"))
        current["Source"] = current.get("source")
        current["Raw Curve State"] = current.get("raw_state", "N/A")
        current["PriceDataQuality"] = current.get("quality", "INVALID")
        current["ContractSelectionQuality"] = [
            _contract_selection_quality(str(asset), method)
            for asset, method in zip(current["asset"], current.get("rollover_method", pd.Series(index=current.index, dtype=object)))
        ]
        current["CurveDataQuality"] = [
            _combined_curve_quality(price_quality, selection_quality)
            for price_quality, selection_quality in zip(current["PriceDataQuality"], current["ContractSelectionQuality"])
        ]
        current["Status"] = current["CurveDataQuality"]
        current["Seasonal Pctl 5Y"] = current.get("Seasonal Pctl 5Y", np.nan)
        current["Seasonal Pctl 10Y"] = current.get("Seasonal Pctl 10Y", np.nan)
        current["Current Month Seasonal Pctl 5Y"] = current["Seasonal Pctl 5Y"]
        current["Current Month Seasonal Pctl 10Y"] = current["Seasonal Pctl 10Y"]
        # These are current-spread percentiles. Historical rows are comparison
        # samples only and are never promoted into the current signal.
        seasonal_results: list[dict[str, Any]] = []
        for _, row in current.iterrows():
            seasonal_results.append(_official_seasonal_result(row, seasonal_history, percentile_with_history))
        current["Official Seasonal Pctl 5Y"] = [item["p5"] for item in seasonal_results]
        current["Official Seasonal Pctl 10Y"] = [item["p10"] for item in seasonal_results]
        current["Official Seasonal State"] = [
            _term_curve_state(item["p10"] if pd.notna(item["p10"]) else item["p5"])
            for item in seasonal_results
        ]
        current["Seasonal Percentile As Of"] = [item["as_of"] for item in seasonal_results]
        current["Seasonal Percentile Status"] = [item["status"] for item in seasonal_results]
        current["Seasonal Percentile 5Y HistoryN"] = [item["history_5y_n"] for item in seasonal_results]
        current["Seasonal Percentile 5Y HistoryStatus"] = [item["history_5y_status"] for item in seasonal_results]
        current["Seasonal Percentile 10Y HistoryN"] = [item["history_10y_n"] for item in seasonal_results]
        current["Seasonal Percentile 10Y HistoryStatus"] = [item["history_10y_status"] for item in seasonal_results]
        current["Seasonal Percentile 10Y Explanation"] = [item["10y_explanation"] for item in seasonal_results]
        current["Current Curve Vendor"] = [item["current_curve_vendor"] for item in seasonal_results]
        current["Seasonal History Source"] = [item["history_source"] for item in seasonal_results]
        current["Seasonal History Vendor"] = [item["history_vendor"] for item in seasonal_results]
        current["Vendor Consistency"] = [item["vendor_consistency"] for item in seasonal_results]
        current["MTD Average Spread"] = current.get("mtd_spread")
        current["MTD Average Leg1 Price"] = current.get("mtd_avg_leg1")
        current["MTD Average Leg2 Price"] = current.get("mtd_avg_leg2")
        current["MTD Daily Observations"] = current.get("mtd_observation_count", 0)
        current["Current Seasonal Status"] = current.get("current_seasonal_status", "N/A")
        current["Source URL"] = current["Source"]
        current["Data Quality"] = current["CurveDataQuality"]
        current["Observed At"] = pd.to_datetime(current.get("observed_at"), errors="coerce", utc=True)
        current["F1 Symbol"] = current.get("active_f1", current.get("leg1_contract"))
        current["F3 Symbol"] = current.get("leg2_contract")
        current["F6 Symbol"] = np.nan
        current["Rollover Date"] = current.get("rollover_date")
        current["Rollover Method"] = current.get("rollover_method")
        current["Days To Expiry"] = current.get("days_to_expiry")
        current["SeasonalPairKey"] = current.get("pair_key")
        f6_symbols = current.loc[current["pair_key"].astype(str).str.endswith("F1/F6"), ["asset", "leg2_contract"]].drop_duplicates("asset").set_index("asset")["leg2_contract"] if "pair_key" in current else pd.Series(dtype=object)
        current["F6 Symbol"] = current["asset"].map(f6_symbols)

    normalized_history = seasonal_history.rename(columns={"Spread": "Spread %", "PairKey": "Structure"}).copy()
    if not normalized_history.empty:
        normalized_history["Date"] = pd.to_datetime(normalized_history["Date"], errors="coerce")
        normalized_history["Month"] = normalized_history["Date"].dt.month
        normalized_history["Asset"] = normalized_history["Asset"].astype(str)
    return normalized_history, current, bundle["diagnostics"]


def _baseline_structure_name(asset: str) -> str:
    if asset in {"WTI", "Natural Gas", "RBOB"}:
        return "F1/F3"
    if asset in {"Copper", "Aluminum"}:
        return "Cash/3M"
    if asset in {"Corn", "Wheat"}:
        return "Dec/Mar"
    if asset == "Soybeans":
        return "Nov/Jan"
    return ""


def _term_quote_fresh(as_of: Any, source: Any) -> bool:
    if pd.isna(as_of):
        return False
    from commodity_cycle.term_structure import quote_is_fresh
    return quote_is_fresh(pd.Timestamp(as_of), max_business_days=3)


def _term_curve_state(value: Any) -> str:
    from commodity_cycle.model import classify_seasonal_curve
    return classify_seasonal_curve(value)


def load_fred_history(api_key: str | None, start: str = "1970-01-01") -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch each FRED series independently so one unavailable series does not hide others."""
    def fetch_one(item: tuple[str, str]) -> tuple[str, pd.DataFrame | None, str]:
        name, series_id = item
        try:
            frame = download_fred_series(series_id, api_key=api_key, observation_start=start)
            frame = frame.rename(columns={"Date": "date", "Value": name})[["date", name]]
            if frame.empty:
                return name, frame, "MISSING"
            age_days = (pd.Timestamp.now().normalize() - pd.to_datetime(frame["date"]).max().normalize()).days
            status = "STALE" if age_days > (150 if name in {"CAPEX", "FPI"} else 60) else "CURRENT"
            return name, frame, status
        except Exception as exc:
            return name, None, f"FAILED: {exc}"
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(fetch_one, FRED_SERIES.items()))
    observations = [frame for _, frame, _ in results if frame is not None]
    statuses = {name: status for name, _, status in results}
    if not observations:
        return pd.DataFrame(), statuses
    combined: pd.DataFrame | None = None
    for frame in observations:
        combined = frame if combined is None else combined.merge(frame, on="date", how="outer")
    assert combined is not None
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"]).sort_values("date").set_index("date")
    return combined, statuses


def _price_at_or_before(series: pd.Series, target: pd.Timestamp) -> float:
    history = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    eligible = history.loc[history.index <= pd.Timestamp(target)]
    return float(eligible.iloc[-1]) if not eligible.empty else np.nan


def price_returns_from_daily(series: pd.Series) -> dict[str, float]:
    """Calculate current-price returns against prices 4/13/26/52 weeks ago."""
    history = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if history.empty:
        return {label: np.nan for label in ("Return 1M", "Return 3M", "Return 6M", "Return 12M")}
    current_date = pd.Timestamp(history.index[-1])
    current_price = float(history.iloc[-1])
    out: dict[str, float] = {}
    for label, weeks in (("Return 1M", 4), ("Return 3M", 13), ("Return 6M", 26), ("Return 12M", 52)):
        reference = _price_at_or_before(history, current_date - pd.Timedelta(weeks=weeks))
        out[label] = current_price / reference - 1.0 if np.isfinite(reference) and reference != 0 else np.nan
    return out


def annualized_curve_spread(
    asset: str,
    leg1_contract: Any,
    leg2_contract: Any,
    curve_spread: Any,
    as_of: Any,
) -> float:
    """Compound the observed leg1/leg2 spread to a one-year equivalent.

    Futures use the actual gap between the two contract expiry dates. LME
    Cash/3M uses its three-month tenor measured from the observation date.
    """
    spread = pd.to_numeric(pd.Series([curve_spread]), errors="coerce").iloc[0]
    observed = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(spread) or pd.isna(observed) or 1.0 + float(spread) <= 0:
        return np.nan

    if asset in {"Copper", "Aluminum"}:
        days = (observed + pd.DateOffset(months=3) - observed).days
    else:
        spec = ENERGY.get(asset) or AGRICULTURE.get(asset)
        if not spec or pd.isna(leg1_contract) or pd.isna(leg2_contract):
            return np.nan
        root = spec[0]
        month_number = {code: month for month, code in MONTH_CODES.items()}

        def expiry_for(symbol: Any) -> pd.Timestamp | None:
            contract = str(symbol).split(".", 1)[0]
            if not contract.startswith(root) or len(contract) < len(root) + 3:
                return None
            month = month_number.get(contract[len(root)])
            try:
                short_year = int(contract[len(root) + 1:len(root) + 3])
            except ValueError:
                return None
            if month is None:
                return None
            year = (observed.year // 100) * 100 + short_year
            if year < observed.year - 20:
                year += 100
            elif year > observed.year + 79:
                year -= 100
            return _contract_expiry(asset, year, month)

        expiry1 = expiry_for(leg1_contract)
        expiry2 = expiry_for(leg2_contract)
        if expiry1 is None or expiry2 is None:
            return np.nan
        days = (expiry2 - expiry1).days

    if days <= 0:
        return np.nan
    return float((1.0 + float(spread)) ** (365.0 / days) - 1.0)


def load_monthly_prices() -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch Yahoo daily continuous-futures prices for displayed price and charts."""
    try:
        import yfinance as yf
    except Exception as exc:
        return pd.DataFrame(), {asset: f"FAILED: yfinance unavailable ({exc})" for asset in PRICE_TICKERS}
    frames: list[pd.Series] = []
    status: dict[str, str] = {}
    for asset, ticker in PRICE_TICKERS.items():
        try:
            frame = yf.download(ticker, period="max", interval="1d", auto_adjust=False, progress=False, threads=False)
            if frame is None or frame.empty:
                status[asset] = "MISSING"
                continue
            close = frame["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            series = pd.to_numeric(close, errors="coerce").dropna()
            series.index = pd.to_datetime(series.index, errors="coerce").tz_localize(None).normalize()
            series = series[~series.index.isna()].groupby(level=0).last()
            if series.empty:
                status[asset] = "MISSING"
                continue
            series.name = asset
            frames.append(series)
            age_days = (pd.Timestamp.now().normalize() - series.index.max().normalize()).days
            freshness = "STALE" if age_days > 7 else "CURRENT"
            status[asset] = f"{freshness}: {series.index.max().date().isoformat()} via Yahoo Finance ({ticker})"
        except Exception as exc:
            status[asset] = f"FAILED: {exc}"
    return (pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()), status


def load_tradingview_performance_prices(
    fallback_prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Fetch TradingView weekly closes, falling back to Yahoo daily closes per asset."""
    try:
        from tradingview_mcp import get_ohlcv_data
    except Exception as exc:
        get_ohlcv_data = None
        client_error = f"TradingView MCP client unavailable ({exc})"
    else:
        client_error = ""

    frames: list[pd.Series] = []
    status: dict[str, str] = {}
    source: dict[str, str] = {}
    for asset, symbol in PERFORMANCE_TV_SYMBOLS.items():
        tv_error = client_error
        series = pd.Series(dtype=float)
        if get_ohlcv_data is None:
            tv_error = client_error
        else:
            for attempt in range(2):
                try:
                    bars = get_ohlcv_data(symbol, interval="1W", count=500, force=True)
                    if bars is None or bars.empty or not {"date", "close"}.issubset(bars.columns):
                        tv_error = f"empty/invalid OHLCV response for {symbol}"
                    else:
                        dates = pd.to_datetime(bars["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
                        closes = pd.to_numeric(bars["close"], errors="coerce")
                        series = pd.Series(closes.to_numpy(), index=dates).dropna()
                        series = series.loc[~series.index.isna()].groupby(level=0).last().sort_index()
                        if series.empty:
                            tv_error = f"empty/invalid OHLCV response for {symbol}"
                        else:
                            break
                except Exception as exc:
                    tv_error = f"{type(exc).__name__}: {exc}"
                if attempt == 0:
                    continue
            if not series.empty:
                series.name = asset
                frames.append(series)
                source[asset] = "TradingView MCP"
                age_days = (pd.Timestamp.now().normalize() - series.index[-1].normalize()).days
                freshness = "STALE" if age_days > 7 else "CURRENT"
                status[asset] = f"{freshness}: {series.index[-1].date().isoformat()} via TradingView MCP weekly ({symbol})"
                continue

        yahoo = pd.Series(dtype=float)
        if fallback_prices is not None and asset in fallback_prices:
            yahoo = pd.to_numeric(fallback_prices[asset], errors="coerce").dropna().sort_index()
        if not yahoo.empty:
            yahoo.index = pd.to_datetime(yahoo.index, errors="coerce", utc=True).tz_localize(None).normalize()
            yahoo = yahoo.loc[~yahoo.index.isna()].groupby(level=0).last().sort_index()
            yahoo.name = asset
            frames.append(yahoo)
            source[asset] = "Yahoo Finance"
            age_days = (pd.Timestamp.now().normalize() - yahoo.index[-1].normalize()).days
            freshness = "STALE" if age_days > 7 else "CURRENT"
            ticker = PRICE_TICKERS[asset]
            status[asset] = (
                f"{freshness}: {yahoo.index[-1].date().isoformat()} via Yahoo Finance fallback ({ticker}); "
                f"TradingView unavailable: {tv_error}"
            )
        else:
            source[asset] = "N/A"
            status[asset] = f"MISSING: TradingView ({tv_error}); Yahoo Finance ({PRICE_TICKERS[asset]}) also unavailable"
    prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    return prices, status, source


def cftc_snapshot_from_master(
    master: pd.DataFrame,
    *,
    source_error: bool = False,
    reference_date: pd.Timestamp | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build current commodity CFTC signals while retaining non-current history for diagnostics."""
    out: dict[str, dict[str, Any]] = {}
    for asset in CFTC_ASSET_MAP:
        series = cftc_asset_series(master, CFTC_ASSET_MAP[asset], "Managed Money")
        contract = cftc_contract_status(
            master,
            CFTC_ASSET_MAP[asset],
            source_error=source_error,
            reference_date=reference_date,
        )
        cfg = cftc_asset_config(CFTC_ASSET_MAP[asset])
        if series.empty:
            out[asset] = {
                "Status": contract["status"],
                "CFTC Status": contract["status"],
                "CFTC Contract Market Code": contract.get("code"),
                "Latest Official CFTC Report Date": contract.get("latest_report_date"),
                "Series Present In Latest Report": "NO",
                "Reason Current Signal Missing": "EXACT_CFTC_SERIES_UNAVAILABLE",
            }
            continue
        row = series.sort_values("Date").iloc[-1]
        is_current = contract["status"] == "CURRENT"
        last_date = pd.to_datetime(row.get("Date"), errors="coerce")
        last_net = row.get("NetPctOI")
        last_pctl = row.get("NetPctOI_5Y_Percentile")
        missing_reason = None
        if not is_current:
            missing_reason = (
                "OUTRIGHT_ALUMINUM_SERIES_NOT_PRESENT_IN_LATEST_CFTC_REPORT"
                if asset == "Aluminum" and contract["status"] == "SERIES_NOT_CURRENT"
                else "EXACT_CFTC_SERIES_NOT_CURRENT"
            )
        out[asset] = {
                "MM Net % OI": last_net if is_current else np.nan,
                "Net Direction": "Net Long" if is_current and pd.notna(last_net) and last_net > 0 else "Net Short" if is_current and pd.notna(last_net) and last_net < 0 else "Neutral" if is_current and pd.notna(last_net) else "N/A",
                "3Y Percentile": row.get("NetPctOI_3Y_Percentile") if is_current else np.nan,
                "5Y Percentile": last_pctl if is_current else np.nan,
                "4W Change": row.get("NetPctOI_4W_Change") if is_current else np.nan,
                "13W Change": row.get("NetPctOI_13W_Change") if is_current else np.nan,
                "Updated Date": last_date if is_current else pd.NaT,
                "CFTC As Of": last_date if is_current else pd.NaT,
                "Series Observation Date": last_date,
                "History Weeks": row.get("History_Weeks"),
                "History Quality": "READY" if float(row.get("History_Weeks", 0) or 0) >= 260 else "INSUFFICIENT_5Y_HISTORY",
                "Status": contract["status"],
                "CFTC Status": contract["status"],
                "CFTC Contract Market Code": contract.get("code") or (cfg.code_patterns[0] if cfg and cfg.code_patterns else None),
                "CFTC Market Name": row.get("Market_Name", row.get("Raw_Contract_Name")),
                "Latest Official CFTC Report Date": contract.get("latest_report_date"),
                "Series Present In Latest Report": "YES" if contract.get("present_in_latest_report") else "NO",
                "CFTC Open Interest": row.get("Open_Interest") if is_current else np.nan,
                "MM Long": row.get("Long") if is_current else np.nan,
                "MM Short": row.get("Short") if is_current else np.nan,
                "MM Spreading": row.get("Spreading") if is_current else np.nan,
                "MM Net": row.get("Net") if is_current else np.nan,
                "MM Traders Long": row.get("Traders_Long") if is_current else np.nan,
                "MM Traders Short": row.get("Traders_Short") if is_current else np.nan,
                "MM Traders Spreading": row.get("Traders_Spread") if is_current else np.nan,
                "Last Available Date": last_date,
                "Last Available MM Net % OI": last_net,
                "Last Available COT 5Y Percentile": last_pctl,
                "Reason Current Signal Missing": missing_reason,
        }
    return out


def load_cftc_snapshot() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Consume the existing positioning store/service without creating another pipeline."""
    try:
        result = load_positioning_data(force_update=False)
        master = result.get("cftc_master", pd.DataFrame())
        source_details = result.get("status", {}).get("CFTC Commodities", {})
        source_error = str(source_details.get("status", "")).startswith("SOURCE_FAILED")
        source_freshness = cftc_latest_status(master, "Disaggregated").get("status", "DATA UNAVAILABLE")
        out = cftc_snapshot_from_master(master, source_error=source_error)
        status = {"CFTC": source_freshness if out else "MISSING"}
        return out, status
    except Exception as exc:
        return {}, {"CFTC": f"FAILED: {exc}"}


def _as_of_alignment(*values: Any) -> str:
    parsed = [pd.to_datetime(value, errors="coerce") for value in values]
    valid = [pd.Timestamp(value).normalize() for value in parsed if pd.notna(value)]
    if len(valid) != len(values):
        return "INCOMPLETE_AS_OF"
    return "ALIGNED" if len(set(valid)) == 1 else "MIXED_AS_OF"


def build_market_confirmation(
    monthly_prices: pd.DataFrame,
    term_history: pd.DataFrame,
    term_current: pd.DataFrame,
    cftc: dict[str, dict[str, Any]],
    performance_prices: pd.DataFrame | None = None,
    performance_sources: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute commodity and sector snapshots from prices, current live curves and canonical CFTC."""
    from commodity_cycle.model import (
        COMMODITY_SECTORS,
        classify_commodity_price_momentum,
        classify_cftc_relative_state,
        classify_sector_price_state,
        classify_seasonal_curve,
        resolve_price_curve_market_state,
        resolve_cftc_qualifier,
    )

    current_rows: list[dict[str, Any]] = []
    performance_prices = performance_prices if performance_prices is not None else pd.DataFrame()
    performance_sources = performance_sources or {}
    for asset in PRICE_TICKERS:
        price = monthly_prices.get(asset, pd.Series(dtype=float)).dropna() if not monthly_prices.empty else pd.Series(dtype=float)
        performance = performance_prices.get(asset, pd.Series(dtype=float)).dropna() if not performance_prices.empty else pd.Series(dtype=float)
        returns = price_returns_from_daily(performance)
        r1, r3, r6, r12 = (returns[label] for label in ("Return 1M", "Return 3M", "Return 6M", "Return 12M"))
        pstate = classify_commodity_price_momentum(r3, r6, r12)
        curve = term_current.loc[term_current["Asset"].eq(asset)] if not term_current.empty else pd.DataFrame()
        if not curve.empty and "Structure" in curve:
            preferred = {"WTI": "F1/F3", "Natural Gas": "F1/F3", "RBOB": "F1/F3", "Corn": "Dec/Mar", "Wheat": "Dec/Mar", "Soybeans": "Nov/Jan", "Copper": "Cash/3M", "Aluminum": "Cash/3M"}.get(asset)
            primary = curve.loc[curve["Structure"].eq(preferred)] if preferred else pd.DataFrame()
            curve = primary if not primary.empty else curve
        curve_row = curve.sort_values("As Of").tail(1).iloc[0].to_dict() if not curve.empty else {}
        seasonal_pctl = curve_row.get("Official Seasonal Pctl 10Y", curve_row.get("Seasonal Pctl 10Y"))
        curve_date = pd.to_datetime(curve_row.get("As Of"), errors="coerce")
        seasonal_5y = curve_row.get("Official Seasonal Pctl 5Y", curve_row.get("Seasonal Pctl 5Y"))
        curve_state = curve_row.get("Official Seasonal State", classify_seasonal_curve(seasonal_pctl))
        raw = curve_row.get("Raw Curve State", curve_row.get("Structure", "N/A"))
        c = cftc.get(asset, {})
        row = {
            "Sector": next((sector for sector, assets in COMMODITY_SECTORS.items() if asset in assets), "N/A"),
            "Commodity": asset,
            "Price": float(price.iloc[-1]) if len(price) else np.nan,
            "Price Date": price.index[-1] if len(price) else pd.NaT,
            "Performance As Of": performance.index[-1] if len(performance) else pd.NaT,
            "Analytics Return Source": performance_sources.get(asset, "N/A"),
            "Return 1M": r1, "Return 3M": r3, "Return 6M": r6, "Return 12M": r12,
            "Price State": pstate,
            "Term Structure As Of": curve_row.get("As Of"),
            "Term Structure Source": curve_row.get("Source"),
            "Term Structure Structure": curve_row.get("Structure"),
            "F1 Symbol": curve_row.get("F1 Symbol"), "F3 Symbol": curve_row.get("F3 Symbol"), "F6 Symbol": curve_row.get("F6 Symbol"),
            "CurveDataQuality": curve_row.get("CurveDataQuality"),
            "Price Data Quality": curve_row.get("PriceDataQuality"),
            "Contract Selection Quality": curve_row.get("ContractSelectionQuality"),
            "MTD Average Spread": curve_row.get("MTD Average Spread"),
            "MTD Daily Observations": curve_row.get("MTD Daily Observations"),
            "Current Seasonal Status": curve_row.get("Current Seasonal Status"),
            "Rollover Date": curve_row.get("Rollover Date"), "Rollover Method": curve_row.get("Rollover Method"),
            "Days To Expiry": curve_row.get("Days To Expiry"),
            "5Y HistoryN": curve_row.get("5Y HistoryN"), "5Y HistoryStartDate": curve_row.get("5Y HistoryStartDate"),
            "5Y HistoryEndDate": curve_row.get("5Y HistoryEndDate"), "5Y HistoryStatus": curve_row.get("5Y HistoryStatus"),
            "10Y HistoryN": curve_row.get("10Y HistoryN"), "10Y HistoryStartDate": curve_row.get("10Y HistoryStartDate"),
            "10Y HistoryEndDate": curve_row.get("10Y HistoryEndDate"), "10Y HistoryStatus": curve_row.get("10Y HistoryStatus"),
            "Leg 1": curve_row.get("Leg1 Contract", curve_row.get("Leg1")), "Leg 2": curve_row.get("Leg2 Contract", curve_row.get("Leg2")),
            "Curve Spread": curve_row.get("Spread %"),
            "Annualized Curve Spread": annualized_curve_spread(
                asset,
                curve_row.get("Leg1 Contract", curve_row.get("Leg1")),
                curve_row.get("Leg2 Contract", curve_row.get("Leg2")),
                curve_row.get("Spread %"),
                curve_row.get("As Of"),
            ),
            "Raw Curve State": raw,
            "Seasonal Percentile 10Y": seasonal_pctl,
            "Seasonal Percentile 5Y": seasonal_5y,
            "Seasonal Percentile As Of": curve_row.get("Seasonal Percentile As Of"),
            "Seasonal Percentile Status": curve_row.get("Seasonal Percentile Status", "N/A"),
            "Seasonal Percentile 5Y HistoryN": curve_row.get("Seasonal Percentile 5Y HistoryN"),
            "Seasonal Percentile 5Y HistoryStatus": curve_row.get("Seasonal Percentile 5Y HistoryStatus"),
            "Seasonal Percentile 10Y HistoryN": curve_row.get("Seasonal Percentile 10Y HistoryN"),
            "Seasonal Percentile 10Y HistoryStatus": curve_row.get("Seasonal Percentile 10Y HistoryStatus"),
            "Seasonal Percentile 10Y Explanation": curve_row.get("Seasonal Percentile 10Y Explanation"),
            "Current Curve Vendor": curve_row.get("Current Curve Vendor"),
            "Seasonal History Source": curve_row.get("Seasonal History Source"),
            "Seasonal History Vendor": curve_row.get("Seasonal History Vendor"),
            "Vendor Consistency": curve_row.get("Vendor Consistency"),
            "Seasonal Relative State": curve_state,
            "Price × Seasonal Curve": resolve_price_curve_market_state(pstate, curve_state),
            **{
                "MM Net % OI": np.nan, "Net Direction": "N/A", "3Y Percentile": np.nan,
                "5Y Percentile": np.nan, "4W Change": np.nan, "13W Change": np.nan,
                "Updated Date": pd.NaT, "History Weeks": np.nan, **c,
            },
            "CFTC Status": c.get("Status", "MISSING"),
            "CFTC Relative State": classify_cftc_relative_state(c.get("5Y Percentile", np.nan)),
            "Relative Positioning State": classify_cftc_relative_state(c.get("5Y Percentile", np.nan)),
            "CFTC Qualifier": "N/A",
        }
        row["CFTC Qualifier"] = resolve_cftc_qualifier(
            [c.get("5Y Percentile", np.nan)], [c.get("MM Net % OI", np.nan)]
        ) if c else "N/A"
        row["As Of Alignment"] = _as_of_alignment(
            row.get("Price Date"),
            row.get("Performance As Of"),
            row.get("Term Structure As Of"),
            row.get("CFTC As Of"),
            row.get("Seasonal Percentile As Of"),
        )
        current_rows.append(row)

    commodity = pd.DataFrame(current_rows)
    sectors = []
    for sector, assets in COMMODITY_SECTORS.items():
        part = commodity.loc[commodity["Commodity"].isin(assets)]
        price_state = classify_sector_price_state(part["Price State"].tolist(), sector)
        values = pd.to_numeric(part["5Y Percentile"], errors="coerce")
        net = pd.to_numeric(part["MM Net % OI"], errors="coerce")
        dispersion = float(values.max() - values.min()) if values.notna().sum() >= 2 else np.nan
        qualifier = resolve_cftc_qualifier(values.tolist(), net.tolist()) if values.notna().any() else "N/A"
        curves = part["Seasonal Relative State"].tolist()
        tight_count = sum(v in {
            "Mild Tight vs Seasonal", "Strong Tight vs Seasonal", "Extreme Tight vs Seasonal"
        } for v in curves)
        curve_percentiles_10y = pd.to_numeric(part["Seasonal Percentile 10Y"], errors="coerce")
        curve_percentiles_5y = pd.to_numeric(part["Seasonal Percentile 5Y"], errors="coerce")
        # Prefer the structural 10Y percentile, but keep the sector model usable
        # while the growing history has only reached the valid 5Y threshold.
        curve_percentiles = curve_percentiles_10y.where(curve_percentiles_10y.notna(), curve_percentiles_5y)
        sector_curve_percentile = float(curve_percentiles.median()) if curve_percentiles.notna().any() else np.nan
        sector_curve_state = classify_seasonal_curve(sector_curve_percentile)
        market_state = resolve_price_curve_market_state(price_state, sector_curve_state)
        if market_state != "N/A" and qualifier != "N/A":
            market_state = f"{market_state} / {qualifier}"
        sectors.append({"Sector": sector, "Bullish Count": int(part["Price State"].isin(["Bullish", "Strong Bullish"]).sum()),
                        "Bearish Count": int(part["Price State"].isin(["Bearish", "Strong Bearish"]).sum()),
                        "Price State": price_state, "Curve Tight Count": tight_count,
                        "Curve Tight Breadth": float(curve_percentiles.gt(60).sum() / curve_percentiles.notna().sum()) if curve_percentiles.notna().any() else np.nan,
                        "Curve Loose Breadth": float(curve_percentiles.lt(40).sum() / curve_percentiles.notna().sum()) if curve_percentiles.notna().any() else np.nan,
                        "Curve Median Percentile": sector_curve_percentile,
                        "Curve State": sector_curve_state,
                        "CFTC Average 5Y Percentile": float(values.mean()) if values.notna().any() else np.nan,
                        "CFTC Median 5Y Percentile": float(values.median()) if values.notna().any() else np.nan,
                        "CFTC Dispersion": dispersion, "CFTC Qualifier": qualifier,
                        "Market Confirmation": market_state})
    return commodity, pd.DataFrame(sectors)


def latest_complete_commodity_cycle_row(history: pd.DataFrame) -> pd.Series:
    """Return the newest month with a fully classified physical FRED regime."""
    if history.empty or "Core State" not in history:
        return pd.Series(dtype=object)
    state = history["Core State"]
    complete = history.loc[state.notna() & state.ne("DATA INCOMPLETE")]
    return complete.iloc[-1] if not complete.empty else pd.Series(dtype=object)


def build_commodity_cycle_history(
    fred: pd.DataFrame,
    prices: pd.DataFrame,
    term_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate the monthly physical/PPI regimes and separate quarterly CAPEX context."""
    from commodity_cycle.model import calculate_capex, calculate_core_state, calculate_ppi_states, calculate_stress

    if fred.empty:
        return pd.DataFrame(), pd.DataFrame(), term_history.copy()
    model = fred.copy().sort_index()
    inventory_columns = [name for name in FRED_SERIES if name not in {"PPIACO", "CPIAUCSL", "CAPEX", "FPI"}]
    for name in FRED_SERIES:
        if name not in model:
            model[name] = np.nan
    model[inventory_columns] = model[inventory_columns].apply(pd.to_numeric, errors="coerce")
    stress = calculate_stress(model[inventory_columns])
    core = calculate_core_state(stress)
    model = model.join(stress, how="left").join(core, how="left")
    ppi = calculate_ppi_states(model.get("PPIACO", pd.Series(index=model.index, dtype=float)), model["Core State"])
    model = model.join(ppi, how="left", rsuffix="_PPI")
    # CAPEX and FPI are both quarterly series: join only within their published quarter.
    capex_inputs = fred[[c for c in ("CAPEX", "FPI") if c in fred]].copy()
    if {"CAPEX", "FPI"}.issubset(capex_inputs.columns):
        capex_inputs.index = pd.to_datetime(capex_inputs.index).to_period("Q").to_timestamp(how="start")
        capex_inputs = capex_inputs.groupby(level=0).last()
        capex_ratio = pd.to_numeric(capex_inputs["CAPEX"], errors="coerce").div(pd.to_numeric(capex_inputs["FPI"], errors="coerce"))
        capex = calculate_capex(capex_ratio)
    else:
        capex = pd.DataFrame()

    # Recompute historical percentiles from only earlier valid same-month/pair observations.
    curve_history = term_history.copy()
    if not curve_history.empty:
        curve_history["Date"] = pd.to_datetime(curve_history["Date"], errors="coerce")
        curve_history["Spread %"] = pd.to_numeric(curve_history["Spread %"], errors="coerce")
        curve_history = curve_history.dropna(subset=["Date"])
        if "Structure" not in curve_history:
            curve_history["Structure"] = curve_history["Asset"].map(_baseline_structure_name)
        curve_history["Month"] = curve_history["Date"].dt.month
        from commodity_cycle.term_structure_pipeline import percentile_with_history
        curve_history["PairKey"] = curve_history["Structure"].astype(str)
        curve_history["Spread"] = curve_history["Spread %"]
        curve_history["Seasonal Pctl 5Y RT"] = np.nan
        curve_history["Seasonal Pctl 10Y RT"] = np.nan
        curve_history["5Y HistoryN"] = 0
        curve_history["10Y HistoryN"] = 0
        curve_history["5Y HistoryStatus"] = "INSUFFICIENT_5Y_HISTORY"
        curve_history["10Y HistoryStatus"] = "INSUFFICIENT_10Y_HISTORY"
        for idx, row in curve_history.iterrows():
            for years, label, col in ((5, "5Y", "Seasonal Pctl 5Y RT"), (10, "10Y", "Seasonal Pctl 10Y RT")):
                info = percentile_with_history(float(row["Spread %"]) if pd.notna(row["Spread %"]) else np.nan,
                                               curve_history,
                                               str(row["Asset"]), int(row["Month"]), str(row["PairKey"]), years,
                                               before=pd.Timestamp(row["Date"]))
                curve_history.at[idx, col] = info["percentile"]
                curve_history.at[idx, f"{label} HistoryN"] = info["history_n"]
                curve_history.at[idx, f"{label} HistoryStartDate"] = info["history_start"]
                curve_history.at[idx, f"{label} HistoryEndDate"] = info["history_end"]
                curve_history.at[idx, f"{label} HistoryStatus"] = info["history_status"]
    return model, capex, curve_history
