from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from fred_client import download_fred_series
from positioning import cftc_asset_series, cftc_latest_status, load_positioning_data

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
CFTC_ASSET_MAP = {
    "WTI": "WTI", "Natural Gas": "Natural Gas", "RBOB": "RBOB",
    "Copper": "Copper", "Aluminum": "Aluminum", "Corn": "Corn",
    "Wheat": "Wheat", "Soybeans": "Soybeans",
}


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
        current["CurveDataQuality"] = current.get("quality", "INVALID")
        current["Status"] = current["CurveDataQuality"]
        current["Seasonal Pctl 5Y"] = current.get("Seasonal Pctl 5Y", np.nan)
        current["Seasonal Pctl 10Y"] = current.get("Seasonal Pctl 10Y", np.nan)
        current["Current Month Seasonal Pctl 5Y"] = current["Seasonal Pctl 5Y"]
        current["Current Month Seasonal Pctl 10Y"] = current["Seasonal Pctl 10Y"]
        # Until MTD reaches five synchronized business-day spreads, keep the last finalized
        # month as the official model state; the current month remains explicitly provisional.
        official5, official10, official_state = [], [], []
        for _, row in current.iterrows():
            if str(row.get("asset")) in {"Corn", "Wheat", "Soybeans"}:
                p5, p10 = row.get("Seasonal Pctl 5Y"), row.get("Seasonal Pctl 10Y")
            elif int(row.get("mtd_observation_count", 0) or 0) >= 5:
                p5, p10 = row.get("Seasonal Pctl 5Y"), row.get("Seasonal Pctl 10Y")
            else:
                finalized = store.last_finalized_for(str(row["asset"]), str(row["pair_key"]))
                if finalized:
                    finalized_date = pd.Timestamp(f"{finalized['month']}-01")
                    p5 = percentile_with_history(float(finalized["monthly_spread"]), seasonal_history,
                                                 str(row["asset"]), finalized_date.month, str(row["pair_key"]), 5,
                                                 before=finalized_date)
                    p10 = percentile_with_history(float(finalized["monthly_spread"]), seasonal_history,
                                                  str(row["asset"]), finalized_date.month, str(row["pair_key"]), 10,
                                                  before=finalized_date)
                    p5 = p5["percentile"]
                    p10 = p10["percentile"]
                else:
                    candidates = seasonal_history.loc[
                        (seasonal_history["Asset"] == row["asset"])
                        & (seasonal_history["PairKey"] == row["pair_key"])
                        & (pd.to_datetime(seasonal_history["Date"], errors="coerce") < pd.Timestamp(row["As Of"]).to_period("M").to_timestamp())
                    ].sort_values("Date")
                    if candidates.empty:
                        p5, p10 = np.nan, np.nan
                    else:
                        latest_hist = candidates.iloc[-1]
                        finalized_date = pd.Timestamp(latest_hist["Date"])
                        p5 = percentile_with_history(float(latest_hist["Spread"]), seasonal_history,
                                                     str(row["asset"]), finalized_date.month, str(row["pair_key"]), 5,
                                                     before=finalized_date)["percentile"]
                        p10 = percentile_with_history(float(latest_hist["Spread"]), seasonal_history,
                                                      str(row["asset"]), finalized_date.month, str(row["pair_key"]), 10,
                                                      before=finalized_date)["percentile"]
            official5.append(p5)
            official10.append(p10)
            official_state.append(_term_curve_state(p10 if pd.notna(p10) else p5))
        current["Official Seasonal Pctl 5Y"] = official5
        current["Official Seasonal Pctl 10Y"] = official10
        current["Official Seasonal State"] = official_state
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


def load_monthly_prices() -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch monthly continuous futures closes; unavailable histories stay absent."""
    try:
        import yfinance as yf
    except Exception as exc:
        return pd.DataFrame(), {asset: f"FAILED: yfinance unavailable ({exc})" for asset in PRICE_TICKERS}
    frames: list[pd.Series] = []
    status: dict[str, str] = {}
    for asset, ticker in PRICE_TICKERS.items():
        try:
            frame = yf.download(ticker, period="max", interval="1mo", auto_adjust=False, progress=False, threads=False)
            if frame is None or frame.empty:
                status[asset] = "MISSING"
                continue
            close = frame["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            series = pd.to_numeric(close, errors="coerce").dropna()
            series.index = pd.to_datetime(series.index, errors="coerce").tz_localize(None).to_period("M").to_timestamp("M")
            series = series[~series.index.isna()].groupby(level=0).last()
            current_month = pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M")
            series = series.loc[series.index.to_period("M") < current_month]
            if series.empty:
                status[asset] = "MISSING"
                continue
            series.name = asset
            frames.append(series)
            age_days = (pd.Timestamp.now().normalize() - series.index.max().normalize()).days
            freshness = "STALE" if age_days > 62 else "CURRENT"
            status[asset] = f"{freshness}: {series.index.max().date().isoformat()} via Yahoo Finance ({ticker})"
        except Exception as exc:
            status[asset] = f"FAILED: {exc}"
    return (pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()), status


def load_cftc_snapshot() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Consume the existing positioning store/service without creating another pipeline."""
    try:
        result = load_positioning_data(force_update=False)
        master = result.get("cftc_master", pd.DataFrame())
        source_freshness = cftc_latest_status(master, "Disaggregated").get("status", "DATA UNAVAILABLE")
        out: dict[str, dict[str, Any]] = {}
        for asset in CFTC_ASSET_MAP:
            series = cftc_asset_series(master, CFTC_ASSET_MAP[asset], "Managed Money")
            if series.empty:
                out[asset] = {}
                continue
            row = series.sort_values("Date").iloc[-1]
            out[asset] = {
                "MM Net % OI": row.get("NetPctOI"),
                "Net Direction": "Net Long" if pd.notna(row.get("NetPctOI")) and row.get("NetPctOI") > 0 else "Net Short" if pd.notna(row.get("NetPctOI")) and row.get("NetPctOI") < 0 else "Flat / N/A",
                "3Y Percentile": row.get("NetPctOI_3Y_Percentile"),
                "5Y Percentile": row.get("NetPctOI_5Y_Percentile"),
                "4W Change": row.get("NetPctOI_4W_Change"),
                "13W Change": row.get("NetPctOI_13W_Change"),
                "Updated Date": pd.to_datetime(row.get("Date"), errors="coerce"),
                "History Weeks": row.get("History_Weeks"),
                "Status": source_freshness,
            }
        status = {"CFTC": source_freshness if out else "MISSING"}
        return out, status
    except Exception as exc:
        return {}, {"CFTC": f"FAILED: {exc}"}


def build_market_confirmation(
    monthly_prices: pd.DataFrame,
    term_history: pd.DataFrame,
    term_current: pd.DataFrame,
    cftc: dict[str, dict[str, Any]],
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
    history_rows: list[pd.DataFrame] = []
    for asset in PRICE_TICKERS:
        price = monthly_prices.get(asset, pd.Series(dtype=float)).dropna() if not monthly_prices.empty else pd.Series(dtype=float)
        r3 = price.iloc[-1] / price.iloc[-4] - 1 if len(price) >= 4 else np.nan
        r6 = price.iloc[-1] / price.iloc[-7] - 1 if len(price) >= 7 else np.nan
        r12 = price.iloc[-1] / price.iloc[-13] - 1 if len(price) >= 13 else np.nan
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
            "Return 3M": r3, "Return 6M": r6, "Return 12M": r12,
            "Price State": pstate,
            "Term Structure As Of": curve_row.get("As Of"),
            "Term Structure Source": curve_row.get("Source"),
            "Term Structure Structure": curve_row.get("Structure"),
            "F1 Symbol": curve_row.get("F1 Symbol"), "F3 Symbol": curve_row.get("F3 Symbol"), "F6 Symbol": curve_row.get("F6 Symbol"),
            "CurveDataQuality": curve_row.get("CurveDataQuality"),
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
            "Raw Curve State": raw,
            "Seasonal Percentile 10Y": seasonal_pctl,
            "Seasonal Percentile 5Y": seasonal_5y,
            "Seasonal Curve State": curve_state,
            "Price × Curve": resolve_price_curve_market_state(pstate, curve_state),
            **{
                "MM Net % OI": np.nan, "Net Direction": "N/A", "3Y Percentile": np.nan,
                "5Y Percentile": np.nan, "4W Change": np.nan, "13W Change": np.nan,
                "Updated Date": pd.NaT, "History Weeks": np.nan, **c,
            },
            "CFTC Status": c.get("Status", "MISSING"),
            "CFTC Relative State": classify_cftc_relative_state(c.get("5Y Percentile", np.nan)),
            "CFTC Qualifier": "N/A",
        }
        row["CFTC Qualifier"] = resolve_cftc_qualifier(
            [c.get("5Y Percentile", np.nan)], [c.get("MM Net % OI", np.nan)]
        ) if c else "N/A"
        current_rows.append(row)

        if len(price):
            h = price.rename("Price").to_frame()
            h["Return 3M"] = h["Price"].div(h["Price"].shift(3)).sub(1)
            h["Return 6M"] = h["Price"].div(h["Price"].shift(6)).sub(1)
            h["Return 12M"] = h["Price"].div(h["Price"].shift(12)).sub(1)
            h["Price State"] = [classify_commodity_price_momentum(a, b, c_) for a, b, c_ in zip(h["Return 3M"], h["Return 6M"], h["Return 12M"])]
            h["Commodity"] = asset
            history_rows.append(h)
    commodity = pd.DataFrame(current_rows)
    sectors = []
    for sector, assets in COMMODITY_SECTORS.items():
        part = commodity.loc[commodity["Commodity"].isin(assets)]
        price_state = classify_sector_price_state(part["Price State"].tolist(), sector)
        values = pd.to_numeric(part["5Y Percentile"], errors="coerce")
        net = pd.to_numeric(part["MM Net % OI"], errors="coerce")
        dispersion = float(values.max() - values.min()) if values.notna().sum() >= 2 else np.nan
        qualifier = resolve_cftc_qualifier(values.tolist(), net.tolist()) if values.notna().any() else "N/A"
        curves = part["Seasonal Curve State"].tolist()
        tight_count = sum(v in {"Tight", "Strong Tightness", "Extreme Tightness"} for v in curves)
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
    inventory_columns = [name for name in FRED_SERIES if name not in {"PPIACO", "CAPEX", "FPI"}]
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
