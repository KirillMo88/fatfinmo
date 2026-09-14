from __future__ import annotations

from datetime import date, datetime, timezone
from html import escape
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finance_core import extract_ohlcv_frame
from bybit_derivatives import BYBIT_STORAGE_PATH, read_bybit_storage
from fund_flows import FundFlowCache, default_fund_flow_cache_path, fetch_etf_com_fund_flow_history
from global_liquidity import (
    CHINA_M2_SERIES_ID,
    ECB_TOTAL_ASSETS_SERIES_ID,
    GLOBAL_LIQUIDITY_CONFIG,
    PBOC_TOTAL_ASSETS_SERIES_ID,
    RAW_COLUMNS,
    WEEKLY_COLUMNS,
    read_global_liquidity,
    raw_series,
    update_global_liquidity,
)
from gold_regime.config import GOLD_REGIME_CONFIG
from gold_regime.etf_flows import aggregate_gold_etf_flows, load_gold_etf_flows
from market_model import download_fred_market_data


START = pd.Timestamp("2016-01-01")
END = min(pd.Timestamp("2026-12-31"), pd.Timestamp(datetime.now(timezone.utc)).tz_convert(None).normalize())
OUTPUT_DIR = Path(os.environ.get("EXPORT_OUTPUT_DIR", "exports"))
OUTPUT_PATH = OUTPUT_DIR / "weekly_basis_ohlc_screener_2016_2026.xls"

YAHOO_OHLC = {
    "SPY": "SPY",
    "BTC-USD": "BTC-USD",
    "Gold": "GLD",
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
    "WTI": "CL=F",
    "IWM": "IWM",
    "XLI": "XLI",
    "XLP": "XLP",
}

FRED_WEEKLY_VALUES = {
    "US2Y": ("DGS2", 1.0),
    "US10Y_REAL": ("DFII10", 1.0),
    "WALCL": ("WALCL", 1000.0),
    "WTREGEN": ("WTREGEN", 1000.0),
    "RRPONTSYD": ("RRPONTSYD", 1.0),
    "EURUSD": ("DEXUSEU", 1.0),
    "USDJPY": ("DEXJPUS", 1.0),
    "USDCNY": ("DEXCHUS", 1.0),
}

MONTHLY_TO_WEEKLY = {
    "US_M2": "us_m2_usd_bn",
    "EA_M2": "ea_m2_usd_bn",
    "China_M2": "china_m2_usd_bn",
    "Japan_M2": "japan_m2_usd_bn",
    "Fed_Assets": "fed_assets_usd_bn",
    "ECB_Assets": "ecb_assets_usd_bn",
    "BoJ_Assets": "boj_assets_usd_bn",
    "PBoC_Assets": "pboc_assets_usd_bn",
}

M2_AND_CB_SOURCE_SERIES = {
    "US_M2": "M2SL",
    "EA_M2": GLOBAL_LIQUIDITY_CONFIG["ecb_m2_key"],
    "China_M2": CHINA_M2_SERIES_ID,
    "Japan_M2": f"{GLOBAL_LIQUIDITY_CONFIG['boj_m2_db']}'{GLOBAL_LIQUIDITY_CONFIG['boj_m2_code']}",
    "Fed_Assets": "WALCL",
    "ECB_Assets": ECB_TOTAL_ASSETS_SERIES_ID,
    "BoJ_Assets": f"{GLOBAL_LIQUIDITY_CONFIG['boj_total_assets_db']}'{GLOBAL_LIQUIDITY_CONFIG['boj_total_assets_code']}",
    "PBoC_Assets": PBOC_TOTAL_ASSETS_SERIES_ID,
}

BTC_ETF_FLOW_START = pd.Timestamp("2024-01-01")
BTC_SPOT_ETF_FLOW_TICKERS = (
    "IBIT",
    "FBTC",
    "GBTC",
    "ARKB",
    "BITB",
    "BTCO",
    "EZBC",
    "HODL",
    "BRRR",
    "BTCW",
)

BYBIT_BTC_COLUMNS = {
    "open_interest_usd": "BTCUSDT_Open_Interest_USD",
    "oi_change_1w_pct": "BTCUSDT_OI_Change_1W",
    "oi_change_4w_pct": "BTCUSDT_OI_Change_4W",
    "funding_rate": "BTCUSDT_Funding_Rate",
    "funding_7d": "BTCUSDT_Funding_7D",
    "funding_28d": "BTCUSDT_Funding_28D",
    "perp_premium_pct": "BTCUSDT_Perpetual_Premium_Basis",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weekly_index = pd.date_range(START, END, freq="W-FRI")
    out = pd.DataFrame({"Date": weekly_index})
    metadata: list[dict[str, str]] = []

    out = out.merge(load_yahoo_weekly_ohlc(metadata), on="Date", how="left")

    raw, monthly, _weekly = read_or_update_global_liquidity()
    out = out.merge(load_fred_weekly_values(raw, weekly_index, metadata), on="Date", how="left")
    out = out.merge(load_monthly_global_values(raw, monthly, weekly_index, metadata), on="Date", how="left")
    out = out.merge(load_gold_flows_and_cot(weekly_index, metadata), on="Date", how="left")
    out = out.merge(load_btc_etf_flow(weekly_index, metadata), on="Date", how="left")
    out = out.merge(load_bybit_btcusdt(weekly_index, metadata), on="Date", how="left")

    out = out.loc[(out["Date"] >= START) & (out["Date"] <= END)].sort_values("Date")
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    write_excel_html(OUTPUT_PATH, out, pd.DataFrame(metadata))
    print(str(OUTPUT_PATH.resolve()))
    print(f"rows={len(out)} cols={len(out.columns)}")


def load_yahoo_weekly_ohlc(metadata: list[dict[str, str]]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for label, ticker in YAHOO_OHLC.items():
        try:
            px = yf.download(
                ticker,
                start=START.strftime("%Y-%m-%d"),
                end=(END + pd.Timedelta(days=4)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            px = pd.DataFrame()
        daily = extract_ohlcv_frame(px, ticker)
        weekly = weekly_ohlcv(daily)
        if weekly.empty:
            metadata.append(meta(label, ticker, "Yahoo Finance", "OHLC unavailable"))
            continue
        renamed = weekly.rename(columns={column: f"{label}_{column}" for column in ["Open", "High", "Low", "Close", "Volume"]})
        renamed = renamed.reset_index().rename(columns={"index": "Date"})
        renamed["Date"] = pd.to_datetime(renamed["Date"]).dt.normalize()
        parts.append(renamed[["Date"] + [f"{label}_{column}" for column in ["Open", "High", "Low", "Close", "Volume"]]])
        metadata.append(meta(label, ticker, "Yahoo Finance", f"{weekly.index.min().date()} to {weekly.index.max().date()}"))
    return merge_parts(parts)


def weekly_ohlcv(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    frame = daily.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.loc[(frame.index >= START) & (frame.index <= END)]
    weekly = frame.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    return weekly.dropna(subset=["Close"])


def read_or_update_global_liquidity() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw, monthly, weekly = read_global_liquidity()
    if raw.empty or monthly.empty:
        raw, monthly, weekly = update_global_liquidity(api_key=os.getenv("FRED_API_KEY") or None, force=True)
    return raw, monthly, weekly


def load_fred_weekly_values(raw: pd.DataFrame, weekly_index: pd.DatetimeIndex, metadata: list[dict[str, str]]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    market_fred = pd.DataFrame()
    for label, (series_id, divisor) in FRED_WEEKLY_VALUES.items():
        series = raw_series(raw, series_id) / divisor
        if series.empty and series_id in {"DGS2", "DFII10"}:
            if market_fred.empty:
                market_fred = download_fred_market_data(api_key=os.getenv("FRED_API_KEY") or None)
            series = fred_frame_series(market_fred, series_id) / divisor
        if series.empty:
            weekly = pd.Series(np.nan, index=weekly_index)
        else:
            series.index = pd.to_datetime(series.index)
            weekly = series.resample("W-FRI").last().reindex(weekly_index)
        parts.append(pd.DataFrame({"Date": weekly_index, label: weekly.to_numpy()}))
        unit = "USD bn" if divisor == 1000.0 else "native/rate"
        metadata.append(meta(label, series_id, "FRED / Global Liquidity raw", unit))
    return merge_parts(parts)


def fred_frame_series(frame: pd.DataFrame, series_id: str) -> pd.Series:
    if frame.empty or not {"Series_ID", "Date", "Value"}.issubset(frame.columns):
        return pd.Series(dtype="float64")
    rows = frame[frame["Series_ID"].astype(str).str.upper().eq(series_id.upper())].copy()
    if rows.empty:
        return pd.Series(dtype="float64")
    rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce")
    rows["Value"] = pd.to_numeric(rows["Value"], errors="coerce")
    rows = rows.dropna(subset=["Date", "Value"]).sort_values("Date")
    return rows.groupby("Date")["Value"].last()


def load_monthly_global_values(
    raw: pd.DataFrame,
    monthly: pd.DataFrame,
    weekly_index: pd.DatetimeIndex,
    metadata: list[dict[str, str]],
) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame({"Date": weekly_index})
    monthly_values = monthly.copy()
    monthly_values["date"] = pd.to_datetime(monthly_values["date"], errors="coerce")
    monthly_values = monthly_values.dropna(subset=["date"]).sort_values("date")
    cols = [column for column in MONTHLY_TO_WEEKLY.values() if column in monthly_values.columns]
    source = monthly_values.set_index("date")[cols].sort_index().ffill()
    weekly = source.resample("W-FRI").ffill().reindex(weekly_index).ffill()
    merged = weekly.reset_index().rename(columns={"index": "Date"})
    merged = merged.rename(columns={column: label for label, column in MONTHLY_TO_WEEKLY.items()})
    source_dates = load_m2_and_cb_source_dates(raw, weekly_index, metadata)
    merged = merged.merge(source_dates, on="Date", how="left")
    for label in MONTHLY_TO_WEEKLY:
        metadata.append(meta(label, MONTHLY_TO_WEEKLY[label], "Global Liquidity monthly layer", "USD bn, forward-filled to weekly Fridays"))
    return merged


def load_m2_and_cb_source_dates(raw: pd.DataFrame, weekly_index: pd.DatetimeIndex, metadata: list[dict[str, str]]) -> pd.DataFrame:
    out = pd.DataFrame({"Date": weekly_index})
    if raw.empty:
        return out
    for label, series_id in M2_AND_CB_SOURCE_SERIES.items():
        aligned = monthly_source_dates(raw, series_id, weekly_index)
        observation_col = f"{label}_Observation_Date"
        release_col = f"{label}_Release_Date"
        out[observation_col] = format_date_series(aligned["observation_date"]).to_numpy()
        out[release_col] = format_date_series(aligned["release_date"]).to_numpy()
        release_count = int(pd.to_datetime(aligned["release_date"], errors="coerce").notna().sum())
        note = "monthly source observation/release dates, forward-filled to weekly Fridays"
        if release_count == 0:
            note += "; release date unavailable in source/cache"
        metadata.append(meta(f"{label}_Observation_Date", series_id, "Global Liquidity raw", note))
        metadata.append(meta(f"{label}_Release_Date", series_id, "Global Liquidity raw", note))
    return out


def monthly_source_dates(raw: pd.DataFrame, series_id: str, weekly_index: pd.DatetimeIndex) -> pd.DataFrame:
    rows = raw[raw["series_id"].astype(str).eq(series_id)].copy()
    if rows.empty:
        return pd.DataFrame({"observation_date": pd.NaT, "release_date": pd.NaT}, index=weekly_index)
    rows["observation_date"] = pd.to_datetime(rows["observation_date"], errors="coerce")
    rows["release_date"] = pd.to_datetime(rows["release_date"], errors="coerce")
    rows["raw_value"] = pd.to_numeric(rows["raw_value"], errors="coerce")
    rows = rows.dropna(subset=["observation_date", "raw_value"]).sort_values(["observation_date", "release_date"])
    if rows.empty:
        return pd.DataFrame({"observation_date": pd.NaT, "release_date": pd.NaT}, index=weekly_index)
    rows["month"] = rows["observation_date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        rows.groupby("month", dropna=False)
        .agg(
            observation_date=("observation_date", "last"),
            release_date=("release_date", last_valid_timestamp),
        )
        .sort_index()
    )
    combined_index = monthly.index.union(weekly_index).sort_values()
    weekly = monthly.reindex(combined_index).ffill().reindex(weekly_index)
    return weekly


def last_valid_timestamp(values: pd.Series) -> pd.Timestamp:
    clean = pd.to_datetime(values, errors="coerce").dropna()
    if clean.empty:
        return pd.NaT
    return pd.Timestamp(clean.iloc[-1])


def format_date_series(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    return dates.dt.strftime("%Y-%m-%d").where(dates.notna(), "")


def load_gold_flows_and_cot(weekly_index: pd.DatetimeIndex, metadata: list[dict[str, str]]) -> pd.DataFrame:
    cfg = GOLD_REGIME_CONFIG
    today = END.date()
    try:
        daily_flows, available, unavailable = load_gold_etf_flows(date(2016, 1, 1), today, cfg)
        etf = aggregate_gold_etf_flows(daily_flows, cfg["etf_tickers"], cfg)
        etf_part = etf[["date", "etf_flow_1w"]].rename(columns={"date": "Date", "etf_flow_1w": "Gold_ETF_Flow"})
        metadata.append(meta("Gold_ETF_Flow", ",".join(cfg["etf_tickers"]), "ETF.com via Gold Regime", f"weekly net flow; available={','.join(available)}; unavailable={','.join(unavailable)}"))
    except Exception as exc:
        etf_part = pd.DataFrame(columns=["Date", "Gold_ETF_Flow"])
        metadata.append(meta("Gold_ETF_Flow", ",".join(cfg["etf_tickers"]), "ETF.com via Gold Regime", f"ERROR: {exc}"))

    try:
        cot_frame, contract = load_gold_cot_light(Path("persistent") / "finance_cache" / "gold_regime")
        cot = cot_frame[["date", "cot_mm_long", "cot_mm_short", "cot_open_interest"]].rename(
            columns={
                "date": "Date",
                "cot_mm_long": "Gold_COT_MM_Long",
                "cot_mm_short": "Gold_COT_MM_Short",
                "cot_open_interest": "Gold_COT_OpenInterest",
            }
        )
        metadata.append(meta("Gold_COT_*", contract or "COMEX Gold", "CFTC via Gold Regime", "weekly report, aligned to Friday"))
    except Exception as exc:
        cot = pd.DataFrame(columns=["Date", "Gold_COT_MM_Long", "Gold_COT_MM_Short", "Gold_COT_OpenInterest"])
        metadata.append(meta("Gold_COT_*", "COMEX Gold", "CFTC via Gold Regime", f"ERROR: {exc}"))

    merged = merge_parts([etf_part, cot])
    if merged.empty:
        return pd.DataFrame({"Date": weekly_index})
    merged["Date"] = pd.to_datetime(merged["Date"]).dt.normalize()
    return pd.DataFrame({"Date": weekly_index}).merge(merged, on="Date", how="left")


def load_gold_cot_light(cache_dir: Path) -> tuple[pd.DataFrame, str | None]:
    cache_path = cache_dir / "cftc_disaggregated_futures_only.csv"
    identifier_path = cache_dir / "cftc_comex_gold_contract.txt"
    if not cache_path.exists():
        raise FileNotFoundError(str(cache_path))
    wanted = [
        "Market_and_Exchange_Names",
        "Report_Date_as_YYYY_MM_DD",
        "Contract_Market_Name",
        "Commodity Name",
        "Open_Interest_All",
        "M_Money_Positions_Long_All",
        "M_Money_Positions_Short_All",
    ]
    saved_identifier = identifier_path.read_text().strip() if identifier_path.exists() else ""
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(cache_path, usecols=wanted, chunksize=50_000, low_memory=False):
        commodity = chunk["Commodity Name"].astype(str).str.upper()
        market = chunk["Market_and_Exchange_Names"].astype(str).str.upper()
        contract = chunk["Contract_Market_Name"].astype(str).str.upper()
        if saved_identifier:
            mask = chunk["Contract_Market_Name"].astype(str).eq(saved_identifier)
        else:
            mask = (
                commodity.eq("GOLD")
                & market.str.contains("COMMODITY EXCHANGE", na=False)
                & contract.str.contains("GOLD", na=False)
                & ~contract.str.contains("MICRO|MINI|OPTION", na=False)
            )
        selected = chunk.loc[mask].copy()
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        return pd.DataFrame(columns=["date", "cot_mm_long", "cot_mm_short", "cot_open_interest"]), saved_identifier or None
    selected = pd.concat(chunks, ignore_index=True)
    contract_name = str(selected["Contract_Market_Name"].dropna().iloc[-1]) if selected["Contract_Market_Name"].notna().any() else saved_identifier or None
    report_dates = pd.to_datetime(selected["Report_Date_as_YYYY_MM_DD"], errors="coerce")
    out = pd.DataFrame(
        {
            "date": (report_dates + pd.Timedelta(days=3)).dt.to_period("W-FRI").dt.end_time.dt.normalize(),
            "cot_mm_long": parse_number(selected["M_Money_Positions_Long_All"]),
            "cot_mm_short": parse_number(selected["M_Money_Positions_Short_All"]),
            "cot_open_interest": parse_number(selected["Open_Interest_All"]),
        }
    )
    out = out.dropna(subset=["date", "cot_mm_long", "cot_mm_short", "cot_open_interest"]).sort_values("date")
    out = out.loc[(out["date"] >= START) & (out["date"] <= END)]
    return out, contract_name


def parse_number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype(str).str.replace(",", "", regex=False), errors="coerce")


def load_btc_etf_flow(weekly_index: pd.DatetimeIndex, metadata: list[dict[str, str]]) -> pd.DataFrame:
    tickers = BTC_SPOT_ETF_FLOW_TICKERS
    cache = FundFlowCache(default_fund_flow_cache_path())
    frames: list[pd.DataFrame] = []
    available: list[str] = []
    unavailable: list[str] = []
    for ticker in tickers:
        try:
            fetched = fetch_etf_com_fund_flow_history(ticker, BTC_ETF_FLOW_START.date(), END.date(), timeout=20.0)
            cache.upsert_observations(fetched)
        except Exception:
            pass
        observations = cache.load_observations(ticker, BTC_ETF_FLOW_START.date())
        if observations:
            available.append(ticker)
            frames.append(pd.DataFrame({"date": pd.to_datetime([obs.date for obs in observations]), "net_flow": [obs.net_flow for obs in observations]}))
        else:
            unavailable.append(ticker)
    if not frames:
        metadata.append(meta("BTC ETF Flow", "+".join(tickers), "ETF.com proxy flow", f"ERROR/unavailable: {','.join(unavailable)}"))
        return pd.DataFrame({"Date": weekly_index, "BTC_ETF_Flow": np.nan})
    daily = pd.concat(frames, ignore_index=True)
    weekly = daily.set_index("date")["net_flow"].resample("W-FRI").sum(min_count=1).rename("BTC_ETF_Flow")
    metadata.append(meta("BTC ETF Flow", "+".join(tickers), "ETF.com proxy flow", f"weekly net flow; available={','.join(available)}; unavailable={','.join(unavailable)}"))
    return pd.DataFrame({"Date": weekly_index}).merge(weekly.reset_index().rename(columns={"date": "Date"}), on="Date", how="left")


def load_bybit_btcusdt(weekly_index: pd.DatetimeIndex, metadata: list[dict[str, str]]) -> pd.DataFrame:
    out = pd.DataFrame({"Date": weekly_index})
    try:
        storage = read_bybit_storage()
    except Exception as exc:
        metadata.append(meta("BTCUSDT Bybit tactical fields", "BTCUSDT", "Bybit persistent storage", f"ERROR: {exc}"))
        return out
    if storage.empty:
        metadata.append(meta("BTCUSDT Bybit tactical fields", "BTCUSDT", "Bybit persistent storage", f"unavailable: {BYBIT_STORAGE_PATH}"))
        return out
    rows = storage[storage["asset"].astype(str).str.upper().eq("BTC-USD")].copy()
    if rows.empty:
        metadata.append(meta("BTCUSDT Bybit tactical fields", "BTCUSDT", "Bybit persistent storage", "BTC-USD unavailable"))
        return out
    rows["Date"] = pd.to_datetime(rows["date"], errors="coerce").dt.normalize()
    rows = rows.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    for source_col in BYBIT_BTC_COLUMNS:
        rows[source_col] = pd.to_numeric(rows[source_col], errors="coerce")
    if "funding_1d" in rows.columns:
        rows["funding_1d"] = pd.to_numeric(rows["funding_1d"], errors="coerce")
        rows["funding_rate"] = rows["funding_rate"].fillna(rows["funding_1d"])
    rows["open_interest_usd"], oi_note = fill_open_interest_usd(rows)
    weekly = rows.set_index("Date")[list(BYBIT_BTC_COLUMNS)].reindex(weekly_index)
    weekly = weekly.rename(columns=BYBIT_BTC_COLUMNS).reset_index().rename(columns={"index": "Date"})
    missing_premium = int(weekly["BTCUSDT_Perpetual_Premium_Basis"].isna().sum())
    metadata.append(
        meta(
            "BTCUSDT Bybit tactical fields",
            "BTCUSDT",
            "Bybit persistent storage",
            (
                f"weekly fields from {rows['Date'].min().date()} to {rows['Date'].max().date()}; "
                f"{oi_note}; Funding Rate falls back to weekly funding_1d where current funding_rate is unavailable; "
                f"Perpetual Premium/Basis missing rows={missing_premium}"
            ),
        )
    )
    return out.merge(weekly, on="Date", how="left")


def fill_open_interest_usd(rows: pd.DataFrame) -> tuple[pd.Series, str]:
    values = pd.to_numeric(rows.get("open_interest_usd", np.nan), errors="coerce").copy()
    changes = pd.to_numeric(rows.get("oi_change_1w_pct", np.nan), errors="coerce")
    if values.notna().sum() > 1 or changes.notna().sum() == 0:
        return values, "Open Interest USD uses stored values"
    known_positions = np.flatnonzero(values.notna().to_numpy())
    if len(known_positions) == 0:
        return values, "Open Interest USD unavailable in storage"
    filled = values.reset_index(drop=True)
    change_values = changes.reset_index(drop=True)
    anchor = int(known_positions[-1])
    for idx in range(anchor - 1, -1, -1):
        next_value = filled.iloc[idx + 1]
        next_change = change_values.iloc[idx + 1]
        if np.isfinite(next_value) and np.isfinite(next_change) and abs(1.0 + next_change) > 1e-12:
            filled.iloc[idx] = next_value / (1.0 + next_change)
    for idx in range(anchor + 1, len(filled)):
        prev_value = filled.iloc[idx - 1]
        current_change = change_values.iloc[idx]
        if np.isfinite(prev_value) and np.isfinite(current_change):
            filled.iloc[idx] = prev_value * (1.0 + current_change)
    filled.index = values.index
    return filled, "Open Interest USD reconstructed from latest stored OI and weekly OI Change 1W"


def merge_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    clean = [part for part in parts if part is not None and not part.empty]
    if not clean:
        return pd.DataFrame(columns=["Date"])
    merged = clean[0]
    for part in clean[1:]:
        merged = merged.merge(part, on="Date", how="outer")
    return merged


def meta(field: str, series: str, source: str, note: str) -> dict[str, str]:
    return {"Field": field, "Series": series, "Source": source, "Notes": note}


def write_excel_html(path: Path, data: pd.DataFrame, metadata: pd.DataFrame) -> None:
    sheets = [
        ("Weekly_Data", data),
        ("Metadata", metadata),
    ]
    worksheet_xml = "\n".join(
        f"<x:ExcelWorksheet><x:Name>{escape(name[:31])}</x:Name><x:WorksheetOptions><x:DisplayGridlines/></x:WorksheetOptions></x:ExcelWorksheet>"
        for name, _ in sheets
    )
    body = []
    for name, frame in sheets:
        body.append(f"<h2>{escape(name)}</h2>")
        body.append(frame.to_html(index=False, border=1, na_rep="", escape=True))
        body.append("<br style='page-break-before:always'>")
    workbook = f"""<html xmlns:o="urn:schemas-microsoft-com:office:office"
xmlns:x="urn:schemas-microsoft-com:office:excel"
xmlns="http://www.w3.org/TR/REC-html40">
<head>
<meta charset="utf-8">
<!--[if gte mso 9]><xml>
<x:ExcelWorkbook>
<x:ExcelWorksheets>
{worksheet_xml}
</x:ExcelWorksheets>
</x:ExcelWorkbook>
</xml><![endif]-->
</head>
<body>
{''.join(body)}
</body>
</html>"""
    path.write_bytes(workbook.encode("utf-8"))


if __name__ == "__main__":
    main()
