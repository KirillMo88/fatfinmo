from __future__ import annotations

import io
import json
import re
from urllib.request import Request, urlopen
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


DISAGGREGATED_URL = "https://publicreporting.cftc.gov/api/v3/views/72hh-3qpy/export.csv"
TFF_URL = "https://publicreporting.cftc.gov/api/v3/views/gpe5-46if/export.csv"
AAII_URL = "https://www.aaii.com/files/surveys/sentiment.xls"
AAII_RESULTS_URL = "https://www.aaii.com/sentimentsurvey/sent_results"
NAAIM_URL = "https://naaim.org/programs/naaim-exposure-index/"
NAAIM_TABLE_URL = "https://index.naaim.org/embeddable/table"

POSITIONING_STORAGE_DIR = Path("persistent") / "positioning"
RAW_DIR = POSITIONING_STORAGE_DIR / "raw"
PROCESSED_DIR = POSITIONING_STORAGE_DIR / "processed"
STATUS_PATH = POSITIONING_STORAGE_DIR / "source_status.json"

CFTC_PERCENTILE_WINDOW = 156
CFTC_PERCENTILE_MIN_PERIODS = 52
CFTC_STALE_DAYS = 10
CFTC_UPDATE_FREQUENCY = "Weekly"
CFTC_SCHEDULED_UPDATE_DAY = "Saturday"
AAII_HISTORICAL_FILENAME = "aaii_historical.xls"
AAII_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/vnd.ms-excel,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
EXCEL_MAX_ROWS = 1_048_000


@dataclass(frozen=True)
class CftcDashboardAsset:
    canonical_asset: str
    asset_group: str
    report_type: str
    default_participant: str
    contract_patterns: tuple[str, ...]
    code_patterns: tuple[str, ...] = ()


CFTC_DASHBOARD_LAYOUT: tuple[tuple[str, ...], ...] = (
    ("GOLD", "SILVER", "WTI", "PLATINUM", "PALLADIUM"),
    ("S&P 500", "NASDAQ-100", "RUSSELL 2000", "VIX"),
    ("UST 2Y", "UST 5Y", "UST 10Y", "UST BOND"),
    ("BTC", "ETH", "SOL", "HYPERLIQUID"),
)

CFTC_DASHBOARD_ASSETS: tuple[CftcDashboardAsset, ...] = (
    CftcDashboardAsset("GOLD", "Commodities", "Disaggregated", "Managed Money", ("GOLD",), ("088691",)),
    CftcDashboardAsset("SILVER", "Commodities", "Disaggregated", "Managed Money", ("SILVER",), ("084691",)),
    CftcDashboardAsset("WTI", "Commodities", "Disaggregated", "Managed Money", ("WTI", "CRUDE OIL, LIGHT SWEET", "LIGHT SWEET CRUDE"), ("067651",)),
    CftcDashboardAsset("PLATINUM", "Commodities", "Disaggregated", "Managed Money", ("PLATINUM",), ("076651",)),
    CftcDashboardAsset("PALLADIUM", "Commodities", "Disaggregated", "Managed Money", ("PALLADIUM",), ("075651",)),
    CftcDashboardAsset("S&P 500", "Equity / Volatility", "TFF", "Asset Manager", ("E-MINI S&P 500", "S&P 500 STOCK INDEX", "S&P 500"), ("13874", "138741")),
    CftcDashboardAsset("NASDAQ-100", "Equity / Volatility", "TFF", "Asset Manager", ("E-MINI NASDAQ-100", "NASDAQ-100", "NASDAQ 100"), ("20974", "209742")),
    CftcDashboardAsset("RUSSELL 2000", "Equity / Volatility", "TFF", "Asset Manager", ("RUSSELL 2000", "E-MINI RUSSELL"), ("239742",)),
    CftcDashboardAsset("VIX", "Equity / Volatility", "TFF", "Asset Manager", ("VIX FUTURES", "VIX"), ("1170E1", "1170E")),
    CftcDashboardAsset("UST 2Y", "Rates", "TFF", "Leveraged Money", ("2-YEAR U.S. TREASURY", "2 YEAR U.S. TREASURY", "2-YEAR TREASURY"), ("042601",)),
    CftcDashboardAsset("UST 5Y", "Rates", "TFF", "Leveraged Money", ("5-YEAR U.S. TREASURY", "5 YEAR U.S. TREASURY", "5-YEAR TREASURY"), ("044601",)),
    CftcDashboardAsset("UST 10Y", "Rates", "TFF", "Leveraged Money", ("10-YEAR U.S. TREASURY", "10 YEAR U.S. TREASURY", "10-YEAR TREASURY"), ("043602",)),
    CftcDashboardAsset("UST BOND", "Rates", "TFF", "Leveraged Money", ("U.S. TREASURY BONDS", "TREASURY BONDS", "ULTRA U.S. TREASURY BOND"), ("020601",)),
    CftcDashboardAsset("BTC", "Crypto", "TFF", "Leveraged Money", ("BITCOIN",), ("133741",)),
    CftcDashboardAsset("ETH", "Crypto", "TFF", "Leveraged Money", ("ETHER", "ETHEREUM"), ("146021",)),
    CftcDashboardAsset("SOL", "Crypto", "TFF", "Leveraged Money", ("SOLANA", "SOL"), ()),
    CftcDashboardAsset("HYPERLIQUID", "Crypto", "TFF", "Leveraged Money", ("HYPERLIQUID", "HYPE"), ()),
)

DISAGGREGATED_PARTICIPANTS: tuple[tuple[str, str, str, str | None], ...] = (
    ("Producer/Merchant", "prod_merc_positions_long_all", "prod_merc_positions_short_all", "prod_merc_positions_spread_all"),
    ("Swap Dealer", "swap_positions_long_all", "swap_positions_short_all", "swap_positions_spread_all"),
    ("Managed Money", "m_money_positions_long_all", "m_money_positions_short_all", "m_money_positions_spread_all"),
    ("Other Reportables", "other_rept_positions_long_all", "other_rept_positions_short_all", "other_rept_positions_spread_all"),
    ("Non-Reportables", "nonrept_positions_long_all", "nonrept_positions_short_all", None),
)

TFF_PARTICIPANTS: tuple[tuple[str, str, str, str | None], ...] = (
    ("Dealer", "dealer_positions_long_all", "dealer_positions_short_all", "dealer_positions_spread_all"),
    ("Asset Manager", "asset_mgr_positions_long_all", "asset_mgr_positions_short_all", "asset_mgr_positions_spread_all"),
    ("Leveraged Money", "lev_money_positions_long_all", "lev_money_positions_short_all", "lev_money_positions_spread_all"),
    ("Other Reportables", "other_rept_positions_long_all", "other_rept_positions_short_all", "other_rept_positions_spread_all"),
    ("Non-Reportables", "nonrept_positions_long_all", "nonrept_positions_short_all", None),
)


def update_positioning_data(force: bool = False) -> dict[str, Any]:
    ensure_positioning_dirs()
    status: dict[str, Any] = read_status()
    raw_disaggregated = download_csv_source(DISAGGREGATED_URL, RAW_DIR / "cftc_disaggregated.csv", "CFTC Commodities", status, force)
    raw_tff = download_csv_source(TFF_URL, RAW_DIR / "cftc_tff.csv", "CFTC Financials", status, force)
    aaii = download_aaii(status, force)
    naaim = download_naaim(status, force)

    master = calculate_cftc_positioning_metrics(resolve_canonical_contracts(normalize_cftc(raw_disaggregated, raw_tff)))
    validations = validate_cftc_master(master)
    save_processed(master, "cftc_master")
    save_processed(cftc_dashboard_frame(master), "cftc_dashboard")
    save_processed(aaii, "aaii")
    save_processed(naaim, "naaim")
    status["CFTC Master"] = {
        "last_updated_utc": now_utc_iso(),
        "rows": int(len(master)),
        "validation_errors": validations,
    }
    write_status(status)
    return {"cftc_master": master, "aaii": aaii, "naaim": naaim, "status": status}


def load_positioning_data(force_update: bool = False) -> dict[str, Any]:
    ensure_positioning_dirs()
    master = read_processed("cftc_master")
    aaii = read_processed("aaii")
    naaim = read_processed("naaim")
    if force_update or master.empty:
        return update_positioning_data(force=True)
    return {"cftc_master": master, "aaii": aaii, "naaim": naaim, "status": read_status()}


def cftc_dashboard_frame(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return master.copy()
    assets = {asset.canonical_asset for asset in CFTC_DASHBOARD_ASSETS}
    return master.loc[(master["Canonical_Asset"].isin(assets)) & (master["Preferred_For_Dashboard"].astype(bool))].copy()


def cftc_categories_for_report(report_type: str) -> list[str]:
    if str(report_type).upper() == "TFF":
        return [item[0] for item in TFF_PARTICIPANTS]
    return [item[0] for item in DISAGGREGATED_PARTICIPANTS]


def cftc_asset_config(asset: str) -> CftcDashboardAsset | None:
    return next((item for item in CFTC_DASHBOARD_ASSETS if item.canonical_asset == asset), None)


def cftc_asset_series(master: pd.DataFrame, asset: str, participant: str) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    subset = master.loc[(master["Canonical_Asset"] == asset) & (master["Preferred_For_Dashboard"].astype(bool))].copy()
    if subset.empty:
        return pd.DataFrame()
    selected = subset.loc[subset["Participant_Category"] == participant].copy()
    if selected.empty:
        selected = subset.loc[subset["Participant_Category"] == str(subset["Participant_Category"].iloc[0])].copy()
    return selected.sort_values("Date")


def cftc_latest_status(master: pd.DataFrame, report_type: str) -> dict[str, Any]:
    if master.empty:
        return {"last_report_date": None, "status": "DATA UNAVAILABLE"}
    dates = pd.to_datetime(master.loc[master["Report_Type"] == report_type, "Date"], errors="coerce").dropna()
    if dates.empty:
        return {"last_report_date": None, "status": "DATA UNAVAILABLE"}
    latest = pd.Timestamp(dates.max()).normalize()
    age = (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - latest).days
    return {"last_report_date": latest.date().isoformat(), "status": "STALE DATA" if age > CFTC_STALE_DAYS else "CURRENT"}


def export_positioning_xlsx(master: pd.DataFrame, aaii: pd.DataFrame, naaim: pd.DataFrame, status: dict[str, Any]) -> bytes:
    output = io.BytesIO()
    dashboard = cftc_dashboard_frame(master)
    metadata = positioning_metadata(status)
    with pd.ExcelWriter(output, engine="xlsxwriter", engine_kwargs={"options": {"constant_memory": True}}) as writer:
        dashboard.to_excel(writer, sheet_name="CFTC_Dashboard", index=False)
        write_excel_chunks(writer, master, "CFTC_Master")
        aaii.to_excel(writer, sheet_name="AAII", index=False)
        naaim.to_excel(writer, sheet_name="NAAIM", index=False)
        metadata.to_excel(writer, sheet_name="Metadata", index=False)
    return output.getvalue()


def write_excel_chunks(writer: pd.ExcelWriter, frame: pd.DataFrame, base_sheet_name: str) -> None:
    if len(frame) <= EXCEL_MAX_ROWS:
        frame.to_excel(writer, sheet_name=base_sheet_name, index=False)
        return
    for chunk_index, start in enumerate(range(0, len(frame), EXCEL_MAX_ROWS), start=1):
        sheet_name = base_sheet_name if chunk_index == 1 else f"{base_sheet_name}_{chunk_index}"
        frame.iloc[start : start + EXCEL_MAX_ROWS].to_excel(writer, sheet_name=sheet_name[:31], index=False)


def market_positioning_snapshot(master: pd.DataFrame, aaii: pd.DataFrame, naaim: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not aaii.empty:
        row = aaii.sort_values("Date").tail(1).iloc[0]
        for column in [
            "AAII_Bullish",
            "AAII_Neutral",
            "AAII_Bearish",
            "AAII_BullBearSpread",
            "AAII_Bullish_3Y_Percentile",
            "AAII_Bearish_3Y_Percentile",
            "AAII_BullBearSpread_3Y_Percentile",
        ]:
            out[column] = row.get(column)
    if not naaim.empty:
        row = naaim.sort_values("Date").tail(1).iloc[0]
        for column in ["NAAIM_Exposure", "NAAIM_3Y_Percentile", "NAAIM_4W_Change", "NAAIM_13W_Change"]:
            out[column] = row.get(column)
    for asset in ["VIX", "S&P 500", "RUSSELL 2000", "UST 2Y", "BTC", "ETH"]:
        for participant in ["Dealer", "Asset Manager", "Leveraged Money", "Non-Reportables"]:
            rows = cftc_asset_series(master, asset, participant)
            if rows.empty:
                continue
            row = rows.tail(1).iloc[0]
            prefix = sanitize_field_name(f"{asset}_{participant}")
            out[f"{prefix}_NetPctOI"] = row.get("NetPctOI")
            out[f"{prefix}_NetPctOI_3Y_Percentile"] = row.get("NetPctOI_3Y_Percentile")
            out[f"{prefix}_NetPctOI_4W_Change"] = row.get("NetPctOI_4W_Change")
            out[f"{prefix}_NetPctOI_13W_Change"] = row.get("NetPctOI_13W_Change")
    return out


def normalize_cftc(disaggregated: pd.DataFrame, tff: pd.DataFrame) -> pd.DataFrame:
    frames = [
        normalize_cftc_report(disaggregated, "Disaggregated", DISAGGREGATED_PARTICIPANTS, "CFTC Disaggregated Futures Only"),
        normalize_cftc_report(tff, "TFF", TFF_PARTICIPANTS, "CFTC TFF Futures Only"),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=cftc_master_columns())
    master = pd.concat(frames, ignore_index=True)
    master = master.dropna(subset=["Date", "Raw_Contract_Name", "Participant_Category", "Report_Type"])
    master = master.drop_duplicates(subset=["Date", "Raw_Contract_Name", "Participant_Category", "Report_Type"], keep="last")
    return master[cftc_master_columns()]


def normalize_cftc_report(raw: pd.DataFrame, report_type: str, participants: tuple[tuple[str, str, str, str | None], ...], source: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cftc_master_columns())
    frame = normalize_columns(raw)
    date_col = first_existing(frame, ["report_date_as_yyyy_mm_dd", "report_date"])
    contract_col = first_existing(frame, ["contract_market_name", "market_and_exchange_names"])
    market_col = first_existing(frame, ["market_and_exchange_names", "contract_market_name"])
    exchange_col = first_existing(frame, ["exchange_name", "market_and_exchange_names"])
    code_col = first_existing(frame, ["cftc_contract_market_code", "cftc_market_code", "commodity_code"])
    oi_col = first_existing(frame, ["open_interest_all", "open_interest"])
    if not date_col or not contract_col or not oi_col:
        return pd.DataFrame(columns=cftc_master_columns())

    base = pd.DataFrame(
        {
            "Date": parse_cftc_dates(frame[date_col]),
            "Raw_Contract_Name": frame[contract_col].astype(str),
            "Exchange": frame[exchange_col].astype(str) if exchange_col else "",
            "CFTC_Code": frame[code_col].astype(str) if code_col else "",
            "Open_Interest": parse_number(frame[oi_col]),
        }
    )
    if market_col and market_col != contract_col:
        base["Raw_Contract_Name"] = (base["Raw_Contract_Name"] + " - " + frame[market_col].astype(str)).str.strip(" -")

    out = []
    for label, long_col, short_col, spread_col in participants:
        if long_col not in frame.columns or short_col not in frame.columns:
            continue
        part = base.copy()
        part["Report_Type"] = report_type
        part["Participant_Category"] = label
        part["Long"] = parse_number(frame[long_col])
        part["Short"] = parse_number(frame[short_col])
        part["Spreading"] = parse_number(frame[spread_col]) if spread_col and spread_col in frame.columns else np.nan
        part["Source"] = source
        out.append(part)
    if not out:
        return pd.DataFrame(columns=cftc_master_columns())
    result = pd.concat(out, ignore_index=True)
    result["Asset_Group"] = "Unmapped"
    result["Canonical_Asset"] = result["Raw_Contract_Name"]
    result["Preferred_For_Dashboard"] = False
    result["Net"] = result["Long"] - result["Short"]
    result["NetPctOI"] = result["Net"] / result["Open_Interest"].replace(0.0, np.nan) * 100.0
    result["LongPctOI"] = result["Long"] / result["Open_Interest"].replace(0.0, np.nan) * 100.0
    result["ShortPctOI"] = result["Short"] / result["Open_Interest"].replace(0.0, np.nan) * 100.0
    for column in ["NetPctOI_3Y_Percentile", "NetPctOI_4W_Change", "NetPctOI_13W_Change", "NetPctOI_3Y_Median"]:
        result[column] = np.nan
    result["History_Weeks"] = 0
    return result[cftc_master_columns()]


def resolve_canonical_contracts(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return master
    out = master.copy()
    out["Preferred_For_Dashboard"] = False
    for cfg in CFTC_DASHBOARD_ASSETS:
        candidates = out.loc[out["Report_Type"] == cfg.report_type].copy()
        if candidates.empty:
            continue
        if cfg.canonical_asset == "WTI":
            wti_mask = (
                (out["Report_Type"] == cfg.report_type)
                & out["CFTC_Code"].astype(str).str.contains("|".join(re.escape(code) for code in cfg.code_patterns), case=False, na=False)
                & out["Raw_Contract_Name"].astype(str).str.upper().str.contains("CRUDE OIL, LIGHT SWEET|WTI-PHYSICAL", regex=True, na=False)
            )
            if wti_mask.any():
                out.loc[wti_mask, "Canonical_Asset"] = cfg.canonical_asset
                out.loc[wti_mask, "Asset_Group"] = cfg.asset_group
                out.loc[wti_mask, "Preferred_For_Dashboard"] = True
                continue
        selected_name = select_contract_name(candidates, cfg)
        if not selected_name:
            continue
        mask = (out["Report_Type"] == cfg.report_type) & (out["Raw_Contract_Name"] == selected_name)
        out.loc[mask, "Canonical_Asset"] = cfg.canonical_asset
        out.loc[mask, "Asset_Group"] = cfg.asset_group
        out.loc[mask, "Preferred_For_Dashboard"] = True
    return out


def calculate_cftc_positioning_metrics(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return master
    out = master.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values(["Report_Type", "Raw_Contract_Name", "Participant_Category", "Date"])
    group_cols = ["Report_Type", "Raw_Contract_Name", "Participant_Category"]
    out["History_Weeks"] = out.groupby(group_cols).cumcount() + 1
    out["NetPctOI_4W_Change"] = out.groupby(group_cols)["NetPctOI"].diff(4)
    out["NetPctOI_13W_Change"] = out.groupby(group_cols)["NetPctOI"].diff(13)
    out["NetPctOI_3Y_Median"] = out.groupby(group_cols)["NetPctOI"].transform(lambda s: s.rolling(CFTC_PERCENTILE_WINDOW, min_periods=CFTC_PERCENTILE_MIN_PERIODS).median())
    out["NetPctOI_3Y_Percentile"] = out.groupby(group_cols)["NetPctOI"].transform(
        lambda s: trailing_percentile(s, CFTC_PERCENTILE_WINDOW, CFTC_PERCENTILE_MIN_PERIODS)
    )
    return out[cftc_master_columns()]


def calculate_aaii_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_aaii_raw(raw)
    if normalized.empty:
        return pd.DataFrame(columns=aaii_columns())
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(normalized["Date"], errors="coerce"),
            "AAII_Bullish": percent_number(normalized["Bullish"]),
            "AAII_Neutral": percent_number(normalized["Neutral"]),
            "AAII_Bearish": percent_number(normalized["Bearish"]),
        }
    ).dropna(subset=["Date"])
    out = out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    out["AAII_BullBearSpread"] = out["AAII_Bullish"] - out["AAII_Bearish"]
    out["AAII_Bullish_3Y_Percentile"] = trailing_percentile(out["AAII_Bullish"], CFTC_PERCENTILE_WINDOW, CFTC_PERCENTILE_MIN_PERIODS)
    out["AAII_Bearish_3Y_Percentile"] = trailing_percentile(out["AAII_Bearish"], CFTC_PERCENTILE_WINDOW, CFTC_PERCENTILE_MIN_PERIODS)
    out["AAII_BullBearSpread_3Y_Percentile"] = trailing_percentile(out["AAII_BullBearSpread"], CFTC_PERCENTILE_WINDOW, CFTC_PERCENTILE_MIN_PERIODS)
    return out[aaii_columns()]


def normalize_aaii_raw(raw: pd.DataFrame | None) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Date", "Bullish", "Neutral", "Bearish"])
    promoted = promote_aaii_embedded_header(raw)
    frame = normalize_columns(promoted)
    date_col = find_column(frame, ["date"]) or find_column(frame, ["reported"]) or find_column(frame, ["week"])
    bullish_col = find_column(frame, ["bullish"])
    neutral_col = find_column(frame, ["neutral"])
    bearish_col = find_column(frame, ["bearish"])
    if not date_col or not bullish_col or not neutral_col or not bearish_col:
        return pd.DataFrame(columns=["Date", "Bullish", "Neutral", "Bearish"])
    return pd.DataFrame(
        {
            "Date": frame[date_col],
            "Bullish": frame[bullish_col],
            "Neutral": frame[neutral_col],
            "Bearish": frame[bearish_col],
        }
    )


def promote_aaii_embedded_header(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    normalized_columns = {str(column).strip().lower() for column in frame.columns}
    if any("date" in column for column in normalized_columns) and {"bullish", "neutral", "bearish"}.issubset(normalized_columns):
        return frame
    for idx in range(min(len(frame), 25)):
        cells = [str(value).strip().lower() for value in frame.iloc[idx].tolist()]
        has_date = any(cell == "date" or "reported" in cell and "date" in cell for cell in cells)
        if has_date and "bullish" in cells and "neutral" in cells and "bearish" in cells:
            header = []
            for pos, value in enumerate(frame.iloc[idx].tolist()):
                text = str(value).strip()
                header.append(text if text and text.lower() != "nan" else f"Column_{pos}")
            out = frame.iloc[idx + 1 :].copy()
            out.columns = header
            return out
    return frame


def calculate_naaim_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=naaim_columns())
    frame = normalize_columns(raw)
    date_col = find_column(frame, ["date", "week"])
    exposure_col = find_column(frame, ["naaim", "exposure", "mean", "average"])
    if not date_col or not exposure_col:
        return pd.DataFrame(columns=naaim_columns())
    out = pd.DataFrame({"Date": pd.to_datetime(frame[date_col], errors="coerce"), "NAAIM_Exposure": parse_number(frame[exposure_col])}).dropna(subset=["Date"])
    out = out.dropna(subset=["NAAIM_Exposure"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    out["NAAIM_3Y_Percentile"] = trailing_percentile(out["NAAIM_Exposure"], CFTC_PERCENTILE_WINDOW, CFTC_PERCENTILE_MIN_PERIODS)
    out["NAAIM_4W_Change"] = out["NAAIM_Exposure"].diff(4)
    out["NAAIM_13W_Change"] = out["NAAIM_Exposure"].diff(13)
    return out[naaim_columns()]


def download_csv_source(url: str, cache_path: Path, label: str, status: dict[str, Any], force: bool = False) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        try:
            return pd.read_csv(cache_path, low_memory=False)
        except Exception:
            pass
    try:
        frame = pd.read_csv(url, low_memory=False)
        frame.to_csv(cache_path, index=False)
        status[label] = {"last_updated_utc": now_utc_iso(), "source": url, "status": "CURRENT", "rows": int(len(frame))}
        annotate_source_schedule(status[label], label)
        return frame
    except Exception as exc:
        status[label] = {"last_updated_utc": now_utc_iso(), "source": url, "status": "SOURCE_FAILED_USING_CACHE", "error": str(exc)}
        annotate_source_schedule(status[label], label)
        if cache_path.exists():
            return pd.read_csv(cache_path, low_memory=False)
        return pd.DataFrame()


def annotate_source_schedule(source_status: dict[str, Any], label: str) -> None:
    if label.startswith("CFTC "):
        source_status["update_frequency"] = CFTC_UPDATE_FREQUENCY
        source_status["scheduled_update_day"] = CFTC_SCHEDULED_UPDATE_DAY


def download_aaii(status: dict[str, Any], force: bool = False) -> pd.DataFrame:
    historical_path = RAW_DIR / AAII_HISTORICAL_FILENAME
    if not force:
        cached = read_processed("aaii")
        if not cached.empty:
            return cached
    cached = read_processed("aaii")
    source_frames: list[pd.DataFrame] = []
    source_details: dict[str, Any] = {
        "last_updated_utc": now_utc_iso(),
        "source": AAII_RESULTS_URL,
        "historical_source": str(historical_path),
    }
    historical_error = None
    live_error = None
    try:
        historical = read_aaii_historical_workbook(historical_path)
        if not historical.empty:
            source_frames.append(historical)
            source_details["historical_rows"] = int(len(historical))
    except Exception as exc:
        historical_error = str(exc)
    try:
        live = download_aaii_live_results()
        if not live.empty:
            source_frames.append(live)
            source_details["live_rows"] = int(len(live))
    except Exception as exc:
        live_error = str(exc)
    if source_frames:
        out = calculate_aaii_metrics(pd.concat(source_frames, ignore_index=True))
        source_details["rows"] = int(len(out))
        if live_error:
            source_details["live_status"] = "SOURCE_FAILED"
            source_details["live_error"] = live_error
        else:
            source_details["live_status"] = "CURRENT"
        if historical_error:
            source_details["historical_error"] = historical_error
        source_details["status"] = "CURRENT" if not live_error else "HISTORICAL_CURRENT_LIVE_FAILED"
        status["AAII"] = source_details
        return out
    try:
        raw_path = RAW_DIR / "aaii.xls"
        request = Request(AAII_URL, headers=AAII_BROWSER_HEADERS)
        with urlopen(request, timeout=30) as response:
            content = response.read()
        raw_path.write_bytes(content)
        raw = pd.read_excel(io.BytesIO(content))
        out = calculate_aaii_metrics(raw)
        status["AAII"] = {"last_updated_utc": now_utc_iso(), "source": AAII_URL, "status": "CURRENT", "rows": int(len(out))}
        return out
    except Exception as exc:
        source_details["status"] = "SOURCE_FAILED_USING_CACHE" if not cached.empty else "SOURCE_FAILED_NO_CACHE"
        source_details["legacy_source"] = AAII_URL
        source_details["legacy_error"] = str(exc)
        if historical_error:
            source_details["historical_error"] = historical_error
        if live_error:
            source_details["live_error"] = live_error
        status["AAII"] = source_details
        return cached


def read_aaii_historical_workbook(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Date", "Bullish", "Neutral", "Bearish"])
    excel = pd.ExcelFile(path)
    sheet_name = "SENTIMENT" if "SENTIMENT" in excel.sheet_names else excel.sheet_names[0]
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    return normalize_aaii_raw(raw)


def download_aaii_live_results() -> pd.DataFrame:
    response = requests.get(AAII_RESULTS_URL, headers=AAII_BROWSER_HEADERS, timeout=30)
    response.raise_for_status()
    return parse_aaii_live_results_html(response.text)


def parse_aaii_live_results_html(html: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html))
    for table in tables:
        normalized = normalize_aaii_raw(table)
        if not normalized.empty:
            parsed_dates = pd.to_datetime(normalized["Date"], errors="coerce")
            if parsed_dates.notna().any():
                return normalized
    return pd.DataFrame(columns=["Date", "Bullish", "Neutral", "Bearish"])


def download_naaim(status: dict[str, Any], force: bool = False) -> pd.DataFrame:
    if not force:
        cached = read_processed("naaim")
        if not cached.empty:
            return cached
    try:
        response = requests.get(
            NAAIM_TABLE_URL,
            headers={"User-Agent": "Mozilla/5.0 Screener positioning pipeline", "Accept": "text/html,application/xhtml+xml"},
            timeout=30,
        )
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text), flavor="lxml")
        best = max(tables, key=lambda frame: score_naaim_table(frame)) if tables else pd.DataFrame()
        out = calculate_naaim_metrics(best)
        status["NAAIM"] = {"last_updated_utc": now_utc_iso(), "source": NAAIM_TABLE_URL, "status": "CURRENT" if not out.empty else "DATA_NOT_FOUND", "rows": int(len(out))}
        return out
    except Exception as exc:
        status["NAAIM"] = {"last_updated_utc": now_utc_iso(), "source": NAAIM_URL, "status": "SOURCE_FAILED_USING_CACHE", "error": str(exc)}
        return read_processed("naaim")


def score_naaim_table(frame: pd.DataFrame) -> int:
    text = " ".join(str(column).lower() for column in frame.columns)
    return sum(token in text for token in ["date", "naaim", "exposure", "average", "mean"]) + len(frame)


def validate_cftc_master(master: pd.DataFrame) -> list[str]:
    errors = []
    if master.empty:
        return ["CFTC master is empty"]
    if master.duplicated(subset=["Date", "Raw_Contract_Name", "Participant_Category", "Report_Type"]).any():
        errors.append("Date + Raw Contract + Participant + Report Type is not unique")
    if (pd.to_numeric(master["Open_Interest"], errors="coerce") <= 0).any():
        errors.append("Open_Interest has non-positive values")
    net_diff = (pd.to_numeric(master["Long"], errors="coerce") - pd.to_numeric(master["Short"], errors="coerce") - pd.to_numeric(master["Net"], errors="coerce")).abs()
    if (net_diff > 1e-6).any():
        errors.append("Net does not equal Long - Short")
    pct = pd.to_numeric(master["NetPctOI_3Y_Percentile"], errors="coerce").dropna()
    if ((pct < 0.0) | (pct > 100.0)).any():
        errors.append("Percentile outside 0-100")
    dates = pd.to_datetime(master["Date"], errors="coerce").dropna()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if (dates > today).any():
        errors.append("Future CFTC dates detected")
    return errors


def positioning_metadata(status: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("Date", "CFTC report date / weekly observation date", "CFTC, AAII, NAAIM", "All", "CFTC source files update weekly on Saturday", "Parsed date; no look-ahead shifting", "Date"),
        ("Participant_Category", "Economic participant category from source report", "CFTC", "Disaggregated / TFF", "Report-specific; no equivalence mapping forced", "Canonical display label only", "Text"),
        ("NetPctOI", "Net position as percent of open interest", "CFTC", "Disaggregated / TFF", "Category-specific", "(Long - Short) / Open Interest * 100", "Percent points"),
        ("NetPctOI_3Y_Percentile", "Point-in-time trailing 3Y percentile", "CFTC", "Disaggregated / TFF", "Category-specific", "Rolling 156W percentile using data available up to t only; min 52W", "0-100"),
        ("AAII_BullBearSpread", "Bullish minus bearish sentiment", "AAII", "Survey", "Individual investors", "Bullish - Bearish", "Percent points"),
        ("NAAIM_Exposure", "NAAIM Exposure Index", "NAAIM", "Survey", "Active investment managers", "Loaded from source table when accessible; cached fallback allowed", "Index"),
    ]
    out = pd.DataFrame(rows, columns=["Column", "Description", "Source", "Report_Type", "Participant definition", "Transformation", "Units"])
    out["Source_Status_JSON"] = json.dumps(status, ensure_ascii=True)
    return out


def save_processed(frame: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DIR / f"{name}.parquet"
    csv_path = PROCESSED_DIR / f"{name}.csv"
    try:
        frame.to_parquet(parquet_path, index=False)
    except Exception:
        frame.to_csv(csv_path, index=False)
    else:
        if csv_path.exists():
            csv_path.unlink()


def read_processed(name: str) -> pd.DataFrame:
    parquet_path = PROCESSED_DIR / f"{name}.parquet"
    csv_path = PROCESSED_DIR / f"{name}.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    return pd.DataFrame()


def read_status() -> dict[str, Any]:
    if not STATUS_PATH.exists():
        return {}
    try:
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        for key in ("CFTC Commodities", "CFTC Financials"):
            if isinstance(status.get(key), dict):
                annotate_source_schedule(status[key], key)
        return status
    except Exception:
        return {}


def write_status(status: dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=True), encoding="utf-8")


def ensure_positioning_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def select_contract_name(candidates: pd.DataFrame, cfg: CftcDashboardAsset) -> str | None:
    contracts = candidates[["Raw_Contract_Name", "CFTC_Code"]].drop_duplicates().copy()
    contracts["raw_upper"] = contracts["Raw_Contract_Name"].astype(str).str.upper()
    contracts["code"] = contracts["CFTC_Code"].astype(str)
    for code in cfg.code_patterns:
        hit = contracts.loc[contracts["code"].str.contains(re.escape(code), case=False, na=False)]
        if not hit.empty:
            return str(hit.iloc[0]["Raw_Contract_Name"])
    for pattern in cfg.contract_patterns:
        hit = contracts.loc[contracts["raw_upper"].str.contains(re.escape(pattern.upper()), na=False)]
        if not hit.empty:
            filtered = hit.loc[~hit["raw_upper"].str.contains("MICRO|MINI OPTION|BALANCE OF MONTH|CALENDAR", na=False)]
            chosen = filtered if not filtered.empty else hit
            return str(chosen.iloc[0]["Raw_Contract_Name"])
    return None


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [
        re.sub(r"_+", "_", str(column).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")).strip("_")
        for column in out.columns
    ]
    return out.loc[:, ~out.columns.duplicated()]


def first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def find_column(frame: pd.DataFrame, tokens: list[str]) -> str | None:
    for column in frame.columns:
        text = str(column).lower()
        if all(token in text for token in tokens):
            return str(column)
    for column in frame.columns:
        text = str(column).lower()
        if any(token in text for token in tokens):
            return str(column)
    return None


def parse_number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False), errors="coerce")


def percent_number(values: pd.Series) -> pd.Series:
    parsed = parse_number(values)
    if parsed.dropna().median() <= 1.0:
        return parsed * 100.0
    return parsed


def parse_cftc_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%Y %b %d %I:%M:%S %p", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], errors="coerce")
    return parsed.dt.normalize()


def trailing_percentile(series: pd.Series, window: int = CFTC_PERCENTILE_WINDOW, min_periods: int = CFTC_PERCENTILE_MIN_PERIODS) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def rank_last(window_values: np.ndarray) -> float:
        clean = window_values[np.isfinite(window_values)]
        if len(clean) < min_periods or not np.isfinite(window_values[-1]):
            return np.nan
        return float((clean <= window_values[-1]).sum() / len(clean) * 100.0)

    return values.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def cftc_master_columns() -> list[str]:
    return [
        "Date",
        "Asset_Group",
        "Canonical_Asset",
        "Raw_Contract_Name",
        "Exchange",
        "CFTC_Code",
        "Report_Type",
        "Participant_Category",
        "Long",
        "Short",
        "Spreading",
        "Open_Interest",
        "Net",
        "NetPctOI",
        "LongPctOI",
        "ShortPctOI",
        "NetPctOI_3Y_Percentile",
        "NetPctOI_4W_Change",
        "NetPctOI_13W_Change",
        "NetPctOI_3Y_Median",
        "History_Weeks",
        "Preferred_For_Dashboard",
        "Source",
    ]


def aaii_columns() -> list[str]:
    return [
        "Date",
        "AAII_Bullish",
        "AAII_Neutral",
        "AAII_Bearish",
        "AAII_BullBearSpread",
        "AAII_Bullish_3Y_Percentile",
        "AAII_Bearish_3Y_Percentile",
        "AAII_BullBearSpread_3Y_Percentile",
    ]


def naaim_columns() -> list[str]:
    return ["Date", "NAAIM_Exposure", "NAAIM_3Y_Percentile", "NAAIM_4W_Change", "NAAIM_13W_Change"]


def sanitize_field_name(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", value)).strip("_")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
