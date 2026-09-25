from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bybit_derivatives import BYBIT_STORAGE_PATH, read_bybit_storage
from business_cycle import (
    BUSINESS_CYCLE_MODEL_VERSION,
    ECONOMY_REGIME_MODEL_VERSION,
    INFLATION_LAYER_MODEL_VERSION,
    build_business_cycle_snapshot,
)
from finance_core import download_completed_ohlcv
from fred_client import download_fred_series_batch
from fund_flows import calculate_fund_flow_history_metrics, load_fund_flow_history
from global_liquidity import read_global_liquidity
from hy_oas import combine_hy_oas_sources, weekly_archive_available_frame
from liquidity_forecast import BREADTH_VERSION, FORECAST_VERSION, read_forecast_snapshot
from rates_financial_conditions import EXPORT_FIELDS as RATES_FC_EXPORT_FIELDS, MODEL_VERSION as RATES_FC_MODEL_VERSION, read_snapshot as read_rates_fc_snapshot, refresh_snapshot as refresh_rates_fc_snapshot
from funding_conditions import EXPORT_FIELDS as FUNDING_EXPORT_FIELDS, MODEL_VERSION as FUNDING_MODEL_VERSION, read_snapshot as read_funding_snapshot, refresh_snapshot as refresh_funding_snapshot
from positioning import CFTC_DASHBOARD_ASSETS, load_positioning_data, read_processed


DEFAULT_START_DATE = "2010-01-01"
MODEL_VERSION_RECOMPUTED = "Recomputed_Current_Model"
MARKET_REGIME_VERSION = "MR_Recomputed_Current_Model"
GOLD_REGIME_VERSION = "GOLD_Recomputed_Current_Model"
BTC_REGIME_VERSION = "BTC_Recomputed_Current_Model"
GLOBAL_LIQUIDITY_VERSION = "GL_Recomputed_Current_Model"
ETF_FLOWS_VERSION = "ETF_FLOWS_V1"
POSITIONING_VERSION = "POSITIONING_1.0"
EXPORT_LAYER_OPTIONS = ["All", "METADATA", "RAW", "DERIVED", "MODEL_OUTPUT"]
EXPORT_CATEGORY_OPTIONS = [
    "All",
    "Dataset",
    "Market OHLC",
    "Markets",
    "Global Liquidity",
    "Rates",
    "Treasury Financing",
    "ETF Fund Flows",
    "Liquidity Forecast",
    "Forward Liquidity Research",
    "FX / USD",
    "Inflation",
    "Rates & Curves",
    "Growth / Business Cycle",
    "Risk / Financial Conditions",
    "Business Cycle",
    "Credit",
    "Market Regime",
    "Rates & Financial Conditions",
    "Funding Conditions",
    "Crypto-native derivatives",
    "CFTC S&P 500",
    "CFTC NASDAQ-100",
    "CFTC VIX",
    "CFTC UST 2Y",
    "CFTC UST 10Y",
    "CFTC GOLD",
    "CFTC WTI",
    "CFTC BTC",
    "AAII",
    "NAAIM",
    "Gold Regime",
    "BTC Regime",
    "Model Versioning",
]
EXPORT_MODEL_USAGE_OPTIONS = [
    "All",
    "Primary key",
    "Traceability",
    "Market, Gold, BTC",
    "Market",
    "Global Macro / Regime inputs",
    "Forward Liquidity",
    "Rates / Forward Rates Pressure",
    "Global Liquidity Regime",
    "Global Liquidity",
    "Funding Liquidity",
    "Treasury Financing",
    "Liquidity Forecast",
    "Forward Liquidity Research",
    "Rates & Financial Conditions",
    "Rates & Financial Conditions / Financial Fragility",
    "Funding Conditions / Financial Fragility",
    "Positioning",
    "Positioning / Tail Risk",
    "Business Cycle / Inflation / Economy Regime",
    "Business Cycle",
    "Credit",
    "Credit / Funding Liquidity Stress",
    "Market Regime",
    "Gold Regime",
    "BTC Regime",
    "Versioning",
]
EXPORT_FRED_SERIES = {
    "CCSA": ("ContinuingClaims", "Continued Claims (Insured Unemployment), SA", "claims", "weekly", "Business Cycle"),
    "UNRATE": ("UnemploymentRate", "Unemployment rate", "%", "monthly", "Business Cycle"),
    "PAYEMS": ("Payrolls", "Nonfarm payrolls", "thousands", "monthly", "Business Cycle"),
    "INDPRO": ("IndustrialProduction", "Industrial production index", "index", "monthly", "Business Cycle"),
    "RSAFS": ("RetailSales", "Advance retail sales", "millions USD", "monthly", "Business Cycle"),
    "W875RX1": ("RealPersonalIncomeExTransfers", "Real personal income excluding transfers", "bn chained USD", "monthly", "Business Cycle"),
    "PCEC96": ("RealPCE", "Real personal consumption expenditures", "bn chained USD", "monthly", "Business Cycle"),
    "BAMLH0A0HYM2": ("HY_OAS", "High yield option-adjusted spread", "%", "daily", "Credit"),
    "BAMLC0A0CM": ("IG_OAS", "Investment grade option-adjusted spread", "%", "daily", "Credit"),
}
CORE_CFTC_EXPORT = [
    ("S&P 500", "Asset Manager", "SP500_AM"),
    ("NASDAQ-100", "Asset Manager", "NASDAQ100_AM"),
    ("VIX", "Asset Manager", "VIX_AM"),
    ("UST 2Y", "Leveraged Money", "UST2Y_LM"),
    ("UST 10Y", "Leveraged Money", "UST10Y_LM"),
    ("GOLD", "Managed Money", "Gold_MM"),
    ("WTI", "Managed Money", "WTI_MM"),
    ("BTC", "Leveraged Money", "BTC_LM"),
]


@dataclass
class SeriesMeta:
    column: str
    layer: str
    category: str
    description: str
    unit: str
    transformation: str
    source: str
    original_frequency: str
    model_usage: str
    release_date_available: bool = False
    point_in_time_safe: str = "Production timing convention"
    forward_filled: bool = True
    notes: str = ""


def build_weekly_macro_research_workbook(
    api_key: str | None = None,
    start_date: str | pd.Timestamp = DEFAULT_START_DATE,
    end_date: str | pd.Timestamp | None = None,
    layers: list[str] | None = None,
    model_usages: list[str] | None = None,
    categories: list[str] | None = None,
) -> tuple[bytes, str]:
    dataset, metadata, positioning_metadata, model_metadata = build_weekly_macro_dataset(api_key, start_date, end_date)
    dataset, metadata = filter_export_components(
        dataset,
        metadata,
        layers=layers,
        model_usages=model_usages,
        categories=categories,
    )
    data_quality = build_data_quality(dataset, metadata)
    validate_weekly_dataset(dataset)
    calculation_date = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
    filename = f"global_macro_research_weekly_2010_{calculation_date}{export_filename_suffix(layers, model_usages, categories)}.xlsx"
    return write_macro_research_workbook(dataset, metadata, data_quality, positioning_metadata, model_metadata), filename


def filter_export_components(
    dataset: pd.DataFrame,
    metadata: list[SeriesMeta],
    *,
    layers: list[str] | None = None,
    model_usages: list[str] | None = None,
    categories: list[str] | None = None,
) -> tuple[pd.DataFrame, list[SeriesMeta]]:
    """Keep the weekly data and its metadata aligned with export filters."""
    selected_layers = {str(value) for value in (layers or []) if str(value).strip() and str(value).lower() != "all"}
    selected_usages = {str(value) for value in (model_usages or []) if str(value).strip() and str(value).lower() != "all"}
    selected_categories = {str(value) for value in (categories or []) if str(value).strip() and str(value).lower() != "all"}

    def include(meta: SeriesMeta) -> bool:
        if meta.column == "Date":
            return True
        if selected_layers and meta.layer not in selected_layers:
            return False
        if selected_usages and meta.model_usage not in selected_usages:
            return False
        if selected_categories and meta.category not in selected_categories:
            return False
        return True

    filtered_metadata = [meta for meta in metadata if include(meta)]
    columns = [meta.column for meta in filtered_metadata if meta.column in dataset.columns]
    if "Date" in dataset.columns and "Date" not in columns:
        columns.insert(0, "Date")
    filtered_dataset = dataset.loc[:, columns].copy()
    return filtered_dataset, filtered_metadata


def export_filename_suffix(
    layers: list[str] | None,
    model_usages: list[str] | None,
    categories: list[str] | None = None,
) -> str:
    parts: list[str] = []
    if any(str(value).lower() != "all" for value in (layers or [])):
        parts.append("layers-filtered")
    if any(str(value).lower() != "all" for value in (model_usages or [])):
        parts.append("usage-filtered")
    if any(str(value).lower() != "all" for value in (categories or [])):
        parts.append("category-filtered")
    return "_" + "_".join(parts) if parts else ""


def build_weekly_macro_dataset(
    api_key: str | None = None,
    start_date: str | pd.Timestamp = DEFAULT_START_DATE,
    end_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[SeriesMeta], pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(start_date).tz_localize(None).normalize()
    end = pd.Timestamp(end_date).tz_localize(None).normalize() if end_date else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    calendar = pd.date_range(start, end, freq="W-FRI")
    if calendar.empty:
        raise ValueError("Weekly calendar cannot be created")
    dataset = pd.DataFrame({"Date": calendar})
    metadata: list[SeriesMeta] = [
        SeriesMeta("Date", "METADATA", "Dataset", "Weekly observation date", "date", "W-FRI calendar", "Internal", "weekly", "Primary key", False, "Yes", False),
    ]
    calculation_ts = pd.Timestamp.now(tz="UTC").tz_localize(None)
    dataset["Data_AsOf_Date"] = end
    dataset["Calculation_Date"] = calculation_ts
    metadata.extend(
        [
            SeriesMeta("Data_AsOf_Date", "METADATA", "Dataset", "Latest source-data timestamp used for export generation", "date", "Export parameter", "Internal", "export", "Traceability", False, "n/a", False),
            SeriesMeta("Calculation_Date", "METADATA", "Dataset", "Workbook generation timestamp", "datetime", "Generated at export time", "Internal", "export", "Traceability", False, "n/a", False),
        ]
    )

    add_market_ohlc(dataset, metadata)
    add_etf_fund_flows(dataset, metadata)
    add_global_macro_series(dataset, metadata, api_key)
    add_global_liquidity(dataset, metadata)
    add_liquidity_forecast(dataset, metadata)
    add_rates_financial_conditions(dataset, metadata, api_key)
    add_funding_conditions(dataset, metadata, api_key)
    positioning_metadata = add_positioning(dataset, metadata)
    add_bybit(dataset, metadata)
    add_business_cycle_model_outputs(dataset, metadata, api_key)
    add_market_regime_outputs(dataset, metadata, api_key)
    add_gold_regime_outputs(dataset, metadata, api_key)
    add_btc_regime_outputs(dataset, metadata)
    add_model_versions(dataset, metadata)
    model_metadata = build_model_metadata()
    dataset = dataset.loc[(dataset["Date"] >= start) & (dataset["Date"] <= end)].copy()
    return dataset, metadata, positioning_metadata, model_metadata


def add_market_ohlc(dataset: pd.DataFrame, metadata: list[SeriesMeta]) -> None:
    instruments = {
        "SPY": "SPY",
        "QQQ": "QQQ",
        "GLD": "GLD",
        "BTCUSD": "BTC-USD",
    }
    for prefix, ticker in instruments.items():
        try:
            daily = download_completed_ohlcv(ticker, period="max")
        except Exception:
            daily = pd.DataFrame()
        weekly = weekly_ohlc(daily)
        for field in ["Open", "High", "Low", "Close"]:
            col = f"{prefix}_{field}"
            add_aligned_series(
                dataset,
                col,
                weekly.get(field, pd.Series(dtype="float64")),
                metadata,
                SeriesMeta(col, "RAW", "Market OHLC", f"{ticker} weekly {field}", "price", "Weekly OHLC from production market history", f"Yahoo / {ticker}", "daily->weekly", "Market, Gold, BTC"),
                ffill=False,
            )
        if prefix in {"SPY", "QQQ", "GLD"}:
            adj_col = f"{prefix}_AdjClose"
            add_aligned_series(
                dataset,
                adj_col,
                weekly.get("Close", pd.Series(dtype="float64")),
                metadata,
                SeriesMeta(adj_col, "RAW", "Market OHLC", f"{ticker} adjusted weekly close", "price", "auto_adjust production close", f"Yahoo / {ticker}", "daily->weekly", "Market"),
                ffill=False,
            )


def add_etf_fund_flows(dataset: pd.DataFrame, metadata: list[SeriesMeta]) -> None:
    """Add absolute ETF flows plus normalized 4W intensity and trailing percentile."""
    dates = pd.to_datetime(dataset.get("Date"), errors="coerce").dropna()
    if dates.empty:
        return
    export_start = dates.min().normalize()
    history_start = max(export_start - pd.Timedelta(weeks=156), pd.Timestamp("2010-01-01"))
    export_end = dates.max().date()
    assets = {
        "SPY": "SPY",
        "QQQ": "QQQ",
        "GLD": "GLD",
        "BTC": "BTC-USD",
    }
    fields = [
        ("ETF_Flow_1W", "ETF_Flow_1W", "1W absolute ETF fund flow", "USD"),
        ("ETF_Flow_4W", "ETF_Flow_4W", "4W absolute ETF fund flow", "USD"),
        ("ETF_Flow_13W", "ETF_Flow_13W", "13W absolute ETF fund flow", "USD"),
        ("ETF_Flow_Intensity_4W", "ETF_Flow_Intensity_4W", "4W ETF fund flow intensity", "% of AUM"),
        ("ETF_Flow_3Y_Pctl", "ETF_Flow_Trailing_3Y_Percentile", "Trailing 3Y ETF fund flow intensity percentile", "0-100"),
    ]
    for prefix, ticker in assets.items():
        try:
            history = load_fund_flow_history(ticker, history_start.date(), export_end)
            metrics = calculate_fund_flow_history_metrics(history)
        except Exception:
            metrics = pd.DataFrame()
        metric_series = metrics.set_index("date") if not metrics.empty and "date" in metrics.columns else metrics
        source = "ETF.com" if ticker != "BTC-USD" else "ETF.com / IBIT + FBTC + GBTC"
        for source_suffix, output_suffix, description, unit in fields:
            output_column = f"{prefix}_{output_suffix}"
            meta = SeriesMeta(
                output_column,
                "RAW",
                "ETF Fund Flows",
                f"{prefix} {description}",
                unit,
                "Weekly aggregation; 4W intensity = 4W net flow / week-end AUM; fallback uses 156W median absolute weekly flow",
                source,
                "daily->weekly",
                "Market, Gold, BTC",
                False,
                "ETF.com observations available as received; trailing percentile uses data available up to t only",
                False,
                "Trailing 3Y percentile window = 156 weeks; minimum 52 observations",
            )
            add_aligned_series(
                dataset,
                output_column,
                metric_series.get(source_suffix, pd.Series(dtype="float64")) if not metric_series.empty else pd.Series(dtype="float64"),
                metadata,
                meta,
                ffill=False,
            )
def add_global_macro_series(dataset: pd.DataFrame, metadata: list[SeriesMeta], api_key: str | None) -> None:
    import global_macro_tab as gm

    fred = gm._download_fred_macro_series(api_key)
    raw_liquidity, monthly_liquidity, weekly_liquidity = read_global_liquidity()
    market = gm._download_market_macro_series()
    specs = []
    specs.extend(gm._fx_specs(raw_liquidity, market))
    specs.extend(gm._inflation_specs(fred))
    specs.extend(gm._rates_specs(fred, weekly_liquidity))
    specs.extend(gm._growth_specs(fred, market))
    specs.extend(gm._risk_specs(fred, market))
    for spec in specs:
        if spec.instrument == "U.S. ISM Services PMI":
            continue
        col = macro_column_name(spec.instrument)
        add_aligned_series(
            dataset,
            col,
            spec.series,
            metadata,
            SeriesMeta(col, "RAW", spec.block, spec.instrument, spec.unit, "Last observation carried forward to weekly calendar", spec.source, spec.frequency, "Global Macro / Regime inputs", False, "No vintage release calendar; production timing convention", True, spec.data_status),
        )

    add_export_fred_series(dataset, metadata, api_key)
    add_derived_changes(dataset, metadata, "DXY", [4, 13, 26], "percent", "Forward Liquidity")
    add_derived_changes(dataset, metadata, "VIX", [4, 13], "absolute", "Forward Liquidity")
    add_derived_changes(dataset, metadata, "MOVE", [4, 13, 26], "absolute", "Forward Liquidity")
    for col in ["US2Y", "US10Y", "DE10Y", "JP10Y"]:
        if col in dataset.columns:
            add_derived_changes(dataset, metadata, col, [4, 13, 26] if col == "US2Y" else [13], "bps", "Rates / Forward Rates Pressure")
    add_credit_spread_changes(dataset, metadata)
    add_business_cycle_derived(dataset, metadata)

    if {"US10Y", "DE10Y", "JP10Y"}.issubset(dataset.columns):
        dataset["GlobalRates_Average"] = dataset[["US10Y", "DE10Y", "JP10Y"]].mean(axis=1, skipna=True)
        metadata.append(SeriesMeta("GlobalRates_Average", "DERIVED", "Rates", "Average of US, Germany and Japan 10Y yields", "%", "mean(US10Y, DE10Y, JP10Y)", "Internal", "weekly", "Forward Liquidity"))
        add_derived_changes(dataset, metadata, "GlobalRates_Average", [13], "bps", "Forward Liquidity")


def add_export_fred_series(dataset: pd.DataFrame, metadata: list[SeriesMeta], api_key: str | None) -> None:
    try:
        fred_frame = download_fred_series_batch(
            EXPORT_FRED_SERIES.keys(),
            api_key=api_key,
            observation_start="1990-01-01",
        )
    except Exception:
        fred_frame = pd.DataFrame(columns=["Series_ID", "Date", "Value"])
    series_ids = fred_frame.get("Series_ID", pd.Series(dtype="object")).astype(str).str.upper()
    for series_id, (out_col, description, unit, frequency, category) in EXPORT_FRED_SERIES.items():
        source = fred_frame.loc[series_ids.eq(series_id)]
        fred_series = pd.Series(
            pd.to_numeric(source.get("Value", pd.Series(dtype="float64")), errors="coerce").values,
            index=pd.to_datetime(source.get("Date", pd.Series(dtype="datetime64[ns]")), errors="coerce"),
        ).dropna()
        series = fred_series
        source_label = f"FRED / {series_id}"
        original_frequency = frequency
        transformation = "FRED observations carried forward to W-FRI calendar"
        if series_id == "BAMLH0A0HYM2":
            archive = load_hy_oas_archive_frame()
            if not archive.empty:
                fred_frame_for_merge = pd.DataFrame(
                    {
                        "Date": pd.to_datetime(fred_series.index, errors="coerce"),
                        "HY_OAS": fred_series.to_numpy(),
                        "HYOASSourceFrequency": "DAILY_FRED",
                    }
                )
                combined = combine_hy_oas_sources(archive, pd.DataFrame(), fred_frame_for_merge)
                series = pd.Series(combined["HY_OAS"].to_numpy(), index=combined["Date"])
                source_label = "Weekly archive + FRED overlay / BAMLH0A0HYM2"
                original_frequency = "weekly archive + daily FRED"
                transformation = "Weekly archive retained for full history; FRED observations override matching newer weeks"
        remove_metadata_for_column(metadata, out_col)
        add_aligned_series(
            dataset,
            out_col,
            series,
            metadata,
            SeriesMeta(
                out_col,
                "RAW",
                category,
                description,
                unit,
                transformation,
                source_label,
                original_frequency,
                "Weekly Macro Research",
                False,
                "No vintage release calendar; production timing convention",
                True,
            ),
        )


def load_hy_oas_archive_frame() -> pd.DataFrame:
    """Load the long weekly HY OAS archive used when FRED has only recent history."""
    candidates = [
        Path(__file__).resolve().parent / "data" / "BAMLH0A0HYM2_weekly.csv",
        Path("data") / "BAMLH0A0HYM2_weekly.csv",
    ]
    for path in candidates:
        try:
            if not path.exists():
                continue
            raw = pd.read_csv(path, usecols=["time", "close"])
            return weekly_archive_available_frame(raw)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
    return pd.DataFrame(columns=["Date", "HY_OAS", "HYOASSourceFrequency"])


def remove_metadata_for_column(metadata: list[SeriesMeta], column: str) -> None:
    metadata[:] = [meta for meta in metadata if meta.column != column]


def add_global_liquidity(dataset: pd.DataFrame, metadata: list[SeriesMeta]) -> None:
    try:
        _, monthly, weekly = read_global_liquidity()
    except Exception:
        monthly, weekly = pd.DataFrame(), pd.DataFrame()
    monthly = prepare_date_frame(monthly)
    weekly = prepare_date_frame(weekly)
    monthly_map = {
        "US_M2": ("us_m2_usd_bn", "United States M2", "bn USD"),
        "EuroArea_M2": ("ea_m2_usd_bn", "Euro Area M2", "bn USD"),
        "China_M2": ("china_m2_usd_bn", "China M2", "bn USD"),
        "Japan_M2": ("japan_m2_usd_bn", "Japan M2", "bn USD"),
        "Global_M2": ("global_m2_usd_bn", "Global M2", "bn USD"),
        "Fed_BalanceSheet": ("fed_assets_usd_bn", "Federal Reserve balance sheet", "bn USD"),
        "ECB_BalanceSheet": ("ecb_assets_usd_bn", "ECB balance sheet", "bn USD"),
        "BOJ_BalanceSheet": ("boj_assets_usd_bn", "BoJ balance sheet", "bn USD"),
        "PBOC_BalanceSheet": ("pboc_assets_usd_bn", "PBoC balance sheet", "bn USD"),
        "Global_CB_Assets": ("global_cb_assets_usd_bn", "Global central bank assets", "bn USD"),
    }
    for col, (source_col, description, unit) in monthly_map.items():
        add_frame_column(dataset, col, monthly, source_col, metadata, "RAW", "Global Liquidity", description, unit, "monthly", "Global Liquidity")
    weekly_map = {
        "WALCL": ("fed_assets_usd_bn", "Federal Reserve assets weekly layer", "bn USD"),
        "TGA": ("tga_usd_bn", "Treasury General Account", "bn USD"),
        "RRP": ("rrp_usd_bn", "Reverse repo", "bn USD"),
        "US_Net_Liquidity": ("us_net_liquidity_usd_bn", "US net liquidity", "bn USD"),
    }
    for col, (source_col, description, unit) in weekly_map.items():
        add_frame_column(dataset, col, weekly, source_col, metadata, "RAW", "US Net Liquidity", description, unit, "weekly", "Funding Liquidity")

    regime = build_global_liquidity_regime_frame_for_export(monthly, weekly)
    derived_map = {
        "GlobalM2_Growth_13W": ("m2_13w", "Global M2 13W growth", "percent"),
        "GlobalM2_Growth_26W": ("m2_26w", "Global M2 26W growth", "percent"),
        "GlobalM2_Growth_52W": ("m2_52w", "Global M2 52W growth", "percent"),
        "GlobalM2_TrendScore": ("m2_13w_pctl", "Global M2 trend score / percentile", "0-100"),
        "M2Impulse": ("m2_impulse", "M2 impulse", "score"),
        "CBImpulse": ("cb_impulse", "Central bank impulse", "score"),
        "USNLImpulse": ("usnl_impulse", "US net liquidity impulse", "score"),
        "GlobalLiquidityScore": ("global_liquidity_score", "Global liquidity score", "0-100"),
        "GlobalLiquidity_Direction_13W": ("direction_13w", "13-week global liquidity score direction", "score points"),
        "GlobalLiquidity_State": ("final_regime_label", "Global liquidity regime", "state"),
        "GlobalLiquidity_Direction_State": ("direction_13w_state", "Global liquidity direction state", "state"),
        "LongLiquidityCycle_Phase": ("long_cycle_phase", "65M liquidity cycle phase", "state"),
    }
    for col, (source_col, description, unit) in derived_map.items():
        add_frame_column(dataset, col, regime, source_col, metadata, "DERIVED", "Global Liquidity", description, unit, "weekly", "Global Liquidity Regime")
    for col in ["GlobalLiquidityScore"]:
        if col in dataset.columns:
            add_derived_changes(dataset, metadata, col, [4, 13, 26], "absolute", "Global Liquidity")
    if "TGA" in dataset.columns:
        dataset["TGA_Change"] = pd.to_numeric(dataset["TGA"], errors="coerce").diff(1)
        metadata.append(SeriesMeta("TGA_Change", "DERIVED", "Treasury Financing", "Weekly TGA change", "bn USD", "current minus prior week", "Internal", "weekly", "Treasury Financing"))


def add_liquidity_forecast(dataset: pd.DataFrame, metadata: list[SeriesMeta]) -> None:
    forecast, status = read_forecast_snapshot()
    mapping = {
        "US10Y_TermPremium": "%", "SOFR": "%", "EFFR": "%", "SOFR_EFFR_Spread": "percentage points",
        "US_BankReserves": "bn USD", "US10Y_TermPremium_26W_Change": "percentage points",
        "US_BankReserves_13W_PctChange": "%", "BankReservesImpulse": "0-100",
        "LiquidityPressureScore": "0-100", "LiquidityPressureBand": "state",
        "PolicyResponseScore": "0-100", "PolicyResponseBand": "state",
        "LiquidityForwardSignal": "state", "LiquidityForecastState": "state",
        "RSP_SPY": "ratio", "RSP_SPY_13W_ChangePct": "%", "RSP_SPY_13W_Percentile": "0-100",
        "IWM_SPY": "ratio", "IWM_SPY_13W_ChangePct": "%", "IWM_SPY_13W_Percentile": "0-100",
        "BreadthParticipationState": "state", "RiskOnConfirmation": "boolean", "RiskOnState": "state",
        "RiskReductionWarning": "state", "LiquidityForecastModelVersion": "version", "BreadthModelVersion": "version",
    }
    if forecast.empty:
        for name in mapping:
            dataset[name] = np.nan
            metadata.append(SeriesMeta(name, "MODEL_OUTPUT", "Liquidity Forecast", name, mapping[name], "Snapshot unavailable", "Production liquidity forecast snapshot", "weekly", "Liquidity Forecast", notes="Run liquidity-forecast refresh job"))
        return
    forecast = prepare_date_frame(forecast, "Date")
    for name, unit in mapping.items():
        add_frame_column(dataset, name, forecast, name, metadata, "MODEL_OUTPUT", "Liquidity Forecast", name, unit, "weekly", "Liquidity Forecast", point_in_time=f"Production snapshot as of {status.get('DataAsOf', 'n/a')}; {status.get('TimingConvention', 'trailing weekly windows')}")


def add_rates_financial_conditions(dataset: pd.DataFrame, metadata: list[SeriesMeta], api_key: str | None) -> None:
    snapshot = read_rates_fc_snapshot()
    if snapshot.history.empty:
        snapshot = refresh_rates_fc_snapshot(api_key)
    history = snapshot.history.set_index("Date")
    dates = pd.to_datetime(dataset["Date"], errors="coerce")
    for field in RATES_FC_EXPORT_FIELDS:
        if field not in history:
            raise ValueError(f"Rates / FC production field missing: {field}")
        dataset[field] = dates.map(history[field])
        metadata.append(SeriesMeta(
            field, "MODEL_OUTPUT", "Rates & Financial Conditions", field,
            "version" if field.endswith("ModelVersion") else "state" if history[field].dtype == "object" else "z-score / percentage points",
            "Exact completed W-FRI observation; no forward fill", "Production Rates / FC snapshot", "weekly",
            "Rates & Financial Conditions / Financial Fragility", False,
            snapshot.status.get("TimingConvention", "Release-lag-aware production history"), False,
        ))


def add_funding_conditions(dataset: pd.DataFrame, metadata: list[SeriesMeta], api_key: str | None) -> None:
    snapshot = read_funding_snapshot()
    if snapshot.weekly.empty:
        snapshot = refresh_funding_snapshot(api_key)
    history = snapshot.weekly.set_index("Date")
    dates = pd.to_datetime(dataset["Date"], errors="coerce")
    for field in FUNDING_EXPORT_FIELDS:
        if field not in history:
            raise ValueError(f"Funding Conditions production field missing: {field}")
        dataset[field] = dates.map(history[field])
        dtype = history[field].dtype
        unit = ("version" if field.endswith("ModelVersion") else
                "flag" if pd.api.types.is_bool_dtype(dtype) else
                "date" if pd.api.types.is_datetime64_any_dtype(dtype) else
                "state" if dtype == "object" else "score / rate")
        metadata.append(SeriesMeta(
            field, "MODEL_OUTPUT", "Funding Conditions", field, unit,
            "Exact completed W-FRI observation; no forward fill", "Production Funding Conditions snapshot",
            "weekly", "Funding Conditions / Financial Fragility", False,
            snapshot.status.get("TimingConvention", "Initial-release production history"), False,
        ))


def add_positioning(dataset: pd.DataFrame, metadata: list[SeriesMeta]) -> pd.DataFrame:
    try:
        master = read_processed("cftc_dashboard")
        aaii = read_processed("aaii")
        naaim = read_processed("naaim")
        if master.empty and aaii.empty and naaim.empty:
            data = load_positioning_data(force_update=False)
            master = data.get("cftc_master", pd.DataFrame())
            aaii = data.get("aaii", pd.DataFrame())
            naaim = data.get("naaim", pd.DataFrame())
    except Exception:
        master, aaii, naaim = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if not aaii.empty:
        frame = prepare_date_frame(aaii, "Date")
        for col in [
            "AAII_Bullish",
            "AAII_Neutral",
            "AAII_Bearish",
            "AAII_BullBearSpread",
            "AAII_Bullish_3Y_Percentile",
            "AAII_Bearish_3Y_Percentile",
            "AAII_BullBearSpread_3Y_Percentile",
        ]:
            add_frame_column(dataset, col, frame, col, metadata, "RAW" if "Percentile" not in col and "Spread" not in col else "DERIVED", "AAII", col, "percent / percentile", "weekly", "Positioning")
        if "AAII_Bullish" in dataset.columns:
            dataset["AAII_Bullish_4W_Change"] = pd.to_numeric(dataset["AAII_Bullish"], errors="coerce").diff(4)
            dataset["AAII_Bearish_4W_Change"] = pd.to_numeric(dataset["AAII_Bearish"], errors="coerce").diff(4)
            dataset["AAII_Spread_4W_Change"] = pd.to_numeric(dataset["AAII_BullBearSpread"], errors="coerce").diff(4)
            for col in ["AAII_Bullish_4W_Change", "AAII_Bearish_4W_Change", "AAII_Spread_4W_Change"]:
                metadata.append(SeriesMeta(col, "DERIVED", "AAII", col, "percentage points", "current minus 4W prior", "Internal", "weekly", "Positioning"))
    if not naaim.empty:
        frame = prepare_date_frame(naaim, "Date")
        for col in ["NAAIM_Exposure", "NAAIM_3Y_Percentile"]:
            add_frame_column(dataset, col, frame, col, metadata, "RAW" if col == "NAAIM_Exposure" else "DERIVED", "NAAIM", col, "index / percentile", "weekly", "Positioning")
        if "NAAIM_Exposure" in dataset.columns:
            dataset["NAAIM_4W_Change"] = pd.to_numeric(dataset["NAAIM_Exposure"], errors="coerce").diff(4)
            dataset["NAAIM_13W_Change"] = pd.to_numeric(dataset["NAAIM_Exposure"], errors="coerce").diff(13)
            for col in ["NAAIM_4W_Change", "NAAIM_13W_Change"]:
                metadata.append(SeriesMeta(col, "DERIVED", "NAAIM", col, "index points", "current minus prior", "Internal", "weekly", "Positioning"))

    positioning_metadata = build_positioning_metadata(master)
    if master.empty:
        return positioning_metadata
    source = master.copy()
    if "Preferred_For_Dashboard" in source.columns:
        source = source[source["Preferred_For_Dashboard"].astype(bool)]
    metrics = [
        ("NetPctOI", "NetPctOI"),
        ("NetPctOI_3Y_Percentile", "3YPercentile"),
        ("NetPctOI_13W_Change", "13WChange"),
    ]
    for asset, participant, prefix in CORE_CFTC_EXPORT:
        group = source.loc[
            source.get("Canonical_Asset", pd.Series(dtype="object")).astype(str).eq(asset)
            & source.get("Participant_Category", pd.Series(dtype="object")).astype(str).eq(participant)
        ].copy()
        if group.empty:
            for _, out_suffix in metrics:
                col = f"{prefix}_{out_suffix}"
                dataset[col] = np.nan
                metadata.append(SeriesMeta(col, "DERIVED", f"CFTC {asset}", f"{asset} {participant} {out_suffix}", "% OI", "Core CFTC macro signal unavailable", "Unified Positioning Pipeline", "weekly", "Positioning / Tail Risk", notes="MISSING"))
            continue
        dated = group.copy()
        dated["Date"] = pd.to_datetime(dated["Date"], errors="coerce")
        dated = dated.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        for source_col, out_suffix in metrics:
            if source_col not in dated.columns:
                continue
            col = f"{prefix}_{out_suffix}"
            add_aligned_series(
                dataset,
                col,
                pd.Series(pd.to_numeric(dated[source_col], errors="coerce").values, index=dated["Date"]),
                metadata,
                SeriesMeta(col, "DERIVED", f"CFTC {asset}", f"{asset} {participant} {out_suffix}", "% OI", "Processed CFTC_Master preferred contract, compact macro universe only", "Unified Positioning Pipeline", "weekly", "Positioning / Tail Risk", False, "Report_Date production convention; publication date not yet available", True),
            )
    return positioning_metadata


def add_bybit(dataset: pd.DataFrame, metadata: list[SeriesMeta]) -> None:
    try:
        frame = read_bybit_storage(BYBIT_STORAGE_PATH)
    except Exception:
        frame = pd.DataFrame()
    if frame.empty:
        return
    btc = frame[frame["asset"].astype(str).eq("BTC-USD")].copy()
    if btc.empty:
        return
    btc = prepare_date_frame(btc, "date")
    mapping = {
        "BTC_Bybit_Funding": ("funding_1d", "BTC Bybit funding 1D"),
        "BTC_Bybit_Funding_7D": ("funding_7d", "BTC Bybit funding 7D"),
        "BTC_Bybit_Funding_28D": ("funding_28d", "BTC Bybit funding 28D"),
        "BTC_Bybit_OpenInterest": ("open_interest_usd", "BTC Bybit open interest USD"),
        "BTC_Bybit_OI_1W_Change": ("oi_change_1w_pct", "BTC Bybit OI 1W change"),
        "BTC_Bybit_OI_4W_Change": ("oi_change_4w_pct", "BTC Bybit OI 4W change"),
        "BTC_Basis": ("perp_premium_pct", "BTC perpetual basis / premium"),
    }
    for out_col, (source_col, description) in mapping.items():
        add_frame_column(dataset, out_col, btc, source_col, metadata, "RAW" if source_col in {"funding_1d", "open_interest_usd"} else "DERIVED", "Crypto-native derivatives", description, "various", "weekly", "BTC Regime")


def add_market_regime_outputs(dataset: pd.DataFrame, metadata: list[SeriesMeta], api_key: str | None) -> None:
    history = build_market_transition_history_for_export(api_key)
    if history.empty:
        return
    frame = prepare_date_frame(history, "Date")
    mapping = {
        "StructuralMarketRegime": ("Market_Regime", "Structural market regime"),
        "FastTransitionRisk": ("Fast_Transition_Risk", "Fast transition risk"),
        "FastRiskState": ("Fast_Transition_State", "Fast risk state"),
        "FastRiskDirection4W": ("Fast_Risk_Direction_4W_State", "Fast risk direction 4W"),
        "MacroTransitionRisk": ("Macro_Transition_Risk", "Macro transition risk"),
        "MacroRiskState": ("Macro_Transition_State", "Macro risk state"),
        "CreditRisk": ("Credit_Risk", "Credit risk"),
        "CreditState": ("Credit_State", "Credit state"),
        "FinalMarketState": ("Final_Market_State", "Final market state"),
        "PositioningRisk": ("PositioningRisk", "Positioning risk overlay"),
        "PositioningState": ("PositioningState", "Positioning risk state"),
        "LiquidityWarning": ("LiquidityWarning", "Liquidity warning channel"),
        "CreditWarning": ("CreditWarning", "Credit warning channel"),
        "FastWarning": ("FastWarning", "Fast transition warning channel"),
        "MacroWarning": ("MacroWarning", "Macro transition warning channel"),
        "TailRiskFlag": ("TailRiskFlag", "Tail risk escalation flag"),
        "TailRiskReason": ("TailRiskReason", "Tail risk machine-readable reason"),
        "PositioningModel_Version": ("PositioningModel_Version", "Positioning model version"),
        "TailRiskModel_Version": ("TailRiskModel_Version", "Tail risk model version"),
        "VIXRisk": ("Fast_Transition_Risk", "VIX/DXY fast risk production composite"),
        "DXYRisk": ("Macro_DXY_Risk", "DXY risk"),
        "US2YRisk": ("US2Y_Risk", "US2Y risk"),
        "GlobalM2Risk": ("Global_M2_Risk_26W", "Global M2 risk"),
        "Credit_13W_Widening_Percentile": ("Credit_Widening_Percentile", "HY OAS 13W widening percentile"),
    }
    for out_col, (source_col, description) in mapping.items():
        add_frame_column(dataset, out_col, frame, source_col, metadata, "MODEL_OUTPUT", "Market Regime", description, "state / 0-100", "weekly", "Market Regime", point_in_time="Recomputed current production model")


def add_gold_regime_outputs(dataset: pd.DataFrame, metadata: list[SeriesMeta], api_key: str | None) -> None:
    mapping = {
        "GoldAlphaScore": ["gold_alpha", "GoldAlphaScore", "gold_alpha_score"],
        "GoldStructuralMacro": ["structural_macro_score", "GoldStructuralMacro"],
        "GoldForwardMacroRisk": ["forward_macro_risk", "GoldForwardMacroRisk"],
        "GoldTacticalFlow": ["tactical_flow_score", "GoldTacticalFlow"],
        "GoldRegimeState": ["gold_regime", "GoldRegimeState"],
        "Gold_DXYBullScore": ["dxy_bull_score"],
        "Gold_RealYieldBullScore": ["real_yield_bull_score"],
        "Gold_US2YBullScore": ["us2y_bull_score"],
        "Gold_US2YRisk": ["us2y_risk"],
        "Gold_WTIRisk": ["wti_risk"],
        "Gold_ETFFlowScore": ["etf_flow_score"],
        "Gold_COTMomentumScore": ["cot_momentum_score"],
        "Gold_LongLiquidityCyclePhase": ["long_liquidity_cycle"],
        "Gold_StructuralDemandContext": ["additional_structural_demand_context"],
        "Gold_GlobalLiquidityContext": ["long_liquidity_cycle"],
    }
    history = load_gold_regime_history_for_export(api_key)
    if history.empty:
        for out_col in mapping:
            dataset[out_col] = np.nan
            metadata.append(
                SeriesMeta(
                    out_col,
                    "MODEL_OUTPUT",
                    "Gold Regime",
                    out_col,
                    "score/state",
                    "Missing optional production history",
                    "Gold Regime",
                    "weekly",
                    "Gold Regime",
                    notes="Live Gold Regime UI recalculation is intentionally not called by batch export",
                )
            )
        return
    date_col = "date" if "date" in history.columns else "Date" if "Date" in history.columns else history.columns[0]
    frame = prepare_date_frame(history, date_col)
    for out_col, candidates in mapping.items():
        source_col = next((col for col in candidates if col in frame.columns), "")
        if source_col:
            add_frame_column(dataset, out_col, frame, source_col, metadata, "MODEL_OUTPUT", "Gold Regime", out_col, "score/state", "weekly", "Gold Regime", point_in_time="Recomputed current production model")
        else:
            dataset[out_col] = np.nan
            metadata.append(SeriesMeta(out_col, "MODEL_OUTPUT", "Gold Regime", out_col, "score/state", "Missing optional production history column", "Gold Regime", "weekly", "Gold Regime", notes="Column unavailable in persisted history"))


def add_btc_regime_outputs(dataset: pd.DataFrame, metadata: list[SeriesMeta]) -> None:
    try:
        _, monthly, weekly = read_global_liquidity()
        liquidity = build_global_liquidity_regime_frame_for_export(monthly, weekly)
        market_history = build_market_transition_history_for_export(None)
        frame = build_btc_macro_frame_for_export(liquidity, market_history)
    except BaseException:
        frame = pd.DataFrame()
    if frame.empty:
        return
    frame = prepare_date_frame(frame, "date")
    mapping = {
        "BTCStructuralMacro": "BTCStructuralMacro",
        "BTCForwardMacroRisk": "BTCForwardMacroRisk",
        "BTC_GlobalM2Bull": "BTCGlobalM2Bull13W",
        "BTC_GlobalM2Risk": "BTCGlobalM2Risk13W",
        "BTC_DXYBull": "BTCDXYBull",
        "BTC_DXYRisk": "Macro_DXY_Risk",
        "BTC_US2YBull": "BTCUS2YBull",
        "BTC_US2YRisk": "US2Y_Risk",
        "BTC_CreditRisk": "Credit_Risk",
        "BTCRegimeState": "BTCRegimeState",
    }
    for out_col, source_col in mapping.items():
        unit = "state" if out_col == "BTCRegimeState" else "0-100"
        add_frame_column(dataset, out_col, frame, source_col, metadata, "MODEL_OUTPUT", "BTC Regime", out_col, unit, "weekly", "BTC Regime", point_in_time="Recomputed current production model")


def add_business_cycle_model_outputs(dataset: pd.DataFrame, metadata: list[SeriesMeta], api_key: str | None) -> None:
    mapping = {
        "SurveyScore": ("SurveyScore", "Business Cycle survey pillar score", "z-score"),
        "LaborScore": ("LaborScore", "Business Cycle labor pillar score", "z-score"),
        "ProductionScore": ("ProductionScore", "Business Cycle production pillar score", "z-score"),
        "DemandIncomeScore": ("DemandIncomeScore", "Business Cycle demand / income pillar score", "z-score"),
        "BusinessCycleLevel": ("BusinessCycleLevel", "BUSINESS_CYCLE_V1 level score", "z-score"),
        "BusinessCycleMomentum": ("BusinessCycleMomentum", "BUSINESS_CYCLE_V1 13W momentum score", "score change"),
        "BusinessCycleCandidateState": ("BusinessCycleCandidateState", "Unconfirmed Business Cycle candidate phase", "state"),
        "BusinessCycleState": ("BusinessCycleState", "2-week confirmed Business Cycle phase", "state"),
        "BusinessCycleTransitionZone": ("BusinessCycleTransitionZone", "Business Cycle transition zone flag", "boolean"),
        "BusinessCycleConfidence": ("BusinessCycleConfidence", "Business Cycle confidence", "state"),
        "BusinessCyclePosition": ("BusinessCyclePosition", "Business Cycle position vs trend", "state"),
        "BusinessCycleDirection": ("BusinessCycleDirection", "Business Cycle growth direction", "state"),
        "LaborCycleState": ("LaborCycleState", "Labor cycle diagnostic state", "state"),
        "ProductivityExpansionFlag": ("ProductivityExpansionFlag", "Productivity / jobless expansion diagnostic flag", "boolean"),
        "MarketPricingScore": ("MarketPricingScore", "Inflation market pricing channel score", "z-score"),
        "ModelImpliedInflationScore": ("ModelImpliedInflationScore", "Inflation model-implied channel score", "z-score"),
        "SurveyInflationScore": ("SurveyInflationScore", "Inflation survey channel score", "z-score"),
        "InflationDirectionScore": ("InflationDirectionScore", "INFLATION_LAYER_V1 direction score", "z-score"),
        "InflationCandidateState": ("InflationCandidateState", "Unconfirmed inflation candidate state", "state"),
        "InflationState": ("InflationState", "2-week confirmed inflation state", "state"),
        "InflationTransitionZone": ("InflationTransitionZone", "Inflation transition zone flag", "boolean"),
        "InflationConfidence": ("InflationConfidence", "Inflation state confidence", "state"),
        "InflationChannelAgreement": ("InflationChannelAgreement", "Inflation channel agreement count", "state"),
        "MarketPricingDirection": ("MarketPricingDirection", "Market-pricing inflation direction", "state"),
        "ModelImpliedDirection": ("ModelImpliedDirection", "Model-implied inflation direction", "state"),
        "SurveyInflationDirection": ("SurveyInflationDirection", "Survey inflation direction", "state"),
        "InflationCurve": ("InflationCurve", "EXPINF1YR minus EXPINF5YR", "percentage points"),
        "InflationCurveChange_13W": ("InflationCurveChange_13W", "13W change in EXPINF curve", "percentage points"),
        "InflationCurveChange_26W": ("InflationCurveChange_26W", "26W change in EXPINF curve", "percentage points"),
        "T5YIFR": ("T5YIFR", "5Y5Y forward inflation expectation rate", "%"),
        "T5YIFR_Change_13W": ("T5YIFR_Change_13W", "13W change in T5YIFR", "percentage points"),
        "T5YIFR_Change_26W": ("T5YIFR_Change_26W", "26W change in T5YIFR", "percentage points"),
        "StructuralInflationDirection": ("StructuralInflationDirection", "T5YIFR structural inflation direction", "state"),
        "RealizedInflationMomentum": ("RealizedInflationMomentum", "Realized inflation confirmation momentum", "z-score"),
        "RealizedInflationDirection": ("RealizedInflationDirection", "Realized inflation direction", "state"),
        "InflationConfirmationStatus": ("InflationConfirmationStatus", "Leading versus realized inflation confirmation status", "state"),
        "EconomyRegime": ("EconomyRegime", "ECONOMY_REGIME_V1 combined growth / inflation regime", "state"),
    }
    try:
        start = pd.to_datetime(dataset["Date"], errors="coerce").min()
        end = pd.to_datetime(dataset["Date"], errors="coerce").max()
        snapshot = build_business_cycle_snapshot(api_key=api_key, start_date=start, end_date=end)
        frame = prepare_date_frame(snapshot.history, "date")
    except Exception:
        frame = pd.DataFrame(columns=["date"])

    for out_col, (source_col, description, unit) in mapping.items():
        add_frame_column(
            dataset,
            out_col,
            frame,
            source_col,
            metadata,
            "MODEL_OUTPUT",
            "Business Cycle",
            description,
            unit,
            "weekly",
            "Business Cycle / Inflation / Economy Regime",
            point_in_time="Recomputed current production model with release-lag convention",
        )


def add_model_versions(dataset: pd.DataFrame, metadata: list[SeriesMeta]) -> None:
    versions = {
        "BusinessCycleModelVersion": BUSINESS_CYCLE_MODEL_VERSION,
        "InflationLayerModelVersion": INFLATION_LAYER_MODEL_VERSION,
        "EconomyRegimeModelVersion": ECONOMY_REGIME_MODEL_VERSION,
        "MarketRegime_Version": MARKET_REGIME_VERSION,
        "GoldRegime_Version": GOLD_REGIME_VERSION,
        "BTCRegime_Version": BTC_REGIME_VERSION,
        "GlobalLiquidity_Version": GLOBAL_LIQUIDITY_VERSION,
        "PositioningModel_Version": POSITIONING_VERSION,
    }
    for col, value in versions.items():
        dataset[col] = value
        metadata.append(SeriesMeta(col, "METADATA", "Model Versioning", col, "text", "Static model version label for exported outputs", "Internal", "export", "Versioning", False, "n/a", False))


def write_macro_research_workbook(
    weekly_data: pd.DataFrame,
    metadata: list[SeriesMeta],
    data_quality: pd.DataFrame,
    positioning_metadata: pd.DataFrame,
    model_metadata: pd.DataFrame,
) -> bytes:
    output = BytesIO()
    metadata_df = pd.DataFrame([m.__dict__ for m in metadata]).rename(
        columns={
            "column": "Column",
            "layer": "Layer",
            "category": "Category",
            "description": "Description",
            "unit": "Unit",
            "transformation": "Transformation",
            "source": "Source",
            "original_frequency": "OriginalFrequency",
            "model_usage": "ModelUsage",
        }
    )
    metadata_df = metadata_df[["Column", "Layer", "Category", "Description", "Unit", "Transformation", "Source", "OriginalFrequency", "ModelUsage"]]
    from openpyxl import Workbook

    workbook = Workbook(write_only=True)
    for sheet_name, frame in {
        "Weekly_Data": weekly_data,
        "Metadata": metadata_df,
        "Data_Quality": data_quality,
        "Positioning_Metadata": positioning_metadata,
        "Model_Metadata": model_metadata,
        "Release_Lag_Info": release_lag_info(metadata),
    }.items():
        write_dataframe_sheet(workbook, sheet_name, frame, freeze_first_column=(sheet_name == "Weekly_Data"))
    workbook.save(output)
    return output.getvalue()


def write_dataframe_sheet(workbook: Any, sheet_name: str, frame: pd.DataFrame, freeze_first_column: bool = False) -> None:
    worksheet = workbook.create_sheet(title=sheet_name[:31])
    worksheet.append([str(column) for column in frame.columns])
    for row in frame.itertuples(index=False, name=None):
        worksheet.append([excel_cell_value(value) for value in row])
    if frame.columns.any():
        last_col = excel_column_name(len(frame.columns))
        worksheet.auto_filter.ref = f"A1:{last_col}{len(frame) + 1}"
        worksheet.freeze_panes = "B2" if freeze_first_column else "A2"
        for idx, column in enumerate(frame.columns, start=1):
            worksheet.column_dimensions[excel_column_name(idx)].width = min(max(12, len(str(column)) + 2), 36)


def excel_cell_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime().replace(tzinfo=None)
    if isinstance(value, np.datetime64):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        return timestamp.to_pydatetime().replace(tzinfo=None)
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def excel_column_name(column_number: int) -> str:
    name = ""
    while column_number:
        column_number, remainder = divmod(column_number - 1, 26)
        name = chr(65 + remainder) + name
    return name


def build_data_quality(dataset: pd.DataFrame, metadata: list[SeriesMeta]) -> pd.DataFrame:
    rows = []
    total = len(dataset)
    dates = pd.to_datetime(dataset["Date"], errors="coerce")
    for meta in metadata:
        col = meta.column
        if col not in dataset.columns:
            continue
        values = dataset[col]
        non_null = int(values.notna().sum())
        valid_dates = dates[values.notna()]
        first = valid_dates.min()
        last = valid_dates.max()
        stale = False
        if pd.notna(last):
            stale = (dates.max() - last).days > 45 and meta.layer != "METADATA"
        coverage = non_null / total if total else np.nan
        quality_status, quality_notes = data_quality_status(meta, coverage, values, dates)
        rows.append(
            {
                "Series": col,
                "Layer": meta.layer,
                "Category": meta.category,
                "FirstDate": first.date() if pd.notna(first) else "",
                "LastDate": last.date() if pd.notna(last) else "",
                "TotalWeeklyRows": total,
                "NonNullRows": non_null,
                "CoveragePct": coverage,
                "Status": quality_status,
                "OriginalFrequency": meta.original_frequency,
                "Source": meta.source,
                "ReleaseDateAvailable": bool(meta.release_date_available),
                "PointInTimeSafe": meta.point_in_time_safe,
                "ForwardFilled": bool(meta.forward_filled),
                "Stale": bool(stale),
                "Notes": quality_notes,
            }
        )
    return pd.DataFrame(rows)


def data_quality_status(meta: SeriesMeta, coverage: float, values: pd.Series, dates: pd.Series) -> tuple[str, str]:
    existing_note = str(meta.notes or "")
    if meta.layer == "METADATA":
        return "OK", existing_note or "Metadata"
    if not np.isfinite(coverage) or coverage == 0:
        status = "OPTIONAL_UNAVAILABLE" if "optional" in existing_note.lower() else "MISSING"
        return status, existing_note or status
    first_valid = dates[values.notna()].min()
    lookback_like = any(token in meta.column for token in ["3Y", "13W", "26W", "52W", "YoY", "Percentile"])
    if lookback_like and pd.notna(first_valid) and (first_valid - dates.min()).days > 35:
        return "EXPECTED_LOOKBACK" if coverage >= 0.50 else "LIMITED_HISTORY", existing_note or "Initial nulls are expected due to rolling lookback"
    if meta.category == "NAAIM" and coverage < 0.90:
        return "LIMITED_HISTORY", existing_note or "LimitedHistory = True"
    if coverage < 0.50:
        return "LIMITED_HISTORY", existing_note or "Coverage below expected minimum"
    return "OK", existing_note or "OK"


def build_positioning_metadata(master: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Canonical_Asset",
        "Asset_Group",
        "Raw_Contract_Name",
        "CFTC_Code",
        "Report_Type",
        "Participant_Category",
        "Preferred_For_Dashboard",
        "FirstDate",
        "LastDate",
        "HistoryWeeks",
        "LimitedHistory",
        "Source",
    ]
    if master.empty:
        return pd.DataFrame(columns=columns)
    group_cols = [
        "Canonical_Asset",
        "Asset_Group",
        "Raw_Contract_Name",
        "CFTC_Code",
        "Report_Type",
        "Participant_Category",
        "Preferred_For_Dashboard",
        "Source",
    ]
    source = master.copy()
    source["Date"] = pd.to_datetime(source["Date"], errors="coerce")
    out = (
        source.groupby(group_cols, dropna=False)
        .agg(FirstDate=("Date", "min"), LastDate=("Date", "max"), HistoryWeeks=("Date", "nunique"))
        .reset_index()
    )
    out["LimitedHistory"] = out["HistoryWeeks"] < 156
    out["FirstDate"] = out["FirstDate"].dt.date
    out["LastDate"] = out["LastDate"].dt.date
    return out[columns]


def build_model_metadata() -> pd.DataFrame:
    rows = [
        ("Business Cycle", BUSINESS_CYCLE_MODEL_VERSION, "BusinessCycleLevel", "Real-economy cycle level", "0.30 * SurveyScore + 0.30 * ProductionScore + 0.25 * DemandIncomeScore + 0.15 * LaborScore", "ISM + CFNAI + labor + production + demand/income", "weekly", "build_business_cycle_snapshot()", "Expanding point-in-time z-scores; no liquidity or inflation inputs"),
        ("Business Cycle", BUSINESS_CYCLE_MODEL_VERSION, "BusinessCycleMomentum", "Real-economy cycle momentum", "Weighted 13W changes in pillar scores", "Survey + production + demand/income + labor pillars", "13W", "build_business_cycle_snapshot()", "2-week confirmation converts candidate phase into production state"),
        ("Inflation Layer", INFLATION_LAYER_MODEL_VERSION, "InflationDirectionScore", "Expectations-led inflation direction", "0.35 * MarketPricingScore + 0.40 * ModelImpliedInflationScore + 0.25 * SurveyInflationScore", "T5YIE + T10YIE + EXPINF1YR + EXPINF5YR + Michigan 1Y", "13-26W", "build_business_cycle_snapshot()", "Realized inflation is confirmation only, not the production state driver"),
        ("Economy Regime", ECONOMY_REGIME_MODEL_VERSION, "EconomyRegime", "Growth direction plus inflation state", "BusinessCycleDirection x InflationState", "BUSINESS_CYCLE_V1 + INFLATION_LAYER_V1", "weekly", "build_business_cycle_snapshot()", "Interpretation layer; does not modify BusinessCycleState or InflationState"),
        ("Market Regime", MARKET_REGIME_VERSION, "FastTransitionRisk", "Short-horizon transition risk", "Existing production function", "VIX + DXY", "1-4W", "calculate_fast_transition_risk_history()", "Recomputed over history using current model"),
        ("Market Regime", MARKET_REGIME_VERSION, "MacroTransitionRisk", "Medium-term macro transition risk", "Existing production function", "DXY + US2Y + Global M2 + liquidity", "8-26W", "calculate_macro_transition_risk_history()", "Recomputed over history using current model"),
        ("Market Regime", MARKET_REGIME_VERSION, "CreditRisk", "Credit stress state", "Existing production function", "HY OAS", "13W", "calculate_credit_stress_confirmation_history()", "Recomputed over history using current model"),
        ("Market Regime", POSITIONING_VERSION, "PositioningRisk", "Positioning vulnerability overlay", "0.50 * AAII_Bearish_3Y_Percentile + 0.50 * VIX_AssetManager_NetPctOI_3Y_Percentile", "AAII + VIX CFTC Asset Manager", "weekly", "calculate_positioning_risk_history()", "Point-in-time trailing 3Y percentiles"),
        ("Market Regime", "TAILRISK_V1", "TailRiskFlag", "Tail risk interaction overlay", "Explicit escalation rules, not weighted composite", "Positioning + Liquidity + Credit + Fast + Macro", "weekly", "calculate_tail_risk_history()", "Separate overlay; does not modify StructuralMarketRegime"),
        ("Global Liquidity", GLOBAL_LIQUIDITY_VERSION, "GlobalLiquidityScore", "Global liquidity impulse score", "Existing production function", "Global M2 + CB assets + US net liquidity", "13W+", "_build_global_liquidity_regime_frame()", "Recomputed over history using current model"),
        ("ETF Fund Flows", ETF_FLOWS_VERSION, "ETF_Flow_Trailing_3Y_Percentile", "ETF flow intensity percentile", "Point-in-time percentile of 4W flow intensity over trailing 156 weeks", "ETF.com fund flows + ETF AUM", "4W / 3Y", "add_etf_fund_flows()", "Minimum 52 weekly observations; BTC aggregates IBIT + FBTC + GBTC"),
        ("Liquidity Forecast", FORECAST_VERSION, "LiquidityForecastState", "Forward liquidity phase", "Pressure/response interaction conditioned on current GLS direction", "DXY + MOVE + US2Y + ISM + CFNAI + claims + term premium + liquidity impulses + reserves", "13-26W", "refresh_forecast_snapshot()", "Persisted production history; trailing 156W percentiles, minimum 52"),
        ("Liquidity Forecast", BREADTH_VERSION, "BreadthParticipationState", "Relative-strength breadth confirmation", "RSP/SPY and IWM/SPY 13W change trailing percentiles", "RSP + IWM + SPY", "13W", "refresh_forecast_snapshot()", "Persisted production history; production threshold 60/40"),
        ("Rates & Financial Conditions", RATES_FC_MODEL_VERSION, "RatesFinancialConditionsRegime", "Rates direction x core financial conditions direction", "0.65 US2Y momentum + 0.35 real yield momentum; core FC direction 0.45 credit + 0.30 DXY + 0.25 MOVE", "DGS2 + DFII10 + HY/IG OAS + DXY + MOVE; NFCI/ANFCI confirmation only", "13W / 26W", "refresh_rates_fc_snapshot()", "Expanding 156W z-scores; NFCI/ANFCI/Fed Funds initial releases; other histories remain current vintages"),
        ("Funding Conditions", FUNDING_MODEL_VERSION, "FundingState", "Rule-based USD funding-system state", "FundingCore = 0.60 money-market stress + 0.40 reserve pressure; MOVE separate", "SOFR99 + DFF + WRESBAL + GDP; IORB/MOVE diagnostics", "daily / weekly", "refresh_funding_snapshot()", "FRED initial releases; expanding point-in-time normalization; production weekly snapshot"),
        ("Gold Regime", GOLD_REGIME_VERSION, "GoldStructuralMacro", "Gold structural macro backdrop", "Existing production function", "DXY + real yield + US2Y", "medium-term", "build_gold_regime_snapshot()", "Exported only when production history exposes column"),
        ("BTC Regime", BTC_REGIME_VERSION, "BTCForwardMacroRisk", "Forward BTC macro headwind", "Existing production function", "Global M2 + DXY + US2Y + Credit", "8-26W", "_build_btc_macro_frame()", "Recomputed over history using current model"),
        ("Positioning", POSITIONING_VERSION, "CFTC wide fields", "Canonical CFTC participant history", "Unified positioning pipeline", "CFTC_Master processed parquet", "weekly", "load_positioning_data()", "Report date convention; publication date unavailable"),
    ]
    return pd.DataFrame(rows, columns=["Model", "Version", "OutputColumn", "Description", "FormulaOrReference", "Inputs", "Horizon", "ProductionFunction", "Notes"])


def release_lag_info(metadata: list[SeriesMeta]) -> pd.DataFrame:
    rows = []
    for meta in metadata:
        if meta.category in {"CFTC Gold", "AAII", "NAAIM"} or meta.category.startswith("CFTC"):
            rows.append({"Series": meta.column, "TimingConvention": meta.point_in_time_safe, "ReleaseDateAvailable": meta.release_date_available, "Notes": meta.notes})
    if not rows:
        rows.append({"Series": "CFTC / AAII / NAAIM", "TimingConvention": "Processed production timing convention", "ReleaseDateAvailable": False, "Notes": "Publication date history not yet persisted"})
    return pd.DataFrame(rows)


def validate_weekly_dataset(dataset: pd.DataFrame) -> None:
    if dataset.empty or "Date" not in dataset.columns:
        raise ValueError("Weekly_Data is empty or missing Date")
    dates = pd.to_datetime(dataset["Date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Date contains invalid values")
    if dates.duplicated().any():
        raise ValueError("Date contains duplicate weekly rows")
    if not dates.is_monotonic_increasing:
        raise ValueError("Date is not ascending")
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if (dates > today + pd.Timedelta(days=7)).any():
        raise ValueError("Date contains future weekly rows")
    for prefix in ["SPY", "QQQ", "GLD", "BTCUSD"]:
        cols = [f"{prefix}_{field}" for field in ["Open", "High", "Low", "Close"]]
        if not set(cols).issubset(dataset.columns):
            continue
        d = dataset[cols].apply(pd.to_numeric, errors="coerce")
        valid = d.dropna()
        if valid.empty:
            continue
        if ((valid[f"{prefix}_Low"] > valid[f"{prefix}_Open"]) | (valid[f"{prefix}_Low"] > valid[f"{prefix}_Close"]) | (valid[f"{prefix}_High"] < valid[f"{prefix}_Open"]) | (valid[f"{prefix}_High"] < valid[f"{prefix}_Close"])).any():
            raise ValueError(f"OHLC validity failed for {prefix}")
    percentile_cols = [col for col in dataset.columns if "Percentile" in col or "3YPercentile" in col]
    for col in percentile_cols:
        values = pd.to_numeric(dataset[col], errors="coerce").dropna()
        if not values.empty and ((values < 0) | (values > 100)).any():
            raise ValueError(f"Percentile outside 0-100: {col}")


def build_global_liquidity_regime_frame_for_export(monthly: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    monthly = prepare_date_frame(monthly)
    weekly = prepare_date_frame(weekly)
    if monthly.empty and weekly.empty:
        return pd.DataFrame()
    monthly_source = liquidity_weekly_from_monthly(
        monthly,
        [
            "us_m2_usd_bn",
            "ea_m2_usd_bn",
            "china_m2_usd_bn",
            "japan_m2_usd_bn",
            "fed_assets_usd_bn",
            "ecb_assets_usd_bn",
            "boj_assets_usd_bn",
            "pboc_assets_usd_bn",
            "last_updated",
        ],
    )
    weekly_source = weekly.set_index("date").sort_index() if not weekly.empty and "date" in weekly.columns else pd.DataFrame()
    index = monthly_source.index.union(weekly_source.index).sort_values()
    if index.empty:
        return pd.DataFrame()
    frame = pd.DataFrame(index=index)
    for column in monthly_source.columns:
        frame[column] = monthly_source[column].reindex(index).ffill()
    for column in ["us_net_liquidity_usd_bn", "fed_assets_usd_bn", "tga_usd_bn", "rrp_usd_bn", "last_updated"]:
        if column in weekly_source.columns:
            frame[f"weekly_{column}" if column in frame.columns else column] = weekly_source[column].reindex(index).ffill()

    m2_components = ["us_m2_usd_bn", "ea_m2_usd_bn", "china_m2_usd_bn", "japan_m2_usd_bn"]
    cb_components = ["fed_assets_usd_bn", "ecb_assets_usd_bn", "boj_assets_usd_bn", "pboc_assets_usd_bn"]
    for column in m2_components + cb_components + ["us_net_liquidity_usd_bn"]:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["global_m2_usd_bn"] = frame[m2_components].sum(axis=1, min_count=4)
    frame["global_cb_assets_usd_bn"] = frame[cb_components].sum(axis=1, min_count=4)
    frame["global_m2_partial_usd_bn"] = frame[m2_components].sum(axis=1, min_count=1)
    frame["global_cb_assets_partial_usd_bn"] = frame[cb_components].sum(axis=1, min_count=1)
    if "weekly_us_net_liquidity_usd_bn" in frame.columns:
        frame["us_net_liquidity_usd_bn"] = frame["weekly_us_net_liquidity_usd_bn"]

    for weeks in [4, 13, 26, 52]:
        frame[f"m2_{weeks}w"] = frame["global_m2_usd_bn"].pct_change(weeks, fill_method=None)
        frame[f"cb_{weeks}w"] = frame["global_cb_assets_usd_bn"].pct_change(weeks, fill_method=None)
    for weeks in [4, 13, 26]:
        frame[f"usnl_{weeks}w"] = frame["us_net_liquidity_usd_bn"].pct_change(weeks, fill_method=None)

    frame["trend_score"] = (
        0.50 * (frame["m2_13w"] > 0).astype(float)
        + 0.30 * (frame["m2_26w"] > 0).astype(float)
        + 0.20 * (frame["m2_52w"] > 0).astype(float)
    )
    frame.loc[frame[["m2_13w", "m2_26w", "m2_52w"]].isna().any(axis=1), "trend_score"] = np.nan
    frame["trend_state"] = frame["trend_score"].map(liquidity_trend_state)
    for base in ["m2_13w", "m2_26w", "m2_52w", "cb_13w", "cb_26w", "cb_52w", "usnl_4w", "usnl_13w", "usnl_26w"]:
        frame[f"{base}_pctl"] = liquidity_trailing_percentile(frame[base])
    frame["m2_impulse"] = 0.50 * frame["m2_13w_pctl"] + 0.30 * frame["m2_26w_pctl"] + 0.20 * frame["m2_52w_pctl"]
    frame["cb_impulse"] = 0.50 * frame["cb_13w_pctl"] + 0.30 * frame["cb_26w_pctl"] + 0.20 * frame["cb_52w_pctl"]
    frame["usnl_impulse"] = 0.50 * frame["usnl_4w_pctl"] + 0.30 * frame["usnl_13w_pctl"] + 0.20 * frame["usnl_26w_pctl"]
    frame["global_liquidity_score"] = 0.50 * frame["m2_impulse"] + 0.25 * frame["cb_impulse"] + 0.25 * frame["usnl_impulse"]
    frame["impulse_state"] = frame["global_liquidity_score"].map(liquidity_score_state)
    frame["direction_13w"] = frame["global_liquidity_score"] - frame["global_liquidity_score"].shift(13)
    frame["direction_13w_state"] = frame["direction_13w"].map(liquidity_direction_state)
    frame["long_cycle_phase"] = [liquidity_long_cycle_phase(date) for date in frame.index]
    frame["long_cycle_value"] = [liquidity_long_cycle_value(date) for date in frame.index]
    frame["cycle_confirmation"] = liquidity_cycle_confirmation(frame)
    frame["final_regime_label"] = frame.apply(liquidity_final_label, axis=1)
    frame["data_status"] = np.where(
        frame[["global_m2_usd_bn", "global_cb_assets_usd_bn", "us_net_liquidity_usd_bn", "global_liquidity_score"]].notna().all(axis=1),
        "CURRENT",
        "PARTIAL_DATA",
    )
    if "last_updated" not in frame.columns or frame["last_updated"].isna().all():
        frame["last_updated"] = frame.get("weekly_last_updated", "n/a")
    return frame.reset_index().rename(columns={"index": "date"})


def build_market_transition_history_for_export(api_key: str | None, source_sink: dict | None = None) -> pd.DataFrame:
    try:
        from market_model import (
            YAHOO_MARKET_TICKERS,
            calculate_confirmations_history,
            calculate_credit_stress_confirmation_history,
            calculate_fast_transition_risk_history,
            calculate_macro_transition_risk_history,
            calculate_overall_transition_status,
            calculate_positioning_risk_history,
            calculate_tail_risk_history,
            classify_global_liquidity_backdrop,
            download_fred_market_data,
            market_model_config,
            weekly_close,
        )
    except Exception:
        return pd.DataFrame()

    yahoo_weekly: dict[str, pd.DataFrame] = {}
    for ticker in YAHOO_MARKET_TICKERS:
        try:
            yahoo_weekly[ticker] = weekly_ohlc(download_completed_ohlcv(ticker, period="max"))
        except Exception:
            yahoo_weekly[ticker] = pd.DataFrame()
    cfg = market_model_config()
    try:
        fred_data = download_fred_series_batch(
            ["WALCL", "RRPONTSYD", "WTREGEN", "DGS2", "DFII10", "BAMLH0A0HYM2"],
            api_key=api_key,
            observation_start="1990-01-01",
        )
    except Exception:
        try:
            fred_data = download_fred_market_data(api_key=api_key)
        except Exception:
            fred_data = pd.DataFrame()
    if source_sink is not None:
        source_sink["yahoo_weekly"] = yahoo_weekly
        source_sink["fred_data"] = fred_data
    global_m2 = global_m2_weekly_series_for_export()
    fast = calculate_fast_transition_risk_history(
        weekly_close(yahoo_weekly.get("^VIX", pd.DataFrame())),
        weekly_close(yahoo_weekly.get("DX-Y.NYB", pd.DataFrame())),
        cfg,
    )
    macro = calculate_macro_transition_risk_history(
        weekly_close(yahoo_weekly.get("DX-Y.NYB", pd.DataFrame())),
        fred_data,
        cfg,
        global_m2=global_m2,
    )
    credit = calculate_credit_stress_confirmation_history(fred_data, cfg)
    confirmations = calculate_confirmations_history(yahoo_weekly, fred_data, cfg)
    parts = [
        fast[["Date", "Fast_Transition_Risk", "Fast_Transition_State", "Fast_Risk_Direction_4W", "Fast_Risk_Direction_4W_State"]] if not fast.empty else pd.DataFrame(columns=["Date"]),
        macro[["Date", "Macro_Transition_Risk", "Macro_Transition_State", "Global_M2_26W", "Global_M2_Bull_Score_26W", "Global_M2_Risk_26W", "Macro_DXY_Risk", "US2Y_Risk"]] if not macro.empty else pd.DataFrame(columns=["Date"]),
        credit[["Date", "Credit_Risk", "Credit_State", "HY_OAS", "HY_OAS_Change_13W", "HY_Level_Percentile", "Credit_Level_State"]] if not credit.empty else pd.DataFrame(columns=["Date"]),
        confirmations[["Date", "Negative_Confirmation_Count"]] if not confirmations.empty else pd.DataFrame(columns=["Date"]),
    ]
    history = parts[0]
    for part in parts[1:]:
        history = pd.merge(history, part, on="Date", how="outer").sort_values("Date")
    if history.empty:
        return pd.DataFrame()
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    structural = prepare_spy_weekly_regime_frame_for_export(yahoo_weekly.get("SPY", pd.DataFrame()), cfg)
    if not structural.empty:
        history = pd.merge(history, structural[["Date", "Market_Regime"]], on="Date", how="left").sort_values("Date")
        history["Market_Regime"] = history["Market_Regime"].ffill()
    try:
        _, monthly, weekly_liquidity = read_global_liquidity()
        liquidity = build_global_liquidity_regime_frame_for_export(monthly, weekly_liquidity)
    except Exception:
        liquidity = pd.DataFrame()
    if not liquidity.empty:
        liquidity_slice = liquidity[["date", "global_liquidity_score", "direction_13w", "direction_13w_state", "long_cycle_phase"]].copy()
        liquidity_slice["Date"] = pd.to_datetime(liquidity_slice["date"], errors="coerce")
        liquidity_slice = liquidity_slice.rename(
            columns={
                "global_liquidity_score": "Global_Liquidity_Score",
                "direction_13w": "Global_Liquidity_Direction_13W",
                "direction_13w_state": "Global_Liquidity_Direction_13W_State",
                "long_cycle_phase": "Long_Liquidity_Cycle",
            }
        ).drop(columns=["date"], errors="ignore")
        history = pd.merge(history, liquidity_slice.dropna(subset=["Date"]), on="Date", how="left").sort_values("Date")
        for col in ["Global_Liquidity_Score", "Global_Liquidity_Direction_13W", "Global_Liquidity_Direction_13W_State", "Long_Liquidity_Cycle"]:
            history[col] = history[col].ffill()
    history["Global_Liquidity_Backdrop"] = history.apply(
        lambda row: classify_global_liquidity_backdrop(
            row.get("Global_Liquidity_Score"),
            row.get("Global_Liquidity_Direction_13W"),
            row.get("Global_Liquidity_Direction_13W_State"),
        ),
        axis=1,
    )
    history["Overall_Transition_Status"] = history.apply(
        lambda row: calculate_overall_transition_status(
            row.get("Fast_Transition_Risk"),
            row.get("Macro_Transition_Risk"),
            row.get("Negative_Confirmation_Count"),
            structural_regime=row.get("Market_Regime", "UNKNOWN"),
            global_liquidity_backdrop=row.get("Global_Liquidity_Backdrop"),
            global_liquidity_score=row.get("Global_Liquidity_Score"),
            global_liquidity_direction_13w=row.get("Global_Liquidity_Direction_13W"),
            global_liquidity_direction_state=row.get("Global_Liquidity_Direction_13W_State"),
            credit_state=row.get("Credit_State", ""),
        ),
        axis=1,
    )
    history["Final_Market_State"] = history["Overall_Transition_Status"]
    try:
        from positioning import read_processed

        positioning_history = calculate_positioning_risk_history(read_processed("aaii"), read_processed("cftc_master"))
    except Exception:
        positioning_history = pd.DataFrame()
    history = calculate_tail_risk_history(history, positioning_history)
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    return history.loc[(history["Date"] >= "2016-01-01") & (history["Date"] <= today)].reset_index(drop=True)


def build_btc_macro_frame_for_export(liquidity: pd.DataFrame, market_history: pd.DataFrame) -> pd.DataFrame:
    if liquidity.empty:
        return pd.DataFrame()
    frame = liquidity.copy()
    if not market_history.empty and "Date" in market_history.columns:
        market_cols = [column for column in ["Date", "Macro_DXY_Risk", "US2Y_Risk", "Credit_Risk"] if column in market_history.columns]
        mh = market_history[market_cols].copy()
        mh["date"] = pd.to_datetime(mh["Date"], errors="coerce")
        mh = mh.dropna(subset=["date"]).drop(columns=["Date"], errors="ignore").sort_values("date")
        frame = pd.merge_asof(frame.sort_values("date"), mh, on="date", direction="backward")
    for column in ["Macro_DXY_Risk", "US2Y_Risk", "Credit_Risk"]:
        if column not in frame.columns:
            frame[column] = np.nan
    m2_bull = pd.to_numeric(frame.get("m2_13w_pctl"), errors="coerce")
    m2_risk = 100.0 - m2_bull
    dxy_risk = pd.to_numeric(frame["Macro_DXY_Risk"], errors="coerce")
    us2y_risk = pd.to_numeric(frame["US2Y_Risk"], errors="coerce")
    credit_risk = pd.to_numeric(frame["Credit_Risk"], errors="coerce")
    frame["BTCGlobalM2Bull13W"] = m2_bull
    frame["BTCGlobalM2Risk13W"] = m2_risk
    frame["BTCDXYBull"] = 100.0 - dxy_risk
    frame["BTCUS2YBull"] = 100.0 - us2y_risk
    frame["BTCStructuralMacro"] = weighted_mean_series([m2_bull, 100.0 - dxy_risk, 100.0 - us2y_risk], [0.40, 0.40, 0.20])
    frame["BTCForwardMacroRisk"] = weighted_mean_series([m2_risk, dxy_risk, us2y_risk, credit_risk], [0.35, 0.30, 0.20, 0.15])
    frame["BTCRegimeState"] = [
        classify_btc_regime_state(structural, forward)
        for structural, forward in zip(frame["BTCStructuralMacro"], frame["BTCForwardMacroRisk"])
    ]
    return frame


def classify_btc_regime_state(structural_macro: float, forward_macro_risk: float) -> str:
    structural = safe_float(structural_macro)
    forward = safe_float(forward_macro_risk)
    if not np.isfinite(structural) or not np.isfinite(forward):
        return "DATA_INCOMPLETE"
    if structural >= 65.0 and forward <= 40.0:
        return "BULLISH_MACRO"
    if structural >= 55.0 and forward <= 60.0:
        return "CONSTRUCTIVE"
    if forward >= 75.0 and structural < 45.0:
        return "HIGH_MACRO_RISK"
    if forward >= 60.0:
        return "MACRO_HEADWIND"
    return "NEUTRAL"


def load_gold_regime_history_for_export(api_key: str | None = None) -> pd.DataFrame:
    from pathlib import Path

    candidates = [
        Path("/app/persistent/gold_regime/gold_regime_history.parquet"),
        Path("/app/persistent/gold_regime/gold_regime_history.csv"),
        Path("persistent/gold_regime/gold_regime_history.parquet"),
        Path("persistent/gold_regime/gold_regime_history.csv"),
    ]
    for path in candidates:
        try:
            if path.exists() and path.suffix.lower() == ".parquet":
                return pd.read_parquet(path)
            if path.exists() and path.suffix.lower() == ".csv":
                return pd.read_csv(path)
        except Exception:
            continue
    try:
        from gold_regime import build_gold_regime_snapshot, gold_regime_config

        snapshot = build_gold_regime_snapshot(gold_alpha=50.0, fred_api_key=api_key, config=gold_regime_config())
        return snapshot.history
    except Exception:
        pass
    return pd.DataFrame()


def liquidity_weekly_from_monthly(monthly: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if monthly.empty or "date" not in monthly.columns:
        return pd.DataFrame()
    use_cols = [column for column in columns if column in monthly.columns]
    if not use_cols:
        return pd.DataFrame()
    source = monthly[["date"] + use_cols].dropna(subset=["date"]).copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source = source.dropna(subset=["date"]).sort_values("date").set_index("date")
    weekly_index = pd.date_range(source.index.min(), source.index.max() + pd.offsets.MonthEnd(1), freq="W-FRI")
    combined_index = source.index.union(weekly_index).sort_values()
    return source.reindex(combined_index).ffill().reindex(weekly_index).ffill()


def prepare_spy_weekly_regime_frame_for_export(spy_weekly: pd.DataFrame, config: dict) -> pd.DataFrame:
    if spy_weekly is None or spy_weekly.empty or "Close" not in spy_weekly.columns:
        return pd.DataFrame()
    cfg = config["structural"]
    weekly = spy_weekly.copy().sort_index()
    close = pd.to_numeric(weekly["Close"], errors="coerce")
    sma40 = close.rolling(int(cfg["spy_sma_weeks"]), min_periods=int(cfg["spy_sma_weeks"])).mean()
    high52w = close.rolling(52, min_periods=26).max()
    vol13w = close.pct_change(fill_method=None).rolling(int(cfg["vol_window_weeks"]), min_periods=int(cfg["vol_window_weeks"])).std() * np.sqrt(52)
    vol_percentile = vol13w.expanding(min_periods=int(cfg["vol_window_weeks"])).apply(lambda values: float((values <= values[-1]).sum() / len(values) * 100.0), raw=True)
    structural_bull = (close > sma40) & ((close / high52w - 1.0) > float(cfg["drawdown_threshold"]))
    high_vol = vol_percentile >= float(cfg["high_vol_percentile"])
    frame = weekly.copy()
    frame["Market_Regime"] = np.select(
        [structural_bull & ~high_vol, structural_bull & high_vol, ~structural_bull & ~high_vol, ~structural_bull & high_vol],
        ["BULL", "BULL_HIGH_VOL", "CORRECTION", "STRESS"],
        default="UNKNOWN",
    )
    frame = frame[frame["Market_Regime"] != "UNKNOWN"].reset_index().rename(columns={"index": "Date"})
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame.dropna(subset=["Date"])


def global_m2_weekly_series_for_export() -> pd.Series:
    try:
        _, monthly, _ = read_global_liquidity()
    except Exception:
        return pd.Series(dtype="float64")
    monthly = prepare_date_frame(monthly)
    if monthly.empty:
        return pd.Series(dtype="float64")
    liquidity = build_global_liquidity_regime_frame_for_export(monthly, pd.DataFrame())
    if liquidity.empty or "global_m2_usd_bn" not in liquidity.columns:
        return pd.Series(dtype="float64")
    return pd.Series(liquidity["global_m2_usd_bn"].values, index=pd.to_datetime(liquidity["date"], errors="coerce")).dropna().sort_index()


def liquidity_trailing_percentile(series: pd.Series, window: int = 156, min_periods: int = 104) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    def rank_last(window_values: np.ndarray) -> float:
        clean = window_values[np.isfinite(window_values)]
        if len(clean) < min_periods or not np.isfinite(window_values[-1]):
            return np.nan
        return float((clean <= window_values[-1]).sum() / len(clean) * 100.0)

    return values.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def liquidity_trend_state(value: Any) -> str:
    if value is None or pd.isna(value):
        return "DATA_INCOMPLETE"
    number = float(value)
    if number >= 0.80:
        return "STRONGLY_EXPANDING"
    if number >= 0.60:
        return "EXPANDING"
    if number >= 0.40:
        return "FLAT_MIXED"
    if number >= 0.20:
        return "CONTRACTING"
    return "STRONGLY_CONTRACTING"


def liquidity_score_state(value: Any) -> str:
    if value is None or pd.isna(value):
        return "DATA_INCOMPLETE"
    number = float(value)
    if number >= 80:
        return "VERY_STRONG"
    if number >= 60:
        return "STRONG"
    if number >= 40:
        return "NEUTRAL"
    if number >= 20:
        return "WEAK"
    return "VERY_WEAK"


def liquidity_direction_state(value: Any) -> str:
    if value is None or pd.isna(value):
        return "DATA_INCOMPLETE"
    number = float(value)
    if number > 10:
        return "ACCELERATING"
    if number >= 5:
        return "IMPROVING"
    if number > -5:
        return "STABLE"
    if number >= -10:
        return "DETERIORATING"
    return "DETERIORATING_FAST"


def liquidity_long_cycle_phase(value: Any) -> str:
    if value is None or pd.isna(value):
        return "DATA_INCOMPLETE"
    phase_pos = liquidity_cycle_months_since_anchor(pd.Timestamp(value)) % 65.0
    if phase_pos < 16.25:
        return "RECOVERY_REACCELERATION"
    if phase_pos < 32.50:
        return "ACCELERATING_EXPANSION"
    if phase_pos < 48.75:
        return "DECELERATING_EXPANSION"
    return "CONTRACTION"


def liquidity_long_cycle_value(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    months = liquidity_cycle_months_since_anchor(pd.Timestamp(value))
    return float(-np.cos((months / 65.0) * 2.0 * np.pi) * 100.0)


def liquidity_cycle_months_since_anchor(value: pd.Timestamp) -> float:
    anchor = pd.Timestamp("2022-10-01")
    date = pd.Timestamp(value)
    return (date.year - anchor.year) * 12 + (date.month - anchor.month) + (date.day - 1) / 30.4375


def liquidity_cycle_confirmation(frame: pd.DataFrame) -> pd.Series:
    roc = pd.to_numeric(frame.get("m2_52w", np.nan), errors="coerce")
    roc_direction = roc.diff(13)
    cycle_direction = pd.Series(frame.get("long_cycle_value", np.nan), index=frame.index).diff(13)
    confirmed = np.sign(roc_direction) == np.sign(cycle_direction)
    return pd.Series(np.where(confirmed, "CYCLE_CONFIRMED", "LIQUIDITY_CYCLE_DIVERGENCE"), index=frame.index).where(
        roc_direction.notna() & cycle_direction.notna(),
        "DATA_INCOMPLETE",
    )


def liquidity_final_label(row: pd.Series) -> str:
    score = row.get("global_liquidity_score")
    direction = row.get("direction_13w")
    trend = str(row.get("trend_state", ""))
    if score is None or direction is None or pd.isna(score) or pd.isna(direction):
        return "DATA_INCOMPLETE"
    score = float(score)
    direction = float(direction)
    if score >= 80 and direction >= 5:
        return "STRONG_EXPANSION"
    if score >= 60 and direction >= -5:
        return "EXPANSION"
    if score >= 40 and direction > 5:
        return "REACCELERATION"
    if score >= 40 and direction < -10:
        return "LIQUIDITY_WARNING"
    if score >= 40:
        return "DECELERATING_EXPANSION" if "EXPANDING" in trend else "NEUTRAL"
    if score < 20:
        return "STRONG_CONTRACTION"
    return "CONTRACTION"


def weighted_mean_series(series_list: list[pd.Series], weights: list[float]) -> pd.Series:
    frame = pd.concat([pd.to_numeric(series, errors="coerce") for series in series_list], axis=1)
    weight_arr = np.asarray(weights, dtype=float)
    values = frame.to_numpy(dtype=float)
    mask = np.isfinite(values)
    weighted = np.where(mask, values * weight_arr, 0.0).sum(axis=1)
    denom = np.where(mask, weight_arr, 0.0).sum(axis=1)
    result = np.divide(weighted, denom, out=np.full(len(frame), np.nan), where=denom > 0)
    return pd.Series(result, index=frame.index)


def safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def weekly_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    d = frame.copy()
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[~d.index.isna()].sort_index()
    return d.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])


def prepare_date_frame(frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if frame is None or frame.empty or date_col not in frame.columns:
        return pd.DataFrame(columns=["date"])
    out = frame.copy()
    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.tz_localize(None)
    return out.dropna(subset=["date"]).sort_values("date")


def add_frame_column(
    dataset: pd.DataFrame,
    out_col: str,
    frame: pd.DataFrame,
    source_col: str,
    metadata: list[SeriesMeta],
    layer: str,
    category: str,
    description: str,
    unit: str,
    original_frequency: str,
    model_usage: str,
    point_in_time: str = "Production timing convention",
) -> None:
    if frame is None or frame.empty or source_col not in frame.columns or "date" not in frame.columns:
        dataset[out_col] = np.nan
        metadata.append(SeriesMeta(out_col, layer, category, description, unit, "Missing optional source", "Production data layer", original_frequency, model_usage, notes="Source unavailable"))
        return
    series = pd.Series(frame[source_col].values, index=pd.to_datetime(frame["date"], errors="coerce")).dropna()
    add_aligned_series(
        dataset,
        out_col,
        series,
        metadata,
        SeriesMeta(out_col, layer, category, description, unit, "Last observation carried forward to weekly calendar", "Production data layer", original_frequency, model_usage, False, point_in_time, True),
    )


def add_aligned_series(
    dataset: pd.DataFrame,
    column: str,
    series: pd.Series,
    metadata: list[SeriesMeta],
    meta: SeriesMeta,
    ffill: bool = True,
) -> None:
    dates = pd.to_datetime(dataset["Date"], errors="coerce")
    if series is None or series.empty:
        dataset[column] = np.nan
        metadata.append(meta)
        return
    source = pd.Series(series.values, index=pd.to_datetime(series.index, errors="coerce")).dropna()
    source = source[~source.index.isna()].sort_index()
    weekly = source.resample("W-FRI").last()
    if ffill:
        aligned = weekly.reindex(dates).ffill()
    else:
        aligned = weekly.reindex(dates)
    dataset[column] = aligned.to_numpy()
    metadata.append(meta)


def add_derived_changes(dataset: pd.DataFrame, metadata: list[SeriesMeta], base_col: str, windows: list[int], change_type: str, model_usage: str) -> None:
    values = pd.to_numeric(dataset.get(base_col), errors="coerce")
    for weeks in windows:
        if change_type == "percent":
            col = f"{base_col}_{weeks}W_Change"
            dataset[col] = values.pct_change(weeks, fill_method=None)
            unit = "percent"
            transformation = f"pct_change({weeks}W)"
        elif change_type == "bps":
            col = f"{base_col}_{weeks}W_Change_bps"
            dataset[col] = values.diff(weeks) * 100.0
            unit = "bps"
            transformation = f"(current - {weeks}W prior) * 100"
        else:
            col = f"{base_col}_{weeks}W_Change"
            dataset[col] = values.diff(weeks)
            unit = "index points"
            transformation = f"current - {weeks}W prior"
        metadata.append(SeriesMeta(col, "DERIVED", "Forward Liquidity Research", f"{weeks}W change in {base_col}", unit, transformation, "Internal", "weekly", model_usage))


def add_business_cycle_derived(dataset: pd.DataFrame, metadata: list[SeriesMeta]) -> None:
    derived = {
        "ISM_Manufacturing_3M_Change": ("ISM_Manufacturing", 13),
        "CFNAI_3M_Change": ("CFNAI", 13),
        "InitialClaims_13W_Change": ("InitialClaims", 13),
        "ContinuingClaims_13W_Change": ("ContinuingClaims", 13),
        "Unemployment_3M_Change": ("UnemploymentRate", 13),
        "Payrolls_3M_Change": ("Payrolls", 13),
        "Payrolls_6M_Change": ("Payrolls", 26),
        "IndustrialProduction_3M_Change": ("IndustrialProduction", 13),
        "RetailSales_3M_Change": ("RetailSales", 13),
        "RealPersonalIncome_3M_Change": ("RealPersonalIncomeExTransfers", 13),
        "RealPCE_3M_Change": ("RealPCE", 13),
    }
    for out_col, (base_col, weeks) in derived.items():
        if base_col in dataset.columns:
            dataset[out_col] = pd.to_numeric(dataset[base_col], errors="coerce").diff(weeks)
            metadata.append(SeriesMeta(out_col, "DERIVED", "Business Cycle", out_col, "index points", f"current - {weeks}W prior", "Internal", "weekly", "Business Cycle"))
    if "InitialClaims" in dataset.columns:
        dataset["InitialClaims_4W_Avg"] = pd.to_numeric(dataset["InitialClaims"], errors="coerce").rolling(4, min_periods=1).mean()
        metadata.append(SeriesMeta("InitialClaims_4W_Avg", "DERIVED", "Business Cycle", "Initial claims 4-week average", "claims", "rolling 4W mean using observations <= t", "Internal", "weekly", "Business Cycle"))
    if "ContinuingClaims" in dataset.columns:
        dataset["ContinuingClaims_4W_Avg"] = pd.to_numeric(dataset["ContinuingClaims"], errors="coerce").rolling(4, min_periods=1).mean()
        metadata.append(SeriesMeta("ContinuingClaims_4W_Avg", "DERIVED", "Business Cycle", "Continuing claims 4-week average", "claims", "rolling 4W mean using observations <= t", "Internal", "weekly", "Business Cycle"))
    for out_col, base_col in {
        "IndustrialProduction_YoY": "IndustrialProduction",
        "RetailSales_YoY": "RetailSales",
        "RealPersonalIncome_YoY": "RealPersonalIncomeExTransfers",
        "RealPCE_YoY": "RealPCE",
    }.items():
        if base_col in dataset.columns:
            dataset[out_col] = pd.to_numeric(dataset[base_col], errors="coerce").pct_change(52, fill_method=None)
            metadata.append(SeriesMeta(out_col, "DERIVED", "Business Cycle", out_col, "percent", "pct_change(52W)", "Internal", "weekly", "Business Cycle"))


def add_credit_spread_changes(dataset: pd.DataFrame, metadata: list[SeriesMeta]) -> None:
    for base_col, windows in {"HY_OAS": [4, 13, 26], "IG_OAS": [4, 13]}.items():
        if base_col not in dataset.columns:
            continue
        values = pd.to_numeric(dataset[base_col], errors="coerce")
        for weeks in windows:
            col = f"{base_col}_{weeks}W_Change"
            if col in dataset.columns:
                continue
            dataset[col] = values.diff(weeks)
            metadata.append(SeriesMeta(col, "DERIVED", "Credit", f"{base_col} {weeks}W change", "percentage points", f"current - {weeks}W prior", "Internal", "weekly", "Credit / Funding Liquidity Stress"))


def macro_column_name(instrument: str) -> str:
    mapping = {
        "U.S. Dollar Index": "DXY",
        "EUR/USD": "EURUSD",
        "USD/JPY": "USDJPY",
        "USD/CNY": "USDCNY",
        "U.S. 5-Year Breakeven Inflation Rate": "T5YIE",
        "U.S. 10-Year Breakeven Inflation Rate": "T10YIE",
        "U.S. 1-Year Inflation Expectations": "1Y_Inflation_Expectation",
        "U.S. 10-Year Real Yield": "US10Y_Real_Yield",
        "Federal Funds Effective Rate": "FedFunds",
        "U.S. 2-Year Treasury Yield": "US2Y",
        "U.S. 10-Year Treasury Yield": "US10Y",
        "U.S. 2Y-10Y Treasury Curve": "US10Y_2Y_Spread",
        "U.S. 3M-10Y Treasury Curve": "US10Y_3M_Spread",
        "Germany 10-Year Government Bond Yield": "DE10Y",
        "France 10-Year Government Bond Yield": "FR10Y",
        "China 10-Year Government Bond Yield": "CN10Y",
        "Japan 10-Year Government Bond Yield": "JP10Y",
        "U.S. ISM Manufacturing PMI": "ISM_Manufacturing",
        "U.S. ISM Services PMI": "ISM_Services",
        "U.S. Initial Jobless Claims": "InitialClaims",
        "Chicago Fed National Activity Index": "CFNAI",
        "WTI Crude Oil": "WTI",
        "Copper": "Copper",
        "CBOE Volatility Index": "VIX",
        "ICE BofA MOVE Index": "MOVE",
        "U.S. High Yield Option-Adjusted Spread": "HY_OAS",
        "U.S. Investment Grade Option-Adjusted Spread": "IG_OAS",
        "Chicago Fed Adjusted National Financial Conditions Index": "ANFCI",
    }
    return mapping.get(instrument, clean_token(instrument))


def cftc_asset_col(asset: str) -> str:
    mapping = {
        "GOLD": "Gold",
        "SILVER": "Silver",
        "WTI": "WTI",
        "PLATINUM": "Platinum",
        "PALLADIUM": "Palladium",
        "S&P 500": "SP500",
        "NASDAQ-100": "Nasdaq100",
        "RUSSELL 2000": "Russell2000",
        "VIX": "VIX",
        "UST 2Y": "UST2Y",
        "UST 5Y": "UST5Y",
        "UST 10Y": "UST10Y",
        "UST BOND": "USTBond",
        "BTC": "BTC",
        "ETH": "ETH",
        "SOL": "SOL",
        "HYPERLIQUID": "Hyperliquid",
    }
    return mapping.get(asset.upper(), clean_token(asset))


def clean_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip())
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")
