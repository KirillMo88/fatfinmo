from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from fred_client import download_fred_series_live
from historical_archive import save_fred_series, save_model_market_history, save_raw_table, save_validated_file


FRED_ARCHIVE_START = "1900-01-01"


def fred_series_registry() -> tuple[str, ...]:
    """Return the union of official FRED inputs consumed by models and charts."""
    from business_cycle import DOWNLOAD_FRED_SERIES
    from commodity_cycle.data import FRED_SERIES as COMMODITY_FRED_SERIES
    from funding_conditions import FRED_IDS as FUNDING_FRED_IDS
    from global_liquidity import FRED_CHINA_M2_LEGACY_SERIES, FRED_FORECAST_SERIES, FRED_GLOBAL_SERIES
    from global_macro_tab import GLOBAL_MACRO_FRED_SERIES
    from gold_regime.gromen import FOREIGN_DEBT_SERIES, US_GOLD_COMPONENTS
    from gold_regime.service import FRED_GOLD_REGIME_SERIES
    from macro_research_export import EXPORT_FRED_SERIES
    from market_cycle import MARKET_CYCLE_FRED_SERIES
    from market_model import FRED_MARKET_SERIES_IDS
    from rates_financial_conditions import FRED_IDS as RATES_FRED_IDS
    from treasury_fiscal_regime import SOURCE_SPECS as TREASURY_FRED_SERIES

    series_ids = {
        *DOWNLOAD_FRED_SERIES.keys(),
        *COMMODITY_FRED_SERIES.values(),
        *FUNDING_FRED_IDS,
        *FRED_FORECAST_SERIES,
        *FRED_GLOBAL_SERIES,
        FRED_CHINA_M2_LEGACY_SERIES,
        *GLOBAL_MACRO_FRED_SERIES,
        *FOREIGN_DEBT_SERIES,
        *US_GOLD_COMPONENTS,
        *FRED_GOLD_REGIME_SERIES,
        *EXPORT_FRED_SERIES.keys(),
        *MARKET_CYCLE_FRED_SERIES,
        *FRED_MARKET_SERIES_IDS,
        *RATES_FRED_IDS,
        *TREASURY_FRED_SERIES.keys(),
        "SOFR",
        "EFFR",
        "THREEFYTP10",
    }
    return tuple(sorted({str(series_id).strip().upper() for series_id in series_ids if str(series_id).strip()}))


def refresh_fred_archive(api_key: str | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    service_unavailable = False
    for series_id in fred_series_registry():
        if service_unavailable:
            results.append({"Provider": "FRED", "Dataset": series_id, "Status": "SKIPPED_SERVICE_UNAVAILABLE"})
            continue
        try:
            frame = download_fred_series_live(
                series_id,
                api_key=api_key,
                observation_start=FRED_ARCHIVE_START,
                timeout=45.0,
            )
            result = save_fred_series(series_id, frame)
            results.append(
                {
                    "Provider": result.provider,
                    "Dataset": result.dataset,
                    "Status": result.status,
                    "Rows": result.rows,
                    "LastObservation": result.last_observation,
                }
            )
        except Exception as exc:
            message = str(exc)
            results.append({"Provider": "FRED", "Dataset": series_id, "Status": "FAILED", "Error": message})
            lowered = message.lower()
            service_unavailable = (
                "non-json response" in lowered
                or "connecterror" in lowered
                or "connection refused" in lowered
                or "name or service not known" in lowered
                or "timed out" in lowered
            )
    return results


def model_market_ticker_registry() -> tuple[str, ...]:
    """Only model/chart dependencies; the broad Table universe is intentionally excluded."""
    from business_cycle import ASSET_TICKERS
    from commodity_cycle.data import PRICE_TICKERS as COMMODITY_PRICE_TICKERS
    from global_macro_tab import GLOBAL_MACRO_MARKET_TICKERS
    from gold_regime.service import YAHOO_GOLD_REGIME_TICKERS
    from liquidity_forecast import PRICE_TICKERS as LIQUIDITY_PRICE_TICKERS
    from market_cycle import MARKET_CYCLE_TICKERS
    from market_model import YAHOO_MARKET_TICKERS
    from rates_financial_conditions import MARKET_TICKERS as RATES_MARKET_TICKERS
    from qqq_current_risk_data import QQQ_CURRENT_RISK_TICKERS

    tickers = {
        *ASSET_TICKERS.values(),
        *COMMODITY_PRICE_TICKERS.values(),
        *GLOBAL_MACRO_MARKET_TICKERS.values(),
        *YAHOO_GOLD_REGIME_TICKERS,
        *LIQUIDITY_PRICE_TICKERS.values(),
        *MARKET_CYCLE_TICKERS,
        *YAHOO_MARKET_TICKERS,
        *RATES_MARKET_TICKERS.values(),
        *QQQ_CURRENT_RISK_TICKERS,
    }
    return tuple(sorted({str(ticker).strip() for ticker in tickers if str(ticker).strip()}))


def refresh_model_market_archive() -> list[dict[str, Any]]:
    from finance_core import download_completed_ohlcv

    results: list[dict[str, Any]] = []
    for ticker in model_market_ticker_registry():
        try:
            frame = download_completed_ohlcv(
                ticker,
                period="max",
                allow_archive_fallback=False,
            )
            result = save_model_market_history(ticker, frame)
            results.append(
                {
                    "Provider": result.provider,
                    "Dataset": result.dataset,
                    "Status": result.status,
                    "Rows": result.rows,
                    "LastObservation": result.last_observation,
                }
            )
        except Exception as exc:
            results.append({"Provider": "MODEL_MARKET", "Dataset": ticker, "Status": "FAILED", "Error": str(exc)})

    for ticker, archive_key in (("SPY", "SPY_RAW"), ("QQQ", "QQQ_RAW"), ("IWM", "IWM_RAW")):
        try:
            frame = download_completed_ohlcv(
                ticker,
                period="max",
                auto_adjust=False,
                allow_archive_fallback=False,
            )
            result = save_model_market_history(archive_key, frame)
            results.append(
                {
                    "Provider": result.provider,
                    "Dataset": result.dataset,
                    "Status": result.status,
                    "Rows": result.rows,
                    "LastObservation": result.last_observation,
                }
            )
        except Exception as exc:
            results.append({"Provider": "MODEL_MARKET", "Dataset": archive_key, "Status": "FAILED", "Error": str(exc)})
    return results


def refresh_tradingview_archive() -> list[dict[str, Any]]:
    """Archive TradingView-only histories required by models."""
    try:
        from tradingview_mcp import get_ohlcv_data

        frame = get_ohlcv_data("INDEX:NDFI", interval="1D", count=5000, force=True)
        candidate = frame.copy()
        candidate["Series_ID"] = "NDFI"
        result = save_raw_table(
            "TRADINGVIEW",
            "NDFI",
            candidate,
            date_column="date",
            value_column="close",
            group_column="Series_ID",
            min_rows=252,
            source="TradingView INDEX:NDFI",
        )
        return [
            {
                "Provider": result.provider,
                "Dataset": result.dataset,
                "Status": result.status,
                "Rows": result.rows,
                "LastObservation": result.last_observation,
            }
        ]
    except Exception as exc:
        return [{"Provider": "TRADINGVIEW", "Dataset": "NDFI", "Status": "FAILED", "Error": str(exc)}]


def refresh_global_liquidity_archive(api_key: str | None = None) -> list[dict[str, Any]]:
    from global_liquidity import update_global_liquidity

    raw, _, _ = update_global_liquidity(api_key=api_key, force=True)
    valid = raw.loc[pd.to_numeric(raw.get("raw_value"), errors="coerce").notna()].copy()
    result = save_raw_table(
        "GLOBAL_LIQUIDITY",
        "RAW_OFFICIAL_SERIES",
        valid,
        date_column="observation_date",
        value_column="raw_value",
        group_column="series_id",
        min_rows=24,
        source="FRED + ECB + BoJ + PBoC and validated fallbacks",
    )
    return [
        {
            "Provider": result.provider,
            "Dataset": result.dataset,
            "Status": result.status,
            "Rows": result.rows,
            "LastObservation": result.last_observation,
        }
    ]


def refresh_world_economy_archive() -> list[dict[str, Any]]:
    from world_economy import WORLD_ECONOMY_STORAGE_DIR, load_world_economy_data
    from world_gdp import WorldGDPService
    from world_trade import GlobalTradeService

    results: list[dict[str, Any]] = []
    load_world_economy_data(force_refresh=True)
    cache_dir = Path(WORLD_ECONOMY_STORAGE_DIR)
    official_files = {
        "BIS_TOTAL_CREDIT": cache_dir / "bis_ws_tc_csv_flat.zip",
        "BIS_DSR": cache_dir / "bis_ws_dsr_csv_flat.zip",
        "OECD_LONG_TERM_RATES": cache_dir / "oecd_long_term_interest_rates.csv",
        "IMF_WEO_COUNTRIES": cache_dir / "imf_weo_countries.xlsx",
        "WORLD_BANK_RESERVES": cache_dir / "world_bank_gold_fx_reserves.csv",
        "SOVEREIGN_CDS": cache_dir / "world_government_bonds_sovereign_cds.html",
        "SOURCE_METADATA": cache_dir / "source_metadata.json",
    }
    for dataset, path in official_files.items():
        try:
            result = save_validated_file("WORLD_ECONOMY", dataset, path)
            results.append({"Provider": result.provider, "Dataset": result.dataset, "Status": result.status})
        except Exception as exc:
            results.append({"Provider": "WORLD_ECONOMY", "Dataset": dataset, "Status": "FAILED", "Error": str(exc)})

    gdp = WorldGDPService().load(force_refresh=True)
    if not gdp.get("data", pd.DataFrame()).empty:
        frame = gdp["data"].copy()
        date_column = "year" if "year" in frame else "date"
        value_candidates = [name for name in ("nominal_gdp_usd", "gdp_ppp_intl_usd", "gdp_per_capita_usd") if name in frame]
        group_column = "iso3" if "iso3" in frame else "country_id"
        if value_candidates and date_column in frame and group_column in frame:
            normalized = frame[[group_column, date_column, value_candidates[0]]].copy()
            normalized[date_column] = pd.to_datetime(normalized[date_column].astype(str) + "-01-01", errors="coerce")
            result = save_raw_table(
                "WORLD_BANK",
                "WORLD_GDP",
                normalized,
                date_column=date_column,
                value_column=value_candidates[0],
                group_column=group_column,
                min_rows=100,
                source="World Bank WDI",
            )
            results.append({"Provider": result.provider, "Dataset": result.dataset, "Status": result.status, "Rows": result.rows})

    trade = GlobalTradeService().load(force_refresh=True)
    if not trade.get("data", pd.DataFrame()).empty:
        frame = trade["data"].copy()
        value_candidates = [name for name in ("real_exports_usd", "nominal_exports_usd", "nominal_gdp_usd") if name in frame]
        group_column = "iso3" if "iso3" in frame else "country_code"
        if value_candidates and "year" in frame and group_column in frame:
            normalized = frame[[group_column, "year", value_candidates[0]]].copy()
            normalized["year"] = pd.to_datetime(normalized["year"].astype(str) + "-01-01", errors="coerce")
            result = save_raw_table(
                "WORLD_BANK",
                "GLOBAL_TRADE",
                normalized,
                date_column="year",
                value_column=value_candidates[0],
                group_column=group_column,
                min_rows=100,
                source="World Bank WDI",
            )
            results.append({"Provider": result.provider, "Dataset": result.dataset, "Status": result.status, "Rows": result.rows})
    return results


def archive_model_input_caches() -> list[dict[str, Any]]:
    """Keep two copies of release-dated inputs that cannot be rebuilt from revised FRED alone."""
    from business_cycle import BUSINESS_CYCLE_FRED_CACHE
    from funding_conditions import STORAGE_DIR as FUNDING_DIR
    from rates_financial_conditions import CREDIT_ARCHIVE, FRED_CACHE, FRED_INITIAL_CACHE
    from treasury_fiscal_regime import STORAGE_DIR as TREASURY_DIR

    candidates: dict[str, Path] = {
        "BUSINESS_CYCLE_FRED_CACHE": Path(BUSINESS_CYCLE_FRED_CACHE),
        "RATES_FRED_CACHE": Path(FRED_CACHE),
        "RATES_FRED_INITIAL_CACHE": Path(FRED_INITIAL_CACHE),
        "RATES_CREDIT_WEEKLY_ARCHIVE": Path(CREDIT_ARCHIVE),
    }
    candidates.update({f"FUNDING_{path.stem.upper()}": path for path in Path(FUNDING_DIR).glob("fred_*.parquet")})
    candidates.update({f"TREASURY_{path.stem.upper()}": path for path in Path(TREASURY_DIR).glob("fred_*.parquet")})
    results: list[dict[str, Any]] = []
    for dataset, path in candidates.items():
        try:
            frame = pd.read_parquet(path)
            if frame.empty:
                raise ValueError("cache is empty")
            result = save_validated_file("MODEL_INPUTS", dataset, path)
            results.append({"Provider": result.provider, "Dataset": result.dataset, "Status": result.status})
        except Exception as exc:
            results.append({"Provider": "MODEL_INPUTS", "Dataset": dataset, "Status": "SKIPPED", "Error": str(exc)})
    return results


def run_weekly_historical_archive(api_key: str | None = None) -> pd.DataFrame:
    """Refresh independent provider archives; one failed provider cannot erase another."""
    results = refresh_fred_archive(api_key=api_key or os.getenv("FRED_API_KEY"))
    results.extend(refresh_model_market_archive())
    results.extend(refresh_tradingview_archive())
    for refresh in (refresh_global_liquidity_archive, refresh_world_economy_archive):
        try:
            if refresh is refresh_global_liquidity_archive:
                results.extend(refresh(api_key=api_key or os.getenv("FRED_API_KEY")))
            else:
                results.extend(refresh())
        except Exception as exc:
            results.append(
                {
                    "Provider": refresh.__name__.removeprefix("refresh_").removesuffix("_archive").upper(),
                    "Dataset": "BUNDLE",
                    "Status": "FAILED",
                    "Error": str(exc),
                }
            )
    results.extend(archive_model_input_caches())
    return pd.DataFrame(results)
