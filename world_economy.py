from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import zipfile

import httpx
import numpy as np
import pandas as pd

from historical_archive import restore_validated_file


WORLD_ECONOMY_TTL_SECONDS = 7 * 24 * 60 * 60
WORLD_ECONOMY_STORAGE_DIR = Path(
    os.environ.get(
        "WORLD_ECONOMY_STORAGE_DIR",
        str(Path(__file__).with_name("persistent") / "world_economy"),
    )
)
BIS_BULK_URL = "https://data.bis.org/static/bulk/WS_TC_csv_flat.zip"
BIS_DSR_BULK_URL = "https://data.bis.org/static/bulk/WS_DSR_csv_flat.zip"
BIS_TOPIC_URL = "https://data.bis.org/topics/TOTAL_CREDIT"
BIS_DSR_TOPIC_URL = "https://data.bis.org/topics/DSR"
OECD_LONG_TERM_RATES_URL = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/.M.IRLT.PA.....?dimensionAtObservation=AllDimensions&format=csvfile"
OECD_LONG_TERM_RATES_EXPLORER_URL = "https://data-explorer.oecd.org/vis?df%5Bds%5D=dsDisseminateFinalDMZ&df%5Bid%5D=DSD_STES%40DF_FINMARK&df%5Bag%5D=OECD.SDD.STES&dq=.M..PA.....&lom=LASTNPERIODS&lo=5&to%5BTIME_PERIOD%5D=false&vw=tb"
WORLD_GOVERNMENT_BONDS_CDS_URL = "https://www.worldgovernmentbonds.com/sovereign-cds/"
WORLD_GOVERNMENT_BONDS_CDS_API_URL = "https://www.worldgovernmentbonds.com/wp-json/cds/v1/main"
WORLD_BANK_RESERVES_URL = "https://api.worldbank.org/v2/country/all/indicator/FI.RES.TOTL.CD"
WORLD_BANK_RESERVES_SOURCE_URL = "https://data.worldbank.org/indicator/FI.RES.TOTL.CD"
# User-supplied annual Russian long-term interest rates. Screenshot dates are
# interpreted as MM/DD/YYYY; only the month is needed by the rate table.
RUSSIAN_LONG_TERM_RATES = (
    (2003, 1, 9.72), (2004, 1, 9.40), (2005, 3, 6.83), (2006, 1, 6.47),
    (2007, 1, 6.27), (2008, 1, 9.49), (2009, 1, 8.00), (2010, 1, 7.44),
    (2011, 3, 8.50), (2012, 1, 6.85), (2013, 1, 7.71), (2014, 1, 14.09),
    (2015, 1, 9.74), (2016, 1, 8.38), (2017, 2, 7.59), (2018, 1, 8.71),
    (2019, 1, 6.23), (2020, 1, 5.91), (2021, 1, 8.46), (2022, 1, 10.34),
    (2023, 2, 12.30), (2024, 1, 15.18), (2025, 1, 14.47), (2026, 5, 16.88),
)
IMF_WEO_URL = "https://data.imf.org/Datasets/WEO"
IMF_WEO_API_URL = "https://data.imf.org/Datasets/WEO"
IMF_INDICATORS = ("NGDPD", "GGXCNL_NGDP", "GGX_NGDP", "BCA_NGDPD", "NGDP_RPCH", "GGXWDG_NGDP")
BIS_TOTAL_CREDIT_SERIES_TEMPLATE = "Q.{bis_iso2}.C.A.M.770.A"
DEFAULT_REFERENCE_YEAR = int(os.environ.get("WORLD_ECONOMY_REFERENCE_YEAR", "2026"))
BIS_AGGREGATE_CODES = {"4T", "5A", "5R", "G2", "W"}
SPECIAL_ISO2_TO_ISO3 = {
    "EL": "GRC",
    "GB": "GBR",
    "HK": "HKG",
    "KR": "KOR",
    "RU": "RUS",
    "TR": "TUR",
    "TW": "TWN",
    "UK": "GBR",
    "US": "USA",
    "XK": "XKX",
}
WEO_VINTAGE_MONTHS = {
    "Jan": "January", "Apr": "April", "Jul": "July", "Oct": "October",
}

TABLE_COLUMNS = [
    "gdp_rank", "country_name", "iso3", "reference_year", "imf_vintage", "nominal_gdp_usd_bn", "gdp_cagr_5y_pct",
    "gdp_cagr_10y_pct", "total_debt_gdp_pct", "bis_observation_quarter",
    "government_debt_gdp_pct", "government_debt_coverage_pct",
    "debt_delta_5y_pp", "debt_delta_10y_pp", "fiscal_balance_pct", "government_expenditure_pct",
    "fiscal_delta_5y_pp", "fiscal_delta_10y_pp", "fiscal_balance_usd_bn", "current_account_pct",
    "current_account_delta_5y_pp", "current_account_delta_10y_pp", "dsr_pct", "dsr_observation_quarter",
    "dsr_delta_5y_pp", "dsr_delta_10y_pp",
    "long_term_interest_rate_pct", "long_term_interest_rate_observation_month",
    "long_term_interest_rate_delta_5y_pp", "long_term_interest_rate_delta_10y_pp",
    "cds_5y_bps", "cds_var_6m_pct", "cds_pd_pct",
]


def load_world_economy_data(
    *,
    reference_year: int | None = None,
    force_refresh: bool = False,
    storage_dir: Path | None = None,
) -> dict[str, Any]:
    """Load validated macro and market data, retaining the last good raw files on refresh errors."""
    reference_year = int(reference_year if reference_year is not None else DEFAULT_REFERENCE_YEAR)
    cache_dir = Path(storage_dir or WORLD_ECONOMY_STORAGE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    bis_path = cache_dir / "bis_ws_tc_csv_flat.zip"
    dsr_path = cache_dir / "bis_ws_dsr_csv_flat.zip"
    oecd_rates_path = cache_dir / "oecd_long_term_interest_rates.csv"
    cds_path = cache_dir / "world_government_bonds_sovereign_cds.html"
    weo_path = cache_dir / "imf_weo_countries.xlsx"
    reserves_path = cache_dir / "world_bank_gold_fx_reserves.csv"
    metadata_path = cache_dir / "source_metadata.json"
    metadata = _read_json(metadata_path)
    warning: str | None = None

    required_cache = (bis_path, dsr_path, weo_path, oecd_rates_path)
    if not all(path.exists() for path in required_cache):
        _restore_world_economy_archive(cache_dir)

    should_refresh = force_refresh or _cache_is_stale(
        (bis_path, dsr_path, weo_path, oecd_rates_path, cds_path, reserves_path)
    )
    if should_refresh:
        try:
            with httpx.Client(timeout=httpx.Timeout(90.0, connect=20.0), follow_redirects=True) as client:
                bis_response = client.get(BIS_BULK_URL)
                bis_response.raise_for_status()
                bis_bytes = bis_response.content
                # Validate before replacing the current raw snapshot.
                _parse_bis_bulk(bis_bytes)

                dsr_response = client.get(BIS_DSR_BULK_URL)
                dsr_response.raise_for_status()
                dsr_bytes = dsr_response.content
                _parse_bis_dsr(dsr_bytes)

                workbook_url = discover_imf_weo_workbook_url(client)
                weo_response = client.get(workbook_url)
                weo_response.raise_for_status()
                weo_bytes = weo_response.content
                _validate_weo_workbook(weo_bytes)
                weo_vintage = _weo_vintage_from_url(workbook_url)

                oecd_response = client.get(OECD_LONG_TERM_RATES_URL)
                oecd_response.raise_for_status()
                oecd_rates = _parse_oecd_long_term_rates(oecd_response.content)
                oecd_rates = _with_russian_long_term_rate_override(oecd_rates)
                oecd_rates_bytes = oecd_rates.to_csv(index=False).encode("utf-8")

                cds_response = client.post(
                    WORLD_GOVERNMENT_BONDS_CDS_API_URL,
                    json={},
                    headers={
                        "Referer": WORLD_GOVERNMENT_BONDS_CDS_URL,
                        "Origin": "https://www.worldgovernmentbonds.com",
                        "X-Requested-With": "XMLHttpRequest",
                        "Accept": "application/json, text/javascript, */*; q=0.01",
                    },
                )
                cds_response.raise_for_status()
                cds_payload = cds_response.json()
                if not cds_payload.get("success") or not isinstance(cds_payload.get("table"), str):
                    raise ValueError("World Government Bonds CDS endpoint returned an invalid table response.")
                cds_html = cds_payload["table"].encode("utf-8")
                _parse_world_government_bonds_cds(cds_html)

                reserves: pd.DataFrame | None = None
                reserves_error: str | None = None
                try:
                    reserves = _fetch_world_bank_reserves(client, reference_year)
                except Exception as exc:
                    reserves_error = str(exc)

            _atomic_write_bytes(bis_path, bis_bytes)
            _atomic_write_bytes(dsr_path, dsr_bytes)
            _atomic_write_bytes(weo_path, weo_bytes)
            _atomic_write_bytes(oecd_rates_path, oecd_rates_bytes)
            _atomic_write_bytes(cds_path, cds_html)
            if reserves is not None:
                _atomic_write_bytes(reserves_path, reserves.to_csv(index=False).encode("utf-8"))
            vintage_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", weo_vintage).strip("_") or "unknown"
            weo_archive_path = cache_dir / f"imf_weo_{vintage_slug}.xlsx"
            if not weo_archive_path.exists():
                _atomic_write_bytes(weo_archive_path, weo_bytes)
            metadata = {
                "bis_source": "BIS Credit to the non-financial sector, F1.1: total credit, % GDP",
                "bis_series_template": BIS_TOTAL_CREDIT_SERIES_TEMPLATE,
                "bis_url": BIS_BULK_URL,
                "bis_dsr_source": "BIS Debt service ratios, private non-financial sector",
                "bis_dsr_topic_url": BIS_DSR_TOPIC_URL,
                "bis_dsr_url": BIS_DSR_BULK_URL,
                "bis_dsr_retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "bis_topic_url": BIS_TOPIC_URL,
                "bis_retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "weo_source": "IMF World Economic Outlook, Countries sheet",
                "weo_url": workbook_url,
                "weo_vintage": weo_vintage,
                "weo_archive_file": weo_archive_path.name,
                "weo_indicators": list(IMF_INDICATORS),
                "weo_retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "oecd_long_term_rates_source": "OECD Financial market — monthly long-term interest rates, percent per annum",
                "oecd_long_term_rates_url": OECD_LONG_TERM_RATES_EXPLORER_URL,
                "oecd_long_term_rates_api_url": OECD_LONG_TERM_RATES_URL,
                "oecd_long_term_rates_retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "russian_long_term_rates_source": "User-provided annual values; override OECD observations for Russia",
                "sovereign_cds_source": "World Government Bonds, Sovereign 5Y CDS; Var 6m and implied PD (40% recovery assumption)",
                "sovereign_cds_url": WORLD_GOVERNMENT_BONDS_CDS_URL,
                "sovereign_cds_retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "world_bank_reserves_source": "World Bank WDI, FI.RES.TOTL.CD — total reserves including gold (current US$)",
                "world_bank_reserves_url": WORLD_BANK_RESERVES_SOURCE_URL,
                "world_bank_reserves_api_url": WORLD_BANK_RESERVES_URL,
                "world_bank_reserves_retrieved_at_utc": (
                    datetime.now(timezone.utc).isoformat() if reserves is not None
                    else metadata.get("world_bank_reserves_retrieved_at_utc")
                ),
            }
            if reserves_error:
                metadata["world_bank_reserves_error"] = reserves_error
            _atomic_write_text(metadata_path, json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
            if reserves_error:
                warning = f"World Bank reserves refresh failed; using cached values if available: {reserves_error}"
        except Exception as exc:
            if not _world_economy_cache_is_valid(bis_path, dsr_path, weo_path, oecd_rates_path):
                _restore_world_economy_archive(cache_dir)
            if not _world_economy_cache_is_valid(bis_path, dsr_path, weo_path, oecd_rates_path):
                raise RuntimeError(f"Could not refresh World Economy sources and no validated cache is available: {exc}") from exc
            warning = f"Refresh failed; showing last validated BIS/IMF snapshot: {exc}"

    if not (bis_path.exists() and dsr_path.exists() and weo_path.exists() and oecd_rates_path.exists()):
        raise RuntimeError("World Economy source cache is incomplete. Use Refresh World Economy to download official data.")

    bis = _parse_bis_bulk(bis_path.read_bytes())
    dsr = _parse_bis_dsr(dsr_path.read_bytes())
    imf = _parse_weo_workbook(weo_path.read_bytes())
    oecd_rates = _with_russian_long_term_rate_override(
        _parse_oecd_long_term_rates(oecd_rates_path.read_bytes())
    )
    cds = _parse_world_government_bonds_cds(cds_path.read_bytes()) if cds_path.exists() else pd.DataFrame(
        columns=["iso3", "country_name", "cds_5y_bps", "cds_var_6m_pct", "cds_pd_pct"]
    )
    reserves = parse_world_bank_reserves_frame(pd.read_csv(reserves_path)) if reserves_path.exists() else pd.DataFrame(
        columns=["iso3", "year", "gold_fx_reserves_usd_bn"]
    )
    table, history, diagnostics, effective_year, year_warning = build_world_economy_frames(
        bis,
        imf,
        dsr=dsr,
        long_term_rates=oecd_rates,
        cds=cds,
        reserves=reserves,
        reference_year=reference_year,
        source_metadata=metadata,
    )
    warnings = [message for message in (warning, year_warning) if message]
    diagnostics["warning"] = " ".join(warnings) if warnings else None
    diagnostics["snapshot_loaded_at_utc"] = datetime.now(timezone.utc).isoformat()
    return {
        "table": table,
        "history": history,
        "diagnostics": diagnostics,
        "metadata": metadata,
        "reference_year": effective_year,
        "requested_reference_year": int(reference_year),
    }


def _restore_world_economy_archive(cache_dir: Path) -> bool:
    restored = False
    mapping = {
        "BIS_TOTAL_CREDIT": cache_dir / "bis_ws_tc_csv_flat.zip",
        "BIS_DSR": cache_dir / "bis_ws_dsr_csv_flat.zip",
        "OECD_LONG_TERM_RATES": cache_dir / "oecd_long_term_interest_rates.csv",
        "IMF_WEO_COUNTRIES": cache_dir / "imf_weo_countries.xlsx",
        "WORLD_BANK_RESERVES": cache_dir / "world_bank_gold_fx_reserves.csv",
        "SOVEREIGN_CDS": cache_dir / "world_government_bonds_sovereign_cds.html",
        "SOURCE_METADATA": cache_dir / "source_metadata.json",
    }
    for dataset, destination in mapping.items():
        if restore_validated_file("WORLD_ECONOMY", dataset, destination):
            restored = True
    return restored


def _world_economy_cache_is_valid(
    bis_path: Path,
    dsr_path: Path,
    weo_path: Path,
    oecd_rates_path: Path,
) -> bool:
    try:
        _parse_bis_bulk(bis_path.read_bytes())
        _parse_bis_dsr(dsr_path.read_bytes())
        _validate_weo_workbook(weo_path.read_bytes())
        _parse_oecd_long_term_rates(oecd_rates_path.read_bytes())
        return True
    except Exception:
        return False


def discover_imf_weo_workbook_url(client: httpx.Client | None = None) -> str:
    own_client = client is None
    selected_client = client or httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        response = selected_client.get(IMF_WEO_URL)
        response.raise_for_status()
        matches = re.findall(
            r"(/-/media/iData/External-Storage/Documents/[^\"'<> ]+/en/WEO[^\"'<> ]+all\.xlsx)",
            response.text,
            flags=re.IGNORECASE,
        )
        if not matches:
            raise RuntimeError("IMF WEO page did not expose the current all-country workbook download.")
        def vintage_key(path: str) -> tuple[int, int]:
            match = re.search(r"WEO([A-Za-z]{3})(20\d{2})all\.xlsx", path, flags=re.IGNORECASE)
            if not match:
                return (0, 0)
            month = {"jan": 1, "apr": 4, "jul": 7, "oct": 10}.get(match.group(1).lower(), 0)
            return int(match.group(2)), month

        latest_path = max(dict.fromkeys(matches), key=vintage_key)
        return "https://data.imf.org" + latest_path
    finally:
        if own_client:
            selected_client.close()


def parse_bis_bulk_csv(csv_bytes: bytes) -> pd.DataFrame:
    """Parse only BIS WS_TC total non-financial-sector credit as a percentage of GDP (F1.1)."""
    with zipfile.ZipFile(io.BytesIO(csv_bytes)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV inside BIS bulk archive, found {len(names)}.")
        frame = pd.read_csv(archive.open(names[0]), dtype=str, low_memory=False)
    columns = {column.split(":", 1)[0]: column for column in frame.columns}
    required = {"FREQ", "BORROWERS_CTY", "TC_BORROWERS", "TC_LENDERS", "VALUATION", "UNIT_TYPE", "TC_ADJUST", "TIME_PERIOD", "OBS_VALUE"}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"BIS WS_TC flat data is missing dimensions: {', '.join(missing)}")
    selected = frame.loc[
        frame[columns["FREQ"]].fillna("").str.startswith("Q:")
        & frame[columns["TC_BORROWERS"]].fillna("").str.startswith("C:")
        & frame[columns["TC_LENDERS"]].fillna("").str.startswith("A:")
        & frame[columns["VALUATION"]].fillna("").str.startswith("M:")
        & frame[columns["UNIT_TYPE"]].fillna("").str.startswith("770:")
        & frame[columns["TC_ADJUST"]].fillna("").str.startswith("A:")
    ].copy()
    if selected.empty:
        raise ValueError("BIS WS_TC download contains no F1.1 total-credit observations (% GDP).")

    def code_part(value: Any) -> str:
        return str(value or "").split(":", 1)[0].strip()

    selected["bis_iso2"] = selected[columns["BORROWERS_CTY"]].map(code_part)
    selected["country_name"] = selected[columns["BORROWERS_CTY"]].astype(str).str.split(":", n=1).str[-1].str.strip()
    selected["period"] = selected[columns["TIME_PERIOD"]].astype(str).str.strip()
    parts = selected["period"].str.extract(r"^(?P<year>\d{4})-Q(?P<quarter>[1-4])$")
    selected["year"] = pd.to_numeric(parts["year"], errors="coerce").astype("Int64")
    selected["quarter"] = pd.to_numeric(parts["quarter"], errors="coerce").astype("Int64")
    selected["debt_gdp_pct"] = pd.to_numeric(selected[columns["OBS_VALUE"]].str.replace(",", "", regex=False), errors="coerce")
    selected["bis_series_id"] = selected["period"].map(lambda _: "")
    selected["bis_series_id"] = "Q." + selected["bis_iso2"] + ".C.A.M.770.A"
    if "OBS_STATUS" in columns:
        selected["observation_status"] = selected[columns["OBS_STATUS"]].fillna("")
    else:
        selected["observation_status"] = ""
    selected["iso3"] = selected["bis_iso2"].map(_bis_iso2_to_iso3)
    selected = selected.loc[
        selected["year"].notna()
        & selected["quarter"].notna()
        & selected["debt_gdp_pct"].notna()
    ].copy()
    output = selected[[
        "iso3", "bis_iso2", "country_name", "year", "quarter", "period", "debt_gdp_pct",
        "bis_series_id", "observation_status",
    ]]
    return output.drop_duplicates(["bis_iso2", "year", "quarter"], keep="last").sort_values(
        ["bis_iso2", "year", "quarter"]
    ).reset_index(drop=True)


def _parse_bis_bulk(data: bytes) -> pd.DataFrame:
    frame = parse_bis_bulk_csv(data)
    if frame.empty or frame["debt_gdp_pct"].notna().sum() < 100:
        raise ValueError("BIS F1.1 snapshot failed the minimum observation validation.")
    return frame


def parse_bis_dsr_bulk_csv(csv_bytes: bytes) -> pd.DataFrame:
    """Parse quarterly BIS private non-financial sector debt service ratios."""
    with zipfile.ZipFile(io.BytesIO(csv_bytes)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV inside BIS DSR bulk archive, found {len(names)}.")
        frame = pd.read_csv(archive.open(names[0]), dtype=str, low_memory=False)
    columns = {column.split(":", 1)[0]: column for column in frame.columns}
    required = {"FREQ", "BORROWERS_CTY", "DSR_BORROWERS", "TIME_PERIOD", "OBS_VALUE", "UNIT_MEASURE", "UNIT_MULT"}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"BIS DSR flat data is missing dimensions: {', '.join(missing)}")

    selected = frame.loc[
        frame[columns["FREQ"]].fillna("").str.startswith("Q:")
        & frame[columns["DSR_BORROWERS"]].fillna("").str.startswith("P:")
        & frame[columns["UNIT_MEASURE"]].fillna("").str.startswith("367:")
        & frame[columns["UNIT_MULT"]].fillna("").str.startswith("0:")
    ].copy()
    if selected.empty:
        raise ValueError("BIS DSR download contains no quarterly private non-financial sector observations.")

    selected["bis_iso2"] = selected[columns["BORROWERS_CTY"]].map(
        lambda value: _safe_text(value).split(":", 1)[0].strip()
    )
    selected["country_name"] = selected[columns["BORROWERS_CTY"]].map(
        lambda value: _safe_text(value).split(":", 1)[-1].strip()
    )
    selected["period"] = selected[columns["TIME_PERIOD"]].astype(str).str.strip()
    parts = selected["period"].str.extract(r"^(?P<year>\d{4})-Q(?P<quarter>[1-4])$")
    selected["year"] = pd.to_numeric(parts["year"], errors="coerce").astype("Int64")
    selected["quarter"] = pd.to_numeric(parts["quarter"], errors="coerce").astype("Int64")
    selected["dsr_pct"] = pd.to_numeric(
        selected[columns["OBS_VALUE"]].str.replace(",", "", regex=False), errors="coerce"
    )
    selected["iso3"] = selected["bis_iso2"].map(_bis_iso2_to_iso3)
    selected = selected.loc[
        selected["year"].notna() & selected["quarter"].notna() & selected["dsr_pct"].notna()
    ].copy()
    return selected[["iso3", "bis_iso2", "country_name", "year", "quarter", "period", "dsr_pct"]].drop_duplicates(
        ["bis_iso2", "year", "quarter"], keep="last"
    ).sort_values(["bis_iso2", "year", "quarter"]).reset_index(drop=True)


def _parse_bis_dsr(data: bytes) -> pd.DataFrame:
    frame = parse_bis_dsr_bulk_csv(data)
    if frame.empty or frame["dsr_pct"].notna().sum() < 100:
        raise ValueError("BIS private non-financial sector DSR snapshot failed the minimum observation validation.")
    return frame


def parse_oecd_long_term_rates_csv(csv_bytes: bytes) -> pd.DataFrame:
    """Parse OECD monthly long-term interest rates (IRLT, percent per annum)."""
    frame = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, low_memory=False)
    columns = {str(column).lstrip("\ufeff").split(":", 1)[0].strip().upper(): column for column in frame.columns}
    normalized = {name.upper(): name for name in frame.columns}
    cached_fields = {"ISO3", "PERIOD", "YEAR", "MONTH", "LONG_TERM_INTEREST_RATE_PCT"}
    if cached_fields.issubset(normalized):
        selected = frame[[normalized[name] for name in ("ISO3", "PERIOD", "YEAR", "MONTH", "LONG_TERM_INTEREST_RATE_PCT")]].copy()
        selected.columns = ["iso3", "period", "year", "month", "long_term_interest_rate_pct"]
        selected["iso3"] = selected["iso3"].map(lambda value: _safe_text(value).strip().upper())
        selected["period"] = selected["period"].map(lambda value: _safe_text(value).strip())
        selected["year"] = pd.to_numeric(selected["year"], errors="coerce").astype("Int64")
        selected["month"] = pd.to_numeric(selected["month"], errors="coerce").astype("Int64")
        selected["long_term_interest_rate_pct"] = pd.to_numeric(
            selected["long_term_interest_rate_pct"].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
        selected = selected.loc[
            selected["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)
            & selected["year"].notna() & selected["month"].between(1, 12)
            & selected["long_term_interest_rate_pct"].notna()
        ].copy()
        if selected.empty:
            raise ValueError("Cached OECD long-term interest rate observations failed date/value validation.")
        return selected[["iso3", "period", "year", "month", "long_term_interest_rate_pct"]].drop_duplicates(
            ["iso3", "year", "month"], keep="last"
        ).sort_values(["iso3", "year", "month"]).reset_index(drop=True)

    required = {"REF_AREA", "FREQ", "MEASURE", "UNIT_MEASURE", "TIME_PERIOD", "OBS_VALUE"}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"OECD long-term interest rates are missing dimensions: {', '.join(missing)}")

    def code(value: Any) -> str:
        return _safe_text(value).split(":", 1)[0].strip().upper()

    selected = frame.loc[
        frame[columns["FREQ"]].map(code).eq("M")
        & frame[columns["MEASURE"]].map(code).eq("IRLT")
        & frame[columns["UNIT_MEASURE"]].map(code).eq("PA")
    ].copy()
    if selected.empty:
        raise ValueError("OECD download contains no monthly IRLT observations in percent per annum.")
    selected["iso3"] = selected[columns["REF_AREA"]].map(code)
    selected["period"] = selected[columns["TIME_PERIOD"]].astype(str).str.strip()
    parts = selected["period"].str.extract(r"^(?P<year>\d{4})-(?P<month>\d{2})$")
    selected["year"] = pd.to_numeric(parts["year"], errors="coerce").astype("Int64")
    selected["month"] = pd.to_numeric(parts["month"], errors="coerce").astype("Int64")
    selected["long_term_interest_rate_pct"] = pd.to_numeric(
        selected[columns["OBS_VALUE"]].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    selected = selected.loc[
        selected["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)
        & selected["year"].notna() & selected["month"].between(1, 12)
        & selected["long_term_interest_rate_pct"].notna()
    ].copy()
    if selected.empty:
        raise ValueError("OECD long-term interest rate observations failed date/value validation.")
    return selected[["iso3", "period", "year", "month", "long_term_interest_rate_pct"]].drop_duplicates(
        ["iso3", "year", "month"], keep="last"
    ).sort_values(["iso3", "year", "month"]).reset_index(drop=True)


def _parse_oecd_long_term_rates(data: bytes) -> pd.DataFrame:
    frame = parse_oecd_long_term_rates_csv(data)
    if frame.empty or frame["long_term_interest_rate_pct"].notna().sum() < 100:
        raise ValueError("OECD long-term interest rate snapshot failed minimum observation validation.")
    return frame


def _with_russian_long_term_rate_override(rates: pd.DataFrame) -> pd.DataFrame:
    """Replace OECD's Russian observations with the user-provided annual series."""
    rows = pd.DataFrame([
        {"iso3": "RUS", "period": f"{year}-{month:02d}", "year": year, "month": month,
         "long_term_interest_rate_pct": value}
        for year, month, value in RUSSIAN_LONG_TERM_RATES
    ])
    base = rates.loc[~rates["iso3"].eq("RUS")].copy()
    return pd.concat([base, rows], ignore_index=True).sort_values(
        ["iso3", "year", "month"], kind="stable"
    ).reset_index(drop=True)


@lru_cache(maxsize=512)
def _bis_iso2_to_iso3(iso2: str) -> str | None:
    if iso2 in SPECIAL_ISO2_TO_ISO3:
        return SPECIAL_ISO2_TO_ISO3[iso2]
    try:
        import pycountry
    except ImportError:
        return None
    country = pycountry.countries.get(alpha_2=iso2)
    return country.alpha_3 if country else None


def _safe_text(value: Any) -> str:
    if value is None or value is pd.NA or pd.isna(value):
        return ""
    return str(value)


def parse_weo_countries_frame(countries: pd.DataFrame) -> pd.DataFrame:
    required = {"COUNTRY.ID", "COUNTRY", "INDICATOR.ID", "LATEST_ACTUAL_ANNUAL_DATA", "SCALE"}
    missing = sorted(required - set(countries.columns))
    if missing:
        raise ValueError(f"IMF WEO Countries sheet is missing fields: {', '.join(missing)}")
    year_columns = [column for column in countries.columns if re.fullmatch(r"\d{4}", str(column))]
    if not year_columns:
        raise ValueError("IMF WEO Countries sheet contains no annual year columns.")
    selected = countries.loc[countries["INDICATOR.ID"].astype(str).isin(IMF_INDICATORS)].copy()
    present = set(selected["INDICATOR.ID"].astype(str))
    missing_indicators = sorted(set(IMF_INDICATORS) - present)
    if missing_indicators:
        raise ValueError(f"IMF WEO workbook is missing required indicators: {', '.join(missing_indicators)}")
    id_columns = ["COUNTRY.ID", "COUNTRY", "INDICATOR.ID", "LATEST_ACTUAL_ANNUAL_DATA", "SCALE"]
    long = selected[id_columns + year_columns].melt(
        id_vars=id_columns,
        value_vars=year_columns,
        var_name="year",
        value_name="raw_value",
    )
    long["iso3"] = long["COUNTRY.ID"].astype(str).str.upper().str.strip()
    long["country_name"] = long["COUNTRY"].astype(str).str.strip()
    long["indicator_id"] = long["INDICATOR.ID"].astype(str).str.strip()
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["latest_actual_year"] = pd.to_numeric(long["LATEST_ACTUAL_ANNUAL_DATA"], errors="coerce").astype("Int64")
    long["value"] = pd.to_numeric(
        long["raw_value"].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    long["scale"] = long["SCALE"].astype(str).str.strip()
    gdp_mask = long["indicator_id"].eq("NGDPD")
    long.loc[gdp_mask, "value"] = long.loc[gdp_mask].apply(
        lambda row: _weo_gdp_to_usd_bn(row["value"], row["scale"]),
        axis=1,
    )
    return long.loc[
        long["value"].notna() & long["year"].notna() & long["iso3"].str.len().eq(3),
        ["iso3", "country_name", "indicator_id", "year", "value", "latest_actual_year", "scale"],
    ].drop_duplicates(["iso3", "indicator_id", "year"], keep="last").reset_index(drop=True)


def _weo_gdp_to_usd_bn(value: Any, scale: Any) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return np.nan
    normalized_scale = str(scale or "").strip().lower()
    if normalized_scale in {"billions", "billion"}:
        return float(number)
    if normalized_scale in {"millions", "million"}:
        return float(number) / 1_000.0
    if normalized_scale in {"units", "unit", "" , "nan"}:
        return float(number) / 1_000_000_000.0
    raise ValueError(f"Unsupported NGDPD WEO scale: {scale}")


def _parse_weo_workbook(workbook_bytes: bytes) -> pd.DataFrame:
    countries = pd.read_excel(io.BytesIO(workbook_bytes), sheet_name="Countries", engine="openpyxl")
    return parse_weo_countries_frame(countries)


def _validate_weo_workbook(workbook_bytes: bytes) -> str:
    countries = pd.read_excel(io.BytesIO(workbook_bytes), sheet_name="Countries", engine="openpyxl")
    parsed = parse_weo_countries_frame(countries)
    if parsed.empty or parsed["iso3"].nunique() < 20:
        raise ValueError("IMF WEO workbook failed the minimum country-coverage validation.")
    return "Unknown WEO vintage"


def _parse_world_government_bonds_cds(html_bytes: bytes) -> pd.DataFrame:
    """Parse the public World Government Bonds CDS table returned by its page endpoint."""
    class TableParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.rows: list[list[dict[str, Any]]] = []
            self.row: list[dict[str, Any]] | None = None
            self.cell: dict[str, Any] | None = None
            self.in_tbody = False

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            attr = dict(attrs)
            if tag == "tbody":
                self.in_tbody = True
            elif tag == "tr" and self.in_tbody:
                self.row = []
            elif tag == "td" and self.row is not None:
                self.cell = {**attr, "classes": [], "text": []}
            elif self.cell is not None and attr.get("class"):
                self.cell["classes"].extend(str(attr["class"]).split())

        def handle_endtag(self, tag: str) -> None:
            if tag == "td" and self.cell is not None and self.row is not None:
                self.row.append(self.cell)
                self.cell = None
            elif tag == "tr" and self.row is not None:
                self.rows.append(self.row)
                self.row = None
            elif tag == "tbody":
                self.in_tbody = False

        def handle_data(self, data: str) -> None:
            if self.cell is not None:
                self.cell["text"].append(data)

    parser = TableParser()
    parser.feed(html_bytes.decode("utf-8", errors="replace"))
    if not parser.rows:
        raise ValueError("World Government Bonds response has no CDS table.")
    records: list[dict[str, Any]] = []
    for cells in parser.rows:
        if len(cells) < 8:
            continue
        country_name = " ".join("".join(cells[1]["text"]).split())
        flag_classes = cells[0]["classes"]
        flag_position = flag_classes.index("flag") if "flag" in flag_classes else -1
        iso2 = flag_classes[flag_position + 1].upper() if flag_position >= 0 and flag_position + 1 < len(flag_classes) else ""
        iso3 = _bis_iso2_to_iso3(iso2) if iso2 else None
        if not iso3:
            continue

        def cell_number(cell: Any, *, prefer_sort_key: bool = True) -> float:
            sort_key = cell.get("sorttable_customkey") if prefer_sort_key else None
            raw = sort_key or " ".join("".join(cell["text"]).split())
            match = re.search(r"[-+]?\d+(?:[,.]\d+)*", raw.replace(",", ""))
            return float(match.group(0)) if match else np.nan

        records.append({
            "iso3": iso3,
            "country_name": country_name,
            "cds_5y_bps": cell_number(cells[3]),
            "cds_var_6m_pct": cell_number(cells[5]),
            # The provider reuses the CDS sort key in the PD column; read its displayed value.
            "cds_pd_pct": cell_number(cells[6], prefer_sort_key=False),
        })
    frame = pd.DataFrame(records)
    if len(frame) < 15 or frame["iso3"].duplicated().any() or frame["cds_5y_bps"].notna().sum() < 15:
        raise ValueError("World Government Bonds CDS table failed country coverage or uniqueness validation.")
    return frame.reset_index(drop=True)


def parse_world_bank_reserves_frame(data: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize World Bank FI.RES.TOTL.CD observations to USD billions by country/year."""
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.json_normalize(data)
    columns = {str(column).strip().casefold(): column for column in frame.columns}
    cached_fields = {"iso3", "year", "gold_fx_reserves_usd_bn"}
    if cached_fields.issubset(columns):
        selected = frame[[columns[name] for name in ("iso3", "year", "gold_fx_reserves_usd_bn")]].copy()
        selected.columns = ["iso3", "year", "gold_fx_reserves_usd_bn"]
        selected["iso3"] = selected["iso3"].astype(str).str.strip().str.upper()
        selected["year"] = pd.to_numeric(selected["year"], errors="coerce")
        selected["gold_fx_reserves_usd_bn"] = pd.to_numeric(
            selected["gold_fx_reserves_usd_bn"], errors="coerce"
        )
    else:
        required = {"countryiso3code", "date", "value"}
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(f"World Bank reserves response is missing fields: {', '.join(missing)}")
        selected = pd.DataFrame({
            "iso3": frame[columns["countryiso3code"]].astype(str).str.strip().str.upper(),
            "year": pd.to_numeric(frame[columns["date"]], errors="coerce"),
            "gold_fx_reserves_usd_bn": pd.to_numeric(frame[columns["value"]], errors="coerce") / 1_000_000_000.0,
        })
        region_column = columns.get("region.id")
        if region_column:
            selected = selected.loc[frame[region_column].astype(str).str.upper().ne("NA")].copy()
    selected = selected.loc[
        selected["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)
        & selected["year"].notna()
        & selected["gold_fx_reserves_usd_bn"].notna()
        & selected["gold_fx_reserves_usd_bn"].ge(0)
    ].copy()
    if selected.empty:
        raise ValueError("World Bank reserves response contains no valid country observations.")
    selected["year"] = selected["year"].astype(int)
    return selected.drop_duplicates(["iso3", "year"], keep="last").sort_values(
        ["iso3", "year"]
    ).reset_index(drop=True)


def _fetch_world_bank_reserves(client: httpx.Client, reference_year: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    params = {"format": "json", "per_page": 20000, "date": f"2000:{int(reference_year)}"}
    page = 1
    pages = 1
    while page <= pages:
        response = client.get(WORLD_BANK_RESERVES_URL, params={**params, "page": page})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[0], dict):
            raise ValueError("World Bank reserves API returned an invalid paginated response.")
        if payload[0].get("message"):
            raise ValueError(f"World Bank reserves API error: {payload[0]['message']}")
        if page == 1:
            pages = int(payload[0].get("pages", 0))
        if isinstance(payload[1], list):
            records.extend(item for item in payload[1] if isinstance(item, dict))
        page += 1
    frame = parse_world_bank_reserves_frame(records)
    if frame["iso3"].nunique() < 40:
        raise ValueError("World Bank reserves data failed the minimum country-coverage validation.")
    return frame


def build_world_economy_frames(
    bis: pd.DataFrame,
    imf: pd.DataFrame,
    *,
    dsr: pd.DataFrame | None = None,
    long_term_rates: pd.DataFrame | None = None,
    cds: pd.DataFrame | None = None,
    reserves: pd.DataFrame | None = None,
    reference_year: int = 2026,
    source_metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], int, str | None]:
    if bis.empty:
        raise ValueError("No BIS total-credit series are available.")
    if imf.empty:
        raise ValueError("No IMF WEO country values are available.")
    bis = bis.copy()
    imf = imf.copy()
    dsr = dsr.copy() if dsr is not None else pd.DataFrame(
        columns=["iso3", "bis_iso2", "country_name", "year", "quarter", "period", "dsr_pct"]
    )
    long_term_rates = long_term_rates.copy() if long_term_rates is not None else pd.DataFrame(
        columns=["iso3", "period", "year", "month", "long_term_interest_rate_pct"]
    )
    cds = cds.copy() if cds is not None else pd.DataFrame(
        columns=["iso3", "country_name", "cds_5y_bps", "cds_var_6m_pct", "cds_pd_pct"]
    )
    reserves = reserves.copy() if reserves is not None else pd.DataFrame(
        columns=["iso3", "year", "gold_fx_reserves_usd_bn"]
    )
    metadata = source_metadata or {}
    bis["year"] = pd.to_numeric(bis["year"], errors="coerce").astype("Int64")
    bis["quarter"] = pd.to_numeric(bis["quarter"], errors="coerce").astype("Int64")
    duplicate_country_quarters = int(
        bis.loc[bis["iso3"].notna()].duplicated(["iso3", "year", "quarter"]).sum()
    )
    if duplicate_country_quarters:
        mapped_bis = bis.loc[bis["iso3"].notna()].copy()
        unmapped_bis = bis.loc[bis["iso3"].isna()].copy()
        try:
            import pycountry
            canonical_iso2 = {
                country.alpha_3: country.alpha_2
                for country in pycountry.countries
                if hasattr(country, "alpha_3") and hasattr(country, "alpha_2")
            }
            mapped_bis["_canonical_iso2"] = mapped_bis["iso3"].map(canonical_iso2)
            mapped_bis["_mapping_priority"] = mapped_bis["bis_iso2"].eq(mapped_bis["_canonical_iso2"]).astype(int)
            mapped_bis = mapped_bis.sort_values("_mapping_priority", ascending=False).drop_duplicates(
                ["iso3", "year", "quarter"], keep="first"
            ).drop(columns=["_canonical_iso2", "_mapping_priority"])
        except ImportError:
            mapped_bis = mapped_bis.drop_duplicates(["iso3", "year", "quarter"], keep="last")
        bis = pd.concat([mapped_bis, unmapped_bis], ignore_index=True, sort=False)
    imf["year"] = pd.to_numeric(imf["year"], errors="coerce").astype("Int64")
    iso3_values = set(bis["iso3"].dropna().astype(str))
    imf = imf.loc[imf["iso3"].isin(iso3_values)].copy()
    year_warning = None
    requested = int(reference_year)
    present = _common_imf_years(imf)
    if requested not in present:
        common = _common_imf_years(imf)
        candidates = [year for year in common if year <= requested]
        if not candidates:
            raise ValueError(f"WEO does not contain GDP and fiscal values for reference year {requested} or an earlier common year.")
        reference_year = max(candidates)
        year_warning = f"WEO reference year {requested} is unavailable; using latest common year {reference_year} explicitly."
    else:
        reference_year = requested

    gdp = _indicator_panel(imf, "NGDPD", "nominal_gdp_usd_bn")
    fiscal = _indicator_panel(imf, "GGXCNL_NGDP", "fiscal_balance_pct")
    expenditure = _indicator_panel(imf, "GGX_NGDP", "government_expenditure_pct")
    government_debt = _indicator_panel(imf, "GGXWDG_NGDP", "government_debt_gdp_pct")
    current_account = _indicator_panel(imf, "BCA_NGDPD", "current_account_pct")
    real = _indicator_panel(imf, "NGDP_RPCH", "real_gdp_growth_pct")
    country_info = imf[["iso3", "country_name"]].drop_duplicates("iso3").set_index("iso3")["country_name"]
    latest_bis = bis.sort_values(["year", "quarter"]).groupby("iso3", as_index=False).tail(1)
    latest_bis = latest_bis.set_index("iso3")
    cds_by_iso3 = cds.drop_duplicates("iso3", keep="last").set_index("iso3")
    universe = sorted(iso3_values)
    rows: list[dict[str, Any]] = []
    for iso3 in universe:
        current_gdp = _value(gdp, iso3, reference_year, "nominal_gdp_usd_bn")
        current_fiscal = _value(fiscal, iso3, reference_year, "fiscal_balance_pct")
        current_expenditure = _value(expenditure, iso3, reference_year, "government_expenditure_pct")
        current_government_debt_gdp = _value(government_debt, iso3, reference_year, "government_debt_gdp_pct")
        reserve_history = reserves.loc[
            reserves["iso3"].eq(iso3)
            & pd.to_numeric(reserves["year"], errors="coerce").le(reference_year)
        ].sort_values("year", ascending=False)
        government_debt_coverage = np.nan
        for reserve_row in reserve_history.itertuples(index=False):
            reserve_year = int(reserve_row.year)
            reserve_value = float(reserve_row.gold_fx_reserves_usd_bn)
            reserve_year_gdp = _value(gdp, iso3, reserve_year, "nominal_gdp_usd_bn")
            reserve_year_debt_gdp = _value(government_debt, iso3, reserve_year, "government_debt_gdp_pct")
            government_debt_usd_bn = reserve_year_gdp * reserve_year_debt_gdp / 100.0
            if np.isfinite(reserve_value) and np.isfinite(government_debt_usd_bn) and government_debt_usd_bn > 0:
                government_debt_coverage = reserve_value / government_debt_usd_bn * 100.0
                break
        current_account_pct = _value(current_account, iso3, reference_year, "current_account_pct")
        current_account_5y = _value(current_account, iso3, reference_year - 5, "current_account_pct")
        current_account_10y = _value(current_account, iso3, reference_year - 10, "current_account_pct")
        gdp5 = _value(gdp, iso3, reference_year - 5, "nominal_gdp_usd_bn")
        gdp10 = _value(gdp, iso3, reference_year - 10, "nominal_gdp_usd_bn")
        debt_row = latest_bis.loc[iso3] if iso3 in latest_bis.index else None
        debt = float(debt_row["debt_gdp_pct"]) if debt_row is not None and pd.notna(debt_row["debt_gdp_pct"]) else np.nan
        debt_year = int(debt_row["year"]) if debt_row is not None else None
        debt_quarter = int(debt_row["quarter"]) if debt_row is not None else None
        debt5 = _bis_matching_value(bis, iso3, debt_year - 5, debt_quarter) if debt_year and debt_quarter else np.nan
        debt10 = _bis_matching_value(bis, iso3, debt_year - 10, debt_quarter) if debt_year and debt_quarter else np.nan
        fiscal5 = _value(fiscal, iso3, reference_year - 5, "fiscal_balance_pct")
        fiscal10 = _value(fiscal, iso3, reference_year - 10, "fiscal_balance_pct")
        dsr_latest = dsr.loc[dsr["iso3"].eq(iso3)].sort_values(["year", "quarter"]).tail(1)
        if dsr_latest.empty:
            dsr_value, dsr_period, dsr_delta_5y, dsr_delta_10y = np.nan, None, np.nan, np.nan
        else:
            dsr_observation = dsr_latest.iloc[0]
            dsr_value = float(dsr_observation["dsr_pct"])
            dsr_year = int(dsr_observation["year"])
            dsr_quarter = int(dsr_observation["quarter"])
            dsr_period = str(dsr_observation["period"])
            dsr_delta_5y = dsr_value - _dsr_matching_value(dsr, iso3, dsr_year - 5, dsr_quarter)
            dsr_delta_10y = dsr_value - _dsr_matching_value(dsr, iso3, dsr_year - 10, dsr_quarter)
        rates_latest = long_term_rates.loc[long_term_rates["iso3"].eq(iso3)].sort_values(["year", "month"]).tail(1)
        if rates_latest.empty:
            rate_value, rate_period, rate_delta_5y, rate_delta_10y = np.nan, None, np.nan, np.nan
        else:
            rate_observation = rates_latest.iloc[0]
            rate_value = float(rate_observation["long_term_interest_rate_pct"])
            rate_year = int(rate_observation["year"])
            rate_month = int(rate_observation["month"])
            rate_period = str(rate_observation["period"])
            if iso3 == "RUS":
                # Russian inputs are annual observations, not monthly OECD data.
                rate_delta_5y = rate_value - _rate_matching_year_value(long_term_rates, iso3, rate_year - 5)
                rate_delta_10y = rate_value - _rate_matching_year_value(long_term_rates, iso3, rate_year - 10)
            else:
                rate_delta_5y = rate_value - _rate_matching_value(long_term_rates, iso3, rate_year - 5, rate_month)
                rate_delta_10y = rate_value - _rate_matching_value(long_term_rates, iso3, rate_year - 10, rate_month)
        cds_row = cds_by_iso3.loc[iso3] if iso3 in cds_by_iso3.index else None
        latest_real = [_value(real, iso3, reference_year - offset, "real_gdp_growth_pct") for offset in (3, 2, 1)]
        avg_real = float(np.mean(latest_real)) if all(np.isfinite(value) for value in latest_real) else np.nan
        row = {
            "country_name": country_info.get(iso3, iso3),
            "iso3": iso3,
            "reference_year": reference_year,
            "imf_vintage": (source_metadata or {}).get("weo_vintage"),
            "nominal_gdp_usd_bn": current_gdp,
            "gdp_cagr_5y_pct": _cagr(current_gdp, gdp5, 5),
            "gdp_cagr_10y_pct": _cagr(current_gdp, gdp10, 10),
            "total_debt_gdp_pct": debt,
            "government_debt_gdp_pct": current_government_debt_gdp,
            "government_debt_coverage_pct": government_debt_coverage,
            "bis_observation_quarter": f"{debt_year}Q{debt_quarter}" if debt_year and debt_quarter else None,
            "debt_delta_5y_pp": debt - debt5 if np.isfinite(debt) and np.isfinite(debt5) else np.nan,
            "debt_delta_10y_pp": debt - debt10 if np.isfinite(debt) and np.isfinite(debt10) else np.nan,
            "fiscal_balance_pct": current_fiscal,
            "government_expenditure_pct": current_expenditure,
            "current_account_pct": current_account_pct,
            "current_account_delta_5y_pp": current_account_pct - current_account_5y if np.isfinite(current_account_pct) and np.isfinite(current_account_5y) else np.nan,
            "current_account_delta_10y_pp": current_account_pct - current_account_10y if np.isfinite(current_account_pct) and np.isfinite(current_account_10y) else np.nan,
            "fiscal_delta_5y_pp": current_fiscal - fiscal5 if np.isfinite(current_fiscal) and np.isfinite(fiscal5) else np.nan,
            "fiscal_delta_10y_pp": current_fiscal - fiscal10 if np.isfinite(current_fiscal) and np.isfinite(fiscal10) else np.nan,
            "fiscal_balance_usd_bn": current_gdp * current_fiscal / 100.0 if np.isfinite(current_gdp) and np.isfinite(current_fiscal) else np.nan,
            "dsr_pct": dsr_value,
            "dsr_observation_quarter": dsr_period,
            "dsr_delta_5y_pp": dsr_delta_5y,
            "dsr_delta_10y_pp": dsr_delta_10y,
            "long_term_interest_rate_pct": rate_value,
            "long_term_interest_rate_observation_month": rate_period,
            "long_term_interest_rate_delta_5y_pp": rate_delta_5y,
            "long_term_interest_rate_delta_10y_pp": rate_delta_10y,
            "cds_5y_bps": float(cds_row["cds_5y_bps"]) if cds_row is not None and pd.notna(cds_row["cds_5y_bps"]) else np.nan,
            "cds_var_6m_pct": float(cds_row["cds_var_6m_pct"]) if cds_row is not None and pd.notna(cds_row["cds_var_6m_pct"]) else np.nan,
            "cds_pd_pct": float(cds_row["cds_pd_pct"]) if cds_row is not None and pd.notna(cds_row["cds_pd_pct"]) else np.nan,
            f"real_gdp_growth_{reference_year - 3}_pct": latest_real[0],
            f"real_gdp_growth_{reference_year - 2}_pct": latest_real[1],
            f"real_gdp_growth_{reference_year - 1}_pct": latest_real[2],
            "avg_real_gdp_growth_3y_pct": avg_real,
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    table["gdp_rank"] = table["nominal_gdp_usd_bn"].rank(method="min", ascending=False, na_option="bottom").astype("Int64")
    real_growth_columns = [f"real_gdp_growth_{year}_pct" for year in (reference_year - 3, reference_year - 2, reference_year - 1)]
    table = table[TABLE_COLUMNS + real_growth_columns + ["avg_real_gdp_growth_3y_pct"]].sort_values(
        ["nominal_gdp_usd_bn", "country_name"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)

    annual = bis.loc[
        bis["quarter"].eq(4) & bis["year"].between(2000, reference_year - 1)
    ].copy()
    annual = annual.rename(columns={"period": "bis_observation_quarter", "debt_gdp_pct": "total_debt_gdp_pct"})
    annual["year"] = annual["year"].astype(int)
    imf_history = imf.pivot_table(
        index=["iso3", "year"],
        columns="indicator_id",
        values="value",
        aggfunc="first",
    ).reset_index()
    imf_latest = imf.pivot_table(
        index=["iso3", "year"],
        columns="indicator_id",
        values="latest_actual_year",
        aggfunc="first",
    ).reset_index()
    history = annual[["iso3", "country_name", "year", "total_debt_gdp_pct", "bis_observation_quarter"]].merge(
        imf_history, on=["iso3", "year"], how="outer"
    ).merge(imf_latest, on=["iso3", "year"], how="left", suffixes=("", "_latest_actual_year"))
    history["country_name"] = history["country_name"].fillna(history["iso3"].map(country_info))
    history["fiscal_balance_pct"] = history.get("GGXCNL_NGDP", np.nan)
    history["government_expenditure_pct"] = history.get("GGX_NGDP", np.nan)
    history["government_debt_gdp_pct"] = history.get("GGXWDG_NGDP", np.nan)
    history["current_account_pct"] = history.get("BCA_NGDPD", np.nan)
    history["nominal_gdp_usd_bn"] = history.get("NGDPD", np.nan)
    history["real_gdp_growth_pct"] = history.get("NGDP_RPCH", np.nan)
    history["fiscal_status"] = np.where(
        history["GGXCNL_NGDP_latest_actual_year"].notna()
        & history["year"].le(history["GGXCNL_NGDP_latest_actual_year"]),
        "IMF historical / estimate",
        "IMF forecast",
    )
    history["gdp_status"] = np.where(
        history["NGDPD_latest_actual_year"].notna()
        & history["year"].le(history["NGDPD_latest_actual_year"]),
        "IMF historical / estimate",
        "IMF forecast",
    )
    history["real_growth_status"] = np.where(
        history["NGDP_RPCH_latest_actual_year"].notna()
        & history["year"].le(history["NGDP_RPCH_latest_actual_year"]),
        "IMF historical / estimate",
        "IMF forecast",
    )
    history["imf_vintage"] = metadata.get("weo_vintage")
    history["fiscal_balance_usd_bn"] = history["nominal_gdp_usd_bn"] * history["fiscal_balance_pct"] / 100.0
    history = history.sort_values(["iso3", "year"]).reset_index(drop=True)

    # Annual means preserve the source frequency for quarterly BIS/BIS DSR and
    # monthly OECD rates, while the trajectory scatter continues to use its
    # established year-end debt observations.
    debt_annual_mean = bis.groupby(["iso3", "year"], as_index=False)["debt_gdp_pct"].mean().rename(
        columns={"debt_gdp_pct": "total_debt_gdp_annual_avg_pct"}
    )
    dsr_annual_mean = dsr.groupby(["iso3", "year"], as_index=False)["dsr_pct"].mean().rename(
        columns={"dsr_pct": "dsr_annual_avg_pct"}
    )
    rate_annual_mean = long_term_rates.groupby(["iso3", "year"], as_index=False)["long_term_interest_rate_pct"].mean().rename(
        columns={"long_term_interest_rate_pct": "long_term_interest_rate_annual_avg_pct"}
    )
    for annual_mean in (debt_annual_mean, dsr_annual_mean, rate_annual_mean):
        history = history.merge(annual_mean, on=["iso3", "year"], how="outer")
    history = history.sort_values(["iso3", "year"]).reset_index(drop=True)
    history["country_name"] = history["country_name"].fillna(history["iso3"].map(country_info))
    reserves_history = reserves[["iso3", "year", "gold_fx_reserves_usd_bn"]].copy()
    reserves_history["year"] = pd.to_numeric(reserves_history["year"], errors="coerce").astype("Int64")
    reserves_history["gold_fx_reserves_usd_bn"] = pd.to_numeric(
        reserves_history["gold_fx_reserves_usd_bn"], errors="coerce"
    )
    reserves_history = reserves_history.dropna(subset=["iso3", "year"]).drop_duplicates(
        ["iso3", "year"], keep="last"
    )
    history = history.merge(reserves_history, on=["iso3", "year"], how="left")
    government_debt_usd_bn = history["nominal_gdp_usd_bn"] * history["government_debt_gdp_pct"] / 100.0
    history["government_debt_coverage_pct"] = np.where(
        government_debt_usd_bn.gt(0),
        history["gold_fx_reserves_usd_bn"] / government_debt_usd_bn * 100.0,
        np.nan,
    )
    history["nominal_gdp_growth_pct"] = history.groupby("iso3")["nominal_gdp_usd_bn"].pct_change() * 100.0
    history["interest_on_total_debt_pct_gdp"] = (
        history["long_term_interest_rate_annual_avg_pct"]
        * history["total_debt_gdp_annual_avg_pct"] / 100.0
    )
    history["debt_yoy_delta_pp"] = history.groupby("iso3")["total_debt_gdp_pct"].diff()
    history["fiscal_yoy_delta_pp"] = history.groupby("iso3")["fiscal_balance_pct"].diff()
    history["is_provisional"] = False

    provisional = bis.loc[bis["year"].eq(reference_year)].sort_values(["year", "quarter"]).groupby("iso3", as_index=False).tail(1)
    provisional = provisional.merge(
        imf_history.loc[imf_history["year"].eq(reference_year), ["iso3", "year", "GGXCNL_NGDP", "GGX_NGDP", "BCA_NGDPD", "NGDPD", "NGDP_RPCH", "GGXWDG_NGDP"]],
        on=["iso3", "year"], how="left",
    )
    provisional = provisional.merge(
        imf_latest.loc[imf_latest["year"].eq(reference_year), ["iso3", "year", "GGXCNL_NGDP", "GGX_NGDP", "BCA_NGDPD", "NGDPD", "NGDP_RPCH", "GGXWDG_NGDP"]],
        on=["iso3", "year"], how="left", suffixes=("", "_latest_actual_year"),
    )
    if not provisional.empty:
        provisional = provisional.rename(columns={"period": "bis_observation_quarter", "debt_gdp_pct": "total_debt_gdp_pct"})
        provisional["country_name"] = provisional["iso3"].map(country_info).fillna(provisional["country_name"])
        provisional["fiscal_balance_pct"] = provisional["GGXCNL_NGDP"]
        provisional["government_expenditure_pct"] = provisional["GGX_NGDP"]
        provisional["government_debt_gdp_pct"] = provisional["GGXWDG_NGDP"]
        provisional["current_account_pct"] = provisional["BCA_NGDPD"]
        provisional["nominal_gdp_usd_bn"] = provisional["NGDPD"]
        provisional["real_gdp_growth_pct"] = provisional["NGDP_RPCH"]
        provisional["fiscal_balance_usd_bn"] = provisional["nominal_gdp_usd_bn"] * provisional["fiscal_balance_pct"] / 100.0
        provisional["debt_yoy_delta_pp"] = np.nan
        provisional["debt_yoy_delta_pp"] = provisional.apply(
            lambda row: row["total_debt_gdp_pct"] - _bis_matching_value(
                bis, row["iso3"], int(row["year"]) - 1, int(row["quarter"])
            ) if np.isfinite(_bis_matching_value(bis, row["iso3"], int(row["year"]) - 1, int(row["quarter"]))) else np.nan,
            axis=1,
        )
        provisional["fiscal_yoy_delta_pp"] = provisional.apply(
            lambda row: row["fiscal_balance_pct"] - _value(
                fiscal, row["iso3"], int(row["year"]) - 1, "fiscal_balance_pct"
            ) if np.isfinite(_value(fiscal, row["iso3"], int(row["year"]) - 1, "fiscal_balance_pct")) else np.nan,
            axis=1,
        )
        provisional["is_provisional"] = True
        fiscal_latest_year = provisional.get(
            "GGXCNL_NGDP_latest_actual_year", pd.Series(np.nan, index=provisional.index)
        )
        gdp_latest_year = provisional.get(
            "NGDPD_latest_actual_year", pd.Series(np.nan, index=provisional.index)
        )
        growth_latest_year = provisional.get(
            "NGDP_RPCH_latest_actual_year", pd.Series(np.nan, index=provisional.index)
        )
        # WEO latest-actual columns use pandas' nullable Int64 dtype. Comparing a
        # provisional year with a missing latest-actual year therefore returns
        # pd.NA, which np.where cannot treat as a boolean.
        provisional["fiscal_status"] = np.where(
            (fiscal_latest_year.notna() & provisional["year"].le(fiscal_latest_year)).fillna(False),
            "IMF historical / estimate",
            "IMF forecast",
        )
        provisional["gdp_status"] = np.where(
            (gdp_latest_year.notna() & provisional["year"].le(gdp_latest_year)).fillna(False),
            "IMF historical / estimate",
            "IMF forecast",
        )
        provisional["real_growth_status"] = np.where(
            (growth_latest_year.notna() & provisional["year"].le(growth_latest_year)).fillna(False),
            "IMF historical / estimate",
            "IMF forecast",
        )
        provisional["imf_vintage"] = metadata.get("weo_vintage")
        for column in history.columns:
            if column not in provisional.columns:
                provisional[column] = np.nan
        history = pd.concat([history, provisional[history.columns]], ignore_index=True, sort=False)
        history = history.sort_values(["iso3", "year", "is_provisional"]).reset_index(drop=True)

    history["debt_gdp_pct"] = history["total_debt_gdp_pct"]
    bis_countries = bis[["bis_iso2", "iso3", "country_name"]].drop_duplicates()
    unmapped = bis_countries.loc[bis_countries["iso3"].isna()]
    mapped = bis_countries.loc[bis_countries["iso3"].notna()]
    coverage = table["nominal_gdp_usd_bn"].notna() & table["fiscal_balance_pct"].notna()
    diagnostics = {
        "bis_series_definition": "BIS WS_TC F1.1 / Q.<ISO2>.C.A.M.770.A (Total credit to non-financial sector, % GDP)",
        "bis_source_url": metadata.get("bis_topic_url", BIS_TOPIC_URL),
        "bis_retrieved_at_utc": metadata.get("bis_retrieved_at_utc"),
        "bis_observations": int(len(bis)),
        "bis_duplicate_country_quarters_dropped": duplicate_country_quarters,
        "bis_countries_total": int(bis["bis_iso2"].nunique()),
        "bis_countries_mapped": int(mapped["iso3"].nunique()),
        "bis_dsr_observations": int(len(dsr)),
        "bis_dsr_countries_mapped": int(dsr["iso3"].dropna().nunique()),
        "bis_unmapped_economies": unmapped[["bis_iso2", "country_name"]].drop_duplicates().to_dict("records"),
        "imf_source_url": metadata.get("weo_url", IMF_WEO_URL),
        "imf_vintage": metadata.get("weo_vintage"),
        "imf_retrieved_at_utc": metadata.get("weo_retrieved_at_utc"),
        "imf_indicators": list(IMF_INDICATORS),
        "world_bank_reserves_source_url": metadata.get("world_bank_reserves_url", WORLD_BANK_RESERVES_SOURCE_URL),
        "world_bank_reserves_retrieved_at_utc": metadata.get("world_bank_reserves_retrieved_at_utc"),
        "world_bank_reserves_countries": int(reserves["iso3"].nunique()) if not reserves.empty else 0,
        "reference_year": int(reference_year),
        "countries_with_reference_gdp_and_fiscal": int(coverage.sum()),
        "countries_in_table": int(len(table)),
        "latest_bis_quarters": bis.groupby("iso3").tail(1).set_index("iso3")["period"].to_dict(),
    }
    return table, history, diagnostics, int(reference_year), year_warning


def _indicator_panel(imf: pd.DataFrame, indicator: str, value_name: str) -> pd.DataFrame:
    selected = imf.loc[imf["indicator_id"].eq(indicator), ["iso3", "year", "value"]].copy()
    return selected.rename(columns={"value": value_name})


def _value(panel: pd.DataFrame, iso3: str, year: int, column: str) -> float:
    values = panel.loc[panel["iso3"].eq(iso3) & panel["year"].eq(year), column]
    return float(values.iloc[0]) if not values.empty and pd.notna(values.iloc[0]) else np.nan


def _bis_matching_value(bis: pd.DataFrame, iso3: str, year: int, quarter: int) -> float:
    values = bis.loc[bis["iso3"].eq(iso3) & bis["year"].eq(year) & bis["quarter"].eq(quarter), "debt_gdp_pct"]
    return float(values.iloc[0]) if not values.empty and pd.notna(values.iloc[0]) else np.nan


def _dsr_matching_value(dsr: pd.DataFrame, iso3: str, year: int, quarter: int) -> float:
    values = dsr.loc[
        dsr["iso3"].eq(iso3) & dsr["year"].eq(year) & dsr["quarter"].eq(quarter), "dsr_pct"
    ]
    return float(values.iloc[0]) if not values.empty and pd.notna(values.iloc[0]) else np.nan


def _rate_matching_value(rates: pd.DataFrame, iso3: str, year: int, month: int) -> float:
    values = rates.loc[
        rates["iso3"].eq(iso3) & rates["year"].eq(year) & rates["month"].eq(month),
        "long_term_interest_rate_pct",
    ]
    return float(values.iloc[0]) if not values.empty and pd.notna(values.iloc[0]) else np.nan


def _rate_matching_year_value(rates: pd.DataFrame, iso3: str, year: int) -> float:
    values = rates.loc[
        rates["iso3"].eq(iso3) & rates["year"].eq(year), "long_term_interest_rate_pct"
    ]
    return float(values.iloc[-1]) if not values.empty and pd.notna(values.iloc[-1]) else np.nan


def _cagr(current: float, prior: float, years: int) -> float:
    if not np.isfinite(current) or not np.isfinite(prior) or current <= 0 or prior <= 0:
        return np.nan
    return float(((current / prior) ** (1.0 / years) - 1.0) * 100.0)


def _common_imf_years(imf: pd.DataFrame) -> set[int]:
    years_by_indicator = {
        indicator: set(imf.loc[imf["indicator_id"].eq(indicator), "year"].dropna().astype(int))
        for indicator in ("NGDPD", "GGXCNL_NGDP")
    }
    return years_by_indicator["NGDPD"] & years_by_indicator["GGXCNL_NGDP"]


def _weo_vintage_from_url(url: str) -> str:
    match = re.search(r"WEO([A-Za-z]{3})(20\d{2})all\.xlsx", url, flags=re.IGNORECASE)
    if not match:
        return "Unknown WEO vintage"
    month = WEO_VINTAGE_MONTHS.get(match.group(1).title(), match.group(1).title())
    return f"{month} {match.group(2)}"


def _cache_is_stale(paths: tuple[Path, ...]) -> bool:
    if any(not path.exists() for path in paths):
        return True
    now = datetime.now(timezone.utc).timestamp()
    return any(now - path.stat().st_mtime > WORLD_ECONOMY_TTL_SECONDS for path in paths)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _atomic_write_text(path: Path, payload: str) -> None:
    _atomic_write_bytes(path, payload.encode("utf-8"))
