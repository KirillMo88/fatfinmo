from __future__ import annotations

from datetime import datetime, timezone
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


WORLD_ECONOMY_TTL_SECONDS = 7 * 24 * 60 * 60
WORLD_ECONOMY_STORAGE_DIR = Path(
    os.environ.get(
        "WORLD_ECONOMY_STORAGE_DIR",
        str(Path(__file__).with_name("persistent") / "world_economy"),
    )
)
BIS_BULK_URL = "https://data.bis.org/static/bulk/WS_TC_csv_flat.zip"
BIS_TOPIC_URL = "https://data.bis.org/topics/TOTAL_CREDIT"
IMF_WEO_URL = "https://data.imf.org/Datasets/WEO"
IMF_WEO_API_URL = "https://data.imf.org/Datasets/WEO"
IMF_INDICATORS = ("NGDPD", "GGXCNL_NGDP", "NGDP_RPCH")
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
    "debt_delta_5y_pp", "debt_delta_10y_pp", "fiscal_balance_pct",
    "fiscal_delta_5y_pp", "fiscal_delta_10y_pp", "fiscal_balance_usd_bn",
]


def load_world_economy_data(
    *,
    reference_year: int | None = None,
    force_refresh: bool = False,
    storage_dir: Path | None = None,
) -> dict[str, Any]:
    """Load validated official BIS and IMF data, retaining the last good raw files on refresh errors."""
    reference_year = int(reference_year if reference_year is not None else DEFAULT_REFERENCE_YEAR)
    cache_dir = Path(storage_dir or WORLD_ECONOMY_STORAGE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    bis_path = cache_dir / "bis_ws_tc_csv_flat.zip"
    weo_path = cache_dir / "imf_weo_countries.xlsx"
    metadata_path = cache_dir / "source_metadata.json"
    metadata = _read_json(metadata_path)
    warning: str | None = None

    should_refresh = force_refresh or _cache_is_stale((bis_path, weo_path))
    if should_refresh:
        try:
            with httpx.Client(timeout=httpx.Timeout(90.0, connect=20.0), follow_redirects=True) as client:
                bis_response = client.get(BIS_BULK_URL)
                bis_response.raise_for_status()
                bis_bytes = bis_response.content
                # Validate before replacing the current raw snapshot.
                _parse_bis_bulk(bis_bytes)

                workbook_url = discover_imf_weo_workbook_url(client)
                weo_response = client.get(workbook_url)
                weo_response.raise_for_status()
                weo_bytes = weo_response.content
                _validate_weo_workbook(weo_bytes)
                weo_vintage = _weo_vintage_from_url(workbook_url)

            _atomic_write_bytes(bis_path, bis_bytes)
            _atomic_write_bytes(weo_path, weo_bytes)
            vintage_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", weo_vintage).strip("_") or "unknown"
            weo_archive_path = cache_dir / f"imf_weo_{vintage_slug}.xlsx"
            if not weo_archive_path.exists():
                _atomic_write_bytes(weo_archive_path, weo_bytes)
            metadata = {
                "bis_source": "BIS Credit to the non-financial sector, F1.1: total credit, % GDP",
                "bis_series_template": BIS_TOTAL_CREDIT_SERIES_TEMPLATE,
                "bis_url": BIS_BULK_URL,
                "bis_topic_url": BIS_TOPIC_URL,
                "bis_retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "weo_source": "IMF World Economic Outlook, Countries sheet",
                "weo_url": workbook_url,
                "weo_vintage": weo_vintage,
                "weo_archive_file": weo_archive_path.name,
                "weo_indicators": list(IMF_INDICATORS),
                "weo_retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_write_text(metadata_path, json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        except Exception as exc:
            if not (bis_path.exists() and weo_path.exists()):
                raise RuntimeError(f"Could not refresh World Economy sources and no validated cache is available: {exc}") from exc
            warning = f"Refresh failed; showing last validated BIS/IMF snapshot: {exc}"

    if not (bis_path.exists() and weo_path.exists()):
        raise RuntimeError("World Economy source cache is incomplete. Use Refresh World Economy to download official data.")

    bis = _parse_bis_bulk(bis_path.read_bytes())
    imf = _parse_weo_workbook(weo_path.read_bytes())
    table, history, diagnostics, effective_year, year_warning = build_world_economy_frames(
        bis,
        imf,
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
    try:
        import pycountry
    except ImportError:
        pycountry = None

    def iso3_for(iso2: str) -> str | None:
        if iso2 in SPECIAL_ISO2_TO_ISO3:
            return SPECIAL_ISO2_TO_ISO3[iso2]
        if pycountry is None:
            return None
        country = pycountry.countries.get(alpha_2=iso2)
        return country.alpha_3 if country else None

    selected["iso3"] = selected["bis_iso2"].map(iso3_for)
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


def build_world_economy_frames(
    bis: pd.DataFrame,
    imf: pd.DataFrame,
    *,
    reference_year: int = 2026,
    source_metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], int, str | None]:
    if bis.empty:
        raise ValueError("No BIS total-credit series are available.")
    if imf.empty:
        raise ValueError("No IMF WEO country values are available.")
    bis = bis.copy()
    imf = imf.copy()
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
    real = _indicator_panel(imf, "NGDP_RPCH", "real_gdp_growth_pct")
    country_info = imf[["iso3", "country_name"]].drop_duplicates("iso3").set_index("iso3")["country_name"]
    latest_bis = bis.sort_values(["year", "quarter"]).groupby("iso3", as_index=False).tail(1)
    latest_bis = latest_bis.set_index("iso3")
    universe = sorted(iso3_values)
    rows: list[dict[str, Any]] = []
    for iso3 in universe:
        current_gdp = _value(gdp, iso3, reference_year, "nominal_gdp_usd_bn")
        current_fiscal = _value(fiscal, iso3, reference_year, "fiscal_balance_pct")
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
            "bis_observation_quarter": f"{debt_year}Q{debt_quarter}" if debt_year and debt_quarter else None,
            "debt_delta_5y_pp": debt - debt5 if np.isfinite(debt) and np.isfinite(debt5) else np.nan,
            "debt_delta_10y_pp": debt - debt10 if np.isfinite(debt) and np.isfinite(debt10) else np.nan,
            "fiscal_balance_pct": current_fiscal,
            "fiscal_delta_5y_pp": current_fiscal - fiscal5 if np.isfinite(current_fiscal) and np.isfinite(fiscal5) else np.nan,
            "fiscal_delta_10y_pp": current_fiscal - fiscal10 if np.isfinite(current_fiscal) and np.isfinite(fiscal10) else np.nan,
            "fiscal_balance_usd_bn": current_gdp * current_fiscal / 100.0 if np.isfinite(current_gdp) and np.isfinite(current_fiscal) else np.nan,
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
    history["debt_yoy_delta_pp"] = history.groupby("iso3")["total_debt_gdp_pct"].diff()
    history["fiscal_yoy_delta_pp"] = history.groupby("iso3")["fiscal_balance_pct"].diff()
    history["is_provisional"] = False

    provisional = bis.loc[bis["year"].eq(reference_year)].sort_values(["year", "quarter"]).groupby("iso3", as_index=False).tail(1)
    provisional = provisional.merge(
        imf_history.loc[imf_history["year"].eq(reference_year), ["iso3", "year", "GGXCNL_NGDP", "NGDPD", "NGDP_RPCH"]],
        on=["iso3", "year"], how="left",
    )
    provisional = provisional.merge(
        imf_latest.loc[imf_latest["year"].eq(reference_year), ["iso3", "year", "GGXCNL_NGDP", "NGDPD", "NGDP_RPCH"]],
        on=["iso3", "year"], how="left", suffixes=("", "_latest_actual_year"),
    )
    if not provisional.empty:
        provisional = provisional.rename(columns={"period": "bis_observation_quarter", "debt_gdp_pct": "total_debt_gdp_pct"})
        provisional["country_name"] = provisional["iso3"].map(country_info).fillna(provisional["country_name"])
        provisional["fiscal_balance_pct"] = provisional["GGXCNL_NGDP"]
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
        provisional["fiscal_status"] = np.where(
            provisional["year"].le(provisional.get("GGXCNL_NGDP_latest_actual_year", pd.Series(np.nan, index=provisional.index))),
            "IMF historical / estimate",
            "IMF forecast",
        )
        provisional["gdp_status"] = np.where(
            provisional["year"].le(provisional.get("NGDPD_latest_actual_year", pd.Series(np.nan, index=provisional.index))),
            "IMF historical / estimate",
            "IMF forecast",
        )
        provisional["real_growth_status"] = np.where(
            provisional["year"].le(provisional.get("NGDP_RPCH_latest_actual_year", pd.Series(np.nan, index=provisional.index))),
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
        "bis_unmapped_economies": unmapped[["bis_iso2", "country_name"]].drop_duplicates().to_dict("records"),
        "imf_source_url": metadata.get("weo_url", IMF_WEO_URL),
        "imf_vintage": metadata.get("weo_vintage"),
        "imf_retrieved_at_utc": metadata.get("weo_retrieved_at_utc"),
        "imf_indicators": list(IMF_INDICATORS),
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
