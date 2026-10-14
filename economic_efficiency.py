from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from world_gdp import WORLD_BANK_API, WORLD_ECONOMY_STORAGE_DIR


PWT_VERSION = "11.0"
PWT_SOURCE_PAGE = "https://www.rug.nl/ggdc/productivity/pwt/"
PWT_XLSX_URL = "https://dataverse.nl/api/access/datafile/554105"
PWT_TTL_SECONDS = 30 * 24 * 60 * 60
SERVICES_INDICATOR = "NV.SRV.TOTL.ZS"
PWT_REQUIRED_COLUMNS = (
    "countrycode",
    "country",
    "year",
    "ctfp",
    "rtfpna",
    "cgdpo",
    "cn",
    "rgdpna",
    "rnna",
    "emp",
    "avh",
)
PWT_METRIC_COLUMNS = ("ctfp", "rtfpna", "cgdpo", "cn", "rgdpna", "rnna", "emp", "avh")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator.div(denominator.where(denominator.gt(0))).replace([np.inf, -np.inf], np.nan)


class EconomicEfficiencyService:
    """Persist official PWT 11.0 and the one additional WDI structural series."""

    def __init__(self, storage_dir: Path | None = None, client: httpx.Client | None = None) -> None:
        root = Path(storage_dir or WORLD_ECONOMY_STORAGE_DIR)
        self.storage_dir = root / "economic_efficiency" / "pwt_11_0"
        self.raw_dir = self.storage_dir / "raw"
        self.processed_dir = self.storage_dir / "processed"
        self.metadata_dir = self.storage_dir / "metadata"
        self.manifest_path = self.metadata_dir / "latest.json"
        self.client = client

    def load(self, force_refresh: bool = False) -> dict[str, Any]:
        if not force_refresh and self._cache_is_fresh():
            return self._cached(status="cached")
        try:
            pwt, services, metadata, raw_xlsx, raw_services = self._download()
            self._save_snapshot(pwt, services, metadata, raw_xlsx, raw_services)
            return {"pwt": pwt, "services": services, "metadata": metadata, "status": "current"}
        except Exception as exc:
            if self.manifest_path.exists():
                cached = self._cached(status="cached")
                cached["warning"] = f"Economic Efficiency refresh failed; showing the last valid snapshot: {exc}"
                return cached
            return {
                "pwt": pd.DataFrame(),
                "services": pd.DataFrame(),
                "metadata": {},
                "status": "unavailable",
                "warning": f"Economic Efficiency data is unavailable: {exc}",
            }

    def _cache_is_fresh(self) -> bool:
        return self.manifest_path.exists() and (
            _utc_now().timestamp() - self.manifest_path.stat().st_mtime
        ) < PWT_TTL_SECONDS

    def _cached(self, *, status: str) -> dict[str, Any]:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        pwt_path = self.processed_dir / manifest["pwt_file"]
        services_path = self.processed_dir / manifest["services_file"]
        for path, expected in (
            (pwt_path, manifest["pwt_sha256"]),
            (services_path, manifest["services_sha256"]),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Persisted Economic Efficiency snapshot failed integrity check: {path.name}")
        metadata = json.loads((self.metadata_dir / manifest["metadata_file"]).read_text(encoding="utf-8"))
        return {
            "pwt": coerce_pwt(pd.read_csv(pwt_path)),
            "services": coerce_services(pd.read_csv(services_path)),
            "metadata": metadata,
            "status": status,
        }

    def _download(self) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], bytes, Any]:
        own_client = self.client is None
        client = self.client or httpx.Client(timeout=180.0, follow_redirects=True)
        try:
            pwt_response = client.get(PWT_XLSX_URL)
            pwt_response.raise_for_status()
            raw_xlsx = pwt_response.content
            pwt = self._read_pwt(raw_xlsx)

            services_response = client.get(
                f"{WORLD_BANK_API}/country/all/indicator/{SERVICES_INDICATOR}",
                params={"format": "json", "per_page": 20000, "date": "1950:2030"},
            )
            services_response.raise_for_status()
            raw_services = services_response.json()
            services = self._normalise_services(raw_services)
        finally:
            if own_client:
                client.close()

        latest_year = int(pwt.loc[pwt[list(PWT_METRIC_COLUMNS)].notna().all(axis=1), "year"].max())
        country_count = int(pwt.loc[pwt["year"].eq(latest_year), "iso3"].nunique())
        if country_count < 150:
            raise ValueError(f"PWT {PWT_VERSION} latest-year coverage is unexpectedly low: {country_count}")
        metadata = {
            "pwt_version": PWT_VERSION,
            "pwt_source_url": PWT_SOURCE_PAGE,
            "pwt_download_url": PWT_XLSX_URL,
            "pwt_license": "CC BY 4.0",
            "pwt_latest_year": latest_year,
            "pwt_country_count": country_count,
            "wdi_indicator": SERVICES_INDICATOR,
            "wdi_source_url": "https://databank.worldbank.org/source/world-development-indicators",
            "retrieved_at_utc": _utc_now().isoformat(),
            "dataset_version": 1,
        }
        return pwt, services, metadata, raw_xlsx, raw_services

    @staticmethod
    def _read_pwt(payload: bytes) -> pd.DataFrame:
        workbook = pd.ExcelFile(BytesIO(payload), engine="openpyxl")
        sheet = "Data" if "Data" in workbook.sheet_names else workbook.sheet_names[0]
        frame = pd.read_excel(workbook, sheet_name=sheet)
        missing = sorted(set(PWT_REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"PWT workbook is missing required columns: {missing}")
        result = coerce_pwt(frame[list(PWT_REQUIRED_COLUMNS)])
        if result.empty or result["iso3"].nunique() < 150:
            raise ValueError("PWT workbook did not pass country coverage validation")
        return result

    @staticmethod
    def _normalise_services(payload: Any) -> pd.DataFrame:
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            raise ValueError("World Bank returned no Services/GDP observations")
        rows: list[dict[str, Any]] = []
        for item in payload[1]:
            iso3 = str(item.get("countryiso3code") or "").strip().upper()
            year = pd.to_numeric(item.get("date"), errors="coerce")
            value = pd.to_numeric(item.get("value"), errors="coerce")
            if iso3 and pd.notna(year):
                rows.append({"iso3": iso3, "year": int(year), "services_gdp_pct": value})
        return coerce_services(pd.DataFrame(rows))

    def _save_snapshot(
        self,
        pwt: pd.DataFrame,
        services: pd.DataFrame,
        metadata: dict[str, Any],
        raw_xlsx: bytes,
        raw_services: Any,
    ) -> None:
        pwt_payload = pwt.to_csv(index=False).encode("utf-8")
        services_payload = services.to_csv(index=False).encode("utf-8")
        pwt_digest = hashlib.sha256(pwt_payload).hexdigest()
        services_digest = hashlib.sha256(services_payload).hexdigest()
        short = pwt_digest[:16]
        pwt_name = f"pwt110_{short}.csv"
        services_name = f"services_{services_digest[:16]}.csv"
        metadata_name = f"snapshot_{short}.json"
        saved_metadata = dict(metadata, pwt_sha256=pwt_digest, services_sha256=services_digest)
        _atomic_write(self.processed_dir / pwt_name, pwt_payload)
        _atomic_write(self.processed_dir / services_name, services_payload)
        _atomic_write(self.raw_dir / "pwt110.xlsx", raw_xlsx)
        _atomic_write(
            self.raw_dir / "wdi_services.json",
            json.dumps(raw_services, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        )
        _atomic_write(
            self.metadata_dir / metadata_name,
            json.dumps(saved_metadata, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        _atomic_write(
            self.manifest_path,
            json.dumps(
                {
                    "pwt_file": pwt_name,
                    "services_file": services_name,
                    "metadata_file": metadata_name,
                    "pwt_sha256": pwt_digest,
                    "services_sha256": services_digest,
                    "activated_at_utc": _utc_now().isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
        )


def coerce_pwt(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "countrycode" in result:
        result["iso3"] = result["countrycode"].astype(str).str.strip().str.upper()
    elif "iso3" in result:
        result["iso3"] = result["iso3"].astype(str).str.strip().str.upper()
        result["countrycode"] = result["iso3"]
    else:
        raise ValueError("PWT data requires countrycode or iso3")
    result["country"] = result.get("country", result["iso3"]).astype(str).str.strip()
    result["year"] = pd.to_numeric(result["year"], errors="raise").astype(int)
    for column in PWT_METRIC_COLUMNS:
        result[column] = pd.to_numeric(result.get(column), errors="coerce")
    return result[["iso3", "countrycode", "country", "year", *PWT_METRIC_COLUMNS]].drop_duplicates(
        ["iso3", "year"], keep="last"
    ).sort_values(["iso3", "year"]).reset_index(drop=True)


def coerce_services(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return pd.DataFrame(columns=["iso3", "year", "services_gdp_pct"])
    result["iso3"] = result["iso3"].astype(str).str.strip().str.upper()
    result["year"] = pd.to_numeric(result["year"], errors="raise").astype(int)
    result["services_gdp_pct"] = pd.to_numeric(result["services_gdp_pct"], errors="coerce")
    return result.drop_duplicates(["iso3", "year"], keep="last").sort_values(["iso3", "year"]).reset_index(drop=True)


def calculate_efficiency_history(pwt: pd.DataFrame) -> pd.DataFrame:
    data = coerce_pwt(pwt)
    result = data.copy()
    result["tfp_level"] = result["ctfp"] * 100.0
    result["capital_productivity_current_pct"] = _safe_ratio(result["cgdpo"], result["cn"]) * 100.0
    result["capital_productivity_real_pct"] = _safe_ratio(result["rgdpna"], result["rnna"]) * 100.0
    hours = result["emp"] * result["avh"]
    result["gdp_per_hour_ppp"] = _safe_ratio(result["cgdpo"], hours)
    result["gdp_per_hour_real"] = _safe_ratio(result["rgdpna"], hours)

    lookback = result[[
        "iso3", "year", "rtfpna", "capital_productivity_real_pct", "gdp_per_hour_real"
    ]].copy()
    lookback["year"] = lookback["year"] + 10
    lookback = lookback.rename(columns={
        "rtfpna": "rtfpna_10y_ago",
        "capital_productivity_real_pct": "capital_productivity_real_10y_ago",
        "gdp_per_hour_real": "gdp_per_hour_real_10y_ago",
    })
    result = result.merge(lookback, on=["iso3", "year"], how="left", validate="one_to_one")
    result["tfp_growth_10y_pct"] = (
        _safe_ratio(result["rtfpna"], result["rtfpna_10y_ago"]) - 1.0
    ) * 100.0
    result["capital_productivity_change_10y_pct"] = (
        _safe_ratio(result["capital_productivity_real_pct"], result["capital_productivity_real_10y_ago"]) - 1.0
    ) * 100.0
    result["gdp_per_hour_change_10y_pct"] = (
        _safe_ratio(result["gdp_per_hour_real"], result["gdp_per_hour_real_10y_ago"]) - 1.0
    ) * 100.0
    return result.replace([np.inf, -np.inf], np.nan)


def resolve_reference_year(pwt: pd.DataFrame, world_gdp: pd.DataFrame) -> int:
    efficiency = calculate_efficiency_history(pwt)
    gdp = world_gdp.copy()
    gdp["year"] = pd.to_numeric(gdp["year"], errors="coerce")
    gdp["nominal_gdp_usd"] = pd.to_numeric(gdp["nominal_gdp_usd"], errors="coerce")
    if "is_country" not in gdp:
        gdp["is_country"] = gdp["iso3"].ne("WLD")
    complete = efficiency[list(PWT_METRIC_COLUMNS)].notna().all(axis=1)
    candidates = sorted(efficiency.loc[complete, "year"].unique(), reverse=True)
    for year in candidates:
        annual_gdp = gdp.loc[gdp["year"].eq(year) & gdp["is_country"]].dropna(subset=["nominal_gdp_usd"])
        top10 = annual_gdp.nlargest(10, "nominal_gdp_usd")
        available = set(efficiency.loc[complete & efficiency["year"].eq(year), "iso3"])
        if len(top10) == 10 and set(top10["iso3"]).issubset(available):
            return int(year)
    raise ValueError("No synchronized PWT/WDI year has complete efficiency data for the nominal-GDP Top 10")


def eligible_countries(pwt: pd.DataFrame, reference_year: int) -> pd.DataFrame:
    data = coerce_pwt(pwt)
    current = data.loc[data["year"].eq(int(reference_year))].copy()
    current = current.loc[current[list(PWT_METRIC_COLUMNS)].notna().all(axis=1)]
    return current[["iso3", "country"]].drop_duplicates("iso3").sort_values("country").reset_index(drop=True)


def _services_snapshot(services: pd.DataFrame, iso_codes: pd.Series, reference_year: int) -> pd.DataFrame:
    data = coerce_services(services)
    data = data.loc[data["iso3"].isin(set(iso_codes)) & data["year"].le(int(reference_year))]
    return (
        data.dropna(subset=["services_gdp_pct"])
        .sort_values(["iso3", "year"])
        .groupby("iso3", as_index=False)
        .tail(1)[["iso3", "year", "services_gdp_pct"]]
        .rename(columns={"year": "services_year"})
    )


def build_efficiency_table(
    pwt: pd.DataFrame,
    services: pd.DataFrame,
    world_gdp: pd.DataFrame,
    *,
    reference_year: int,
    selected_iso3: str | None = None,
) -> pd.DataFrame:
    history = calculate_efficiency_history(pwt)
    current = history.loc[history["year"].eq(int(reference_year))].copy()
    gdp = world_gdp.copy()
    gdp["year"] = pd.to_numeric(gdp["year"], errors="coerce")
    gdp["nominal_gdp_usd"] = pd.to_numeric(gdp["nominal_gdp_usd"], errors="coerce")
    if "is_country" not in gdp:
        gdp["is_country"] = gdp["iso3"].ne("WLD")
    ranked = (
        gdp.loc[gdp["year"].eq(int(reference_year)) & gdp["is_country"]]
        .dropna(subset=["nominal_gdp_usd"])
        .sort_values("nominal_gdp_usd", ascending=False)
        .drop_duplicates("iso3", keep="first")
        .reset_index(drop=True)
    )
    ranked["global_gdp_rank"] = np.arange(1, len(ranked) + 1)
    top = ranked.head(10).merge(current, on="iso3", how="left", validate="one_to_one")
    top["position"] = top["global_gdp_rank"].map(lambda value: f"#{int(value)}")
    top["is_selected"] = top["iso3"].eq(str(selected_iso3 or "").upper())

    selected_code = str(selected_iso3 or "").upper()
    if selected_code and selected_code not in set(top["iso3"]):
        selected_metric = current.loc[current["iso3"].eq(selected_code)]
        if not selected_metric.empty:
            selected_gdp = ranked.loc[ranked["iso3"].eq(selected_code)]
            if selected_gdp.empty:
                selected_gdp = pd.DataFrame([{
                    "iso3": selected_code,
                    "country_name": selected_metric.iloc[0]["country"],
                    "nominal_gdp_usd": np.nan,
                    "global_gdp_rank": np.nan,
                }])
            appended = selected_gdp.iloc[[0]].merge(selected_metric, on="iso3", how="left")
            appended["position"] = "Selected"
            appended["is_selected"] = True
            top = pd.concat([top, appended], ignore_index=True, sort=False)

    structural = _services_snapshot(services, top["iso3"], reference_year)
    top = top.merge(structural, on="iso3", how="left", validate="many_to_one")
    top["services_is_older"] = top["services_year"].notna() & top["services_year"].ne(int(reference_year))
    if "country_name" not in top:
        top["country_name"] = top["country"]
    top["country_name"] = top["country_name"].fillna(top["country"])
    return top.reset_index(drop=True)


def build_country_history(
    pwt: pd.DataFrame,
    services: pd.DataFrame,
    world_gdp: pd.DataFrame,
    *,
    iso3: str,
    reference_year: int,
    observations: int = 30,
) -> pd.DataFrame:
    history = calculate_efficiency_history(pwt)
    selected = history.loc[
        history["iso3"].eq(iso3.upper()) & history["year"].le(int(reference_year))
    ].sort_values("year").tail(observations).copy()
    structural = coerce_services(services).rename(columns={"services_gdp_pct": "services_gdp_pct"})
    selected = selected.merge(structural, on=["iso3", "year"], how="left", validate="one_to_one")
    gdp = world_gdp[["iso3", "year", "nominal_gdp_usd"]].copy()
    gdp["year"] = pd.to_numeric(gdp["year"], errors="coerce").astype("Int64")
    selected = selected.merge(gdp, on=["iso3", "year"], how="left", validate="one_to_one")
    return selected.sort_values("year").reset_index(drop=True)


def load_economic_efficiency_data(force_refresh: bool = False) -> dict[str, Any]:
    return EconomicEfficiencyService().load(force_refresh=force_refresh)
