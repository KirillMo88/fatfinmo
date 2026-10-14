from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import httpx
import numpy as np
import pandas as pd


WORLD_TRADE_TTL_SECONDS = 180 * 24 * 60 * 60
WORLD_ECONOMY_STORAGE_DIR = Path(
    os.environ.get(
        "WORLD_ECONOMY_STORAGE_DIR",
        str(Path(__file__).with_name("persistent") / "world_economy"),
    )
)
WTO_API_BASE_URL = os.environ.get("WTO_API_BASE_URL", "https://api.wto.org/timeseries/v1")
WDI_BASE_URL = "https://api.worldbank.org/v2/country/WLD/indicator"
GACC_MONTHLY_URL = "https://english.customs.gov.cn/statics/report/monthly.html"

REGION_ORDER = [
    "ASEAN Plus", "EU", "US", "Americas ex US", "CIS",
    "Middle East", "Africa", "Other", "Hong Kong",
]


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


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write(path, frame.to_csv(index=False).encode("utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_fresh(path: Path, ttl_seconds: int = WORLD_TRADE_TTL_SECONDS) -> bool:
    return path.exists() and (_utc_now().timestamp() - path.stat().st_mtime) < ttl_seconds


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "gb18030"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to decode {path.name}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _normalise_header(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    return text.strip("_")


def _find_column(columns: Iterable[object], aliases: Iterable[str]) -> object | None:
    normalised = {_normalise_header(column): column for column in columns}
    for alias in aliases:
        key = _normalise_header(alias)
        if key in normalised:
            return normalised[key]
    for key, original in normalised.items():
        if any(_normalise_header(alias) in key for alias in aliases):
            return original
    return None


class GlobalTradeService:
    """World Bank world-trade history with a six-month persistent cache."""

    INDICATORS = {
        "real_exports_usd": "NE.EXP.GNFS.KD",
        "nominal_exports_usd": "NE.EXP.GNFS.CD",
        "real_gdp_growth_pct": "NY.GDP.MKTP.KD.ZG",
        "nominal_gdp_usd": "NY.GDP.MKTP.CD",
    }

    def __init__(self, storage_dir: Path | None = None, client: httpx.Client | None = None) -> None:
        self.storage_dir = Path(storage_dir or WORLD_ECONOMY_STORAGE_DIR) / "world_trade"
        self.raw = self.storage_dir / "raw"
        self.processed = self.storage_dir / "processed"
        self.data_path = self.processed / "global_trade.csv"
        self.metadata_path = self.processed / "metadata.json"
        self.client = client

    def load(self, force_refresh: bool = False) -> dict[str, Any]:
        if not force_refresh and _is_fresh(self.data_path) and self.metadata_path.exists():
            return self._cached()
        try:
            frame, metadata = self._download()
            _write_frame(self.data_path, frame)
            _write_json(self.metadata_path, metadata)
            return {"data": frame, "metadata": metadata, "status": "current"}
        except Exception as exc:
            if self.data_path.exists() and self.metadata_path.exists():
                cached = self._cached()
                cached["status"] = "cached"
                cached["warning"] = f"Refresh failed; showing the last valid version: {exc}"
                return cached
            return {
                "data": pd.DataFrame(), "metadata": {}, "status": "unavailable",
                "warning": f"Global trade history is unavailable: {exc}",
            }

    def _cached(self) -> dict[str, Any]:
        return {"data": pd.read_csv(self.data_path), "metadata": _read_json(self.metadata_path), "status": "cached"}

    def _download(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        last_completed_year = _utc_now().year - 1
        own_client = self.client is None
        client = self.client or httpx.Client(timeout=35.0, follow_redirects=True)
        try:
            series: list[pd.DataFrame] = []
            source_dates: dict[str, str | None] = {}
            for label, code in self.INDICATORS.items():
                response = client.get(
                    f"{WDI_BASE_URL}/{code}",
                    params={"format": "json", "per_page": 2000, "date": f"1970:{last_completed_year}"},
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
                    raise ValueError(f"World Bank returned no observations for {code}")
                rows = pd.DataFrame(payload[1])
                rows["year"] = pd.to_numeric(rows["date"], errors="coerce")
                rows[label] = pd.to_numeric(rows["value"], errors="coerce")
                series.append(rows[["year", label]].dropna(subset=["year"]))
                source_dates[code] = payload[0].get("lastupdated") if isinstance(payload[0], dict) else None
        finally:
            if own_client:
                client.close()

        frame = series[0]
        for item in series[1:]:
            frame = frame.merge(item, on="year", how="outer", validate="one_to_one")
        frame["year"] = frame["year"].astype(int)
        frame = frame.loc[frame["year"].between(1970, last_completed_year)].sort_values("year")
        frame["real_exports_growth_pct"] = frame["real_exports_usd"].pct_change(fill_method=None) * 100.0
        frame["real_exports_growth_5y_avg_pct"] = frame["real_exports_growth_pct"].rolling(5, min_periods=5).mean()
        frame["nominal_exports_usd_trn"] = frame["nominal_exports_usd"] / 1e12
        frame["real_exports_usd_trn"] = frame["real_exports_usd"] / 1e12
        frame["exports_gdp_pct"] = frame["nominal_exports_usd"] / frame["nominal_gdp_usd"] * 100.0
        if frame["nominal_exports_usd"].notna().sum() < 20:
            raise ValueError("World Bank series did not pass the minimum history check")
        metadata = {
            "retrieved_at_utc": _utc_now().isoformat(),
            "provider": "World Bank",
            "source": "World Bank World Development Indicators",
            "source_url": "https://databank.worldbank.org/source/world-development-indicators",
            "indicators": self.INDICATORS,
            "source_last_updated": source_dates,
            "latest_completed_year": last_completed_year,
            "observation_period": f"{int(frame['year'].min())}-{int(frame['year'].max())}",
            "unit": "Source USD levels; derived percent changes and ratios",
            "currency": "USD", "validation_status": "valid", "dataset_version": 1,
            "forecast": self._load_wto_forecast(),
        }
        return frame.reset_index(drop=True), metadata

    def _load_wto_forecast(self) -> dict[str, Any] | None:
        """Load an optional sourced WTO forecast; invalid or stale-looking inputs are omitted."""
        path = self.raw / "wto_forecast.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        required = {"forecast_year", "growth_pct", "publication_date", "source_url"}
        if not required.issubset(frame.columns) or frame.empty:
            return None
        frame["forecast_year"] = pd.to_numeric(frame["forecast_year"], errors="coerce")
        frame["growth_pct"] = pd.to_numeric(frame["growth_pct"], errors="coerce")
        frame["publication_date"] = pd.to_datetime(frame["publication_date"], errors="coerce", utc=True)
        frame = frame.dropna(subset=["forecast_year", "growth_pct", "publication_date"])
        frame = frame.loc[frame["source_url"].astype(str).str.contains(r"(^|\.)wto\.org/", case=False, regex=True)]
        if frame.empty:
            return None
        latest = frame.sort_values("publication_date").iloc[-1]
        if int(latest["forecast_year"]) < _utc_now().year:
            return None
        return {
            "year": int(latest["forecast_year"]), "growth_pct": float(latest["growth_pct"]),
            "publication_date": latest["publication_date"].date().isoformat(),
            "source_url": str(latest["source_url"]),
            "concept": "WTO Merchandise Trade Volume Forecast — Goods Only",
        }


class TradeBalanceService:
    """Validated WTO annual merchandise trade balances."""

    EXPORT_CODE = "ITS_MTV_AX"
    IMPORT_CODE = "ITS_MTV_AM"
    TOTAL_MERCHANDISE_CODE = "TO"
    WORLD_PARTNER_CODE = "000"
    API_MAX_RECORDS = 1_000_000
    AGGREGATE_NAMES = {
        "world", "total", "all economies", "european union", "eu", "asean",
        "africa", "asia", "europe", "middle east", "north america", "south america",
    }
    AGGREGATE_NAME_PATTERN = re.compile(
        r"(?:\bmembers?\b|\bgroup of\b|\bg[- ]?\d+\b|\bbrics?\b|\bopec\b|"
        r"\borganization of\b|\bcooperation council\b|\bassociation for\b|"
        r"\bagreement\b|\bcustoms union\b|\bcommon market\b|\beconomic community\b|"
        r"\bdeveloped economies\b|\bdeveloping economies\b|\bleast[- ]developed\b|"
        r"\bgeographical region\b)",
        flags=re.IGNORECASE,
    )
    DATASET_VERSION = 2

    def __init__(self, storage_dir: Path | None = None, client: httpx.Client | None = None) -> None:
        self.storage_dir = Path(storage_dir or WORLD_ECONOMY_STORAGE_DIR) / "trade_balances"
        self.raw = self.storage_dir / "raw"
        self.processed = self.storage_dir / "processed"
        self.data_path = self.processed / "trade_balances.csv"
        self.metadata_path = self.processed / "metadata.json"
        self.reporters_path = self.processed / "individual_reporters.json"
        self.client = client
        self._individual_reporters: dict[str, dict[str, str]] | None = None

    def load(self, force_refresh: bool = False) -> dict[str, Any]:
        if not force_refresh and _is_fresh(self.data_path) and self.metadata_path.exists():
            metadata = _read_json(self.metadata_path)
            if int(metadata.get("dataset_version", 0)) >= self.DATASET_VERSION:
                return self._result("cached")
        errors: list[str] = []
        candidates: list[tuple[pd.DataFrame, str, dict[str, Any]]] = []
        if os.environ.get("WTO_API_KEY"):
            try:
                candidates.append((self._download_api(), "WTO Timeseries API", {
                    "provider": "World Trade Organization", "source_url": WTO_API_BASE_URL,
                    "dataset_codes": [self.EXPORT_CODE, self.IMPORT_CODE],
                }))
            except Exception as exc:
                errors.append(f"WTO API: {exc}")
        else:
            errors.append("WTO_API_KEY is not configured")
        for path in sorted(self.raw.glob("*"), key=lambda item: item.stat().st_mtime, reverse=True):
            if path.suffix.lower() not in {".csv", ".xls", ".xlsx"}:
                continue
            try:
                candidates.append((self._normalise_official_table(_read_table(path)), f"WTO file: {path.name}", {
                    "provider": "World Trade Organization", "source_document": path.name,
                    "checksum_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }))
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
        for frame, source, provenance in candidates:
            try:
                validated, metadata = self.validate(frame)
                metadata.update(provenance)
                metadata.update({"source": source, "retrieved_at_utc": _utc_now().isoformat()})
                _write_frame(self.data_path, validated)
                _write_json(self.metadata_path, metadata)
                return {"data": validated, "metadata": metadata, "status": "current"}
            except Exception as exc:
                errors.append(f"{source}: {exc}")
        if self.data_path.exists() and self.metadata_path.exists():
            result = self._result("cached")
            if errors:
                result["warning"] = "Refresh failed; showing the last validated WTO year. " + " | ".join(errors[-3:])
            return result
        return {
            "data": pd.DataFrame(), "metadata": {}, "status": "unavailable",
            "warning": "WTO trade balances are incomplete. Configure WTO_API_KEY or add an official CSV/XLSX to "
                       f"{self.raw}." + ((" " + " | ".join(errors[-3:])) if errors else ""),
        }

    def _result(self, status: str) -> dict[str, Any]:
        return {"data": pd.read_csv(self.data_path), "metadata": _read_json(self.metadata_path), "status": status}

    def _download_api(self) -> pd.DataFrame:
        key = os.environ["WTO_API_KEY"]
        own_client = self.client is None
        client = self.client or httpx.Client(timeout=60.0, follow_redirects=True)
        frames = []
        last_completed_year = _utc_now().year - 1
        try:
            headers = {"Ocp-Apim-Subscription-Key": key}
            individual_reporters = self._download_individual_reporters(client, headers)
            individual_codes = set(individual_reporters)
            for code, field in ((self.EXPORT_CODE, "exports_usd_mn"), (self.IMPORT_CODE, "imports_usd_mn")):
                response = client.get(
                    f"{WTO_API_BASE_URL.rstrip('/')}/data",
                    params={
                        "i": code,
                        "r": "all",
                        "p": self.WORLD_PARTNER_CODE,
                        "ps": f"2000-{last_completed_year}",
                        "pc": self.TOTAL_MERCHANDISE_CODE,
                        "fmt": "json",
                        "mode": "full",
                        "head": "M",
                        "lang": 1,
                        "meta": "false",
                        "off": 0,
                        "max": self.API_MAX_RECORDS,
                    },
                    headers=headers,
                )
                response.raise_for_status()
                try:
                    payload = response.json()
                except UnicodeDecodeError:
                    payload = json.loads(response.content.decode("cp1252"))
                if isinstance(payload, list):
                    records = payload
                elif isinstance(payload, dict):
                    records = payload.get("Dataset") or payload.get("dataset") or payload.get("data") or payload
                else:
                    raise ValueError(f"Unexpected WTO API payload type for {code}: {type(payload).__name__}")
                part = pd.DataFrame(records)
                if part.empty:
                    raise ValueError(f"No records for {code}")
                if len(part) >= self.API_MAX_RECORDS:
                    raise ValueError(
                        f"WTO response for {code} reached the {self.API_MAX_RECORDS:,}-record API limit"
                    )
                year_col = _find_column(part.columns, ["Year", "PeriodCode", "period"])
                economy_col = _find_column(part.columns, ["ReportingEconomy", "Reporter", "economy"])
                code_col = _find_column(part.columns, ["ReportingEconomyCode", "ReporterCode", "iso3"])
                value_col = _find_column(part.columns, ["Value", "value"])
                if year_col is None or economy_col is None or value_col is None:
                    raise ValueError(f"Unexpected WTO API schema for {code}")
                normal = pd.DataFrame({
                    "year": pd.to_numeric(part[year_col], errors="coerce"),
                    "economy": part[economy_col].astype(str).str.strip(),
                    "economy_code": part[code_col].astype(str).str.strip() if code_col is not None else "",
                    field: pd.to_numeric(part[value_col], errors="coerce"),
                })
                normal["economy_code"] = normal["economy_code"].map(self._normalise_economy_code)
                normal = normal.loc[normal["economy_code"].isin(individual_codes | {"0"})].copy()
                frames.append(normal)
        finally:
            if own_client:
                client.close()
        merged = frames[0].merge(frames[1], on=["year", "economy", "economy_code"], how="outer")
        merged["is_individual_economy"] = merged["economy_code"].isin(individual_codes)
        return merged

    def _download_individual_reporters(
        self, client: httpx.Client, headers: dict[str, str]
    ) -> dict[str, dict[str, str]]:
        response = client.get(
            f"{WTO_API_BASE_URL.rstrip('/')}/reporters",
            params={"ig": "individual", "lang": 1},
            headers=headers,
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except UnicodeDecodeError:
            payload = json.loads(response.content.decode("cp1252"))
        records = payload if isinstance(payload, list) else payload.get("data", []) if isinstance(payload, dict) else []
        reporters: dict[str, dict[str, str]] = {}
        for item in records:
            code = self._normalise_economy_code(item.get("code"))
            if not code:
                continue
            reporters[code] = {
                "iso3": str(item.get("iso3A") or "").strip().upper(),
                "name": str(item.get("name") or "").strip(),
            }
        if len(reporters) < 150:
            raise ValueError(f"WTO individual-reporter reference contains only {len(reporters)} economies")
        self._individual_reporters = reporters
        _write_json(self.reporters_path, {
            "retrieved_at_utc": _utc_now().isoformat(),
            "source": f"{WTO_API_BASE_URL.rstrip('/')}/reporters?ig=individual",
            "reporters": reporters,
        })
        return reporters

    @staticmethod
    def _normalise_economy_code(value: object) -> str:
        text = str(value or "").strip().upper()
        if re.fullmatch(r"\d+(?:\.0+)?", text):
            return str(int(float(text)))
        return text

    def _load_individual_reporter_codes(self) -> set[str]:
        if self._individual_reporters is not None:
            return set(self._individual_reporters)
        if not self.reporters_path.exists():
            return set()
        payload = _read_json(self.reporters_path)
        reporters = payload.get("reporters", {})
        if not isinstance(reporters, dict):
            return set()
        self._individual_reporters = {
            self._normalise_economy_code(code): value for code, value in reporters.items()
        }
        return set(self._individual_reporters)

    @classmethod
    def _aggregate_name_mask(cls, names: pd.Series) -> pd.Series:
        normalized = names.astype(str).str.strip().str.lower()
        return normalized.isin(cls.AGGREGATE_NAMES) | normalized.str.contains(
            cls.AGGREGATE_NAME_PATTERN, na=False
        )

    def _normalise_official_table(self, frame: pd.DataFrame) -> pd.DataFrame:
        year_col = _find_column(frame.columns, ["year", "period", "period_year"])
        economy_col = _find_column(frame.columns, ["economy", "reporting economy", "reporter", "country", "economy_name"])
        code_col = _find_column(frame.columns, ["economy_code", "reporter_code", "reporting economy code", "iso3"])
        export_col = _find_column(frame.columns, ["exports_usd_mn", "exports million usd", "merchandise exports", "export value"])
        import_col = _find_column(frame.columns, ["imports_usd_mn", "imports million usd", "merchandise imports", "import value"])
        if year_col is not None and economy_col is not None and export_col is not None and import_col is not None:
            return pd.DataFrame({
                "year": pd.to_numeric(frame[year_col], errors="coerce"),
                "economy": frame[economy_col].astype(str).str.strip(),
                "economy_code": frame[code_col].astype(str).str.strip() if code_col is not None else "",
                "exports_usd_mn": pd.to_numeric(frame[export_col], errors="coerce"),
                "imports_usd_mn": pd.to_numeric(frame[import_col], errors="coerce"),
            })
        indicator_col = _find_column(frame.columns, ["indicator", "flow", "trade_flow"])
        value_col = _find_column(frame.columns, ["value", "trade_value", "value_usd_mn"])
        if None in (year_col, economy_col, indicator_col, value_col):
            raise ValueError("Required year/economy/export/import columns were not found")
        long = pd.DataFrame({
            "year": pd.to_numeric(frame[year_col], errors="coerce"),
            "economy": frame[economy_col].astype(str).str.strip(),
            "economy_code": frame[code_col].astype(str).str.strip() if code_col is not None else "",
            "flow": frame[indicator_col].astype(str).str.lower(),
            "value": pd.to_numeric(frame[value_col], errors="coerce"),
        })
        long["flow"] = np.where(long["flow"].str.contains("export"), "exports_usd_mn", np.where(long["flow"].str.contains("import"), "imports_usd_mn", ""))
        long = long.loc[long["flow"].ne("")]
        return long.pivot_table(index=["year", "economy", "economy_code"], columns="flow", values="value", aggfunc="first").reset_index()

    def validate(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        required = {"year", "economy", "exports_usd_mn", "imports_usd_mn"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
        clean = frame.copy()
        clean["year"] = pd.to_numeric(clean["year"], errors="coerce")
        clean["exports_usd_mn"] = pd.to_numeric(clean["exports_usd_mn"], errors="coerce")
        clean["imports_usd_mn"] = pd.to_numeric(clean["imports_usd_mn"], errors="coerce")
        clean["economy"] = clean["economy"].astype(str).str.strip()
        if "economy_code" not in clean:
            clean["economy_code"] = ""
        clean["economy_code"] = clean["economy_code"].map(self._normalise_economy_code)
        clean = clean.dropna(subset=["year", "exports_usd_mn", "imports_usd_mn"])
        clean = clean.loc[clean["year"].le(_utc_now().year - 1)]
        lower_name = clean["economy"].str.lower()
        world = clean.loc[lower_name.isin({"world", "total", "all economies"}) | clean["economy_code"].isin({"000", "0"})].copy()
        economies = clean.loc[~clean.index.isin(world.index)].copy()
        classification_source = "aggregate-label fallback"
        if "is_individual_economy" in economies:
            marker = economies["is_individual_economy"]
            if marker.dtype != bool:
                marker = marker.astype(str).str.lower().isin({"true", "1", "yes"})
            economies = economies.loc[marker].copy()
            classification_source = "WTO /reporters?ig=individual"
        else:
            individual_codes = self._load_individual_reporter_codes()
            coded = economies["economy_code"].ne("")
            if individual_codes and coded.any():
                economies = economies.loc[~coded | economies["economy_code"].isin(individual_codes)].copy()
                classification_source = "cached WTO /reporters?ig=individual"
            economies = economies.loc[~self._aggregate_name_mask(economies["economy"])].copy()
        key = np.where(economies["economy_code"].ne(""), economies["economy_code"], economies["economy"].str.lower())
        economies["_key"] = key
        if economies.duplicated(["year", "_key"]).any():
            raise ValueError("Duplicate economy-year rows detected")

        valid_years: list[int] = []
        checks: dict[int, dict[str, Any]] = {}
        previous_cohort: set[str] | None = None
        for year in sorted(economies["year"].dropna().astype(int).unique()):
            year = int(year)
            current = economies.loc[economies["year"].eq(year)]
            total_row = world.loc[world["year"].eq(year)]
            world_exports = float(total_row["exports_usd_mn"].iloc[0]) if not total_row.empty else np.nan
            world_imports = float(total_row["imports_usd_mn"].iloc[0]) if not total_row.empty else np.nan
            export_coverage = current["exports_usd_mn"].sum() / world_exports if world_exports > 0 else np.nan
            import_coverage = current["imports_usd_mn"].sum() / world_imports if world_imports > 0 else np.nan
            count_ok = len(current) >= 150
            coverage_ok = export_coverage >= 0.95 and import_coverage >= 0.95
            cohort_ok = previous_cohort is None or previous_cohort.issubset(set(current["_key"]))
            is_valid = bool(count_ok and coverage_ok and cohort_ok)
            checks[year] = {
                "economies": int(len(current)), "export_coverage_pct": export_coverage * 100,
                "import_coverage_pct": import_coverage * 100, "prior_cohort_present": cohort_ok,
                "valid": is_valid,
            }
            if is_valid:
                valid_years.append(year)
            turnover = current.assign(_turnover=current["exports_usd_mn"] + current["imports_usd_mn"])
            balance = current.assign(_balance=current["exports_usd_mn"] - current["imports_usd_mn"])
            if is_valid:
                previous_cohort = set(turnover.nlargest(30, "_turnover")["_key"]) | set(balance.nlargest(10, "_balance")["_key"]) | set(balance.nsmallest(10, "_balance")["_key"])
        if not valid_years:
            raise ValueError("No completed year passed the 150-economy and 95% WTO coverage checks")
        selected_year = max(valid_years)
        ten_year_target = selected_year - 10
        ten_year = ten_year_target if ten_year_target in valid_years else max((year for year in valid_years if year <= ten_year_target), default=None)
        if ten_year is None:
            ten_year = min(valid_years)
        output = economies.drop(columns="_key").copy()
        output["trade_balance_usd_bn"] = (output["exports_usd_mn"] - output["imports_usd_mn"]) / 1000.0
        metadata = {
            "latest_valid_year": selected_year, "ten_year_comparison_year": int(ten_year),
            "validation": checks, "valid_years": valid_years,
            "definition": "Merchandise exports minus merchandise imports, USD billions",
            "unit": "USD million (source); USD billion (derived balance)", "currency": "USD",
            "observation_period": f"{int(output['year'].min())}-{int(output['year'].max())}",
            "economy_classification": classification_source,
            "validation_status": "valid", "dataset_version": self.DATASET_VERSION,
        }
        return output.reset_index(drop=True), metadata

    @staticmethod
    def top_balances(frame: pd.DataFrame, year: int) -> pd.DataFrame:
        selected = frame.loc[pd.to_numeric(frame["year"], errors="coerce").eq(year)].copy()
        selected = selected.loc[~TradeBalanceService._aggregate_name_mask(selected["economy"])].copy()
        if "is_individual_economy" in selected:
            marker = selected["is_individual_economy"]
            if marker.dtype != bool:
                marker = marker.astype(str).str.lower().isin({"true", "1", "yes"})
            selected = selected.loc[marker].copy()
        selected["trade_balance_usd_bn"] = pd.to_numeric(selected["trade_balance_usd_bn"], errors="coerce")
        positive = selected.loc[selected["trade_balance_usd_bn"].gt(0)].nlargest(5, "trade_balance_usd_bn")
        negative = selected.loc[selected["trade_balance_usd_bn"].lt(0)].nsmallest(5, "trade_balance_usd_bn")
        return pd.concat([positive, negative]).sort_values("trade_balance_usd_bn", ascending=True).reset_index(drop=True)


class ChinaExportGeographyService:
    """Historical seed plus validated GACC destination data."""

    REGION_ALIASES = {
        "asean plus": "ASEAN Plus", "asean": "ASEAN Plus", "eu": "EU", "european union": "EU",
        "us": "US", "usa": "US", "united states": "US", "americas ex us": "Americas ex US",
        "cis": "CIS", "middle east": "Middle East", "africa": "Africa", "other": "Other",
        "hong kong": "Hong Kong", "hong kong sar": "Hong Kong",
    }
    ASEAN_PLUS = {"BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "VNM", "JPN", "KOR", "AUS", "NZL", "IND"}
    EU27 = {"AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"}
    CIS = {"ARM", "AZE", "BLR", "KAZ", "KGZ", "MDA", "RUS", "TJK", "TKM", "UZB"}
    MIDDLE_EAST = {"BHR", "IRN", "IRQ", "ISR", "JOR", "KWT", "LBN", "OMN", "PSE", "QAT", "SAU", "SYR", "TUR", "ARE", "YEM"}
    AFRICA = {"DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", "COM", "COD", "COG", "CIV", "DJI", "EGY", "GNQ", "ERI", "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR", "LBY", "MDG", "MWI", "MLI", "MRT", "MUS", "MAR", "MOZ", "NAM", "NER", "NGA", "RWA", "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO", "TUN", "UGA", "ZMB", "ZWE"}
    AMERICAS = {"ARG", "BOL", "BRA", "CAN", "CHL", "COL", "ECU", "GUY", "PRY", "PER", "SUR", "URY", "VEN", "MEX", "BLZ", "CRI", "SLV", "GTM", "HND", "NIC", "PAN", "ATG", "BHS", "BRB", "CUB", "DMA", "DOM", "GRD", "HTI", "JAM", "KNA", "LCA", "VCT", "TTO"}

    def __init__(self, storage_dir: Path | None = None, seed_path: Path | None = None, client: httpx.Client | None = None) -> None:
        self.storage_dir = Path(storage_dir or WORLD_ECONOMY_STORAGE_DIR) / "china_exports"
        self.historical_dir = self.storage_dir / "historical"
        self.raw = self.storage_dir / "raw"
        self.processed = self.storage_dir / "processed"
        self.metadata_dir = self.storage_dir / "metadata"
        self.seed_path = Path(seed_path or Path(__file__).with_name("data") / "china_export_regions_2000_2010_2020_2025_full.csv")
        self.client = client
        self.history_path = self.historical_dir / "regional_history.csv"
        self.historical_metadata_path = self.historical_dir / "metadata.json"
        self.dynamic_path = self.processed / "regional_exports.csv"
        self.country_path = self.processed / "country_exports.csv"
        self.metadata_path = self.metadata_dir / "latest.json"
        self.check_path = self.metadata_dir / "last_check.json"

    def load(self, force_refresh: bool = False) -> dict[str, Any]:
        historical = self._historical()
        warning = None
        refreshed = False
        cached_dynamic = self.dynamic_path.exists() and not force_refresh and _is_fresh(self.dynamic_path)
        checked_recently = not force_refresh and _is_fresh(self.check_path)
        if not cached_dynamic and not checked_recently:
            try:
                refreshed = self._refresh_from_raw()
                _write_json(self.check_path, {
                    "checked_at_utc": _utc_now().isoformat(),
                    "status": "updated" if refreshed else "no_official_file_found",
                })
            except Exception as exc:
                warning = f"GACC refresh failed; the previous valid version was retained: {exc}"
                _write_json(self.check_path, {"checked_at_utc": _utc_now().isoformat(), "status": "failed", "error": str(exc)})
        dynamic = pd.read_csv(self.dynamic_path) if self.dynamic_path.exists() else pd.DataFrame(columns=historical.columns)
        metadata = _read_json(self.metadata_path) if self.metadata_path.exists() else {}
        combined = historical.copy() if dynamic.empty else pd.concat([historical, dynamic], ignore_index=True)
        combined = combined.sort_values(["year", "region"]).drop_duplicates(["year", "region"], keep="last")
        fallback_year = 2025 if 2025 in set(pd.to_numeric(historical["year"], errors="coerce")) else None
        dynamic_latest = int(pd.to_numeric(dynamic["year"], errors="coerce").max()) if not dynamic.empty else None
        use_fallback = fallback_year is not None and (dynamic_latest is None or dynamic_latest < fallback_year)
        if use_fallback:
            metadata = {
                **metadata,
                "latest_valid_year": fallback_year,
                "source": "Supplied historical table (2025 fallback)",
                "latest_data_is_fallback": True,
            }
        return {
            "data": combined.reset_index(drop=True), "metadata": metadata,
            "status": "fallback" if use_fallback else ("current" if refreshed else "cached"),
            "warning": warning,
        }

    def _historical(self) -> pd.DataFrame:
        if self.history_path.exists():
            cached = pd.read_csv(self.history_path)
            if {2000, 2010, 2020, 2025}.issubset(set(pd.to_numeric(cached["year"], errors="coerce"))):
                return cached
        if not self.seed_path.exists():
            raise FileNotFoundError(f"Historical China export seed is missing: {self.seed_path}")
        source = pd.read_csv(self.seed_path)
        if "Region" not in source.columns or "TOTAL" not in set(source["Region"]):
            raise ValueError("Historical China export seed has an invalid schema")
        rows = []
        for year in (2000, 2010, 2020, 2025):
            value_col = next((column for column in source.columns if str(year) in column and "US$bn" in column), None)
            share_col = next((column for column in source.columns if str(year) in column and "share" in column.lower()), None)
            if value_col is None or share_col is None:
                raise ValueError(f"Historical China export seed is missing {year}")
            for region in REGION_ORDER:
                row = source.loc[source["Region"].eq(region)]
                if row.empty:
                    raise ValueError(f"Historical China export seed is missing {region}")
                source_label = "Supplied historical table (fallback)" if year == 2025 else "Historical research dataset"
                rows.append({"year": year, "region": region, "exports_usd_bn": float(row[value_col].iloc[0]), "share_pct": float(row[share_col].iloc[0]), "source": source_label})
            total = float(source.loc[source["Region"].eq("TOTAL"), value_col].iloc[0])
            if abs(sum(item["exports_usd_bn"] for item in rows if item["year"] == year) - total) > max(0.1, total * 0.001):
                raise ValueError(f"Historical China export values do not reconcile for {year}")
        history = pd.DataFrame(rows)
        _write_frame(self.history_path, history)
        _write_json(self.historical_metadata_path, {
            "provider": "Supplied historical research dataset", "source_document": self.seed_path.name,
            "checksum_sha256": hashlib.sha256(self.seed_path.read_bytes()).hexdigest(),
            "observation_period": "2000, 2010, 2020, 2025", "unit": "USD billion and percent",
            "currency": "USD", "transformation": "Wide regional table normalized to region-year rows",
            "validation_status": "valid", "dataset_version": 1, "imported_at_utc": _utc_now().isoformat(),
        })
        return history

    def _refresh_from_raw(self) -> bool:
        files = [path for path in self.raw.glob("*") if path.suffix.lower() in {".csv", ".xls", ".xlsx"}]
        if not files:
            files = self._download_official_candidates()
        if not files:
            return False
        errors: list[str] = []
        annual_totals = self._annual_usd_totals(files)
        for path in sorted(files, key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                parsed = self._normalise_gacc(_read_table(path), source_name=path.name, annual_totals=annual_totals)
                validated, metadata = self.validate(parsed)
                metadata.update({
                    "provider": "General Administration of Customs of China (GACC)",
                    "source": f"GACC official file: {path.name}", "source_document": path.name,
                    "source_url": GACC_MONTHLY_URL, "retrieved_at_utc": _utc_now().isoformat(),
                    "checksum_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "validation_status": "valid", "dataset_version": 1,
                })
                _write_frame(self.country_path, parsed)
                _write_frame(self.dynamic_path, validated)
                _write_json(self.metadata_path, metadata)
                return True
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
        raise ValueError(" | ".join(errors[-3:]))

    def _annual_usd_totals(self, files: list[Path]) -> dict[int, float]:
        totals: dict[int, float] = {}
        for path in files:
            if not re.search(r"summary|annual|total", path.name, flags=re.I):
                continue
            try:
                frame = _read_table(path)
                year_col = _find_column(frame.columns, ["year", "calendar year", "年份", "年度"])
                value_col = _find_column(frame.columns, ["total_exports_usd_bn", "annual total usd", "exports usd", "export total usd"])
                year_match = re.search(r"(?:19|20)\d{2}", path.name)
                if value_col is None or (year_col is None and year_match is None):
                    continue
                values = self._usd_values_to_bn(frame[value_col], value_col)
                for index, value in values.items():
                    year = int(pd.to_numeric(frame.loc[index, year_col], errors="coerce")) if year_col is not None else int(year_match.group(0))
                    if np.isfinite(value) and value > 0:
                        totals[year] = float(value)
            except Exception:
                continue
        return totals

    def _download_official_candidates(self) -> list[Path]:
        """Best-effort discovery; GACC layout changes must never remove cached valid data."""
        explicit_url = os.environ.get("GACC_EXPORT_URL")
        own_client = self.client is None
        client = self.client or httpx.Client(timeout=35.0, follow_redirects=True)
        try:
            urls: list[str] = []
            if explicit_url:
                urls.append(explicit_url)
            else:
                response = client.get(GACC_MONTHLY_URL)
                response.raise_for_status()
                links = re.findall(r'href=["\']([^"\']+\.(?:xlsx?|csv)(?:\?[^"\']*)?)["\']', response.text, flags=re.I)
                preferred = [link for link in links if re.search(r"country|region|partner|export|import|trade", link, flags=re.I)]
                urls.extend(urljoin(GACC_MONTHLY_URL, link) for link in (preferred or links)[-6:])
            downloaded: list[Path] = []
            for url in urls:
                response = client.get(url)
                response.raise_for_status()
                suffix = Path(urlparse(url).path).suffix.lower()
                if suffix not in {".csv", ".xls", ".xlsx"} or len(response.content) < 100:
                    continue
                target = self.raw / f"gacc_auto_{len(downloaded) + 1}{suffix}"
                _atomic_write(target, response.content)
                downloaded.append(target)
            return downloaded
        except Exception:
            return []
        finally:
            if own_client:
                client.close()

    def _normalise_gacc(
        self, frame: pd.DataFrame, source_name: str = "", annual_totals: dict[int, float] | None = None,
    ) -> pd.DataFrame:
        year_col = _find_column(frame.columns, ["year", "calendar year", "年份", "年度"])
        region_col = _find_column(frame.columns, ["region", "区域"])
        destination_col = _find_column(frame.columns, ["destination", "country", "partner", "贸易伙伴", "国家_地区"])
        iso_col = _find_column(frame.columns, ["iso3", "country_code", "partner_code"])
        usd_col = _find_column(frame.columns, ["exports_usd_bn", "exports usd", "export usd", "usd bn", "export value usd", "美元"])
        cny_col = _find_column(frame.columns, ["exports_cny", "value cny", "人民币"])
        total_usd_col = _find_column(frame.columns, ["total_exports_usd_bn", "annual total usd"])
        year_from_name = re.search(r"(?:19|20)\d{2}", source_name)
        if (year_col is None and year_from_name is None) or (region_col is None and destination_col is None) or (usd_col is None and cny_col is None):
            raise ValueError("Required year/destination/value columns were not found")
        parsed = pd.DataFrame({
            "year": pd.to_numeric(frame[year_col], errors="coerce") if year_col is not None else int(year_from_name.group(0)),
        }, index=frame.index)
        parsed["region"] = frame[region_col].astype(str).str.strip() if region_col is not None else None
        parsed["destination"] = frame[destination_col].astype(str).str.strip() if destination_col is not None else None
        parsed["iso3"] = frame[iso_col].astype(str).str.upper().str.strip() if iso_col is not None else ""
        if usd_col is not None:
            parsed["exports_usd_bn"] = self._usd_values_to_bn(frame[usd_col], usd_col)
        else:
            cny = pd.to_numeric(frame[cny_col], errors="coerce")
            if total_usd_col is not None:
                total_usd = self._usd_values_to_bn(frame[total_usd_col], total_usd_col)
            elif annual_totals:
                total_usd = parsed["year"].map(annual_totals)
            else:
                raise ValueError("CNY data requires an official annual total in USD")
            parsed["exports_usd_bn"] = cny / cny.groupby(parsed["year"]).transform("sum") * total_usd
        return parsed

    @staticmethod
    def _usd_values_to_bn(values: pd.Series, column: object) -> pd.Series:
        unit_label = _normalise_header(column)
        if "billion" in unit_label or "bn" in unit_label:
            divisor = 1.0
        elif "million" in unit_label or "mn" in unit_label:
            divisor = 1_000.0
        elif "thousand" in unit_label:
            divisor = 1_000_000.0
        else:
            raise ValueError("USD export column must declare billion, million, or thousand units")
        return pd.to_numeric(values, errors="coerce") / divisor

    def validate(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        clean = frame.copy().dropna(subset=["year", "exports_usd_bn"])
        clean["year"] = pd.to_numeric(clean["year"], errors="raise").astype(int)
        clean = clean.loc[clean["year"].le(_utc_now().year - 1)]
        if clean.empty:
            raise ValueError("No completed calendar year is present")
        clean["exports_usd_bn"] = pd.to_numeric(clean["exports_usd_bn"], errors="raise")
        if (clean["exports_usd_bn"] < 0).any():
            raise ValueError("Negative exports are not valid")
        if clean["region"].notna().any():
            clean["region"] = clean["region"].astype(str).str.strip().str.lower().map(self.REGION_ALIASES)
            if clean["region"].isna().any():
                raise ValueError("Unknown regional labels are present")
        else:
            aggregate_labels = {
                "world", "total", "asia", "europe", "asean", "european union", "eu",
                "africa", "latin america", "north america", "oceania", "china", "mainland china",
            }
            clean = clean.loc[~clean["destination"].astype(str).str.lower().str.strip().isin(aggregate_labels)].copy()
            clean["region"] = [self.classify_destination(code, name) for code, name in zip(clean["iso3"], clean["destination"])]
        if clean.duplicated(["year", "iso3"]).any() and clean["iso3"].ne("").any():
            raise ValueError("Duplicate destination-year rows detected")
        grouped = clean.groupby(["year", "region"], as_index=False)["exports_usd_bn"].sum()
        years = []
        output = []
        for year, period in grouped.groupby("year"):
            values = period.set_index("region")["exports_usd_bn"].reindex(REGION_ORDER, fill_value=0.0)
            total = float(values.sum())
            if total <= 0:
                continue
            if (values.eq(0).sum() > 3):
                raise ValueError(f"Too many empty regional buckets in {year}")
            years.append(int(year))
            for region, value in values.items():
                output.append({"year": int(year), "region": region, "exports_usd_bn": float(value), "share_pct": float(value / total * 100.0), "source": "GACC"})
        if not years:
            raise ValueError("No year passed GACC validation")
        latest = max(years)
        return pd.DataFrame(output), {"latest_valid_year": latest, "valid_years": years, "region_order": REGION_ORDER}

    @classmethod
    def classify_destination(cls, iso3: object, name: object = "") -> str:
        code = str(iso3 or "").upper().strip()
        label = str(name or "").lower().strip()
        if code in {"HKG", "344"} or label in {"hong kong", "hong kong sar", "中国香港", "香港"}:
            return "Hong Kong"
        if code == "USA" or label in {"united states", "usa", "us", "美国"}:
            return "US"
        if code in cls.ASEAN_PLUS:
            return "ASEAN Plus"
        if code in cls.EU27:
            return "EU"
        if code in cls.CIS:
            return "CIS"
        if code in cls.MIDDLE_EAST:
            return "Middle East"
        if code in cls.AFRICA:
            return "Africa"
        if code in cls.AMERICAS:
            return "Americas ex US"
        return "Other"


def load_world_trade_module(force_refresh: bool = False, storage_dir: Path | None = None) -> dict[str, Any]:
    """Load all three independent services; one unavailable source never blocks the others."""
    return {
        "global_trade": GlobalTradeService(storage_dir).load(force_refresh),
        "trade_balances": TradeBalanceService(storage_dir).load(force_refresh),
        "china_exports": ChinaExportGeographyService(storage_dir).load(force_refresh),
    }
