from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd


WORLD_GDP_TTL_SECONDS = 30 * 24 * 60 * 60
WORLD_BANK_API = "https://api.worldbank.org/v2"
WORLD_ECONOMY_STORAGE_DIR = Path(
    os.environ.get(
        "WORLD_ECONOMY_STORAGE_DIR",
        str(Path(__file__).with_name("persistent") / "world_economy"),
    )
)


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


class WorldGDPService:
    """Download, validate, persist, and transform World Bank WDI GDP data."""

    INDICATORS = {
        "nominal_gdp_usd": "NY.GDP.MKTP.CD",
        "gdp_ppp_intl_usd": "NY.GDP.MKTP.PP.CD",
        "gdp_per_capita_usd": "NY.GDP.PCAP.CD",
    }
    METRIC_COLUMNS = {
        "nominal": "nominal_gdp_usd",
        "ppp": "gdp_ppp_intl_usd",
    }
    METRIC_START_YEARS = {"nominal": 1960, "ppp": 1990}
    MIN_ECONOMIES = 150

    def __init__(self, storage_dir: Path | None = None, client: httpx.Client | None = None) -> None:
        self.storage_dir = Path(storage_dir or WORLD_ECONOMY_STORAGE_DIR) / "world_gdp"
        self.raw_dir = self.storage_dir / "raw"
        self.processed_dir = self.storage_dir / "processed"
        self.metadata_dir = self.storage_dir / "metadata"
        self.manifest_path = self.metadata_dir / "latest.json"
        self.client = client

    def load(self, force_refresh: bool = False) -> dict[str, Any]:
        if not force_refresh and self._cache_is_fresh():
            return self._cached(status="cached")
        try:
            frame, metadata, raw_payloads = self._download()
            self._save_snapshot(frame, metadata, raw_payloads)
            return {"data": frame, "metadata": metadata, "status": "current"}
        except Exception as exc:
            if self.manifest_path.exists():
                cached = self._cached(status="cached")
                cached["warning"] = f"World Bank refresh failed; showing the last valid snapshot: {exc}"
                return cached
            return {
                "data": pd.DataFrame(),
                "metadata": {},
                "status": "unavailable",
                "warning": f"World GDP data is unavailable: {exc}",
            }

    def _cache_is_fresh(self) -> bool:
        return (
            self.manifest_path.exists()
            and (_utc_now().timestamp() - self.manifest_path.stat().st_mtime) < WORLD_GDP_TTL_SECONDS
        )

    def _cached(self, *, status: str) -> dict[str, Any]:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        data_path = self.processed_dir / manifest["dataset_file"]
        metadata_path = self.metadata_dir / manifest["metadata_file"]
        payload = data_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != manifest["sha256"]:
            raise ValueError("The persisted World GDP snapshot failed its integrity check")
        frame = self._coerce_frame(pd.read_csv(data_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {"data": frame, "metadata": metadata, "status": status}

    def _download(self) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
        last_completed_year = _utc_now().year - 1
        own_client = self.client is None
        client = self.client or httpx.Client(timeout=60.0, follow_redirects=True)
        raw_payloads: dict[str, Any] = {}
        try:
            metadata_response = client.get(
                f"{WORLD_BANK_API}/country",
                params={"format": "json", "per_page": 500},
            )
            metadata_response.raise_for_status()
            country_payload = metadata_response.json()
            raw_payloads["countries"] = country_payload
            country_metadata = self._normalise_country_metadata(country_payload)

            series_frames: list[pd.DataFrame] = []
            source_last_updated: dict[str, str | None] = {}
            for column, indicator in self.INDICATORS.items():
                response = client.get(
                    f"{WORLD_BANK_API}/country/all/indicator/{indicator}",
                    params={
                        "format": "json",
                        "per_page": 20000,
                        "date": f"1960:{last_completed_year}",
                    },
                )
                response.raise_for_status()
                payload = response.json()
                raw_payloads[indicator] = payload
                series_frames.append(self._normalise_indicator(payload, column))
                header = payload[0] if isinstance(payload, list) and payload else {}
                source_last_updated[indicator] = header.get("lastupdated") if isinstance(header, dict) else None
        finally:
            if own_client:
                client.close()

        frame = series_frames[0]
        for item in series_frames[1:]:
            frame = frame.merge(item, on=["iso3", "year"], how="outer", validate="one_to_one")
        frame = frame.merge(country_metadata, on="iso3", how="inner", validate="many_to_one")
        frame = self._coerce_frame(frame)
        frame = frame.loc[frame["year"].between(1960, last_completed_year)].copy()
        frame = frame.sort_values(["year", "iso3"]).reset_index(drop=True)

        reference_year, coverage = self.latest_common_year(frame, return_coverage=True)
        self._validate_history(frame, reference_year)
        retrieved_at = _utc_now().isoformat()
        metadata = {
            "provider": "World Bank",
            "source": "World Development Indicators",
            "source_url": "https://databank.worldbank.org/source/world-development-indicators",
            "indicators": self.INDICATORS,
            "original_units": {
                "nominal_gdp_usd": "Current US dollars",
                "gdp_ppp_intl_usd": "Current international dollars",
                "gdp_per_capita_usd": "Current US dollars per person",
            },
            "retrieved_at_utc": retrieved_at,
            "source_last_updated": source_last_updated,
            "reference_year": reference_year,
            "latest_completed_calendar_year": last_completed_year,
            "coverage": coverage,
            "validation_status": "valid",
            "dataset_version": 1,
            "periods": {
                "nominal_gdp_usd": self._period(frame, "nominal_gdp_usd"),
                "gdp_ppp_intl_usd": self._period(frame, "gdp_ppp_intl_usd"),
                "gdp_per_capita_usd": self._period(frame, "gdp_per_capita_usd"),
            },
        }
        return frame, metadata, raw_payloads

    @staticmethod
    def _normalise_country_metadata(payload: Any) -> pd.DataFrame:
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            raise ValueError("World Bank returned no country metadata")
        rows: list[dict[str, Any]] = []
        for item in payload[1]:
            iso3 = str(item.get("id") or "").strip().upper()
            if not iso3:
                continue
            region = item.get("region") or {}
            region_id = str(region.get("id") or "").strip()
            region_name = str(region.get("value") or "").strip()
            rows.append({
                "iso3": iso3,
                "country_name": str(item.get("name") or iso3).strip(),
                "is_country": bool(region_id and region_id != "NA" and region_name),
                "region_name": region_name,
            })
        frame = pd.DataFrame(rows).drop_duplicates("iso3", keep="last")
        if "WLD" not in set(frame["iso3"]):
            raise ValueError("World Bank metadata does not contain the WLD aggregate")
        return frame

    @staticmethod
    def _normalise_indicator(payload: Any, column: str) -> pd.DataFrame:
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            raise ValueError(f"World Bank returned no observations for {column}")
        rows: list[dict[str, Any]] = []
        for item in payload[1]:
            iso3 = str(item.get("countryiso3code") or "").strip().upper()
            year = pd.to_numeric(item.get("date"), errors="coerce")
            value = pd.to_numeric(item.get("value"), errors="coerce")
            if iso3 and pd.notna(year):
                rows.append({"iso3": iso3, "year": int(year), column: value})
        if not rows:
            raise ValueError(f"World Bank observations could not be normalized for {column}")
        return pd.DataFrame(rows).drop_duplicates(["iso3", "year"], keep="last")

    @staticmethod
    def _coerce_frame(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["iso3"] = result["iso3"].astype(str).str.upper().str.strip()
        result["year"] = pd.to_numeric(result["year"], errors="raise").astype(int)
        for column in WorldGDPService.INDICATORS:
            if column not in result:
                result[column] = np.nan
            result[column] = pd.to_numeric(result[column], errors="coerce")
        if "is_country" in result:
            if result["is_country"].dtype != bool:
                result["is_country"] = result["is_country"].astype(str).str.lower().isin({"true", "1"})
        else:
            result["is_country"] = result["iso3"].ne("WLD")
        return result

    @classmethod
    def latest_common_year(
        cls, frame: pd.DataFrame, *, return_coverage: bool = False
    ) -> int | tuple[int, dict[str, int]]:
        data = cls._coerce_frame(frame)
        last_completed_year = _utc_now().year - 1
        candidates = sorted(
            (int(year) for year in data.loc[data["year"].le(last_completed_year), "year"].unique()),
            reverse=True,
        )
        for year in candidates:
            annual = data.loc[data["year"].eq(year)]
            world = annual.loc[annual["iso3"].eq("WLD")]
            if world.empty or world[list(cls.INDICATORS)].iloc[0].isna().any():
                continue
            countries = annual.loc[annual["is_country"]]
            coverage = {column: int(countries[column].notna().sum()) for column in cls.INDICATORS}
            if any(count < cls.MIN_ECONOMIES for count in coverage.values()):
                continue
            top10 = countries.dropna(subset=["nominal_gdp_usd"]).nlargest(10, "nominal_gdp_usd")
            if len(top10) != 10 or top10[list(cls.INDICATORS)].isna().any().any():
                continue
            return (year, coverage) if return_coverage else year
        raise ValueError(
            "No completed year passed the WLD, 150-economy, and top-10 common-series coverage checks"
        )

    @classmethod
    def _validate_history(cls, frame: pd.DataFrame, reference_year: int) -> None:
        data = cls._coerce_frame(frame)
        world = data.loc[data["iso3"].eq("WLD")]
        minimum_years = {
            "nominal_gdp_usd": 1960,
            "gdp_ppp_intl_usd": 1990,
            "gdp_per_capita_usd": 1960,
        }
        for column, expected_start in minimum_years.items():
            observed = world.loc[world[column].notna() & world["year"].le(reference_year), "year"]
            if observed.empty or int(observed.min()) > expected_start:
                raise ValueError(f"{column} does not provide the required history from {expected_start}")
        for metric in cls.METRIC_COLUMNS:
            cls.build_additive_history(data, metric=metric, reference_year=reference_year)

    @staticmethod
    def _period(frame: pd.DataFrame, column: str) -> str | None:
        observed = frame.loc[frame[column].notna(), "year"]
        if observed.empty:
            return None
        return f"{int(observed.min())}-{int(observed.max())}"

    def _save_snapshot(
        self, frame: pd.DataFrame, metadata: dict[str, Any], raw_payloads: dict[str, Any]
    ) -> None:
        csv_payload = frame.to_csv(index=False).encode("utf-8")
        digest = hashlib.sha256(csv_payload).hexdigest()
        short = digest[:16]
        dataset_name = f"world_gdp_{short}.csv"
        metadata_name = f"snapshot_{short}.json"
        metadata = dict(metadata)
        metadata["sha256"] = digest
        metadata["dataset_file"] = dataset_name
        _atomic_write(self.processed_dir / dataset_name, csv_payload)
        _atomic_write(
            self.metadata_dir / metadata_name,
            json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        for name, payload in raw_payloads.items():
            safe_name = name.replace(".", "_")
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            _atomic_write(self.raw_dir / f"{safe_name}.json.gz", gzip.compress(encoded))
        manifest = {
            "dataset_file": dataset_name,
            "metadata_file": metadata_name,
            "sha256": digest,
            "activated_at_utc": _utc_now().isoformat(),
        }
        _atomic_write(
            self.manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    @classmethod
    def build_top10_table(
        cls, frame: pd.DataFrame, *, reference_year: int | None = None
    ) -> pd.DataFrame:
        data = cls._coerce_frame(frame)
        year = int(reference_year if reference_year is not None else cls.latest_common_year(data))
        current = data.loc[data["year"].eq(year) & data["is_country"]].copy()
        current = current.dropna(subset=["nominal_gdp_usd"]).nlargest(10, "nominal_gdp_usd").copy()
        current["rank"] = np.arange(1, len(current) + 1)
        base = (
            data.loc[data["year"].eq(year - 10) & data["is_country"], ["iso3", *cls.INDICATORS]]
            .drop_duplicates("iso3", keep="last")
            .set_index("iso3")
        )
        for column in cls.INDICATORS:
            base_values = current["iso3"].map(base[column] if column in base else pd.Series(dtype=float))
            valid = current[column].notna() & base_values.notna() & base_values.gt(0)
            current[f"{column}_change_10y_pct"] = np.where(
                valid, (current[column] / base_values - 1.0) * 100.0, np.nan
            )
        current["nominal_gdp_usd_trn"] = current["nominal_gdp_usd"] / 1e12
        current["gdp_ppp_intl_usd_trn"] = current["gdp_ppp_intl_usd"] / 1e12
        return current[[
            "rank", "iso3", "country_name", "nominal_gdp_usd_trn",
            "nominal_gdp_usd_change_10y_pct", "gdp_ppp_intl_usd_trn",
            "gdp_ppp_intl_usd_change_10y_pct", "gdp_per_capita_usd",
            "gdp_per_capita_usd_change_10y_pct",
        ]].reset_index(drop=True)

    @classmethod
    def build_additive_history(
        cls,
        frame: pd.DataFrame,
        *,
        metric: str,
        reference_year: int,
        selected_iso3: str | None = None,
    ) -> pd.DataFrame:
        if metric not in cls.METRIC_COLUMNS:
            raise ValueError(f"Unsupported additive GDP metric: {metric}")
        data = cls._coerce_frame(frame)
        column = cls.METRIC_COLUMNS[metric]
        start_year = cls.METRIC_START_YEARS[metric]
        selected_iso3 = selected_iso3.upper() if selected_iso3 else None
        rows: list[dict[str, Any]] = []
        for year in range(start_year, int(reference_year) + 1):
            annual = data.loc[data["year"].eq(year)]
            world_values = annual.loc[annual["iso3"].eq("WLD"), column].dropna()
            if world_values.empty or float(world_values.iloc[0]) <= 0:
                continue
            world_value = float(world_values.iloc[0])
            ranked = (
                annual.loc[annual["is_country"] & annual[column].notna()]
                .sort_values(column, ascending=False)
                .reset_index(drop=True)
            )
            ranked["rank"] = np.arange(1, len(ranked) + 1)
            top10 = ranked.head(10)
            displayed = top10.copy()
            outside_selected = False
            if selected_iso3 and selected_iso3 not in set(top10["iso3"]):
                selected = ranked.loc[ranked["iso3"].eq(selected_iso3)]
                if not selected.empty:
                    displayed = pd.concat([displayed, selected.iloc[[0]]], ignore_index=True)
                    outside_selected = True
            displayed_total = float(displayed[column].sum())
            other = world_value - displayed_total
            tolerance = max(abs(world_value) * 1e-9, 1.0)
            if other < -tolerance:
                raise ValueError(
                    f"{metric} country values exceed the WLD aggregate materially in {year}: {other:,.0f}"
                )
            if abs(other) <= tolerance:
                other = 0.0
            for item in displayed.itertuples(index=False):
                value = float(getattr(item, column))
                is_selected_outside = bool(
                    outside_selected and selected_iso3 and item.iso3 == selected_iso3 and item.iso3 not in set(top10["iso3"])
                )
                rows.append({
                    "year": year,
                    "iso3": item.iso3,
                    "country_name": item.country_name,
                    "value": value,
                    "share_pct": value / world_value * 100.0,
                    "rank": int(item.rank),
                    "selected_outside_top10": is_selected_outside,
                    "world_value": world_value,
                })
            rows.append({
                "year": year,
                "iso3": "OTHER",
                "country_name": "Other",
                "value": other,
                "share_pct": other / world_value * 100.0,
                "rank": np.nan,
                "selected_outside_top10": False,
                "world_value": world_value,
            })
        result = pd.DataFrame(rows)
        if result.empty:
            raise ValueError(f"No valid {metric} history is available")
        reconciliation = result.groupby("year", as_index=False).agg(
            displayed_value=("value", "sum"), world_value=("world_value", "first"), share_pct=("share_pct", "sum")
        )
        if not np.allclose(reconciliation["displayed_value"], reconciliation["world_value"], rtol=1e-9, atol=1.0):
            raise ValueError(f"{metric} absolute history does not reconcile to WLD")
        if not np.allclose(reconciliation["share_pct"], 100.0, rtol=0, atol=1e-8):
            raise ValueError(f"{metric} share history does not reconcile to 100%")
        return result

    @classmethod
    def build_per_capita_history(
        cls,
        frame: pd.DataFrame,
        *,
        reference_year: int,
        selected_iso3: str | None = None,
    ) -> pd.DataFrame:
        data = cls._coerce_frame(frame)
        data = data.loc[data["year"].between(1960, int(reference_year))].copy()
        if selected_iso3:
            country_codes = [selected_iso3.upper()]
        else:
            latest = data.loc[data["year"].eq(int(reference_year)) & data["is_country"]]
            country_codes = latest.dropna(subset=["nominal_gdp_usd"]).nlargest(10, "nominal_gdp_usd")["iso3"].tolist()
        selected = data.loc[data["iso3"].isin([*country_codes, "WLD"])].copy()
        selected = selected.loc[selected["gdp_per_capita_usd"].notna()]
        world = (
            selected.loc[selected["iso3"].eq("WLD"), ["year", "gdp_per_capita_usd"]]
            .drop_duplicates("year", keep="last")
            .rename(columns={"gdp_per_capita_usd": "world_gdp_per_capita_usd"})
        )
        selected = selected.merge(world, on="year", how="left", validate="many_to_one")
        selected["relative_world_pct"] = (
            selected["gdp_per_capita_usd"] / selected["world_gdp_per_capita_usd"] * 100.0
        )
        selected.loc[selected["iso3"].eq("WLD"), "country_name"] = "World average"
        return selected[[
            "year", "iso3", "country_name", "gdp_per_capita_usd",
            "world_gdp_per_capita_usd", "relative_world_pct",
        ]].sort_values(["year", "iso3"]).reset_index(drop=True)


def load_world_gdp_data(force_refresh: bool = False) -> dict[str, Any]:
    return WorldGDPService().load(force_refresh=force_refresh)
