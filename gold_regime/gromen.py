from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
from typing import Any, Iterable

import httpx
import numpy as np
import pandas as pd

from fred_client import download_fred_series_batch
from .models import LukeGromenGoldSnapshot


TROY_OZ_PER_TONNE = 32_150.7466
FOREIGN_DEBT_SWITCH_DATE = pd.Timestamp("2003-02-01")
US_GOLD_COMPONENT_SWITCH_DATE = pd.Timestamp("2012-01-01")
HISTORICAL_US_GOLD_OZ = 261_499_000.0
EPS_J = -0.77
EPS_R = 0.44
JEWELLERY_FLOOR_TONNES = 400.0
RECYCLED_CAP_TONNES = 3_000.0
STATIC_TARGET_SHARES = (0.25, 0.50, 0.75, 1.00)
ADAPTIVE_THETAS = (1.00, 0.75, 0.50, 0.35)
BASE_THETA = 0.50
GDP_COVERAGE_MIN = 0.95
PRIOR_SURPLUS_COVERAGE_MIN = 0.95
IMF_WEO_DATASET_URL = "https://data.imf.org/Datasets/WEO"

FOREIGN_DEBT_SERIES = ("FDHBFIN", "FORTREASPOS69995")
US_GOLD_COMPONENTS = (
    "FKKYGTA",
    "DNVCOGTA",
    "WPNYGTA",
    "MHTGWSALQ",
    "FRVGBSAM",
    "FRDGBSAM",
    "FRVGCSAM",
    "FRDGCSAM",
)
WGC_VALUE_COLUMNS = (
    "total_supply_tonnes",
    "total_mine_supply_tonnes",
    "recycled_gold_tonnes",
    "jewellery_fabrication_tonnes",
    "technology_tonnes",
    "investment_tonnes",
    "central_banks_tonnes",
    "otc_and_other_tonnes",
    "lbma_gold_price_usd_oz",
)
WGC_FLOW_COLUMNS = tuple(column for column in WGC_VALUE_COLUMNS if column != "lbma_gold_price_usd_oz")


def build_luke_gromen_snapshot(
    gold_price: pd.Series,
    fred_api_key: str | None,
    cache_dir: Path,
    wgc_data_dir: Path | None = None,
) -> LukeGromenGoldSnapshot:
    """Build the isolated Luke Gromen models without mutating any existing Gold Regime inputs."""
    warnings: list[str] = []
    wgc_dir = wgc_data_dir or Path("data") / "gold_regime"
    wgc_annual, wgc_quarterly = load_wgc_data(wgc_dir)
    wgc_metadata = load_wgc_metadata(wgc_dir)
    wgc_annual = prepare_wgc_annual(wgc_annual)
    wgc_ytd = build_wgc_ytd_display(wgc_quarterly)

    try:
        fred = download_fred_series_batch(
            (*FOREIGN_DEBT_SERIES, *US_GOLD_COMPONENTS),
            api_key=fred_api_key,
            observation_start="1970-01-01",
        )
        us_coverage = build_us_gold_coverage_history(gold_price, fred)
    except Exception as exc:
        us_coverage = pd.DataFrame()
        warnings.append(f"U.S. Gold Coverage data unavailable: {exc}")

    try:
        global_ca = load_imf_global_positive_ca(cache_dir)
        source_warning = global_ca.attrs.get("warning")
        if source_warning:
            warnings.append(str(source_warning))
    except Exception as exc:
        global_ca = pd.DataFrame()
        warnings.append(f"IMF WEO current-account data unavailable: {exc}")

    try:
        world_bank_ca = load_world_bank_positive_ca_crosscheck(cache_dir)
        world_bank_warning = world_bank_ca.attrs.get("warning")
        if world_bank_warning:
            warnings.append(str(world_bank_warning))
        if not global_ca.empty and not world_bank_ca.empty:
            crosscheck = world_bank_ca[["year", "global_positive_ca_usd"]].rename(
                columns={"global_positive_ca_usd": "world_bank_positive_ca_usd"}
            )
            global_ca = global_ca.merge(crosscheck, on="year", how="left")
            global_ca["world_bank_difference_pct"] = (
                global_ca["world_bank_positive_ca_usd"] / global_ca["global_positive_ca_usd"] - 1.0
            )
    except Exception as exc:
        warnings.append(f"World Bank historical cross-check unavailable: {exc}")

    calibration_year = select_calibration_year(wgc_annual, global_ca)
    model2_history = build_model2_history(wgc_annual, global_ca)
    static = pd.DataFrame()
    adaptive = pd.DataFrame()
    convergence = pd.DataFrame()
    spot_gold_price = latest_number(gold_price)
    current: dict[str, Any] = {
        "gold_price": spot_gold_price,
        "gold_price_date": latest_date(gold_price),
        "spot_gold_price": spot_gold_price,
        "gold_price_source": "TradingView TVC:GOLD (weekly)",
        "calibration_year": calibration_year,
        "wgc_latest_completed_year": latest_completed_wgc_year(wgc_annual),
        "wgc_data_as_of": wgc_metadata.get("data_as_of"),
        "wgc_latest_published_quarter": wgc_metadata.get("latest_published_quarter"),
    }

    if not us_coverage.empty:
        valid = us_coverage.dropna(subset=["coverage_ratio", "gold_price", "foreign_debt_usd", "us_gold_oz"])
        if not valid.empty:
            row = valid.iloc[-1]
            current.update(row.to_dict())
            for target in (0.20, 0.30, 0.40, 0.50):
                current[f"required_price_{int(target * 100)}pct"] = required_gold_price(
                    row["foreign_debt_usd"], row["us_gold_oz"], target
                )
        latest_month = us_coverage.iloc[-1]
        if latest_month["date"] >= US_GOLD_COMPONENT_SWITCH_DATE and pd.isna(latest_month["us_gold_oz"]):
            warnings.append("All eight U.S. Treasury gold components are not available for the latest month; no historical fallback was used.")

    if calibration_year is None:
        warnings.append("Model 2 has no year with both completed WGC annual data and valid GlobalPositiveCA coverage.")
    else:
        wgc_row = wgc_annual.loc[wgc_annual["year"] == calibration_year].iloc[-1]
        ca_row = global_ca.loc[global_ca["year"] == calibration_year].iloc[-1]
        static = calculate_static_scenarios(float(ca_row["global_positive_ca_usd"]), float(wgc_row["broad_flow_tonnes"]))
        adaptive = calculate_adaptive_matrix(float(ca_row["global_positive_ca_usd"]), wgc_row)
        current.update(
            {
                "global_positive_ca_usd": float(ca_row["global_positive_ca_usd"]),
                "ca_gdp_coverage": float(ca_row["gdp_coverage"]),
                "ca_prior_surplus_coverage": float(ca_row["prior_year_surplus_coverage"]),
                "ca_available_economy_count": int(ca_row["available_economy_count"]),
                "ca_total_eligible_economy_count": int(ca_row["total_eligible_economy_count"]),
                "ca_data_status": str(ca_row["data_status"]),
                "ca_source": str(ca_row.get("source") or "IMF WEO"),
                "core_mds": float(wgc_row["core_mds"]),
                "broad_mds": float(wgc_row["broad_mds"]),
                "balance_gap_tonnes": float(wgc_row["balance_gap_tonnes"]),
                "balance_tolerance_tonnes": float(wgc_row["balance_tolerance_tonnes"]),
                "balance_valid": bool(wgc_row["balance_valid"]),
                "core_gmar": float(wgc_row["core_flow_tonnes"] * TROY_OZ_PER_TONNE * wgc_row["lbma_gold_price_usd_oz"] / ca_row["global_positive_ca_usd"]),
                "broad_gmar": float(wgc_row["broad_flow_tonnes"] * TROY_OZ_PER_TONNE * wgc_row["lbma_gold_price_usd_oz"] / ca_row["global_positive_ca_usd"]),
            }
        )
        convergence = build_convergence_panel(current["spot_gold_price"], static, adaptive)

    if not model2_history.empty:
        wgc_annual = wgc_annual.merge(
            model2_history[["year", "global_positive_ca_usd", "core_gmar", "broad_gmar"]],
            on="year",
            how="left",
        )

    return LukeGromenGoldSnapshot(
        current=current,
        us_coverage_history=us_coverage,
        wgc_annual=wgc_annual,
        wgc_ytd=wgc_ytd,
        global_ca_history=global_ca,
        static_scenarios=static,
        adaptive_matrix=adaptive,
        convergence=convergence,
        warnings=warnings,
    )


def load_wgc_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_path = data_dir / "wgc_gold_balance_annual.csv"
    quarterly_path = data_dir / "wgc_gold_balance_quarterly.csv"
    if not annual_path.exists() or not quarterly_path.exists():
        raise FileNotFoundError(f"Normalized WGC data not found in {data_dir}")
    annual = pd.read_csv(annual_path)
    quarterly = pd.read_csv(quarterly_path)
    for frame in (annual, quarterly):
        for column in WGC_VALUE_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce").astype("Int64")
    annual["is_complete"] = annual["is_complete"].astype(str).str.lower().eq("true")
    quarterly["year"] = pd.to_numeric(quarterly["year"], errors="coerce").astype("Int64")
    quarterly["quarter"] = pd.to_numeric(quarterly["quarter"], errors="coerce").astype("Int64")
    quarterly["is_published"] = quarterly["is_published"].astype(str).str.lower().eq("true")
    return annual, quarterly


def load_wgc_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "wgc_gold_balance_metadata.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def prepare_wgc_annual(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["core_flow_tonnes"] = out["investment_tonnes"] + out["central_banks_tonnes"]
    out["broad_flow_tonnes"] = out["core_flow_tonnes"] + out["otc_and_other_tonnes"]
    out["core_mds"] = out["core_flow_tonnes"] / out["total_supply_tonnes"]
    out["broad_mds"] = out["broad_flow_tonnes"] / out["total_supply_tonnes"]
    physical_demand = (
        out["jewellery_fabrication_tonnes"]
        + out["technology_tonnes"]
        + out["investment_tonnes"]
        + out["central_banks_tonnes"]
        + out["otc_and_other_tonnes"]
    )
    out["balance_gap_tonnes"] = out["total_supply_tonnes"] - physical_demand
    out["balance_tolerance_tonnes"] = np.maximum(1.0, 0.0002 * out["total_supply_tonnes"].abs())
    out["balance_valid"] = out["balance_gap_tonnes"].abs() <= out["balance_tolerance_tonnes"]
    return out


def build_wgc_ytd_display(quarterly: pd.DataFrame) -> pd.DataFrame:
    published = quarterly.loc[quarterly["is_published"]].copy()
    if published.empty:
        return pd.DataFrame()
    latest_year = int(published["year"].max())
    current = published.loc[published["year"] == latest_year].sort_values("quarter")
    count = int(current["quarter"].nunique())
    if count <= 0 or count >= 4:
        return pd.DataFrame()
    row: dict[str, Any] = {
        "year": latest_year,
        "published_quarters": count,
        "through_period": str(current["period"].iloc[-1]),
        "display_only": True,
    }
    for column in WGC_FLOW_COLUMNS:
        row[column] = float(current[column].sum(min_count=count) * 4.0 / count)
    row["lbma_gold_price_usd_oz"] = float(current["lbma_gold_price_usd_oz"].mean())
    return pd.DataFrame([row])


def build_us_gold_coverage_history(gold_price: pd.Series, fred: pd.DataFrame) -> pd.DataFrame:
    if gold_price is None or gold_price.empty:
        return pd.DataFrame()
    normalized = fred.copy()
    normalized["Series_ID"] = normalized["Series_ID"].astype(str).str.upper()
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    normalized["Value"] = pd.to_numeric(normalized["Value"], errors="coerce")
    normalized = normalized.dropna(subset=["Date"])
    normalized["period"] = normalized["Date"].dt.to_period("M")
    pivot = normalized.pivot_table(index="period", columns="Series_ID", values="Value", aggfunc="last").sort_index()

    price = pd.Series(pd.to_numeric(gold_price, errors="coerce").to_numpy(), index=pd.to_datetime(gold_price.index, errors="coerce"))
    price = price.dropna().sort_index()
    price_monthly = price.groupby(price.index.to_period("M")).last()
    if price_monthly.empty:
        return pd.DataFrame()
    periods = pd.period_range("1970-01", price_monthly.index.max(), freq="M")
    out = pd.DataFrame(index=periods)
    out["gold_price"] = price_monthly.reindex(periods)

    fd = pivot.get("FDHBFIN", pd.Series(dtype="float64")).reindex(periods).ffill() * 1_000_000_000.0
    fort = pivot.get("FORTREASPOS69995", pd.Series(dtype="float64")).reindex(periods).ffill() * 1_000_000.0
    dates = periods.to_timestamp()
    before_debt_switch = dates < FOREIGN_DEBT_SWITCH_DATE
    out["foreign_debt_usd"] = np.where(before_debt_switch, fd, fort)
    out["foreign_debt_source"] = np.where(before_debt_switch, "FDHBFIN", "FORTREASPOS69995")

    components = pivot.reindex(index=periods, columns=list(US_GOLD_COMPONENTS))
    component_sum = components.sum(axis=1, min_count=len(US_GOLD_COMPONENTS))
    before_gold_switch = dates < US_GOLD_COMPONENT_SWITCH_DATE
    out["us_gold_oz"] = np.where(before_gold_switch, HISTORICAL_US_GOLD_OZ, component_sum)
    out["us_gold_source"] = np.where(before_gold_switch, "Historical constant", "8 FRED components")
    out["us_gold_value_usd"] = out["us_gold_oz"] * out["gold_price"]
    out["coverage_ratio"] = out["us_gold_value_usd"] / out["foreign_debt_usd"]
    out.insert(0, "date", dates)
    return out.reset_index(drop=True)


def required_gold_price(foreign_debt_usd: float, us_gold_oz: float, target_coverage: float) -> float:
    if not all(np.isfinite(float(value)) for value in (foreign_debt_usd, us_gold_oz, target_coverage)) or us_gold_oz <= 0:
        return np.nan
    return float(target_coverage * foreign_debt_usd / us_gold_oz)


def load_imf_global_positive_ca(cache_dir: Path, force_refresh: bool = False) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "imf_weo_positive_current_account.csv"
    metadata_path = cache_dir / "imf_weo_positive_current_account_metadata.json"
    max_age_seconds = 7 * 24 * 60 * 60
    if cache_path.exists() and not force_refresh:
        age = datetime.now(timezone.utc).timestamp() - cache_path.stat().st_mtime
        if age <= max_age_seconds:
            return _read_imf_ca_cache(cache_path, metadata_path)
    try:
        frame, metadata = download_imf_weo_current_account()
        frame.to_csv(cache_path, index=False)
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        frame.attrs.update(metadata)
        return frame
    except Exception as exc:
        if cache_path.exists():
            frame = _read_imf_ca_cache(cache_path, metadata_path)
            frame.attrs["warning"] = f"Using stale IMF WEO cache after refresh failure: {exc}"
            return frame
        raise


def discover_imf_weo_workbook_url(client: httpx.Client | None = None) -> str:
    own_client = client is None
    selected_client = client or httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        response = selected_client.get(IMF_WEO_DATASET_URL)
        response.raise_for_status()
        matches = re.findall(
            r"(/-/media/iData/External-Storage/Documents/[^\"'<> ]+/en/WEO[^\"'<> ]+all\.xlsx)",
            response.text,
            flags=re.IGNORECASE,
        )
        if not matches:
            raise RuntimeError("The IMF WEO dataset page did not expose the current Countries workbook.")
        return f"https://data.imf.org{matches[0]}"
    finally:
        if own_client:
            selected_client.close()


def download_imf_weo_current_account(workbook_url: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        url = workbook_url or discover_imf_weo_workbook_url(client)
        response = client.get(url)
        response.raise_for_status()
    countries = pd.read_excel(io.BytesIO(response.content), sheet_name="Countries")
    publication_dates = pd.to_datetime(countries.get("PUBLICATION_DATE"), errors="coerce")
    publication_date = publication_dates.dropna().max()
    publication_year = int(publication_date.year) if pd.notna(publication_date) else datetime.now(timezone.utc).year
    frame = calculate_imf_positive_ca_from_countries(countries, publication_year=publication_year)
    metadata = {
        "source": "IMF World Economic Outlook",
        "source_url": url,
        "publication_date": publication_date.date().isoformat() if pd.notna(publication_date) else None,
        "publication_year": publication_year,
        "bca_indicator": "BCA",
        "gdp_indicator": "NGDPD",
        "countries_sheet_only": True,
    }
    return frame, metadata


def calculate_imf_positive_ca_from_countries(countries: pd.DataFrame, publication_year: int) -> pd.DataFrame:
    required = {"COUNTRY.ID", "COUNTRY", "INDICATOR.ID", "LATEST_ACTUAL_ANNUAL_DATA"}
    missing = sorted(required - set(countries.columns))
    if missing:
        raise ValueError(f"IMF WEO Countries sheet is missing: {', '.join(missing)}")
    year_columns = sorted(column for column in countries.columns if isinstance(column, int))
    if not year_columns:
        raise ValueError("IMF WEO Countries sheet has no annual value columns.")

    bca = _weo_indicator_long(countries, "BCA", "current_account_usd", year_columns)
    gdp = _weo_indicator_long(countries, "NGDPD", "gdp_usd", year_columns)
    eligible_codes = set(bca["country_code"]) | set(gdp["country_code"])
    return calculate_imf_positive_ca(
        current_account=bca,
        gdp=gdp,
        eligible_economy_codes=eligible_codes,
        years=year_columns,
        publication_year=publication_year,
    )


def _weo_indicator_long(
    countries: pd.DataFrame,
    indicator_id: str,
    value_name: str,
    year_columns: list[int],
) -> pd.DataFrame:
    selected = countries.loc[countries["INDICATOR.ID"].astype(str).eq(indicator_id)].copy()
    if selected.empty:
        raise ValueError(f"IMF WEO Countries sheet has no {indicator_id} rows.")
    id_columns = ["COUNTRY.ID", "COUNTRY", "LATEST_ACTUAL_ANNUAL_DATA", "SCALE"]
    long = selected[id_columns + year_columns].melt(id_vars=id_columns, value_vars=year_columns, var_name="year", value_name=value_name)
    long = long.rename(
        columns={
            "COUNTRY.ID": "country_code",
            "COUNTRY": "country_name",
            "LATEST_ACTUAL_ANNUAL_DATA": "latest_actual_year",
        }
    )
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["latest_actual_year"] = pd.to_numeric(long["latest_actual_year"], errors="coerce")
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce") * long["SCALE"].map(_weo_scale_multiplier)
    return long.drop(columns=["SCALE"])


def _weo_scale_multiplier(value: Any) -> float:
    scale = str(value or "").strip().lower()
    if scale == "billions":
        return 1_000_000_000.0
    if scale == "millions":
        return 1_000_000.0
    if scale in {"units", "", "nan"}:
        return 1.0
    raise ValueError(f"Unsupported IMF WEO scale: {value}")


def calculate_imf_positive_ca(
    current_account: pd.DataFrame,
    gdp: pd.DataFrame,
    eligible_economy_codes: Iterable[str],
    years: Iterable[int],
    publication_year: int,
) -> pd.DataFrame:
    """Apply the IMF WEO coverage gate without treating missing BCA observations as zero."""
    eligible = set(eligible_economy_codes)
    ca = current_account.loc[current_account["country_code"].isin(eligible)].copy()
    gdp_values = gdp.loc[gdp["country_code"].isin(eligible)].copy()
    joined = gdp_values[["country_code", "year", "gdp_usd"]].merge(
        ca[["country_code", "year", "current_account_usd", "latest_actual_year"]],
        on=["country_code", "year"],
        how="outer",
    )
    positive_by_country_year = ca.assign(positive_ca_usd=ca["current_account_usd"].clip(lower=0))
    rows: list[dict[str, Any]] = []
    current_year = datetime.now(timezone.utc).year
    for year in sorted(int(value) for value in years):
        sample = joined.loc[joined["year"] == year].copy()
        available = sample["current_account_usd"].notna()
        total_gdp = sample["gdp_usd"].dropna().sum(min_count=1)
        covered_gdp = sample.loc[available, "gdp_usd"].dropna().sum(min_count=1)
        gdp_coverage = float(covered_gdp / total_gdp) if pd.notna(total_gdp) and total_gdp > 0 else np.nan
        positive_ca = sample.loc[available, "current_account_usd"].clip(lower=0).sum(min_count=1)

        prior = positive_by_country_year.loc[positive_by_country_year["year"] == year - 1].copy()
        prior_total = prior["positive_ca_usd"].sum(min_count=1)
        available_codes = set(sample.loc[available, "country_code"])
        prior_covered = prior.loc[prior["country_code"].isin(available_codes), "positive_ca_usd"].sum(min_count=1)
        prior_surplus_coverage = (
            float(prior_covered / prior_total)
            if pd.notna(prior_total) and prior_total > 0 and pd.notna(prior_covered)
            else np.nan
        )

        if year >= publication_year:
            data_status = "FORECAST"
        else:
            latest_actual = pd.to_numeric(sample.loc[available, "latest_actual_year"], errors="coerce")
            data_status = "ACTUAL" if not latest_actual.empty and latest_actual.notna().all() and (latest_actual >= year).all() else "ESTIMATE"
        completed = year < current_year
        rows.append(
            {
                "year": year,
                "global_positive_ca_usd": positive_ca,
                "available_economy_count": int(available.sum()),
                "total_eligible_economy_count": len(eligible),
                "gdp_coverage": gdp_coverage,
                "prior_year_surplus_coverage": prior_surplus_coverage,
                "data_status": data_status,
                "is_completed_calendar_year": completed,
                "is_valid": bool(
                    pd.notna(positive_ca)
                    and gdp_coverage >= GDP_COVERAGE_MIN
                    and prior_surplus_coverage >= PRIOR_SURPLUS_COVERAGE_MIN
                ),
                "source": "IMF WEO",
            }
        )
    return pd.DataFrame(rows)


def _read_imf_ca_cache(path: Path, metadata_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    for column in ("is_valid", "is_completed_calendar_year"):
        frame[column] = frame[column].astype(str).str.lower().eq("true")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(metadata, dict):
            frame.attrs.update(metadata)
    return frame


def load_world_bank_positive_ca_crosscheck(cache_dir: Path, force_refresh: bool = False) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "world_bank_current_account_gdp.csv"
    max_age_seconds = 7 * 24 * 60 * 60
    if cache_path.exists() and not force_refresh:
        age = datetime.now(timezone.utc).timestamp() - cache_path.stat().st_mtime
        if age <= max_age_seconds:
            return _read_world_bank_ca_cache(cache_path)
    try:
        frame = download_world_bank_current_account()
        frame.to_csv(cache_path, index=False)
        return frame
    except Exception as exc:
        if cache_path.exists():
            frame = _read_world_bank_ca_cache(cache_path)
            frame.attrs["warning"] = f"Using stale World Bank cache after refresh failure: {exc}"
            return frame
        raise


def download_world_bank_current_account(start_year: int = 2010, end_year: int | None = None) -> pd.DataFrame:
    end = end_year or datetime.now(timezone.utc).year
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        country_payload = _world_bank_get(client, "https://api.worldbank.org/v2/country", {"format": "json", "per_page": "400"})
        countries = country_payload[1] if len(country_payload) > 1 else []
        actual_codes = {
            str(item.get("id"))
            for item in countries
            if str((item.get("region") or {}).get("id") or "").upper() != "NA"
        }
        ca = _world_bank_indicator(client, "BN.CAB.XOKA.CD", start_year, end)
        gdp = _world_bank_indicator(client, "NY.GDP.MKTP.CD", start_year, end)

    ca = ca.loc[ca["country_code"].isin(actual_codes)].rename(columns={"value": "current_account_usd"})
    gdp = gdp.loc[gdp["country_code"].isin(actual_codes)].rename(columns={"value": "gdp_usd"})
    return calculate_world_bank_positive_ca(ca, gdp, actual_codes, range(start_year, end + 1))


def calculate_world_bank_positive_ca(
    current_account: pd.DataFrame,
    gdp: pd.DataFrame,
    actual_country_codes: Iterable[str],
    years: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Aggregate positive current accounts while preserving missing observations as missing."""
    actual_codes = set(actual_country_codes)
    ca = current_account.loc[current_account["country_code"].isin(actual_codes)].copy()
    gdp_values = gdp.loc[gdp["country_code"].isin(actual_codes)].copy()
    joined = gdp_values.merge(ca, on=["country_code", "year"], how="outer")
    rows: list[dict[str, Any]] = []
    universe_count = len(actual_codes)
    selected_years = list(years) if years is not None else sorted(pd.to_numeric(joined["year"], errors="coerce").dropna().astype(int).unique())
    for year in selected_years:
        sample = joined.loc[joined["year"] == year].copy()
        ca_available = sample["current_account_usd"].notna()
        country_coverage = float(ca_available.sum() / universe_count) if universe_count else np.nan
        total_gdp = sample["gdp_usd"].dropna().sum(min_count=1)
        covered_gdp = sample.loc[ca_available, "gdp_usd"].dropna().sum(min_count=1)
        gdp_coverage = float(covered_gdp / total_gdp) if pd.notna(total_gdp) and total_gdp > 0 else np.nan
        positive_ca = sample.loc[ca_available, "current_account_usd"].clip(lower=0).sum(min_count=1)
        rows.append(
            {
                "year": year,
                "global_positive_ca_usd": positive_ca,
                "country_coverage": country_coverage,
                "gdp_coverage": gdp_coverage,
                "covered_countries": int(ca_available.sum()),
                "country_universe": universe_count,
                "is_valid": bool(
                    pd.notna(positive_ca)
                ),
                "source": "World Bank BN.CAB.XOKA.CD",
            }
        )
    return pd.DataFrame(rows)


def _world_bank_get(client: httpx.Client, url: str, params: dict[str, str]) -> list[Any]:
    response = client.get(url, params=params)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f"Unexpected World Bank response from {url}")
    return payload


def _world_bank_indicator(client: httpx.Client, indicator: str, start_year: int, end_year: int) -> pd.DataFrame:
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
    payload = _world_bank_get(
        client,
        url,
        {"format": "json", "per_page": "20000", "date": f"{start_year}:{end_year}"},
    )
    rows = [
        {
            "country_code": str(item.get("countryiso3code") or item.get("country", {}).get("id") or ""),
            "year": pd.to_numeric(item.get("date"), errors="coerce"),
            "value": pd.to_numeric(item.get("value"), errors="coerce"),
        }
        for item in payload[1]
    ]
    frame = pd.DataFrame(rows)
    frame["year"] = frame["year"].astype("Int64")
    return frame


def _read_world_bank_ca_cache(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["is_valid"] = frame["is_valid"].astype(str).str.lower().eq("true")
    return frame


def select_calibration_year(wgc_annual: pd.DataFrame, global_ca: pd.DataFrame) -> int | None:
    if wgc_annual.empty or global_ca.empty:
        return None
    wgc_years = set(pd.to_numeric(wgc_annual.loc[wgc_annual["is_complete"], "year"], errors="coerce").dropna().astype(int))
    allowed_status = global_ca["data_status"].astype(str).str.upper().isin({"ACTUAL", "ESTIMATE"})
    completed = global_ca.get(
        "is_completed_calendar_year",
        pd.to_numeric(global_ca["year"], errors="coerce") < datetime.now(timezone.utc).year,
    ).astype(bool)
    ca_years = set(
        pd.to_numeric(global_ca.loc[global_ca["is_valid"] & allowed_status & completed, "year"], errors="coerce")
        .dropna()
        .astype(int)
    )
    common = sorted(wgc_years & ca_years)
    return common[-1] if common else None


def latest_completed_wgc_year(wgc_annual: pd.DataFrame) -> int | None:
    years = pd.to_numeric(wgc_annual.loc[wgc_annual["is_complete"], "year"], errors="coerce").dropna()
    return int(years.max()) if not years.empty else None


def build_model2_history(wgc_annual: pd.DataFrame, global_ca: pd.DataFrame) -> pd.DataFrame:
    if wgc_annual.empty or global_ca.empty:
        return pd.DataFrame()
    allowed_status = global_ca["data_status"].astype(str).str.upper().isin({"ACTUAL", "ESTIMATE"})
    completed_year = global_ca.get(
        "is_completed_calendar_year",
        pd.to_numeric(global_ca["year"], errors="coerce") < datetime.now(timezone.utc).year,
    ).astype(bool)
    valid_ca = global_ca.loc[global_ca["is_valid"] & allowed_status & completed_year].copy()
    completed = wgc_annual.loc[wgc_annual["is_complete"]].copy()
    merged = completed.merge(valid_ca, on="year", how="inner", validate="one_to_one")
    if merged.empty:
        return merged
    merged["core_gold_flow_value_usd"] = merged["core_flow_tonnes"] * TROY_OZ_PER_TONNE * merged["lbma_gold_price_usd_oz"]
    merged["broad_gold_flow_value_usd"] = merged["broad_flow_tonnes"] * TROY_OZ_PER_TONNE * merged["lbma_gold_price_usd_oz"]
    merged["core_gmar"] = merged["core_gold_flow_value_usd"] / merged["global_positive_ca_usd"]
    merged["broad_gmar"] = merged["broad_gold_flow_value_usd"] / merged["global_positive_ca_usd"]
    return merged.sort_values("year").reset_index(drop=True)


def calculate_static_scenarios(global_positive_ca_usd: float, broad_flow_tonnes: float) -> pd.DataFrame:
    names = ("Normalization", "Monetization", "Strong Monetization", "Monetary Reset")
    rows = []
    for name, share in zip(names, STATIC_TARGET_SHARES):
        price = share * global_positive_ca_usd / (broad_flow_tonnes * TROY_OZ_PER_TONNE)
        rows.append({"scenario": name, "target_share": share, "implied_gold_price": float(price)})
    return pd.DataFrame(rows)


def monetary_gold_flow_tonnes(price: float, base: pd.Series) -> float:
    p0 = float(base["lbma_gold_price_usd_oz"])
    jewellery = max(JEWELLERY_FLOOR_TONNES, float(base["jewellery_fabrication_tonnes"]) * (price / p0) ** EPS_J)
    recycled = min(RECYCLED_CAP_TONNES, float(base["recycled_gold_tonnes"]) * (price / p0) ** EPS_R)
    return float(base["total_mine_supply_tonnes"] + recycled - base["technology_tonnes"] - jewellery)


def solve_adaptive_price(
    global_positive_ca_usd: float,
    base: pd.Series,
    target_share: float,
    theta: float,
    lower: float = 1_000.0,
    upper: float = 100_000.0,
) -> float:
    def objective(price: float) -> float:
        flow = monetary_gold_flow_tonnes(price, base)
        return price * TROY_OZ_PER_TONNE * theta * flow - target_share * global_positive_ca_usd

    lo, hi = float(lower), float(upper)
    f_lo, f_hi = objective(lo), objective(hi)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo == 0:
        return lo if f_lo == 0 else np.nan
    if f_lo * f_hi > 0:
        return np.nan
    for _ in range(120):
        mid = (lo + hi) / 2.0
        f_mid = objective(mid)
        if not np.isfinite(f_mid):
            return np.nan
        if abs(f_mid) <= max(1.0, target_share * global_positive_ca_usd * 1e-10):
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0


def calculate_adaptive_matrix(global_positive_ca_usd: float, base: pd.Series) -> pd.DataFrame:
    rows = []
    for theta in ADAPTIVE_THETAS:
        row: dict[Any, Any] = {"theta": theta}
        for share in STATIC_TARGET_SHARES:
            row[share] = solve_adaptive_price(global_positive_ca_usd, base, share, theta)
        rows.append(row)
    return pd.DataFrame(rows)


def build_convergence_panel(current_gold_price: float, static: pd.DataFrame, adaptive: pd.DataFrame) -> pd.DataFrame:
    rows = [{"valuation": "Current TradingView", "implied_gold_price": current_gold_price}]
    for record in static.to_dict("records"):
        rows.append({"valuation": f"Static — {record['scenario']}", "implied_gold_price": record["implied_gold_price"]})
    base_row = adaptive.loc[np.isclose(pd.to_numeric(adaptive["theta"]), BASE_THETA)]
    if not base_row.empty:
        for share in STATIC_TARGET_SHARES:
            rows.append({"valuation": f"Adaptive θ=50%, s={share:.0%}", "implied_gold_price": float(base_row.iloc[0][share])})
    out = pd.DataFrame(rows)
    out["vs_current_pct"] = out["implied_gold_price"] / current_gold_price - 1.0 if np.isfinite(current_gold_price) and current_gold_price > 0 else np.nan
    return out


def latest_number(series: pd.Series | Iterable[float]) -> float:
    values = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else np.nan


def latest_date(series: pd.Series) -> pd.Timestamp | None:
    if series is None or series.empty:
        return None
    dates = pd.to_datetime(series.dropna().index, errors="coerce")
    dates = dates[dates.notna()]
    return pd.Timestamp(dates.max()) if len(dates) else None
