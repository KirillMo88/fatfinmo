from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from global_liquidity import read_global_liquidity, update_global_liquidity
from treasury_fiscal_regime import STORAGE_DIR, TreasuryFiscalSnapshot, read_snapshot as read_treasury_snapshot


MODEL_VERSION = "TREASURY_FUNDING_POLICY_V1_1"
FISCALDATA_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
MSPD_MARKET_ENDPOINT = "/v1/debt/mspd/mspd_table_3_market"
MSPD_TOTAL_ENDPOINT = "/v1/debt/mspd/mspd_table_1"
MSPD_START = "2015-01-01"
FORECAST_YEARS = tuple(range(2027, 2032))
POLICY_WEIGHTS = {
    "CBImpulsePercentile": 1 / 3,
    "USNLImpulsePercentile": 1 / 3,
    "BankReservesImpulsePercentile": 1 / 3,
}
REFINANCING_REGIME_START = pd.Timestamp("2020-01-01")
REFINANCING_WEIGHTS = {"pressure_3m_percentile": 0.60, "pressure_6m_percentile": 0.40}

MSPD_RAW_PATH = STORAGE_DIR / "mspd_market_raw.parquet"
MSPD_TOTALS_PATH = STORAGE_DIR / "mspd_totals_raw.parquet"
FUNDING_MONTHLY_PATH = STORAGE_DIR / "treasury_funding_monthly.parquet"
FUNDING_ANNUAL_PATH = STORAGE_DIR / "treasury_funding_annual.parquet"
FUNDING_FORECAST_PATH = STORAGE_DIR / "treasury_funding_forecast.parquet"
POLICY_RESPONSE_PATH = STORAGE_DIR / "policy_response_weekly.parquet"
FUNDING_POLICY_STATUS_PATH = STORAGE_DIR / "funding_policy_status.json"
FUNDING_POLICY_CONFIG_PATH = STORAGE_DIR / "funding_policy_config.json"

SECURITY_TYPES = {
    "Bills Maturity Value": "Bills",
    "Notes": "Notes",
    "Bonds": "Bonds",
    "Inflation-Protected Securities": "TIPS",
    "Floating Rate Notes": "FRN",
}
TENORS = {
    "BILL_1Y": (1, "Bills"),
    "NOTE_2Y": (2, "Notes"),
    "NOTE_3Y": (3, "Notes"),
    "NOTE_5Y": (5, "Notes"),
    "NOTE_7Y": (7, "Notes"),
    "NOTE_10Y": (10, "Notes"),
    "BOND_20Y": (20, "Bonds"),
    "BOND_30Y": (30, "Bonds"),
    "FRN_2Y": (2, "FRN"),
    "TIPS_5Y": (5, "TIPS"),
    "TIPS_10Y": (10, "TIPS"),
    "TIPS_30Y": (30, "TIPS"),
}


def default_config() -> dict[str, Any]:
    return {
        "marketable_financing_share": 1.0,
        "cbo_source": "CBO February 2026 baseline, USD billions",
        "cbo_deficit_bn": {"2027": 1887.0, "2028": 2080.0, "2029": 2020.0, "2030": 2201.0, "2031": 2286.0},
        "reference_curve_date": "2026 reference curve assumption",
        "reference_yield_pct": {
            "BILL_1Y": 3.70, "NOTE_2Y": 3.75, "NOTE_3Y": 3.78, "NOTE_5Y": 3.90,
            "NOTE_7Y": 4.05, "NOTE_10Y": 4.20, "BOND_20Y": 4.65, "BOND_30Y": 4.75,
            "FRN_2Y": 3.85, "TIPS_5Y": 1.45, "TIPS_10Y": 1.75, "TIPS_30Y": 2.15,
        },
        "issuance_shares": {
            "LONG": {
                "BILL_1Y": .62, "NOTE_2Y": .04, "NOTE_3Y": .025, "NOTE_5Y": .08,
                "NOTE_7Y": .045, "NOTE_10Y": .065, "BOND_20Y": .035, "BOND_30Y": .04,
                "FRN_2Y": .01, "TIPS_5Y": .01, "TIPS_10Y": .015, "TIPS_30Y": .015,
            },
            "BASE": {
                "BILL_1Y": .75, "NOTE_2Y": .06, "NOTE_3Y": .03, "NOTE_5Y": .045,
                "NOTE_7Y": .025, "NOTE_10Y": .03, "BOND_20Y": .01, "BOND_30Y": .015,
                "FRN_2Y": .015, "TIPS_5Y": .005, "TIPS_10Y": .008, "TIPS_30Y": .007,
            },
            "SHORT": {
                "BILL_1Y": .82, "NOTE_2Y": .07, "NOTE_3Y": .04, "NOTE_5Y": .02,
                "NOTE_7Y": .01, "NOTE_10Y": .01, "BOND_20Y": .002, "BOND_30Y": .003,
                "FRN_2Y": .021, "TIPS_5Y": .001, "TIPS_10Y": .001, "TIPS_30Y": .002,
            },
        },
    }


@dataclass
class TreasuryFundingPolicySnapshot:
    monthly: pd.DataFrame
    annual: pd.DataFrame
    forecast: pd.DataFrame
    policy: pd.DataFrame
    status: dict[str, Any]
    config: dict[str, Any]


def _deep_fill(default: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in default.items():
        if isinstance(value, dict):
            candidate = supplied.get(key, {})
            result[key] = _deep_fill(value, candidate if isinstance(candidate, dict) else {})
        else:
            result[key] = supplied.get(key, value)
    return result


def load_config() -> dict[str, Any]:
    supplied: dict[str, Any] = {}
    if FUNDING_POLICY_CONFIG_PATH.exists():
        try:
            supplied = json.loads(FUNDING_POLICY_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            supplied = {}
    return _deep_fill(default_config(), supplied)


def validate_config(config: dict[str, Any]) -> None:
    share = float(config["marketable_financing_share"])
    if not 0 <= share <= 1:
        raise ValueError("MarketableFinancingShare must be between 0 and 1")
    for scenario, values in config["issuance_shares"].items():
        missing = set(TENORS) - set(values)
        if missing:
            raise ValueError(f"{scenario} is missing tenors: {sorted(missing)}")
        total = sum(float(values[tenor]) for tenor in TENORS)
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError(f"{scenario} issuance shares sum to {total:.8f}, not 1.0")
        if any(float(values[tenor]) < 0 for tenor in TENORS):
            raise ValueError(f"{scenario} has a negative issuance share")
    if set(map(int, config["cbo_deficit_bn"])) != set(FORECAST_YEARS):
        raise ValueError("CBO deficit configuration must contain 2027-2031")
    if set(config["reference_yield_pct"]) != set(TENORS):
        raise ValueError("Reference curve must contain every configured tenor")


def save_config(config: dict[str, Any]) -> None:
    validate_config(config)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = FUNDING_POLICY_CONFIG_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.replace(temporary, FUNDING_POLICY_CONFIG_PATH)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _fiscaldata_page(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=90.0, headers={"User-Agent": "fatfinmo-treasury-funding/1.0"}) as client:
                response = client.get(f"{FISCALDATA_BASE}{endpoint}", params=params)
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"FiscalData request failed: {error}")


def download_fiscaldata(endpoint: str, start: str = MSPD_START) -> pd.DataFrame:
    common = {"filter": f"record_date:gte:{start}", "sort": "record_date", "page[size]": 10000}
    first = _fiscaldata_page(endpoint, {**common, "page[number]": 1})
    pages = int(first.get("meta", {}).get("total-pages", 1))
    expected = int(first.get("meta", {}).get("total-count", len(first.get("data", []))))
    rows = list(first.get("data", []))
    for page in range(2, pages + 1):
        rows.extend(_fiscaldata_page(endpoint, {**common, "page[number]": page}).get("data", []))
    if len(rows) != expected:
        raise RuntimeError(f"FiscalData pagination incomplete: downloaded {len(rows)} of {expected}")
    return pd.DataFrame(rows)


def load_mspd_raw(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    fresh = (MSPD_RAW_PATH.exists() and MSPD_TOTALS_PATH.exists() and
             time.time() - min(MSPD_RAW_PATH.stat().st_mtime, MSPD_TOTALS_PATH.stat().st_mtime) < 20 * 3600)
    if fresh and not refresh:
        return pd.read_parquet(MSPD_RAW_PATH), pd.read_parquet(MSPD_TOTALS_PATH), "FISCALDATA_CACHE"
    try:
        market = download_fiscaldata(MSPD_MARKET_ENDPOINT)
        totals = download_fiscaldata(MSPD_TOTAL_ENDPOINT)
        _atomic_parquet(market, MSPD_RAW_PATH)
        _atomic_parquet(totals, MSPD_TOTALS_PATH)
        return market, totals, "FISCALDATA_CURRENT"
    except Exception:
        if MSPD_RAW_PATH.exists() and MSPD_TOTALS_PATH.exists():
            return pd.read_parquet(MSPD_RAW_PATH), pd.read_parquet(MSPD_TOTALS_PATH), "STALE_FISCALDATA_CACHE"
        raise


def normalize_mspd_market(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {
        "record_date", "security_class1_desc", "security_class2_desc", "issue_date",
        "maturity_date", "outstanding_amt", "issued_amt", "interest_rate_pct", "yield_pct",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"MSPD schema missing fields: {sorted(missing)}")
    frame = raw.copy()
    frame["RecordDate"] = pd.to_datetime(frame["record_date"], errors="coerce").dt.normalize()
    frame["IssueDate"] = pd.to_datetime(frame["issue_date"], errors="coerce").dt.normalize()
    frame["MaturityDate"] = pd.to_datetime(frame["maturity_date"], errors="coerce").dt.normalize()
    frame["CUSIP"] = frame["security_class2_desc"].astype("string").str.strip()
    frame["SecurityType"] = frame["security_class1_desc"].map(SECURITY_TYPES)
    for source, target in (("outstanding_amt", "OutstandingMil"), ("issued_amt", "IssuedMil"),
                           ("interest_rate_pct", "InterestRatePct"), ("yield_pct", "YieldPct")):
        frame[target] = pd.to_numeric(frame[source], errors="coerce")
    frame["EffectiveRatePct"] = frame["InterestRatePct"]
    floating = frame["SecurityType"].isin(["Bills", "FRN"])
    frame.loc[floating, "EffectiveRatePct"] = frame.loc[floating, "YieldPct"]
    eligible = frame["SecurityType"].notna() & frame["CUSIP"].str.fullmatch(r"[0-9A-Z]{9}", na=False)
    detail = frame.loc[eligible].copy()
    securities = detail.loc[detail["OutstandingMil"].notna()].copy()
    negative = int(securities["OutstandingMil"].lt(0).sum())
    invalid_maturity = int(securities["MaturityDate"].le(securities["RecordDate"]).sum())
    securities = securities.loc[securities["OutstandingMil"].gt(0) &
                                securities["MaturityDate"].gt(securities["RecordDate"])].copy()
    duplicates = int(securities.duplicated(["RecordDate", "CUSIP"], keep=False).sum())
    securities = securities.sort_values(["RecordDate", "CUSIP", "OutstandingMil"]).drop_duplicates(
        ["RecordDate", "CUSIP"], keep="last"
    )
    diagnostics = {
        "DuplicateCUSIPRows": duplicates,
        "NegativeOutstandingRows": negative,
        "InvalidMaturityRows": invalid_maturity,
        "ExcludedNonSecurityRows": int(len(raw) - len(detail)),
    }
    return securities.reset_index(drop=True), diagnostics


def _expanding_pit_percentile(values: pd.Series, dates: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    normalized_dates = pd.to_datetime(dates, errors="coerce").dt.normalize()
    output = pd.Series(np.nan, index=values.index, dtype="float64")
    eligible = normalized_dates.ge(REFINANCING_REGIME_START) & numeric.notna()
    history: list[float] = []
    for index in values.index:
        if not eligible.loc[index]:
            continue
        current = float(numeric.loc[index])
        history.append(current)
        output.loc[index] = sum(value <= current for value in history) / len(history) * 100
    return output


def build_near_term_refinancing(securities: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build monthly near-term pressure from validated security snapshots."""
    columns = [
        "record_date", "next_3m_rollover", "next_6m_rollover", "trailing_12m_rollover",
        "next_3m_monthly_run_rate", "next_6m_monthly_run_rate", "trailing_12m_monthly_run_rate",
        "pressure_ratio_3m", "pressure_ratio_6m", "pressure_3m_percentile",
        "pressure_6m_percentile", "near_term_refinancing_pressure",
    ]
    component_names = {"Bills": "bills", "Notes": "notes", "Bonds": "bonds", "TIPS": "tips", "FRN": "frn"}
    for horizon in (3, 6):
        columns.extend(f"next_{horizon}m_{name}" for name in component_names.values())
    if securities.empty:
        return pd.DataFrame(columns=columns), {"NearTermRows": 0}

    maturity_conflicts = int(securities.groupby("CUSIP")["MaturityDate"].nunique().gt(1).sum())
    ordered = securities.sort_values(["CUSIP", "RecordDate", "OutstandingMil"])
    maturity_ledger = ordered.drop_duplicates("CUSIP", keep="last").copy()
    duplicate_ledger_rows = int(maturity_ledger.duplicated("CUSIP").sum())
    rows: list[dict[str, Any]] = []
    for date, snapshot in securities.groupby("RecordDate", sort=True):
        row: dict[str, Any] = {"record_date": date}
        for horizon in (3, 6):
            end = date + pd.DateOffset(months=horizon)
            forward = snapshot.loc[snapshot["MaturityDate"].gt(date) & snapshot["MaturityDate"].le(end)]
            row[f"next_{horizon}m_rollover"] = forward["OutstandingMil"].sum() / 1e6
            for security_type, name in component_names.items():
                row[f"next_{horizon}m_{name}"] = (
                    forward.loc[forward["SecurityType"].eq(security_type), "OutstandingMil"].sum() / 1e6
                )
        trailing_start = date - pd.DateOffset(months=12)
        realized = maturity_ledger.loc[
            maturity_ledger["MaturityDate"].gt(trailing_start) & maturity_ledger["MaturityDate"].le(date)
        ]
        row["trailing_12m_rollover"] = realized["OutstandingMil"].sum() / 1e6
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("record_date").reset_index(drop=True)
    frame["next_3m_monthly_run_rate"] = frame["next_3m_rollover"] / 3
    frame["next_6m_monthly_run_rate"] = frame["next_6m_rollover"] / 6
    frame["trailing_12m_monthly_run_rate"] = frame["trailing_12m_rollover"] / 12
    valid_denominator = frame["trailing_12m_rollover"].gt(0)
    frame["pressure_ratio_3m"] = np.where(
        valid_denominator, 4 * frame["next_3m_rollover"] / frame["trailing_12m_rollover"], np.nan
    )
    frame["pressure_ratio_6m"] = np.where(
        valid_denominator, 2 * frame["next_6m_rollover"] / frame["trailing_12m_rollover"], np.nan
    )
    frame["pressure_3m_percentile"] = _expanding_pit_percentile(frame["pressure_ratio_3m"], frame["record_date"])
    frame["pressure_6m_percentile"] = _expanding_pit_percentile(frame["pressure_ratio_6m"], frame["record_date"])
    complete = frame[list(REFINANCING_WEIGHTS)].notna().all(axis=1)
    frame["near_term_refinancing_pressure"] = np.nan
    frame.loc[complete, "near_term_refinancing_pressure"] = sum(
        frame.loc[complete, column] * weight for column, weight in REFINANCING_WEIGHTS.items()
    )

    component_errors: dict[str, float] = {}
    for horizon in (3, 6):
        component_total = frame[[f"next_{horizon}m_{name}" for name in component_names.values()]].sum(axis=1)
        component_errors[f"Next{horizon}MComponentMaxError"] = float(
            (component_total - frame[f"next_{horizon}m_rollover"]).abs().max()
        )
    regime = frame.loc[frame["record_date"].ge(REFINANCING_REGIME_START)]
    ratios = regime[["pressure_ratio_3m", "pressure_ratio_6m"]].dropna()
    percentiles = regime[["pressure_3m_percentile", "pressure_6m_percentile"]].dropna()
    scores = regime["near_term_refinancing_pressure"].dropna()
    diagnostics = {
        "NearTermRows": int(len(frame)),
        "RefinancingRegimeStart": REFINANCING_REGIME_START.date().isoformat(),
        "RefinancingWeightSum": sum(REFINANCING_WEIGHTS.values()),
        "MaturityLedgerDuplicateRows": duplicate_ledger_rows,
        "CUSIPMaturityDateConflicts": maturity_conflicts,
        "Next3MWithinNext6M": bool(frame["next_3m_rollover"].le(frame["next_6m_rollover"] + 1e-12).all()),
        "Trailing12MPositiveInRegime": bool(regime["trailing_12m_rollover"].gt(0).all()),
        "PressureRatiosFinitePositive": bool(np.isfinite(ratios.to_numpy()).all() and ratios.gt(0).all().all()),
        "PressurePercentilesValid": bool(percentiles.apply(lambda series: series.between(0, 100).all()).all()),
        "NearTermScoreValid": bool(scores.between(0, 100).all()),
        "RefinancingPercentilesPITSafe": True,
        "Pre2020Use": "Trailing 12M realized maturity denominator only; excluded from percentile normalization.",
        **component_errors,
    }
    return frame[columns], diagnostics


def published_marketable_totals(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["RecordDate"] = pd.to_datetime(frame["record_date"], errors="coerce").dt.normalize()
    frame["PublishedTotalMil"] = pd.to_numeric(frame["total_mil_amt"], errors="coerce")
    frame = frame.loc[frame["security_type_desc"].eq("Total Marketable")]
    return frame[["RecordDate", "PublishedTotalMil"]].dropna().drop_duplicates("RecordDate", keep="last")


def _historical_marginal_rate(detail: pd.DataFrame, date: pd.Timestamp) -> tuple[float, float]:
    issued = detail.loc[
        detail["IssueDate"].between(date - pd.Timedelta(days=365), date, inclusive="both") &
        detail["IssuedMil"].gt(0) & detail["EffectiveRatePct"].notna()
    ]
    denominator = detail.loc[
        detail["IssueDate"].between(date - pd.Timedelta(days=365), date, inclusive="both") &
        detail["IssuedMil"].gt(0), "IssuedMil"
    ].sum()
    covered = issued["IssuedMil"].sum()
    if covered <= 0:
        return np.nan, 0.0
    return float(np.average(issued["EffectiveRatePct"], weights=issued["IssuedMil"])), float(covered / denominator * 100) if denominator else 0.0


def build_treasury_funding_history(
    raw_market: pd.DataFrame,
    raw_totals: pd.DataFrame,
    fiscal: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    securities, diagnostics = normalize_mspd_market(raw_market)
    normalized_detail = raw_market.copy()
    normalized_detail["RecordDate"] = pd.to_datetime(normalized_detail["record_date"], errors="coerce").dt.normalize()
    normalized_detail["IssueDate"] = pd.to_datetime(normalized_detail["issue_date"], errors="coerce").dt.normalize()
    normalized_detail["SecurityType"] = normalized_detail["security_class1_desc"].map(SECURITY_TYPES)
    normalized_detail["IssuedMil"] = pd.to_numeric(normalized_detail["issued_amt"], errors="coerce")
    normalized_detail["InterestRatePct"] = pd.to_numeric(normalized_detail["interest_rate_pct"], errors="coerce")
    normalized_detail["YieldPct"] = pd.to_numeric(normalized_detail["yield_pct"], errors="coerce")
    normalized_detail["EffectiveRatePct"] = normalized_detail["InterestRatePct"]
    normalized_detail.loc[normalized_detail["SecurityType"].isin(["Bills", "FRN"]), "EffectiveRatePct"] = normalized_detail["YieldPct"]
    normalized_detail = normalized_detail.loc[normalized_detail["SecurityType"].notna()]
    aggregate = raw_market.copy()
    aggregate["RecordDate"] = pd.to_datetime(aggregate["record_date"], errors="coerce").dt.normalize()
    aggregate["AggregateOutstandingMil"] = pd.to_numeric(aggregate["outstanding_amt"], errors="coerce")
    labels = aggregate["security_class2_desc"].astype("string").str.strip()
    matured = labels.str.match(r"^(Total )?Matured Treasury", na=False)
    ffb = aggregate["security_class1_desc"].eq("Federal Financing Bank")
    published_adjustments = aggregate.loc[matured | ffb].groupby("RecordDate")["AggregateOutstandingMil"].sum()
    published = published_marketable_totals(raw_totals).set_index("RecordDate")["PublishedTotalMil"]
    fiscal_source = pd.DataFrame()
    if fiscal is not None and not fiscal.empty and {"AvailableDate", "Deficit12M"}.issubset(fiscal.columns):
        fiscal_source = fiscal[["AvailableDate", "Deficit12M"]].dropna().sort_values("AvailableDate")
    rows: list[dict[str, Any]] = []
    for date, snapshot in securities.groupby("RecordDate", sort=True):
        total_mil = snapshot["OutstandingMil"].sum()
        remaining_days = (snapshot["MaturityDate"] - date).dt.days
        rollover = snapshot.loc[remaining_days.between(1, 365)]
        row: dict[str, Any] = {"Date": date, "TotalMarketableDebt": total_mil / 1e6}
        for security_type in SECURITY_TYPES.values():
            key = security_type if security_type != "Bills" else "Bills"
            row[f"{key}Outstanding"] = snapshot.loc[snapshot["SecurityType"].eq(security_type), "OutstandingMil"].sum() / 1e6
            row[f"Rollover{key}"] = rollover.loc[rollover["SecurityType"].eq(security_type), "OutstandingMil"].sum() / 1e6
        row["Rollover12M"] = rollover["OutstandingMil"].sum() / 1e6
        row["RolloverIntensity"] = row["Rollover12M"] / row["TotalMarketableDebt"] * 100 if row["TotalMarketableDebt"] else np.nan
        for security_type in SECURITY_TYPES.values():
            key = security_type if security_type != "Bills" else "Bills"
            row[f"Rollover{key}Share"] = row[f"Rollover{key}"] / row["Rollover12M"] * 100 if row["Rollover12M"] else np.nan
        row["WAMMonths"] = np.average(remaining_days / 365.25 * 12, weights=snapshot["OutstandingMil"])
        row["BillShare"] = row["BillsOutstanding"] / row["TotalMarketableDebt"] * 100
        rated = snapshot.loc[snapshot["EffectiveRatePct"].notna()]
        covered = rated["OutstandingMil"].sum()
        row["PortfolioAvgRate"] = (np.average(rated["EffectiveRatePct"], weights=rated["OutstandingMil"])
                                   if covered > 0 else np.nan)
        row["PortfolioRateCoveragePct"] = covered / total_mil * 100 if total_mil else np.nan
        marginal, marginal_coverage = _historical_marginal_rate(normalized_detail.loc[normalized_detail["RecordDate"].eq(date)], date)
        row["MarginalFundingRate"] = marginal
        row["MarginalRateCoveragePct"] = marginal_coverage
        row["PublishedMarketableDebt"] = published.get(date, np.nan) / 1e6
        row["PublishedExclusionAdjustment"] = published_adjustments.get(date, 0.0) / 1e6
        row["AdjustedPublishedMarketableDebt"] = row["PublishedMarketableDebt"] - row["PublishedExclusionAdjustment"]
        row["ReconciliationErrorPct"] = ((row["TotalMarketableDebt"] / row["PublishedMarketableDebt"] - 1) * 100
                                          if pd.notna(row["PublishedMarketableDebt"]) and row["PublishedMarketableDebt"] else np.nan)
        row["ReconciliationFlag"] = "PASS" if pd.notna(row["ReconciliationErrorPct"]) and abs(row["ReconciliationErrorPct"]) <= .5 else "FAIL"
        row["AdjustedReconciliationErrorPct"] = (
            (row["TotalMarketableDebt"] / row["AdjustedPublishedMarketableDebt"] - 1) * 100
            if pd.notna(row["AdjustedPublishedMarketableDebt"]) and row["AdjustedPublishedMarketableDebt"] else np.nan
        )
        row["AdjustedReconciliationFlag"] = (
            "PASS" if pd.notna(row["AdjustedReconciliationErrorPct"]) and abs(row["AdjustedReconciliationErrorPct"]) <= .5 else "FAIL"
        )
        if not fiscal_source.empty:
            available = fiscal_source.loc[fiscal_source["AvailableDate"].le(date), "Deficit12M"]
            row["NewNetFinancing"] = float(available.iloc[-1]) / 1000 if not available.empty else np.nan
        else:
            row["NewNetFinancing"] = np.nan
        row["GrossFinancingRequirement"] = row["Rollover12M"] + row["NewNetFinancing"] if pd.notna(row["NewNetFinancing"]) else np.nan
        row["GFRToDebt"] = row["GrossFinancingRequirement"] / row["TotalMarketableDebt"] * 100 if pd.notna(row["GrossFinancingRequirement"]) else np.nan
        rows.append(row)
    history = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    near_term, near_term_diagnostics = build_near_term_refinancing(securities)
    history = history.merge(near_term, left_on="Date", right_on="record_date", how="left")
    history["RolloverYoYAbs"] = history["Rollover12M"].diff(12)
    history["RolloverYoYPct"] = history["Rollover12M"].pct_change(12, fill_method=None) * 100
    history["RolloverIntensityYoYDeltaPP"] = history["RolloverIntensity"].diff(12)
    history["RolloverWithinDebt"] = history["Rollover12M"].le(history["TotalMarketableDebt"])
    history["Next6MWithinDebt"] = history["next_6m_rollover"].le(history["TotalMarketableDebt"])
    expected_near_term = (
        history["pressure_3m_percentile"] * REFINANCING_WEIGHTS["pressure_3m_percentile"]
        + history["pressure_6m_percentile"] * REFINANCING_WEIGHTS["pressure_6m_percentile"]
    )
    near_term_equal = np.isclose(
        history["near_term_refinancing_pressure"], expected_near_term, equal_nan=True, atol=1e-10
    ).all()
    diagnostics.update({
        "Months": int(len(history)),
        "ReconciliationFailures": int(history["ReconciliationFlag"].eq("FAIL").sum()),
        "AdjustedReconciliationFailures": int(history["AdjustedReconciliationFlag"].eq("FAIL").sum()),
        "RolloverExceedsDebtRows": int((~history["RolloverWithinDebt"]).sum()),
        "Next6MExceedsDebtRows": int((~history["Next6MWithinDebt"]).sum()),
        "NearTermCompositeFormulaValid": bool(near_term_equal),
        "LatestReconciliationErrorPct": float(history.iloc[-1]["ReconciliationErrorPct"]) if not history.empty else np.nan,
        "LatestAdjustedReconciliationErrorPct": float(history.iloc[-1]["AdjustedReconciliationErrorPct"]) if not history.empty else np.nan,
        "NearTermRefinancing": near_term_diagnostics,
    })
    return history, diagnostics


def annual_actuals(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly.copy()
    frame = monthly.copy()
    frame["Year"] = frame["Date"].dt.year
    annual = frame.sort_values("Date").groupby("Year", as_index=False).tail(1).reset_index(drop=True)
    latest_year = int(annual["Year"].max())
    annual["PeriodLabel"] = annual["Year"].astype(str)
    if annual.iloc[-1]["Date"].month < 12:
        annual.loc[annual["Year"].eq(latest_year), "PeriodLabel"] = f"{latest_year} Current / Latest"
    annual["Scenario"] = "BASE"
    return annual


def _portfolio_from_latest(securities: pd.DataFrame) -> pd.DataFrame:
    latest_date = securities["RecordDate"].max()
    frame = securities.loc[securities["RecordDate"].eq(latest_date)].copy()
    return pd.DataFrame({
        "MaturityDate": frame["MaturityDate"],
        "Amount": frame["OutstandingMil"] / 1e6,
        "Rate": frame["EffectiveRatePct"],
        "SecurityType": frame["SecurityType"],
        "Tenor": "EXISTING",
    })


def _issue_cohorts(amount: float, year: int, shares: dict[str, float], curve: dict[str, float]) -> pd.DataFrame:
    issue_date = pd.Timestamp(year=year, month=6, day=30)
    rows = []
    for tenor, (years, security_type) in TENORS.items():
        rows.append({
            "MaturityDate": issue_date + pd.DateOffset(years=years),
            "Amount": amount * float(shares[tenor]),
            "Rate": float(curve[tenor]),
            "SecurityType": security_type,
            "Tenor": tenor,
        })
    return pd.DataFrame(rows)


def run_cohort_forecast(securities: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    validate_config(config)
    latest_date = securities["RecordDate"].max()
    latest = securities.loc[securities["RecordDate"].eq(latest_date)].copy()
    start = _portfolio_from_latest(latest)
    rows: list[dict[str, Any]] = []
    curve = config["reference_yield_pct"]
    for scenario, shares in config["issuance_shares"].items():
        portfolio = start.copy()
        bridge_end = pd.Timestamp(year=min(FORECAST_YEARS) - 1, month=12, day=31)
        bridge = portfolio["MaturityDate"].le(bridge_end)
        bridge_rollover = portfolio.loc[bridge, "Amount"].sum()
        portfolio = portfolio.loc[~bridge].copy()
        if bridge_rollover > 0:
            portfolio = pd.concat([portfolio, _issue_cohorts(bridge_rollover, bridge_end.year, shares, curve)], ignore_index=True)
        for year in FORECAST_YEARS:
            year_start = pd.Timestamp(year=year, month=1, day=1)
            year_end = pd.Timestamp(year=year, month=12, day=31)
            opening = portfolio["Amount"].sum()
            matures = portfolio["MaturityDate"].between(year_start, year_end, inclusive="both")
            rollover = portfolio.loc[matures, "Amount"].sum()
            portfolio = portfolio.loc[~matures].copy()
            net_financing = float(config["cbo_deficit_bn"][str(year)]) / 1000 * float(config["marketable_financing_share"])
            required = rollover + net_financing
            portfolio = pd.concat([portfolio, _issue_cohorts(required, year, shares, curve)], ignore_index=True)
            closing = portfolio["Amount"].sum()
            remaining_months = (portfolio["MaturityDate"] - year_end).dt.days / 365.25 * 12
            valid_rate = portfolio["Rate"].notna()
            row = {
                "Year": year, "Scenario": scenario, "PrincipalRollover": rollover,
                "OpeningDebt": opening, "NewNetFinancing": net_financing,
                "GrossFinancingRequirement": required,
                "GFRToDebt": required / opening * 100 if opening else np.nan,
                "RolloverIntensity": rollover / opening * 100 if opening else np.nan,
                "ClosingDebt": closing, "AccountingGap": closing - opening - net_financing,
                "WAMMonths": np.average(remaining_months, weights=portfolio["Amount"]),
                "BillShare": portfolio.loc[portfolio["SecurityType"].eq("Bills"), "Amount"].sum() / closing * 100,
                "PortfolioAvgRate": np.average(portfolio.loc[valid_rate, "Rate"], weights=portfolio.loc[valid_rate, "Amount"]),
                "PortfolioRateCoveragePct": portfolio.loc[valid_rate, "Amount"].sum() / closing * 100,
                "MarginalFundingRate": sum(float(shares[tenor]) * float(curve[tenor]) for tenor in TENORS),
            }
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["Year", "Scenario"]).reset_index(drop=True)


def trailing_percentile(values: pd.Series, window: int = 156, minimum: int = 52) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")

    def rank(sample: np.ndarray) -> float:
        valid = sample[np.isfinite(sample)]
        if len(valid) < minimum or not np.isfinite(sample[-1]):
            return np.nan
        return float((valid <= sample[-1]).sum() / len(valid) * 100)

    return numeric.rolling(window, min_periods=minimum).apply(rank, raw=True)


def _asof_series(source: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    values = source.dropna().sort_index()
    if values.empty:
        return pd.Series(np.nan, index=index, dtype="float64")
    values.index = pd.to_datetime(values.index).tz_localize(None)
    values = values[~values.index.duplicated(keep="last")]
    return values.reindex(values.index.union(index)).sort_index().ffill().reindex(index)


def build_policy_response(monthly_cb: pd.DataFrame, liquidity: pd.DataFrame) -> pd.DataFrame:
    if liquidity.empty:
        return pd.DataFrame()
    base = liquidity[["Date", "NetLiquidity", "WRESBAL"]].copy()
    base["Date"] = pd.to_datetime(base["Date"], errors="coerce").dt.normalize()
    base = base.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
    index = pd.DatetimeIndex(base.index)
    cb = pd.Series(dtype="float64")
    if not monthly_cb.empty and {"date", "global_cb_assets_usd_bn"}.issubset(monthly_cb.columns):
        data = monthly_cb[["date", "global_cb_assets_usd_bn"]].copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
        data["AvailableDate"] = data["date"] + pd.offsets.MonthEnd(1)
        cb = pd.Series(pd.to_numeric(data["global_cb_assets_usd_bn"], errors="coerce").to_numpy(),
                       index=data["AvailableDate"])
    frame = pd.DataFrame(index=index)
    frame["GlobalCBAssets"] = _asof_series(cb, index)
    frame["USNetLiquidity"] = pd.to_numeric(base["NetLiquidity"], errors="coerce")
    frame["BankReserves"] = pd.to_numeric(base["WRESBAL"], errors="coerce")
    frame["CBImpulse13W"] = frame["GlobalCBAssets"].pct_change(13, fill_method=None) * 100
    frame["USNLImpulse13W"] = frame["USNetLiquidity"].pct_change(13, fill_method=None) * 100
    frame["BankReservesImpulse13W"] = frame["BankReserves"].pct_change(13, fill_method=None) * 100
    for raw_col, output in (
        ("CBImpulse13W", "CBImpulsePercentile"),
        ("USNLImpulse13W", "USNLImpulsePercentile"),
        ("BankReservesImpulse13W", "BankReservesImpulsePercentile"),
    ):
        frame[output] = trailing_percentile(frame[raw_col])
    complete = frame[list(POLICY_WEIGHTS)].notna().all(axis=1)
    frame["PolicyResponseScore"] = np.nan
    frame.loc[complete, "PolicyResponseScore"] = sum(
        frame.loc[complete, column] * weight for column, weight in POLICY_WEIGHTS.items()
    )
    score = frame["PolicyResponseScore"]
    frame["PolicyResponseRegime"] = np.select(
        [score.lt(40), score.gt(60), score.between(40, 60, inclusive="both")],
        ["WEAK", "STRONG", "MID"], default="DATA UNAVAILABLE",
    )
    frame = frame.reset_index().rename(columns={"index": "Date"})
    return frame


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def read_snapshot() -> TreasuryFundingPolicySnapshot:
    status = json.loads(FUNDING_POLICY_STATUS_PATH.read_text(encoding="utf-8")) if FUNDING_POLICY_STATUS_PATH.exists() else {}
    return TreasuryFundingPolicySnapshot(
        _read(FUNDING_MONTHLY_PATH), _read(FUNDING_ANNUAL_PATH), _read(FUNDING_FORECAST_PATH),
        _read(POLICY_RESPONSE_PATH), status, load_config(),
    )


def refresh_snapshot(
    api_key: str | None = None,
    refresh: bool = False,
    treasury_snapshot: TreasuryFiscalSnapshot | None = None,
) -> TreasuryFundingPolicySnapshot:
    config = load_config()
    validate_config(config)
    market_raw, totals_raw, source_state = load_mspd_raw(refresh=refresh)
    base_snapshot = treasury_snapshot or read_treasury_snapshot()
    if base_snapshot.weekly.empty:
        raise RuntimeError("Treasury & Fiscal weekly snapshot is required before funding refresh")
    monthly, treasury_diagnostics = build_treasury_funding_history(market_raw, totals_raw, base_snapshot.fiscal)
    annual = annual_actuals(monthly)
    securities, _ = normalize_mspd_market(market_raw)
    forecast = run_cohort_forecast(securities, config)
    _, global_monthly, _ = read_global_liquidity()
    if global_monthly.empty:
        _, global_monthly, _ = update_global_liquidity(api_key=api_key, force=False)
    policy = build_policy_response(global_monthly, base_snapshot.liquidity)
    policy_range_ok = bool(policy[[*POLICY_WEIGHTS, "PolicyResponseScore"]].apply(
        lambda column: column.dropna().between(0, 100).all()
    ).all()) if not policy.empty else False
    status = {
        "ModelVersion": MODEL_VERSION,
        "CalculatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "MSPDSourceState": source_state,
        "MSPDMarketRows": int(len(market_raw)),
        "MSPDTotalRows": int(len(totals_raw)),
        "TreasuryDiagnostics": treasury_diagnostics,
        "ScenarioWeightSums": {
            scenario: sum(float(value) for value in shares.values())
            for scenario, shares in config["issuance_shares"].items()
        },
        "PolicyWeightSum": sum(POLICY_WEIGHTS.values()),
        "PolicyRangesValid": policy_range_ok,
        "TreasuryPITSafe": True,
        "PolicyPITSafe": True,
        "MissingValuesZeroFilled": False,
        "PolicyTiming": "Monthly global CB assets become available after month completion; weekly US liquidity and reserves use source AvailableDate.",
    }
    if not monthly.empty:
        latest_near_term = monthly.iloc[-1]
        status["NearTermRefinancingLatest"] = {
            key: (float(latest_near_term[key]) if pd.notna(latest_near_term[key]) else None)
            for key in (
                "next_3m_rollover", "next_6m_rollover", "trailing_12m_rollover",
                "pressure_ratio_3m", "pressure_ratio_6m", "pressure_3m_percentile",
                "pressure_6m_percentile", "near_term_refinancing_pressure",
            )
        }
    for frame, path in ((monthly, FUNDING_MONTHLY_PATH), (annual, FUNDING_ANNUAL_PATH),
                        (forecast, FUNDING_FORECAST_PATH), (policy, POLICY_RESPONSE_PATH)):
        _atomic_parquet(frame, path)
    temporary = FUNDING_POLICY_STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(temporary, FUNDING_POLICY_STATUS_PATH)
    return TreasuryFundingPolicySnapshot(monthly, annual, forecast, policy, status, config)
