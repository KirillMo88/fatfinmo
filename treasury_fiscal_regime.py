from __future__ import annotations

import bisect
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fred_client import download_fred_series
from funding_conditions import STORAGE_DIR as FUNDING_STORAGE_DIR
from funding_conditions import _load_initial_release, read_snapshot as read_funding_snapshot


MODEL_VERSION = "TREASURY_FISCAL_REGIME_V1"
STORAGE_DIR = Path("persistent") / "treasury_fiscal_regime"
SNAPSHOT_PATH = STORAGE_DIR / "weekly_history.parquet"
LIQUIDITY_PATH = STORAGE_DIR / "liquidity_history.parquet"
FISCAL_PATH = STORAGE_DIR / "fiscal_history.parquet"
FINANCING_PATH = STORAGE_DIR / "financing_history.parquet"
STATUS_PATH = STORAGE_DIR / "status.json"

FAST_LIQUIDITY_WEEKS = 4
MEDIUM_LIQUIDITY_WEEKS = 13
SLOW_LIQUIDITY_WEEKS = 26
FAST_LIQUIDITY_WEIGHT = 0.30
MEDIUM_LIQUIDITY_WEIGHT = 0.50
SLOW_LIQUIDITY_WEIGHT = 0.20
LIQUIDITY_INJECTION_HIGH = 0.50
LIQUIDITY_INJECTION_LOW = 0.15
LIQUIDITY_DRAIN_LOW = -0.15
LIQUIDITY_DRAIN_HIGH = -0.50
FISCAL_FAST_MONTHS = 3
FISCAL_MEDIUM_MONTHS = 6
FISCAL_STRUCTURAL_MONTHS = 12
FAST_FISCAL_WEIGHT = 0.50
MEDIUM_FISCAL_WEIGHT = 0.30
STRUCTURAL_FISCAL_WEIGHT = 0.20
FISCAL_POSITIVE_THRESHOLD = 0.25
FISCAL_NEGATIVE_THRESHOLD = -0.25
SUPPLY_ELEVATED_PCTL = 0.75
SUPPLY_HIGH_PCTL = 0.90
BILL_HEAVY_THRESHOLD = 0.60
DURATION_HEAVY_THRESHOLD = 0.40
DRIVER_DOMINANCE_THRESHOLD = 0.60
RRP_DEPLETED_BN = 50.0
FINANCING_MAP_TRAIL_QUARTERS = 8
POLICY_MIX_TRAIL_YEARS = 10
MIN_WEEKLY_HISTORY = 52
MIN_MONTHLY_HISTORY = 24
MIN_QUARTERLY_HISTORY = 8
FISCAL_LAG_DAYS_AFTER_MONTH_END = 20
FINANCING_LAG_DAYS_AFTER_QUARTER_END = 80

SOURCE_SPECS = {
    "WALCL": ("weekly", "USD millions", "liquidity"),
    "WTREGEN": ("weekly", "USD millions", "liquidity"),
    "WDTGAL": ("weekly", "USD millions", "liquidity"),
    "RRPONTSYD": ("daily", "USD billions", "liquidity"),
    "WRESBAL": ("weekly", "USD millions", "liquidity"),
    "MTSR133FMS": ("monthly", "USD millions", "fiscal"),
    "MTSO133FMS": ("monthly", "USD millions", "fiscal"),
    "MTSDS133FMS": ("monthly", "USD millions", "fiscal"),
    "GDP": ("quarterly", "USD billions SAAR", "shared"),
    "FGTSL": ("quarterly", "USD millions", "financing"),
    "BOGZ1FL313161110Q": ("quarterly", "USD millions", "financing"),
}
REQUIRED_LIQUIDITY = ("WALCL", "WTREGEN", "RRPONTSYD", "WRESBAL", "GDP")
REQUIRED_FISCAL = ("MTSR133FMS", "MTSO133FMS", "GDP")
REQUIRED_FINANCING = ("FGTSL", "BOGZ1FL313161110Q", "GDP")


@dataclass
class TreasuryFiscalSnapshot:
    weekly: pd.DataFrame
    liquidity: pd.DataFrame
    fiscal: pd.DataFrame
    financing: pd.DataFrame
    status: dict


def pit_z(values: pd.Series, minimum: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    previous = numeric.shift()
    mean = previous.expanding(min_periods=minimum).mean()
    std = previous.expanding(min_periods=minimum).std(ddof=0).replace(0, np.nan)
    return (numeric - mean) / std


def pit_percentile(values: pd.Series, minimum: int) -> pd.Series:
    result = []
    previous: list[float] = []
    for value in pd.to_numeric(values, errors="coerce"):
        result.append(bisect.bisect_right(previous, float(value)) / len(previous)
                      if pd.notna(value) and len(previous) >= minimum else np.nan)
        if pd.notna(value):
            bisect.insort(previous, float(value))
    return pd.Series(result, index=values.index, dtype="float64")


def _empty_source() -> pd.DataFrame:
    return pd.DataFrame(columns=["ObservationDate", "AvailableDate", "Value"])


def _clean_source(source: pd.DataFrame) -> pd.DataFrame:
    if source is None or source.empty:
        return _empty_source()
    frame = source[["ObservationDate", "AvailableDate", "Value"]].copy()
    frame["ObservationDate"] = pd.to_datetime(frame["ObservationDate"], errors="coerce").dt.normalize()
    frame["AvailableDate"] = pd.to_datetime(frame["AvailableDate"], errors="coerce").dt.normalize()
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    return frame.dropna().sort_values(["ObservationDate", "AvailableDate"]).drop_duplicates(
        "ObservationDate", keep="first"
    ).reset_index(drop=True)


def _asof(source: pd.DataFrame, dates: pd.Series | pd.DatetimeIndex, column: str = "Value") -> pd.Series:
    target = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if source.empty:
        return pd.Series(np.nan, index=range(len(target)), dtype="float64")
    available = source.sort_values(["AvailableDate", "ObservationDate"]).drop_duplicates(
        "AvailableDate", keep="last"
    )
    values = pd.Series(available[column].to_numpy(), index=pd.DatetimeIndex(available["AvailableDate"]))
    aligned = values.reindex(values.index.union(target)).sort_index().ffill().reindex(target)
    return pd.Series(aligned.to_numpy(), dtype=aligned.dtype)


def _source_asof(source: pd.DataFrame, dates: pd.Series | pd.DatetimeIndex) -> pd.Series:
    return _asof(source, dates)


def _released_value(source: pd.DataFrame, column: str, scale: float = 1.0) -> pd.DataFrame:
    result = _clean_source(source)
    result[column] = result["Value"] * scale
    return result[["ObservationDate", "AvailableDate", column]]


def _millions_from_initial_vintage(source: pd.DataFrame) -> pd.DataFrame:
    result = source.copy()
    if result.empty:
        return result
    raw = pd.to_numeric(result["Value"], errors="coerce")
    switch = raw.gt(100_000) & raw.shift().lt(10_000) & raw.div(raw.shift()).gt(100)
    if switch.any():
        position = int(np.flatnonzero(switch.to_numpy())[0])
        result.loc[result.index[:position], "Value"] = raw.iloc[:position] * 1000
    elif raw.max() < 10_000:
        result["Value"] = raw * 1000
    elif raw.min() <= 10_000:
        raise ValueError("Mixed reserve/TGA units without a detectable vintage-scale transition")
    return result


def _state_from_percentile(value: float, low: str = "LOW", middle: str = "NORMAL") -> str:
    if pd.isna(value):
        return "DATA UNAVAILABLE"
    if value >= 0.90:
        return "HIGH"
    if value >= 0.75:
        return "ELEVATED"
    if value >= 0.50:
        return middle
    return low


def _liquidity_state(value: float) -> str:
    if pd.isna(value):
        return "DATA UNAVAILABLE"
    if value > LIQUIDITY_INJECTION_HIGH:
        return "INJECTION"
    if value > LIQUIDITY_INJECTION_LOW:
        return "MILD INJECTION"
    if value < LIQUIDITY_DRAIN_HIGH:
        return "DRAIN"
    if value < LIQUIDITY_DRAIN_LOW:
        return "MILD DRAIN"
    return "NEUTRAL"


def _fiscal_state(value: float) -> str:
    if pd.isna(value):
        return "DATA UNAVAILABLE"
    if value > FISCAL_POSITIVE_THRESHOLD:
        return "POSITIVE / ACCELERATING"
    if value < FISCAL_NEGATIVE_THRESHOLD:
        return "NEGATIVE / DECELERATING"
    return "NEUTRAL"


def _driver(frame: pd.DataFrame, horizon: int) -> pd.Series:
    contributions = frame[[f"FedImpulse_{horizon}W", f"TGAImpulse_{horizon}W", f"RRPImpulse_{horizon}W"]].abs()
    denominator = contributions.sum(axis=1, min_count=3).replace(0, np.nan)
    shares = contributions.div(denominator, axis=0)
    for label, column in zip(("Fed", "TGA", "RRP"), shares):
        frame[f"{label}ContributionShare_{horizon}W"] = shares[column]
    maximum = shares.max(axis=1)
    names = shares.fillna(-1).idxmax(axis=1).map({
        f"FedImpulse_{horizon}W": "FED-DRIVEN",
        f"TGAImpulse_{horizon}W": "TGA-DRIVEN",
        f"RRPImpulse_{horizon}W": "RRP-DRIVEN",
    })
    result = names.where(maximum.gt(DRIVER_DOMINANCE_THRESHOLD), "MIXED")
    return result.where(denominator.notna(), "DATA UNAVAILABLE")


def build_liquidity(sources: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame({"Date": dates})
    frame["WALCL"] = _source_asof(sources["WALCL"], dates) / 1000
    frame["WTREGEN"] = _source_asof(_millions_from_initial_vintage(sources["WTREGEN"]), dates) / 1000
    frame["WDTGAL"] = _source_asof(sources["WDTGAL"], dates) / 1000
    frame["RRP"] = _source_asof(sources["RRPONTSYD"], dates)
    frame["WRESBAL"] = _source_asof(_millions_from_initial_vintage(sources["WRESBAL"]), dates) / 1000
    frame["GDP"] = _source_asof(sources["GDP"], dates)
    frame["NetLiquidity"] = frame["WALCL"] - frame["WTREGEN"] - frame["RRP"]
    frame["NetLiquidityLevel"] = frame["NetLiquidity"] / frame["GDP"]
    frame["NetLiquidityLevelPercentile"] = pit_percentile(frame["NetLiquidityLevel"], MIN_WEEKLY_HISTORY)
    frame["TGAFastGap"] = frame["WDTGAL"] - frame["WTREGEN"]
    frame["RRPBufferRaw"] = frame["RRP"] / frame["GDP"]
    frame["RRPBufferPercentile"] = pit_percentile(frame["RRPBufferRaw"], MIN_WEEKLY_HISTORY)
    frame["ReserveBufferRaw"] = frame["WRESBAL"] / frame["GDP"]
    frame["ReserveBufferPercentile"] = pit_percentile(frame["ReserveBufferRaw"], MIN_WEEKLY_HISTORY)
    frame["RRPBuffer"] = frame.apply(
        lambda row: "DATA UNAVAILABLE" if pd.isna(row["RRPBufferPercentile"]) else
        "DEPLETED" if row["RRP"] <= RRP_DEPLETED_BN and row["RRPBufferPercentile"] < 0.25 else
        "LOW BUFFER" if row["RRPBufferPercentile"] < 0.25 else
        "HIGH BUFFER" if row["RRPBufferPercentile"] >= 0.75 else "NORMAL BUFFER", axis=1,
    )
    for horizon, label in ((FAST_LIQUIDITY_WEEKS, "Fast"), (MEDIUM_LIQUIDITY_WEEKS, "Medium"),
                           (SLOW_LIQUIDITY_WEEKS, "Slow")):
        frame[f"{label}LiquidityImpulse"] = frame["NetLiquidity"].diff(horizon)
        frame[f"{label}LiquidityZ"] = pit_z(frame[f"{label}LiquidityImpulse"], MIN_WEEKLY_HISTORY)
        frame[f"FedImpulse_{horizon}W"] = frame["WALCL"].diff(horizon)
        frame[f"TGAImpulse_{horizon}W"] = -frame["WTREGEN"].diff(horizon)
        frame[f"RRPImpulse_{horizon}W"] = -frame["RRP"].diff(horizon)
        frame[f"LiquidityDriver_{horizon}W"] = _driver(frame, horizon)
    frame["TreasuryLiquidityImpulse"] = (
        FAST_LIQUIDITY_WEIGHT * frame["FastLiquidityZ"] +
        MEDIUM_LIQUIDITY_WEIGHT * frame["MediumLiquidityZ"] +
        SLOW_LIQUIDITY_WEIGHT * frame["SlowLiquidityZ"]
    )
    frame["TreasuryLiquidityState"] = frame["TreasuryLiquidityImpulse"].map(_liquidity_state)
    frame["LiquidityDriver"] = frame[f"LiquidityDriver_{MEDIUM_LIQUIDITY_WEEKS}W"]
    return frame


def _monthly_pair(sources: dict[str, pd.DataFrame], first: str, second: str) -> pd.DataFrame:
    left = _clean_source(sources[first]).rename(columns={"AvailableDate": "FirstAvailable", "Value": first})
    right = _clean_source(sources[second]).rename(columns={"AvailableDate": "SecondAvailable", "Value": second})
    merged = left[["ObservationDate", "FirstAvailable", first]].merge(
        right[["ObservationDate", "SecondAvailable", second]], on="ObservationDate", how="outer", validate="one_to_one"
    )
    if merged.empty:
        return merged
    months = pd.date_range(merged["ObservationDate"].min(), merged["ObservationDate"].max(), freq="MS")
    merged = merged.set_index("ObservationDate").reindex(months).rename_axis("ObservationDate").reset_index()
    merged["AvailableDate"] = merged[["FirstAvailable", "SecondAvailable"]].max(axis=1)
    merged.loc[merged[[first, second]].isna().any(axis=1), "AvailableDate"] = pd.NaT
    return merged


def build_fiscal(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fiscal = _monthly_pair(sources, "MTSR133FMS", "MTSO133FMS")
    if fiscal.empty:
        return pd.DataFrame(columns=["ObservationDate", "AvailableDate"])
    fiscal["MonthlyReceipts"] = fiscal["MTSR133FMS"] / 1000
    fiscal["MonthlyOutlays"] = fiscal["MTSO133FMS"] / 1000
    fiscal["Receipts12M"] = fiscal["MonthlyReceipts"].rolling(12, min_periods=12).sum()
    fiscal["Outlays12M"] = fiscal["MonthlyOutlays"].rolling(12, min_periods=12).sum()
    fiscal["Deficit12M"] = fiscal["Outlays12M"] - fiscal["Receipts12M"]
    fiscal["GDP"] = _asof(sources["GDP"], fiscal["AvailableDate"].fillna(pd.Timestamp("1900-01-01")))
    fiscal["FiscalStanceRaw"] = fiscal["Deficit12M"] / fiscal["GDP"]
    fiscal["FiscalStancePercentile"] = pit_percentile(fiscal["FiscalStanceRaw"], MIN_MONTHLY_HISTORY)
    fiscal["FiscalStanceState"] = fiscal["FiscalStancePercentile"].map(
        lambda value: _state_from_percentile(value, middle="MODERATE")
    )
    for months, label in ((FISCAL_FAST_MONTHS, "Fast"), (FISCAL_MEDIUM_MONTHS, "Medium"),
                          (FISCAL_STRUCTURAL_MONTHS, "Structural")):
        fiscal[f"{label}FiscalImpulseRaw"] = fiscal["FiscalStanceRaw"].diff(months)
        fiscal[f"{label}FiscalZ"] = pit_z(fiscal[f"{label}FiscalImpulseRaw"], MIN_MONTHLY_HISTORY)
        fiscal[f"SpendingImpulse_{months}M"] = (fiscal["Outlays12M"] / fiscal["GDP"]).diff(months)
        fiscal[f"RevenueImpulse_{months}M"] = -(fiscal["Receipts12M"] / fiscal["GDP"]).diff(months)
    fiscal["SpendingImpulse"] = fiscal[f"SpendingImpulse_{FISCAL_FAST_MONTHS}M"]
    fiscal["RevenueImpulse"] = fiscal[f"RevenueImpulse_{FISCAL_FAST_MONTHS}M"]
    fiscal["FiscalImpulse"] = (
        FAST_FISCAL_WEIGHT * fiscal["FastFiscalZ"] +
        MEDIUM_FISCAL_WEIGHT * fiscal["MediumFiscalZ"] +
        STRUCTURAL_FISCAL_WEIGHT * fiscal["StructuralFiscalZ"]
    )
    fiscal["FiscalImpulseState"] = fiscal["FiscalImpulse"].map(_fiscal_state)
    deficit = _clean_source(sources["MTSDS133FMS"])
    fiscal["MTSReportedDeficit"] = _asof(deficit, fiscal["AvailableDate"].fillna(pd.Timestamp("1900-01-01"))) / -1000
    return fiscal


def build_financing(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    financing = _monthly_pair(sources, "FGTSL", "BOGZ1FL313161110Q")
    if financing.empty:
        return pd.DataFrame(columns=["ObservationDate", "AvailableDate"])
    financing = financing.loc[financing["ObservationDate"].dt.month.isin([1, 4, 7, 10])].reset_index(drop=True)
    financing["FGTSL"] = financing["FGTSL"] / 1000
    financing["Bills"] = financing["BOGZ1FL313161110Q"] / 1000
    financing["NonBillDebt"] = financing["FGTSL"] - financing["Bills"]
    financing["GDP"] = _asof(sources["GDP"], financing["AvailableDate"].fillna(pd.Timestamp("1900-01-01")))
    for level, prefix in (("FGTSL", "Total"), ("Bills", "Bill"), ("NonBillDebt", "Duration")):
        financing[f"{prefix}NetIssuance4Q"] = financing[level].diff(4)
        financing[f"{prefix}SupplyLoadRaw"] = financing[f"{prefix}NetIssuance4Q"] / financing["GDP"]
        financing[f"{prefix}SupplyPercentile"] = pit_percentile(
            financing[f"{prefix}SupplyLoadRaw"], MIN_QUARTERLY_HISTORY
        )
        financing[f"{prefix}SupplyState"] = financing[f"{prefix}SupplyPercentile"].map(_state_from_percentile)
    positive = financing["TotalNetIssuance4Q"].gt(0)
    financing["BillFinancingShare"] = (financing["BillNetIssuance4Q"] /
                                       financing["TotalNetIssuance4Q"].where(positive))
    financing["DurationFinancingShare"] = 1 - financing["BillFinancingShare"]
    mix = pd.Series("DATA UNAVAILABLE", index=financing.index)
    mix.loc[financing["TotalNetIssuance4Q"].le(0)] = "CONTRACTION / NOT APPLICABLE"
    mix.loc[positive & financing["BillFinancingShare"].gt(BILL_HEAVY_THRESHOLD)] = "BILL HEAVY"
    mix.loc[positive & financing["BillFinancingShare"].between(DURATION_HEAVY_THRESHOLD, BILL_HEAVY_THRESHOLD)] = "BALANCED"
    mix.loc[positive & financing["BillFinancingShare"].lt(DURATION_HEAVY_THRESHOLD)] = "DURATION HEAVY"
    mix.loc[positive & financing["BillNetIssuance4Q"].lt(0)] = "STRONGLY DURATION HEAVY"
    financing["FinancingMix"] = mix
    return financing


def _absorption(row: pd.Series) -> str:
    required = ("ReservePressurePercentile", "MoneyMarketStressPercentile", "PersistentFundingFlag", "FundingState")
    if any(pd.isna(row.get(field)) for field in required) or row["RRPBuffer"] == "DATA UNAVAILABLE":
        return "DATA UNAVAILABLE"
    if (row["ReservePressurePercentile"] >= 0.75 or bool(row["PersistentFundingFlag"]) or
            row["FundingState"] in {"PERSISTENT FUNDING PRESSURE", "SYSTEMIC FUNDING STRESS"}):
        return "TIGHT"
    if (row["ReservePressurePercentile"] < 0.50 and row["MoneyMarketStressPercentile"] < 0.50 and
            row["RRPBuffer"] in {"NORMAL BUFFER", "HIGH BUFFER"}):
        return "AMPLE"
    return "NORMAL"


def _financing_pressure(row: pd.Series) -> str:
    total, duration, absorption = row["TotalSupplyPercentile"], row["DurationSupplyPercentile"], row["AbsorptionCapacity"]
    if pd.isna(total) or pd.isna(duration) or absorption == "DATA UNAVAILABLE":
        return "DATA UNAVAILABLE"
    confirmed = row["FundingState"] in {"PERSISTENT FUNDING PRESSURE", "SYSTEMIC FUNDING STRESS"}
    if duration >= SUPPLY_HIGH_PCTL and absorption == "TIGHT" and confirmed:
        return "SEVERE"
    if duration >= SUPPLY_HIGH_PCTL and (total >= SUPPLY_ELEVATED_PCTL or absorption == "TIGHT"):
        return "HIGH"
    if duration >= SUPPLY_ELEVATED_PCTL or (total >= SUPPLY_ELEVATED_PCTL and absorption != "AMPLE"):
        return "ELEVATED"
    if total < 0.50 and duration < 0.50:
        return "LOW"
    return "MODERATE"


def _policy_mix(row: pd.Series) -> str:
    fiscal = row["FiscalImpulseState"]
    liquidity = row["TreasuryLiquidityState"]
    pressure = row["TreasuryFinancingPressure"]
    if any(not isinstance(state, str) or state == "DATA UNAVAILABLE"
           for state in (fiscal, liquidity, pressure)):
        return "DATA UNAVAILABLE"
    fiscal_sign = 1 if fiscal.startswith("POSITIVE") else -1 if fiscal.startswith("NEGATIVE") else 0
    liquidity_sign = 1 if "INJECTION" in liquidity else -1 if "DRAIN" in liquidity else 0
    high_pressure = pressure in {"ELEVATED", "HIGH", "SEVERE"}
    if fiscal_sign > 0:
        if liquidity_sign == 0 and not high_pressure:
            return "FISCAL SUPPORT / LIQUIDITY NEUTRAL"
        return ("FISCAL & LIQUIDITY SUPPORT" if liquidity_sign > 0 and not high_pressure
                else "FISCAL SUPPORT / TREASURY PRESSURE")
    if fiscal_sign < 0:
        if liquidity_sign > 0 and not high_pressure:
            return "LIQUIDITY SUPPORT / FISCAL PRESSURE"
        return ("FISCAL PRESSURE / LIQUIDITY NEUTRAL" if liquidity_sign == 0 and not high_pressure
                else "FISCAL & TREASURY PRESSURE")
    if liquidity_sign > 0 and not high_pressure:
        return "LIQUIDITY SUPPORT / FISCAL NEUTRAL"
    if liquidity_sign < 0 or high_pressure:
        return "TREASURY PRESSURE / FISCAL NEUTRAL"
    return "NEUTRAL POLICY MIX"


def _funding_history(funding: pd.DataFrame) -> pd.DataFrame:
    if funding is None or funding.empty:
        return pd.DataFrame()
    required = ["Date", "ReservePressure", "MoneyMarketStress", "PersistentFundingFlag", "FundingState", "CollateralStress"]
    if any(column not in funding for column in required):
        return pd.DataFrame()
    data = funding[required].copy().sort_values("Date").reset_index(drop=True)
    data["ReservePressurePercentile"] = pit_percentile(data["ReservePressure"], MIN_WEEKLY_HISTORY)
    data["MoneyMarketStressPercentile"] = pit_percentile(data["MoneyMarketStress"], MIN_WEEKLY_HISTORY)
    return data


def build_snapshot(
    sources: dict[str, pd.DataFrame], funding: pd.DataFrame, end_date: pd.Timestamp | None = None,
) -> TreasuryFiscalSnapshot:
    cleaned = {series_id: _clean_source(sources.get(series_id, _empty_source())) for series_id in SOURCE_SPECS}
    end = pd.Timestamp(end_date or pd.Timestamp.now(tz="UTC").tz_localize(None)).normalize()
    dates = pd.date_range("2010-01-01", end, freq="W-FRI")
    liquidity = build_liquidity(cleaned, dates)
    fiscal = build_fiscal(cleaned)
    financing = build_financing(cleaned)
    weekly = liquidity.copy()
    for lower, fields in (
        (fiscal, ("MonthlyReceipts", "MonthlyOutlays", "Receipts12M", "Outlays12M", "Deficit12M",
                  "FiscalStanceRaw", "FiscalStancePercentile", "FiscalStanceState",
                  "FastFiscalImpulseRaw", "MediumFiscalImpulseRaw", "StructuralFiscalImpulseRaw",
                  "FastFiscalZ", "MediumFiscalZ", "StructuralFiscalZ", "FiscalImpulse", "FiscalImpulseState",
                  "SpendingImpulse", "RevenueImpulse", "MTSReportedDeficit",
                  "SpendingImpulse_3M", "SpendingImpulse_6M", "SpendingImpulse_12M",
                  "RevenueImpulse_3M", "RevenueImpulse_6M", "RevenueImpulse_12M")),
        (financing, ("FGTSL", "Bills", "NonBillDebt", "TotalNetIssuance4Q", "BillNetIssuance4Q",
                     "DurationNetIssuance4Q", "TotalSupplyLoadRaw", "BillSupplyLoadRaw",
                     "DurationSupplyLoadRaw", "TotalSupplyPercentile", "BillSupplyPercentile",
                     "DurationSupplyPercentile", "BillFinancingShare", "DurationFinancingShare",
                     "FinancingMix")),
    ):
        prefix = "Fiscal" if lower is fiscal else "Financing"
        lower = lower.copy()
        if lower.empty:
            weekly[f"{prefix}ObservationDate"] = pd.NaT
            weekly[f"{prefix}AvailableDate"] = pd.NaT
            for field in fields:
                weekly[field] = np.nan
            continue
        lower[f"{prefix}ObservationDate"] = lower["ObservationDate"]
        lower[f"{prefix}AvailableDate"] = lower["AvailableDate"]
        for field in (*fields, f"{prefix}ObservationDate", f"{prefix}AvailableDate"):
            weekly[field] = _asof(lower.dropna(subset=["AvailableDate"]), dates, field)
    funding_frame = _funding_history(funding)
    if funding_frame.empty:
        for field in ("ReservePressurePercentile", "MoneyMarketStressPercentile", "PersistentFundingFlag",
                      "FundingState", "CollateralStress"):
            weekly[field] = np.nan
    else:
        funding_frame = funding_frame.rename(columns={"Date": "AvailableDate"})
        funding_frame["ObservationDate"] = funding_frame["AvailableDate"]
        for field in ("ReservePressurePercentile", "MoneyMarketStressPercentile", "PersistentFundingFlag",
                      "FundingState", "CollateralStress"):
            weekly[field] = _asof(funding_frame, dates, field)
    weekly["AbsorptionCapacity"] = weekly.apply(_absorption, axis=1)
    weekly["AbsorptionTightness"] = weekly["AbsorptionCapacity"].map({"AMPLE": 25, "NORMAL": 50, "TIGHT": 85})
    weekly["TreasuryFinancingPressure"] = weekly.apply(_financing_pressure, axis=1)
    weekly["PolicyMix"] = weekly.apply(_policy_mix, axis=1)
    weekly["TreasuryFiscalModelVersion"] = MODEL_VERSION
    return TreasuryFiscalSnapshot(weekly, liquidity, fiscal, financing, {})


def load_sources(api_key: str | None = None, refresh: bool = False) -> tuple[dict[str, pd.DataFrame], dict]:
    sources: dict[str, pd.DataFrame] = {}
    status: dict[str, dict] = {}
    for series_id, (frequency, unit, domain) in SOURCE_SPECS.items():
        cache_dir = FUNDING_STORAGE_DIR if series_id in {"WRESBAL", "GDP"} else STORAGE_DIR
        try:
            source, state = _load_initial_release(series_id, api_key, refresh, cache_dir=cache_dir)
            source = _clean_source(source)
        except Exception as exc:
            if series_id in {"WRESBAL", "GDP"}:
                source, state = _empty_source(), f"DATA UNAVAILABLE: {type(exc).__name__}"
            else:
                try:
                    source, state = _load_delayed_current_vintage(series_id, api_key, refresh)
                except Exception as fallback_exc:
                    source, state = _empty_source(), f"DATA UNAVAILABLE: {type(fallback_exc).__name__}"
        sources[series_id] = source
        last = source.iloc[-1] if not source.empty else None
        status[series_id] = {
            "State": state, "Frequency": frequency, "Unit": unit, "Domain": domain,
            "ObservationDate": str(last["ObservationDate"].date()) if last is not None else None,
            "AvailableDate": str(last["AvailableDate"].date()) if last is not None else None,
            "Source": f"https://fred.stlouisfed.org/series/{series_id}",
        }
    return sources, status


def _load_delayed_current_vintage(
    series_id: str, api_key: str | None, refresh: bool,
) -> tuple[pd.DataFrame, str]:
    path = STORAGE_DIR / f"fred_{series_id.lower()}_delayed.parquet"
    cached = pd.read_parquet(path) if path.exists() else _empty_source()
    fresh = path.exists() and pd.Timestamp.now().timestamp() - path.stat().st_mtime < 20 * 3600
    if fresh and not refresh and not cached.empty:
        return _clean_source(cached), "FRED_REVISED_HISTORY_DELAYED_CACHE"
    try:
        frame = download_fred_series(series_id, api_key=api_key, observation_start="2010-01-01")
        result = pd.DataFrame({
            "ObservationDate": pd.to_datetime(frame["Date"], errors="coerce"),
            "Value": pd.to_numeric(frame["Value"], errors="coerce"),
        }).dropna()
        frequency = SOURCE_SPECS[series_id][0]
        if frequency == "daily":
            result["AvailableDate"] = result["ObservationDate"] + pd.offsets.BDay(1)
        elif frequency == "monthly":
            result["AvailableDate"] = result["ObservationDate"] + pd.offsets.MonthEnd(0) + pd.Timedelta(
                days=FISCAL_LAG_DAYS_AFTER_MONTH_END
            )
        elif frequency == "quarterly":
            result["AvailableDate"] = result["ObservationDate"] + pd.offsets.QuarterEnd(0) + pd.Timedelta(
                days=FINANCING_LAG_DAYS_AFTER_QUARTER_END
            )
        else:
            raise RuntimeError(f"No conservative release lag for {series_id}")
        result = _clean_source(result)
        if result.empty:
            raise RuntimeError(f"{series_id} has no numeric observations")
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        result.to_parquet(path, index=False)
        return result, "FRED_REVISED_HISTORY_CONSERVATIVE_LAG"
    except Exception:
        if not cached.empty:
            return _clean_source(cached), "STALE_FRED_REVISED_HISTORY_DELAYED_CACHE"
        raise


def refresh_snapshot(api_key: str | None = None, refresh: bool = False) -> TreasuryFiscalSnapshot:
    sources, source_status = load_sources(api_key, refresh)
    funding = read_funding_snapshot().weekly
    snapshot = build_snapshot(sources, funding)
    status = {
        "ModelVersion": MODEL_VERSION,
        "CalculatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "SourceStatus": source_status,
        "FundingStatus": "DATA UNAVAILABLE" if funding.empty else "REUSED FUNDING CONDITIONS WEEKLY OUTPUT",
        "TimingConvention": "FRED first-release vintages; lower-frequency values become usable only on AvailableDate.",
    }
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    for frame, path in ((snapshot.weekly, SNAPSHOT_PATH), (snapshot.liquidity, LIQUIDITY_PATH),
                        (snapshot.fiscal, FISCAL_PATH), (snapshot.financing, FINANCING_PATH)):
        temporary = path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(temporary, STATUS_PATH)
    snapshot.status = status
    return snapshot


def read_snapshot() -> TreasuryFiscalSnapshot:
    def read(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return TreasuryFiscalSnapshot(
        read(SNAPSHOT_PATH), read(LIQUIDITY_PATH), read(FISCAL_PATH), read(FINANCING_PATH),
        json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.exists() else {},
    )
