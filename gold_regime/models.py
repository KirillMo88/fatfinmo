from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Freshness:
    last_updated: date | None
    data_age_days: int | None
    status: str


@dataclass(frozen=True)
class GoldStructuralMacro2Snapshot:
    current: dict[str, Any]
    history: pd.DataFrame


@dataclass(frozen=True)
class GoldAISCValuationSnapshot:
    current: dict[str, Any]
    history: pd.DataFrame


@dataclass(frozen=True)
class LukeGromenGoldSnapshot:
    current: dict[str, Any]
    us_coverage_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    wgc_annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    wgc_ytd: pd.DataFrame = field(default_factory=pd.DataFrame)
    global_ca_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    static_scenarios: pd.DataFrame = field(default_factory=pd.DataFrame)
    adaptive_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    convergence: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GoldRegimeSnapshot:
    current: dict[str, Any]
    history: pd.DataFrame
    etf_unavailable_tickers: list[str] = field(default_factory=list)
    etf_available_tickers: list[str] = field(default_factory=list)
    cot_contract_market_name: str | None = None
    freshness: dict[str, Freshness] = field(default_factory=dict)
    structural_macro2: GoldStructuralMacro2Snapshot | None = None
    aisc_valuation: GoldAISCValuationSnapshot | None = None
    luke_gromen: LukeGromenGoldSnapshot | None = None
    gold_cycle_history: pd.DataFrame = field(default_factory=pd.DataFrame)
