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
class GoldRegimeSnapshot:
    current: dict[str, Any]
    history: pd.DataFrame
    etf_unavailable_tickers: list[str] = field(default_factory=list)
    etf_available_tickers: list[str] = field(default_factory=list)
    cot_contract_market_name: str | None = None
    freshness: dict[str, Freshness] = field(default_factory=dict)
