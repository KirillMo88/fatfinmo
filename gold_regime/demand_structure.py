from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEMAND_CATEGORIES = ["Jewellery", "Technology", "Investment", "Central Banks", "OTC and other"]
DEMAND_TONNES_COLUMNS = {
    "Jewellery": "jewellery_fabrication_tonnes",
    "Technology": "technology_tonnes",
    "Investment": "investment_tonnes",
    "Central Banks": "central_banks_tonnes",
    "OTC and other": "otc_and_other_tonnes",
}
WGC_QUARTERLY_FILENAME = "wgc_gold_balance_quarterly.csv"


def build_demand_structure_frame(data_dir: Path | None = None) -> pd.DataFrame:
    """Build Demand Structure from the normalized quarterly WGC Gold Balance data."""
    source_dir = data_dir or Path(__file__).resolve().parents[1] / "data" / "gold_regime"
    source_path = source_dir / WGC_QUARTERLY_FILENAME
    if not source_path.exists():
        raise FileNotFoundError(f"Normalized WGC quarterly data not found: {source_path}")

    source = pd.read_csv(source_path)
    required_columns = {
        "period",
        "is_published",
        "total_supply_tonnes",
        *DEMAND_TONNES_COLUMNS.values(),
    }
    missing_columns = sorted(required_columns - set(source.columns))
    if missing_columns:
        raise ValueError(f"WGC quarterly data is missing required columns: {', '.join(missing_columns)}")

    published = source["is_published"].astype(str).str.lower().eq("true")
    source = source.loc[published].copy()
    periods = pd.PeriodIndex(source["period"].astype(str), freq="Q-DEC")
    if not periods.is_unique:
        raise ValueError("WGC quarterly data contains duplicate periods")
    source["_period"] = periods
    source = source.sort_values("_period").reset_index(drop=True)
    periods = pd.PeriodIndex(source["_period"], freq="Q-DEC")

    total_supply = pd.to_numeric(source["total_supply_tonnes"], errors="coerce")
    valid_supply = total_supply.where(total_supply.gt(0))
    jewellery = pd.to_numeric(source["jewellery_fabrication_tonnes"], errors="coerce")
    technology = pd.to_numeric(source["technology_tonnes"], errors="coerce")
    investment = pd.to_numeric(source["investment_tonnes"], errors="coerce")
    central_banks = pd.to_numeric(source["central_banks_tonnes"], errors="coerce")
    otc_and_other = pd.to_numeric(source["otc_and_other_tonnes"], errors="coerce")

    shares = {
        "Jewellery": jewellery.div(valid_supply),
        "Technology": technology.div(valid_supply),
        "Investment": investment.div(valid_supply),
        "Central Banks": central_banks.div(valid_supply),
        "OTC and other": otc_and_other.div(valid_supply),
    }

    changes: dict[str, dict[int, np.ndarray]] = {}
    for category, column in DEMAND_TONNES_COLUMNS.items():
        current = pd.Series(pd.to_numeric(source[column], errors="coerce").to_numpy(), index=periods)
        changes[category] = {
            lag: current.to_numpy() - current.reindex(periods - lag).to_numpy()
            for lag in (1, 4)
        }

    rows: list[dict[str, Any]] = []
    for index, period in enumerate(periods):
        date_value = period.end_time.normalize()
        quarter_label = f"Q{period.quarter}'{period.year % 100:02d}"
        for category in DEMAND_CATEGORIES:
            rows.append(
                {
                    "date": date_value,
                    "quarter": quarter_label,
                    "category": category,
                    "demand_share": shares[category].iloc[index],
                    "demand_12m_change_tn": changes[category][4][index],
                    "demand_3m_change_tn": changes[category][1][index],
                }
            )

    frame = pd.DataFrame(rows)
    for column in ("demand_share", "demand_12m_change_tn", "demand_3m_change_tn"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame
