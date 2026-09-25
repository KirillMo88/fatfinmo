import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from global_liquidity import read_global_liquidity
import app

raw, monthly, weekly = read_global_liquidity()
monthly = app._liquidity_prepare_dates(monthly)
weekly = app._liquidity_prepare_dates(weekly)
regime = app._build_global_liquidity_regime_frame(monthly, weekly)
latest = app._liquidity_latest_row_with_value(regime, "global_liquidity_score")
print(regime.shape)
print(
    latest.get("date"),
    latest.get("global_liquidity_score"),
    latest.get("direction_13w_state"),
    latest.get("long_cycle_phase"),
    latest.get("final_regime_label"),
    latest.get("data_status"),
)
