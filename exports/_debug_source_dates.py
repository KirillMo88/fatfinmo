import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exports.build_weekly_basis_ohlc_export import END, START, monthly_source_dates, format_date_series
from global_liquidity import read_global_liquidity

raw, _monthly, _weekly = read_global_liquidity()
weekly_index = pd.date_range(START, END, freq="W-FRI")
for series_id in ["M2SL", "PBOC_TOTAL_ASSETS", "BS01'MABJMTA"]:
    aligned = monthly_source_dates(raw, series_id, weekly_index)
    print(series_id)
    print(aligned.tail(8).to_string())
    print(format_date_series(aligned["observation_date"]).tail(8).to_string())
