from __future__ import annotations

import numpy as np
import pandas as pd


def relative_price_series(asset_close: pd.Series, benchmark_close: pd.Series) -> pd.Series:
    """Return an aligned asset/benchmark price series suitable for return calculations."""
    asset = pd.to_numeric(asset_close, errors="coerce").rename("asset")
    benchmark = pd.to_numeric(benchmark_close, errors="coerce").rename("benchmark")
    aligned = pd.concat([asset, benchmark], axis=1, join="inner").dropna()
    if aligned.empty:
        return pd.Series(dtype="float64", name="relative_close")
    ratio = aligned["asset"] / aligned["benchmark"].replace(0.0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    ratio.name = "relative_close"
    return ratio
