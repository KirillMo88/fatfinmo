from pathlib import Path

import numpy as np
import pandas as pd

import app as application
from app import _liquidity_cycle_maturity_pct, _liquidity_ordinary_roc


def test_liquidity_summary_roc_is_plain_percentage_change():
    frame = pd.DataFrame({"value": [100.0, 102.0, 105.0, 110.0]})

    assert np.isclose(_liquidity_ordinary_roc(frame, "value", 1), 110.0 / 105.0 * 100.0 - 100.0)
    assert np.isclose(_liquidity_ordinary_roc(frame, "value", 3), 10.0)


def test_liquidity_cycle_maturity_uses_october_2022_trough_and_65_month_length():
    maturity = _liquidity_cycle_maturity_pct(pd.Timestamp("2026-09-18"))
    expected_months = 47 + 17 / 30.4375

    assert np.isclose(maturity, expected_months / 65.0 * 100.0)
    assert round(maturity) == 73


def test_asset_allocation_cycle_image_is_bundled_with_app():
    image = Path(application.__file__).resolve().parent / "assets" / "asset-allocation-cycle.png"

    assert image.is_file()
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
