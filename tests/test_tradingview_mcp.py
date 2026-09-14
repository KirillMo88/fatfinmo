import numpy as np
import pandas as pd

import tradingview_mcp as tv


def test_normalize_cny_to_100mn_from_trillion_units():
    value, note = tv.normalize_cny_to_100mn(223.6, unit="T CNY", expected="cnm2")

    assert np.isclose(value, 2_236_000.0)
    assert "trillion" in note


def test_normalize_cny_to_100mn_rejects_unknown_scale():
    value, note = tv.normalize_cny_to_100mn(7.5, unit="index", expected="cnm2")

    assert np.isnan(value)
    assert "unit validation failed" in note


def test_parse_economic_payload_preserves_zero_and_release_date():
    result = tv.parse_economic_payload(
        "ECONOMICS:USBCOI",
        {
            "metadata": {"description": "U.S. ISM Manufacturing PMI", "unit": "index", "frequency": "monthly"},
            "data": [
                {"date": "2024-01-01", "value": 0, "releaseDate": "2024-02-01"},
                {"date": "2024-02-01", "actual": 49.1, "releaseDate": "2024-03-01"},
            ],
        },
    )

    assert result.description == "U.S. ISM Manufacturing PMI"
    assert result.unit == "index"
    assert result.frame["value"].tolist() == [0.0, 49.1]
    assert result.frame["release_date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-02-01", "2024-03-01"]


def test_validate_economic_result_detects_partial_history():
    result = tv.TradingViewEconomicResult(
        symbol="ECONOMICS:USNMPMI",
        frame=pd.DataFrame({"date": [pd.Timestamp("2026-08-01")], "value": [52.0]}),
        description="ISM Services",
        unit="index",
        scale="",
        frequency="monthly",
        data_status="OK",
        source_mode="MCP_PRIMARY",
        notes="",
    )

    valid, status = tv.validate_economic_result(result, min_observations=24, max_stale_days=120)

    assert not valid
    assert status == "PARTIAL"
