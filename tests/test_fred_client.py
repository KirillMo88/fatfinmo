import numpy as np
import pandas as pd
import pytest

from fred_client import (
    FED_LIQUIDITY_SERIES_ID,
    FRED_DEFAULT_OBSERVATION_START,
    FRED_SERIES_IDS,
    FredApiError,
    FredSeriesRequest,
    calculate_fed_liquidity,
    download_fred_series,
    fred_observations_params,
    get_fred_api_key,
    normalize_fred_series_id,
    parse_fred_observations,
)


def test_requested_fred_series_are_configured_uppercase():
    assert FRED_DEFAULT_OBSERVATION_START == "2010-01-01"
    assert FRED_SERIES_IDS == (
        "DGS2",
        "T5YIE",
        "DFII10",
        "M2SL",
        "WALCL",
        "RRPONTSYD",
        "WTREGEN",
    )


def test_normalize_fred_series_id_strips_and_uppercases():
    assert normalize_fred_series_id(" dgs2 ") == "DGS2"
    assert normalize_fred_series_id("m2sl") == "M2SL"


def test_get_fred_api_key_uses_explicit_key_before_environment(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "env_key")

    assert get_fred_api_key(" explicit_key ") == "explicit_key"


def test_get_fred_api_key_requires_configuration(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    with pytest.raises(FredApiError):
        get_fred_api_key()


def test_fred_observations_params_include_json_and_date_range():
    params = fred_observations_params(
        FredSeriesRequest("dgs2", observation_start="2016-01-01", observation_end="2026-12-31"),
        "secret",
    )

    assert params == {
        "series_id": "DGS2",
        "api_key": "secret",
        "file_type": "json",
        "sort_order": "asc",
        "observation_start": "2016-01-01",
        "observation_end": "2026-12-31",
    }


def test_download_fred_series_uses_2010_default_start(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"observations": [{"date": "2010-01-01", "value": "1.0"}]}

    def fake_get(url, params, timeout):
        captured["params"] = params
        return Response()

    monkeypatch.setenv("FRED_API_KEY", "env_key")
    monkeypatch.setattr("fred_client.httpx.get", fake_get)

    parsed = download_fred_series("dgs2")

    assert captured["params"]["observation_start"] == "2010-01-01"
    assert parsed["Date"].iloc[0] == pd.Timestamp("2010-01-01")


def test_parse_fred_observations_converts_dates_and_missing_values():
    payload = {
        "observations": [
            {"date": "2026-01-02", "value": "4.25"},
            {"date": "2026-01-05", "value": "."},
        ]
    }

    parsed = parse_fred_observations("dgs2", payload)

    assert parsed["Series_ID"].tolist() == ["DGS2", "DGS2"]
    assert parsed["Date"].tolist() == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")]
    assert parsed["Value"].iloc[0] == 4.25
    assert np.isnan(parsed["Value"].iloc[1])


def test_calculate_fed_liquidity_forward_fills_components():
    source = pd.DataFrame(
        [
            {"Series_ID": "WALCL", "Date": "2010-01-01", "Value": 100.0},
            {"Series_ID": "RRPONTSYD", "Date": "2010-01-01", "Value": 10.0},
            {"Series_ID": "WTREGEN", "Date": "2010-01-01", "Value": 1.0},
            {"Series_ID": "RRPONTSYD", "Date": "2010-01-02", "Value": 12.0},
            {"Series_ID": "WTREGEN", "Date": "2010-01-08", "Value": 3.0},
        ]
    )

    calculated = calculate_fed_liquidity(source)

    assert calculated["Series_ID"].unique().tolist() == [FED_LIQUIDITY_SERIES_ID]
    assert calculated["Date"].tolist() == [
        pd.Timestamp("2010-01-01"),
        pd.Timestamp("2010-01-02"),
        pd.Timestamp("2010-01-08"),
    ]
    assert calculated["Value"].tolist() == [89.0, 87.0, 85.0]


def test_parse_fred_observations_raises_api_error():
    with pytest.raises(FredApiError):
        parse_fred_observations("DGS2", {"error_code": 400, "error_message": "Bad request"})
