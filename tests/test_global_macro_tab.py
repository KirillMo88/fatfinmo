import pandas as pd

import global_macro_tab as macro
from fred_client import FredApiError
from global_macro_tab import MacroSeriesSpec, _macro_change_color_row, _series_row


def test_series_row_formats_percent_point_and_state_horizons():
    dates = pd.date_range("2024-01-05", periods=170, freq="W-FRI")
    percent_series = pd.Series(range(100, 270), index=dates, dtype="float64")
    bps_series = pd.Series([4.0 + idx * 0.01 for idx in range(170)], index=dates)
    state_series = pd.Series(["A"] * 166 + ["B"] * 4, index=dates)
    now = pd.Timestamp("2027-04-09")

    percent_row = _series_row(
        MacroSeriesSpec("Markets", "Test Price", "TEST", "percent", "number", "weekly", "price", percent_series),
        now,
    )
    bps_row = _series_row(
        MacroSeriesSpec("Rates", "Test Yield", "TEST", "bps", "percent", "weekly", "percentage points", bps_series),
        now,
    )
    state_row = _series_row(
        MacroSeriesSpec("Global Liquidity", "Test State", "TEST", "state", "number", "weekly", "state", state_series),
        now,
    )

    assert list(percent_row.keys())[2:9] == ["Current", "1W", "1M", "3M", "6M", "12M", "36M"]
    assert percent_row["1W"].endswith("%")
    assert bps_row["1W"] == "+0.01%"
    assert state_row["Current"] == "B"
    assert state_row["1M"] == "A"


def test_macro_change_color_row_uses_directional_rules():
    positive_row = pd.Series({"Instrument": "S&P 500 ETF", "1W": "+3.0%", "1M": "-1.0%"})
    inverse_row = pd.Series({"Instrument": "U.S. Dollar Index", "1W": "+3.0%", "1M": "-1.0%"})

    positive_styles = _macro_change_color_row(positive_row, "Markets")
    inverse_styles = _macro_change_color_row(inverse_row, "FX / USD")

    assert "#4ade80" in positive_styles[1]
    assert "#fdba74" in positive_styles[2]
    assert "#ef4444" in inverse_styles[1]
    assert "#86efac" in inverse_styles[2]


def test_fred_macro_uses_saved_archive_during_non_json_outage(tmp_path, monkeypatch):
    archive = tmp_path / "fred_series.parquet"
    pd.DataFrame({
        "Series_ID": ["T5YIE", "DGS2"],
        "Date": ["2026-10-19", "2026-10-19"],
        "Value": [2.3, 3.5],
    }).to_parquet(archive)
    monkeypatch.setattr(macro, "FRED_MACRO_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(macro, "FRED_MACRO_ARCHIVE_PATHS", (archive,))
    calls = []

    def unavailable(series_id, **_kwargs):
        calls.append(series_id)
        raise FredApiError(f"FRED returned a non-JSON response for {series_id}.")

    monkeypatch.setattr(macro, "download_fred_series", unavailable)
    result = macro._download_fred_macro_series("test_key")

    assert calls == ["T5YIE"]
    assert result["T5YIE"].iloc[-1] == 2.3
    assert result["DGS2"].iloc[-1] == 3.5
    assert result["DGS3MO"].empty


def test_fred_macro_saves_successful_observation_for_future_outage(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "FRED_MACRO_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(macro, "FRED_MACRO_ARCHIVE_PATHS", ())

    def download(series_id, **_kwargs):
        if series_id == "T5YIE":
            return pd.DataFrame({"Date": ["2026-10-20"], "Value": [2.4]})
        raise FredApiError(f"FRED returned a non-JSON response for {series_id}.")

    monkeypatch.setattr(macro, "download_fred_series", download)
    first = macro._download_fred_macro_series("test_key")
    assert first["T5YIE"].iloc[-1] == 2.4
    assert (tmp_path / "cache" / "T5YIE.csv").exists()

    monkeypatch.setattr(macro, "download_fred_series", lambda sid, **_kwargs: (_ for _ in ()).throw(FredApiError(f"FRED returned a non-JSON response for {sid}.")))
    second = macro._download_fred_macro_series("test_key")
    assert second["T5YIE"].iloc[-1] == 2.4
