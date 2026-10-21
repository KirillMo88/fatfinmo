from __future__ import annotations

import json

import pandas as pd
import pytest

import fred_client
import finance_core
from historical_archive import (
    HistoricalArchiveError,
    read_fred_series,
    read_model_market_history,
    read_version,
    save_fred_series,
    save_model_market_history,
    save_raw_table,
)


def fred_frame(series_id: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Series_ID": series_id,
            "Date": pd.date_range("2024-01-01", periods=len(values), freq="D"),
            "Value": values,
        }
    )


def test_archive_keeps_only_last_and_previous_good(tmp_path):
    first = fred_frame("DGS2", [1.0, 2.0, 3.0])
    second = fred_frame("DGS2", [1.0, 2.0, 4.0, 5.0])
    third = fred_frame("DGS2", [1.0, 2.0, 4.0, 5.0, 6.0])

    assert save_fred_series("DGS2", first, root=tmp_path).status == "CREATED"
    assert save_fred_series("DGS2", second, root=tmp_path).status == "ROTATED"
    assert save_fred_series("DGS2", third, root=tmp_path).status == "ROTATED"

    current = read_version("FRED", "DGS2", root=tmp_path)
    previous = read_version("FRED", "DGS2", version="previous_good", root=tmp_path)
    assert current["Value"].tolist() == third["Value"].tolist()
    assert previous["Value"].tolist() == second["Value"].tolist()
    assert len(list((tmp_path / "FRED" / "DGS2").glob("*.parquet"))) == 2


def test_archive_rejects_truncated_history_without_rotating(tmp_path):
    full = fred_frame("WALCL", list(range(100)))
    truncated = full.tail(20).copy()
    save_fred_series("WALCL", full, root=tmp_path)

    with pytest.raises(HistoricalArchiveError, match="lost too much history"):
        save_fred_series("WALCL", truncated, root=tmp_path)

    stored = read_fred_series("WALCL", root=tmp_path)
    assert len(stored) == 100
    assert not (tmp_path / "FRED" / "WALCL" / "previous_good.parquet").exists()


def test_fred_client_uses_archive_when_live_source_is_unavailable(tmp_path, monkeypatch):
    save_fred_series("DGS10", fred_frame("DGS10", [4.0, 4.1, 4.2]), root=tmp_path)
    monkeypatch.setenv("HISTORICAL_ARCHIVE_DIR", str(tmp_path))
    monkeypatch.setattr(
        fred_client,
        "download_fred_series_live",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(fred_client.FredApiError("maintenance")),
    )

    result = fred_client.download_fred_series("DGS10", observation_start="2024-01-02")

    assert result["Value"].tolist() == [4.1, 4.2]
    assert result.attrs["source_status"].startswith("ARCHIVED")
    assert result.attrs["live_error"] == "maintenance"


def test_raw_table_rotation_preserves_source_columns(tmp_path):
    frame = pd.DataFrame(
        {
            "series_id": ["ECB_M2"] * 3,
            "observation_date": pd.date_range("2024-01-01", periods=3, freq="MS"),
            "raw_value": [10.0, 11.0, 12.0],
            "unit": ["EUR millions"] * 3,
        }
    )
    result = save_raw_table(
        "GLOBAL_LIQUIDITY",
        "RAW_OFFICIAL_SERIES",
        frame,
        date_column="observation_date",
        value_column="raw_value",
        group_column="series_id",
        root=tmp_path,
    )
    stored = pd.read_parquet(result.path)
    metadata = json.loads(result.path.with_name("metadata.json").read_text(encoding="utf-8"))
    assert stored["unit"].tolist() == ["EUR millions"] * 3
    assert metadata["last_good"]["rows"] == 3


def test_weekly_registry_covers_major_model_domains():
    from historical_archive_job import fred_series_registry

    registered = set(fred_series_registry())
    assert {"WALCL", "M2SL", "DGS2", "DGS10", "SOFR", "GDP", "PPIACO"} <= registered
    assert {"BOGZ1FL663067003Q", "R4247IM163SCEN", "NAPM", "BAMLH0A0HYM2"} <= registered


def test_model_market_archive_is_selective_and_is_used_as_download_fallback(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    history = pd.DataFrame(
        {
            "Open": range(60),
            "High": range(1, 61),
            "Low": range(60),
            "Close": range(1, 61),
            "Volume": [1000] * 60,
        },
        index=dates,
    )
    save_model_market_history("SPY", history, root=tmp_path)
    monkeypatch.setenv("HISTORICAL_ARCHIVE_DIR", str(tmp_path))
    monkeypatch.setattr(finance_core.yf, "download", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(finance_core.time, "sleep", lambda *_args, **_kwargs: None)

    restored = finance_core.download_completed_ohlcv("SPY", period="max")

    assert len(restored) == 60
    assert restored["Close"].iloc[-1] == 60
    assert restored.attrs["source_status"].startswith("ARCHIVED")
    assert len(read_model_market_history("SPY", root=tmp_path)) == 60

    from historical_archive_job import model_market_ticker_registry

    registered = set(model_market_ticker_registry())
    assert {"SPY", "^VIX", "DX-Y.NYB", "CL=F", "HG=F", "BTC-USD"} <= registered
    assert "AAPL" not in registered
