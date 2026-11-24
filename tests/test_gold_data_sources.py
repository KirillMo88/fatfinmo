from __future__ import annotations

import pandas as pd

import tradingview_mcp
from gold_regime import precious_metals
from gold_regime import trading_system_data


def _series(value: float) -> pd.Series:
    return pd.Series(
        [value, value * 1.1],
        index=pd.to_datetime(["2026-01-02", "2026-01-09"]),
        dtype="float64",
    )


def _ohlc(value: float) -> pd.DataFrame:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-02-02"])
    return pd.DataFrame(
        {
            "open": [value, value + 1.0, value + 2.0],
            "high": [value + 2.0, value + 3.0, value + 4.0],
            "low": [value - 2.0, value - 1.0, value],
            "close": [value + 1.0, value + 2.0, value + 3.0],
        },
        index=dates,
    )


def test_precious_metals_uses_yahoo_futures_and_only_xauusd_from_tradingview(monkeypatch):
    yahoo = {
        "GLD": _series(100.0),
        "GDX": _series(50.0),
        "SIL": _series(40.0),
        "SLV": _series(20.0),
        "SI=F": _series(30.0),
        "PA=F": _series(900.0),
        "PL=F": _series(1_200.0),
    }
    monkeypatch.setattr(
        precious_metals,
        "_download_yahoo_weekly",
        lambda: (yahoo, {ticker: "Yahoo Finance" for ticker in yahoo}),
    )
    tradingview_calls: list[str] = []

    def fake_tradingview(symbol: str, **_kwargs) -> pd.DataFrame:
        tradingview_calls.append(symbol)
        return pd.DataFrame(
            {
                "date": ["2026-01-02", "2026-01-09"],
                "close": [4_000.0, 4_100.0],
            }
        )

    monkeypatch.setattr(tradingview_mcp, "get_ohlcv_data", fake_tradingview)

    history, status = precious_metals.load_precious_metals_ratios()

    assert tradingview_calls == ["OANDA:XAUUSD"]
    assert precious_metals.YAHOO_METAL_TICKERS == {
        "SILVER": "SI=F",
        "PALLADIUM": "PA=F",
        "PLATINUM": "PL=F",
    }
    assert list(precious_metals.TRADINGVIEW_SYMBOLS.values()) == ["OANDA:XAUUSD"]
    assert history[["Silver / XAUUSD", "Palladium / XAUUSD", "Platinum / XAUUSD"]].notna().all().all()
    assert status["SI=F"] == "Yahoo Finance"
    assert status["PA=F"] == "Yahoo Finance"
    assert status["PL=F"] == "Yahoo Finance"


def test_trading_system_uses_yahoo_for_silver_and_platinum(monkeypatch, tmp_path):
    tradingview_calls: list[tuple[str, str]] = []
    yahoo_calls: list[str] = []

    def fake_tradingview(symbol: str, interval: str, _status: dict[str, str]) -> pd.DataFrame:
        tradingview_calls.append((symbol, interval))
        return _ohlc(4_000.0)

    def fake_yahoo(symbol: str, _cache_dir) -> tuple[pd.DataFrame, str]:
        yahoo_calls.append(symbol)
        values = {"SI=F": 30.0, "PL=F": 1_200.0, "3GOL.L": 100.0, "3SIL.L": 100.0}
        return _ohlc(values[symbol]), " · test"

    monkeypatch.setattr(trading_system_data, "_tradingview_bars", fake_tradingview)
    monkeypatch.setattr(trading_system_data, "_incremental_yahoo_ohlc", fake_yahoo)
    monkeypatch.setattr(trading_system_data, "_yahoo_currency", lambda _ticker: "USD")
    monkeypatch.chdir(tmp_path)

    output = trading_system_data.load_trading_system_data()

    assert tradingview_calls == [
        ("TVC:GOLD", "1D"),
        ("TVC:GOLD", "1mo"),
        ("TVC:GOLD", "1W"),
    ]
    assert "SI=F" in yahoo_calls
    assert "PL=F" in yahoo_calls
    assert not output["daily"]["SILVER"].empty
    assert not output["daily"]["PLATINUM"].empty
    assert not output["monthly"]["SILVER"].empty
    assert not output["monthly"]["PLATINUM"].empty
    assert "Yahoo Finance" in output["status"]["SI=F 1D"]
    assert "Yahoo Finance" in output["status"]["PL=F 1D"]
