from __future__ import annotations

import numpy as np
import pandas as pd

from bybit_derivatives import (
    BYBIT_ASSET_MAP,
    BYBIT_CONFIG,
    build_weekly_layer,
    classify_oi_price_regime,
    status_frame,
    trailing_percentile,
    update_all_bybit_assets,
    validate_bybit_instrument,
)


class FakeBybitClient:
    def __init__(self, instrument_status: str = "Trading", contract_type: str = "LinearPerpetual") -> None:
        self.instrument_status = instrument_status
        self.contract_type = contract_type
        self.calls: list[tuple[str, dict]] = []

    def get(self, path: str, params: dict) -> dict:
        self.calls.append((path, params))
        if path == "/v5/market/instruments-info":
            return {
                "retCode": 0,
                "result": {
                    "category": params["category"],
                    "list": [
                        {
                            "symbol": params["symbol"],
                            "status": self.instrument_status,
                            "contractType": self.contract_type,
                            "baseCoin": "BTC",
                            "quoteCoin": "USDT",
                        }
                    ],
                },
            }
        raise AssertionError(f"Unexpected path: {path}")


def test_asset_mapping_is_explicit_for_tron() -> None:
    assert BYBIT_ASSET_MAP["TRX-USD"]["symbol"] == "TRXUSDT"


def test_target_history_start_is_march_2018() -> None:
    assert BYBIT_CONFIG["target_start_date"] == "2018-03-01"


def test_validate_bybit_instrument_accepts_active_linear_perpetual_without_auth() -> None:
    client = FakeBybitClient()
    result = validate_bybit_instrument("BTC-USD", client)
    assert result["available"] is True
    assert result["data_status"] == "CURRENT"
    assert client.calls == [
        (
            "/v5/market/instruments-info",
            {"category": "linear", "symbol": "BTCUSDT", "limit": 1000},
        )
    ]


def test_validate_bybit_instrument_marks_non_trading_unavailable() -> None:
    result = validate_bybit_instrument("BTC-USD", FakeBybitClient(instrument_status="PreLaunch"))
    assert result["available"] is False
    assert result["data_status"] == "INSTRUMENT_UNAVAILABLE"


def test_trailing_percentile_is_point_in_time_and_requires_minimum_history() -> None:
    series = pd.Series(np.arange(110, dtype=float))
    result = trailing_percentile(series, window=156, min_periods=104)
    assert np.isnan(result.iloc[102])
    assert result.iloc[-1] == 100.0


def test_oi_price_regime_uses_price_and_oi_direction_together() -> None:
    price = pd.Series([0.01, 0.01, -0.01, -0.01])
    oi = pd.Series([0.01, -0.01, 0.01, -0.01])
    assert classify_oi_price_regime(price, oi).tolist() == [
        "PRICE_UP_OI_UP",
        "PRICE_UP_OI_DOWN",
        "PRICE_DOWN_OI_UP",
        "PRICE_DOWN_OI_DOWN",
    ]


def test_build_weekly_layer_keeps_long_format_and_metrics() -> None:
    dates = pd.date_range("2023-01-01", periods=820, freq="D")
    price = pd.Series(np.linspace(100.0, 180.0, len(dates)), index=dates)
    price.index.name = "Date"
    oi = pd.DataFrame(
        {
            "timestamp": dates,
            "open_interest_raw": np.linspace(1000.0, 2500.0, len(dates)),
        }
    )
    funding_timestamps = pd.date_range("2023-01-01", periods=820 * 3, freq="8h")
    funding = pd.DataFrame(
        {
            "timestamp": funding_timestamps,
            "funding_rate": np.full(len(funding_timestamps), 0.0001),
        }
    )
    ticker = {
        "timestamp": int(pd.Timestamp("2025-04-01", tz="UTC").timestamp() * 1000),
        "lastPrice": "181",
        "markPrice": "180",
        "indexPrice": "179",
        "openInterest": "2600",
        "openInterestValue": "468000",
        "fundingRate": "0.0001",
        "volume24h": "100000",
        "turnover24h": "18000000",
    }

    weekly = build_weekly_layer("BTC-USD", {}, oi, funding, ticker, price)

    assert set(["timestamp", "date", "asset", "exchange_symbol", "open_interest_usd"]).issubset(weekly.columns)
    assert weekly["asset"].eq("BTC-USD").all()
    assert weekly["exchange_symbol"].eq("BTCUSDT").all()
    assert weekly["open_interest_usd"].dropna().iloc[-1] > 0
    assert weekly["oi_change_4w_percentile"].notna().any()
    assert weekly["funding_28d_percentile"].notna().any()


def test_status_frame_uses_nan_values_not_zeroes() -> None:
    frame = status_frame("HYPE-USD", "INSTRUMENT_UNAVAILABLE")
    assert frame.loc[0, "data_status"] == "INSTRUMENT_UNAVAILABLE"
    assert pd.isna(frame.loc[0, "open_interest_usd"])


def test_force_update_rebuilds_storage_without_old_short_history(monkeypatch, tmp_path) -> None:
    path = tmp_path / "weekly_bybit_derivatives.csv"
    status_frame("BTC-USD", "CURRENT").assign(date=pd.Timestamp("2026-02-13")).to_csv(path, index=False)

    def fake_update(asset, existing, client=None):
        assert existing.empty
        return status_frame(asset, "CURRENT").assign(date=pd.Timestamp("2018-03-02"), history_start_date="2018-03-02")

    monkeypatch.setattr("bybit_derivatives.update_bybit_asset", fake_update)
    monkeypatch.setattr("bybit_derivatives.BYBIT_ASSET_MAP", {"BTC-USD": BYBIT_ASSET_MAP["BTC-USD"]})

    updated = update_all_bybit_assets(path=path, force=True)

    assert updated["history_start_date"].iloc[0] == "2018-03-02"
