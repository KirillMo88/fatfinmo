import numpy as np
import pandas as pd
import pytest

import app


def test_ai_group_fast_performance_averages_component_returns(monkeypatch):
    dates = pd.to_datetime(["2026-09-11", "2026-09-14"])
    frames = {
        "AAA": pd.DataFrame({"Close": [100.0, 110.0]}, index=dates),
        "BBB": pd.DataFrame({"Close": [200.0, 190.0]}, index=dates),
    }

    monkeypatch.setitem(app.AI_UNIVERSE, "Memory", ["AAA", "BBB"])
    monkeypatch.setattr(app, "download_performance_ohlcv", lambda ticker, **_: frames[ticker])

    result = app.get_ai_group_performance_metrics("Memory")

    assert result is not None
    assert result[0] == pytest.approx(2.5)
    assert np.isnan(result[1])


def _ohlcv(close_values, dates):
    close = pd.Series(close_values, index=pd.to_datetime(dates), dtype="float64")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1.0,
        }
    )


def test_ai_group_ohlcv_keeps_member_weight_during_short_gap():
    frames = {
        "AAA": _ohlcv([100.0, 110.0, 120.0], ["2026-09-10", "2026-09-11", "2026-09-14"]),
        "BBB": _ohlcv([200.0, 220.0], ["2026-09-10", "2026-09-11"]),
    }

    normalized = app._normalize_ai_member_frames(frames)
    result = app._aggregate_ai_group_ohlcv(normalized)

    # BBB is flat on the missing date, so the group moves from 110 to 115,
    # rather than jumping to AAA's standalone value of 120.
    assert result.loc[pd.Timestamp("2026-09-14"), "Close"] == pytest.approx(115.0)


def test_ai_group_overlay_uses_slow_snapshot_price_when_fast_price_unavailable(monkeypatch):
    previous = pd.DataFrame(
        [{"Group": "Equity", "Subgroup": "AI", "Ticker": "Memory", "CurrentPrice": 50.0}]
    )
    slow = pd.DataFrame(
        [
            {
                "Group": "Equity",
                "Subgroup": "AI",
                "Ticker": "Memory",
                "CurrentPrice": 100.0,
                "PerfRef_1D": 90.0,
                "PerfRef_1W": np.nan,
                "PerfRef_1M": np.nan,
                "PerfRef_3M": np.nan,
                "PerfRef_6M": np.nan,
                "PerfRef_12M": np.nan,
            }
        ]
    )
    monkeypatch.setattr(app, "read_snapshot_frame", lambda *_: (previous, {"RefreshBucket": -1}))
    monkeypatch.setattr(app, "lightweight_current_price", lambda *_args, **_kwargs: np.nan)
    monkeypatch.setattr(app, "atomic_write_snapshot", lambda *_args, **_kwargs: None)

    result, _, _ = app.compute_performance_table(slow, "test-ai-overlay", refresh_nonce=1)

    assert result.iloc[0]["CurrentPrice"] == pytest.approx(100.0)
    assert result.iloc[0]["Perf_1D_%"] == pytest.approx((100.0 / 90.0 - 1.0) * 100.0)
