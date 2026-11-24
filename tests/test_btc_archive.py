import pandas as pd

from btc_cycle import merge_btc_mcp_weekly_history
from btc_data import read_btc_weekly_archive


def test_bundled_btc_archive_has_requested_window():
    archive = read_btc_weekly_archive()

    assert len(archive) == 840
    assert archive["date"].min() == pd.Timestamp("2009-10-11")
    assert archive["date"].max() == pd.Timestamp("2025-12-28")


def test_btc_archive_is_authoritative_before_2026_and_yahoo_is_live_after():
    archive = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-28"]),
            "open": [88_627.48],
            "high": [90_546.45],
            "low": [86_382.25],
            "close": [87_884.07],
            "volume": [1.0],
        }
    )
    yahoo = pd.DataFrame(
        {
            "Open": [90_000.0, 91_000.0, 92_000.0],
            "High": [91_000.0, 92_000.0, 93_000.0],
            "Low": [89_000.0, 90_000.0, 91_000.0],
            "Close": [90_500.0, 91_500.0, 92_500.0],
            "Volume": [10.0, 11.0, 12.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]),
    )
    tradingview = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05"]),
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [1.0],
        }
    )

    merged = merge_btc_mcp_weekly_history(
        yahoo,
        tradingview,
        today="2026-01-09",
        archive_weekly=archive,
    )

    assert merged.loc[merged["date"].eq("2025-12-28"), "Close"].iloc[0] == 87_884.07
    assert merged.loc[merged["date"].eq("2026-01-02"), "Close"].iloc[0] == 91_500.0
    assert not merged["Close"].eq(1.5).any()
