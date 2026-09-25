from __future__ import annotations

import pandas as pd
import httpx
from io import StringIO

from fred_client import download_fred_series
from tradingview_mcp import call_tool, get_economic_data


def main() -> None:
    try:
        fred = download_fred_series("BAMLH0A0HYM2", observation_start="1993-01-01")
        print("FRED", len(fred), fred["Date"].min(), fred["Date"].max(), fred["Value"].notna().sum())
    except Exception as exc:
        print("FRED_ERROR", type(exc).__name__)
    try:
        response = httpx.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            params={"id": "BAMLH0A0HYM2", "cosd": "1993-01-01"},
            timeout=30,
            follow_redirects=True,
        )
        response.raise_for_status()
        csv = pd.read_csv(StringIO(response.text))
        print("FRED_CSV", len(csv), csv.iloc[:, 0].min(), csv.iloc[:, 0].max(), csv.iloc[:, 1].notna().sum())
    except Exception as exc:
        print("FRED_CSV_ERROR", type(exc).__name__)

    for symbol in ["S5FI", "S5TH"]:
        try:
            result = get_economic_data(symbol, date_from="1990-01-01")
            first = result.frame["date"].min() if not result.frame.empty else None
            last = result.frame["date"].max() if not result.frame.empty else None
            print(symbol, result.data_status, result.frequency, len(result.frame), first, last)
        except Exception as exc:
            print(symbol, "ERROR", type(exc).__name__)
        for qualified in [f"INDEX:{symbol}", symbol]:
            try:
                payload = call_tool(
                    "get_ohlcv",
                    {"symbol": qualified, "interval": "1D", "count": 5000, "summary": False},
                )
                bars = payload.get("bars") or payload.get("data") or payload.get("candles") or []
                print("OHLCV", qualified, list(payload)[:10], len(bars))
                if bars:
                    print("OHLCV_RANGE", qualified, bars[0].get("t"), bars[-1].get("t"))
                break
            except Exception as exc:
                print("OHLCV", qualified, "ERROR", type(exc).__name__)
    for qualified in ["FRED:BAMLH0A0HYM2", "BAMLH0A0HYM2"]:
        try:
            payload = call_tool(
                "get_ohlcv",
                {"symbol": qualified, "interval": "1D", "count": 5000, "summary": False},
            )
            bars = payload.get("bars") or payload.get("data") or payload.get("candles") or []
            print("HY_OHLCV", qualified, len(bars))
            if bars:
                print("HY_OHLCV_RANGE", qualified, bars[0].get("t"), bars[-1].get("t"))
            break
        except Exception as exc:
            print("HY_OHLCV", qualified, "ERROR", type(exc).__name__)

    cache = pd.read_csv("/app/persistent/tradingview_mcp/economic_history.csv")
    print("CACHE_COLS", list(cache.columns))
    if "symbol" in cache:
        summary = cache.groupby("symbol").agg(rows=("symbol", "size"), first=("date", "min"), last=("date", "max"))
        print(summary.to_string())


if __name__ == "__main__":
    main()
