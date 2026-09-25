from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, "/app")

from fund_flows import ETF_COM_SOURCE, fetch_etf_com_fund_flow_history


TICKERS = ["GLD", "IAU", "GLDM"]
START_DATE = date(2016, 1, 1)
REQUESTED_END_DATE = date(2026, 12, 31)


def build_weekly_fund_flows() -> pd.DataFrame:
    end_date = min(REQUESTED_END_DATE, datetime.now(timezone.utc).date())
    rows = []
    for ticker in TICKERS:
        observations = fetch_etf_com_fund_flow_history(ticker, START_DATE, end_date)
        if not observations:
            rows.append(
                {
                    "Week_End": pd.NaT,
                    "Ticker": ticker,
                    "Weekly_Net_Flow_USD": pd.NA,
                    "Weekly_Net_Flow_Millions": pd.NA,
                    "Observations": 0,
                    "First_Observation": pd.NaT,
                    "Last_Observation": pd.NaT,
                    "Source": ETF_COM_SOURCE,
                }
            )
            continue

        daily = pd.DataFrame(
            {
                "Date": pd.to_datetime([obs.date for obs in observations]),
                "Ticker": [ticker] * len(observations),
                "Net_Flow_USD": [obs.net_flow for obs in observations],
            }
        )
        weekly = (
            daily.set_index("Date")
            .resample("W-FRI")
            .agg(
                Weekly_Net_Flow_USD=("Net_Flow_USD", "sum"),
                Observations=("Net_Flow_USD", "count"),
                First_Observation=("Net_Flow_USD", lambda values: values.index.min()),
                Last_Observation=("Net_Flow_USD", lambda values: values.index.max()),
            )
            .reset_index()
            .rename(columns={"Date": "Week_End"})
        )
        weekly = weekly.loc[weekly["Observations"] > 0].copy()
        weekly["Ticker"] = ticker
        weekly["Weekly_Net_Flow_Millions"] = weekly["Weekly_Net_Flow_USD"] / 1_000_000.0
        weekly["Source"] = ETF_COM_SOURCE
        rows.extend(
            weekly[
                [
                    "Week_End",
                    "Ticker",
                    "Weekly_Net_Flow_USD",
                    "Weekly_Net_Flow_Millions",
                    "Observations",
                    "First_Observation",
                    "Last_Observation",
                    "Source",
                ]
            ].to_dict("records")
        )

    output = pd.DataFrame(rows)
    output["Week_End"] = pd.to_datetime(output["Week_End"]).dt.strftime("%Y-%m-%d")
    output["First_Observation"] = pd.to_datetime(output["First_Observation"]).dt.strftime("%Y-%m-%d")
    output["Last_Observation"] = pd.to_datetime(output["Last_Observation"]).dt.strftime("%Y-%m-%d")
    output = output.sort_values(["Ticker", "Week_End"], na_position="last").reset_index(drop=True)
    return output


def main() -> None:
    output_dir = Path("/app/exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "weekly_fund_flows_GLD_IAU_GLDM_2016_2026.csv"
    frame = build_weekly_fund_flows()
    frame.to_csv(output_path, index=False)
    print(output_path)
    print(f"rows={len(frame)}")
    print(frame.groupby("Ticker", dropna=False)["Observations"].sum().to_string())
    print(frame.groupby("Ticker", dropna=False)["Week_End"].agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()
