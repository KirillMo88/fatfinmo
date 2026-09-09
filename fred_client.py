from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import httpx
import numpy as np
import pandas as pd


FRED_BASE_URL = "https://api.stlouisfed.org/fred"
FRED_DEFAULT_OBSERVATION_START = "2010-01-01"
FRED_SERIES_IDS = (
    "DGS2",
    "T5YIE",
    "DFII10",
    "M2SL",
    "WALCL",
    "RRPONTSYD",
    "WTREGEN",
)
FED_LIQUIDITY_SERIES_ID = "FED_LIQUIDITY"
FED_LIQUIDITY_COMPONENTS = ("WALCL", "RRPONTSYD", "WTREGEN")


class FredApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class FredSeriesRequest:
    series_id: str
    observation_start: str | None = None
    observation_end: str | None = None


def normalize_fred_series_id(series_id: str) -> str:
    return str(series_id).strip().upper()


def get_fred_api_key(api_key: str | None = None) -> str:
    key = (api_key or os.environ.get("FRED_API_KEY") or "").strip()
    if not key:
        raise FredApiError("FRED_API_KEY is not configured.")
    return key


def fred_observations_params(request: FredSeriesRequest, api_key: str) -> dict[str, str]:
    params = {
        "series_id": normalize_fred_series_id(request.series_id),
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
    }
    if request.observation_start:
        params["observation_start"] = request.observation_start
    if request.observation_end:
        params["observation_end"] = request.observation_end
    return params


def parse_fred_observations(series_id: str, payload: dict) -> pd.DataFrame:
    if "error_code" in payload:
        message = payload.get("error_message") or "Unknown FRED API error."
        raise FredApiError(str(message))

    observations = payload.get("observations", [])
    frame = pd.DataFrame(observations)
    if frame.empty:
        return pd.DataFrame(columns=["Series_ID", "Date", "Value"])

    out = pd.DataFrame(
        {
            "Series_ID": normalize_fred_series_id(series_id),
            "Date": pd.to_datetime(frame["date"], errors="coerce"),
            "Value": pd.to_numeric(frame["value"].replace(".", np.nan), errors="coerce"),
        }
    )
    return out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def calculate_fed_liquidity(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["Series_ID", "Date", "Value"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    normalized = frame.copy()
    normalized["Series_ID"] = normalized["Series_ID"].map(normalize_fred_series_id)
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    normalized["Value"] = pd.to_numeric(normalized["Value"], errors="coerce")
    normalized = normalized.dropna(subset=["Date"])

    pivot = (
        normalized[normalized["Series_ID"].isin(FED_LIQUIDITY_COMPONENTS)]
        .pivot_table(index="Date", columns="Series_ID", values="Value", aggfunc="last")
        .sort_index()
        .ffill()
    )
    missing = [series_id for series_id in FED_LIQUIDITY_COMPONENTS if series_id not in pivot.columns]
    if missing:
        raise FredApiError(f"Cannot calculate {FED_LIQUIDITY_SERIES_ID}; missing: {', '.join(missing)}.")

    value = pivot["WALCL"] - pivot["RRPONTSYD"] - pivot["WTREGEN"]
    out = pd.DataFrame(
        {
            "Series_ID": FED_LIQUIDITY_SERIES_ID,
            "Date": value.index,
            "Value": value,
        }
    )
    return out.dropna(subset=["Value"]).reset_index(drop=True)[columns]


def download_fred_series(
    series_id: str,
    api_key: str | None = None,
    observation_start: str | None = None,
    observation_end: str | None = None,
    timeout: float = 20.0,
) -> pd.DataFrame:
    key = get_fred_api_key(api_key)
    request = FredSeriesRequest(
        series_id=normalize_fred_series_id(series_id),
        observation_start=observation_start or FRED_DEFAULT_OBSERVATION_START,
        observation_end=observation_end,
    )
    params = fred_observations_params(request, key)
    try:
        response = httpx.get(f"{FRED_BASE_URL}/series/observations", params=params, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise FredApiError(f"FRED request failed for {request.series_id}: {exc}") from exc
    return parse_fred_observations(request.series_id, response.json())


def download_fred_series_batch(
    series_ids: Iterable[str] = FRED_SERIES_IDS,
    api_key: str | None = None,
    observation_start: str | None = None,
    observation_end: str | None = None,
    timeout: float = 20.0,
) -> pd.DataFrame:
    key = get_fred_api_key(api_key)
    normalized_series_ids = [normalize_fred_series_id(series_id) for series_id in series_ids]
    frames = [
        download_fred_series(
            series_id,
            api_key=key,
            observation_start=observation_start or FRED_DEFAULT_OBSERVATION_START,
            observation_end=observation_end,
            timeout=timeout,
        )
        for series_id in normalized_series_ids
    ]
    if not frames:
        return pd.DataFrame(columns=["Series_ID", "Date", "Value"])
    data = pd.concat(frames, ignore_index=True)
    requested = set(normalized_series_ids)
    if set(FED_LIQUIDITY_COMPONENTS).issubset(requested):
        data = pd.concat([data, calculate_fed_liquidity(data)], ignore_index=True)
    return data
