from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from io import StringIO
from pathlib import Path
import re
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


MACRO_SURPRISES_MODEL_VERSION = "MACRO_SURPRISES_V1"
MACRO_SURPRISES_STORAGE_DIR = Path("persistent") / "macro_surprises"
MACRO_SURPRISES_RELEASES_PATH = MACRO_SURPRISES_STORAGE_DIR / "releases.parquet"
MACRO_SURPRISES_SEED_PATH = Path("data") / "PMI.xlsx"

GROWTH_WEIGHTS = {"PMI": 0.50, "Retail Sales": 0.30, "Initial Jobless Claims": 0.20}
HALF_LIVES = {"PMI": 28.0, "Retail Sales": 28.0, "Initial Jobless Claims": 14.0, "CPI": 28.0}
MIN_HISTORY = {"PMI": 24, "Retail Sales": 24, "Initial Jobless Claims": 52, "CPI": 24}
SURPRISE_THRESHOLD = 0.20

SEED_BLOCKS = {
    "PMI": (0, 5),
    "CPI": (8, 13),
    "Initial Jobless Claims": (16, 21),
    "Retail Sales": (23, 28),
}

INVESTING_SOURCES = {
    "PMI": {
        "url": "https://www.investing.com/economic-calendar/ism-manufacturing-pmi-173",
        "definition": "ISM Manufacturing PMI",
    },
    "Retail Sales": {
        "url": "https://www.investing.com/economic-calendar/retail-sales-256",
        "definition": "Retail Sales MoM",
    },
    "Initial Jobless Claims": {
        "url": "https://www.investing.com/economic-calendar/initial-jobless-claims-294",
        "definition": "Initial Jobless Claims",
    },
    # The supplied workbook contains CPI YoY. The requested CPI MoM page is
    # intentionally not spliced into that history; a matching Investing series is used.
    "CPI": {
        "url": "https://www.investing.com/economic-calendar/cpi-733",
        "definition": "CPI YoY",
        "requested_url": "https://www.investing.com/economic-calendar/united-states-consumer-price-index-(cpi)-mom-69",
    },
}

RELEASE_COLUMNS = [
    "Indicator",
    "SeriesDefinition",
    "ReleaseDate",
    "ReleaseTime",
    "ReleaseTimestamp",
    "Actual",
    "Forecast",
    "Previous",
    "Source",
    "SourceURL",
    "EventKey",
    "ReleaseKey",
    "RevisionNumber",
    "RevisionDetected",
    "RetrievedAt",
]


@dataclass
class MacroSurprisesSnapshot:
    releases: pd.DataFrame
    history: pd.DataFrame
    current: dict[str, Any]
    latest_releases: pd.DataFrame
    diagnostics: pd.DataFrame
    source_status: dict[str, str]


def build_macro_surprises_snapshot(
    business_history: pd.DataFrame,
    *,
    refresh_live: bool = False,
    as_of: str | pd.Timestamp | None = None,
    seed_path: str | Path = MACRO_SURPRISES_SEED_PATH,
    storage_path: str | Path = MACRO_SURPRISES_RELEASES_PATH,
) -> MacroSurprisesSnapshot:
    as_of_ts = _normalize_date(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    releases, initialized, seed_meta = load_or_initialize_releases(seed_path=seed_path, storage_path=storage_path)
    source_status = {indicator: "SEED HISTORY" for indicator in INVESTING_SOURCES}

    if refresh_live or initialized:
        live_frames: list[pd.DataFrame] = []
        for indicator, spec in INVESTING_SOURCES.items():
            try:
                frame = fetch_investing_releases(indicator, spec["url"], spec["definition"])
                live_frames.append(frame)
                source_status[indicator] = f"CURRENT / {len(frame)} visible releases checked"
            except Exception as exc:
                source_status[indicator] = f"LIVE UPDATE FAILED: {type(exc).__name__}"
        if live_frames:
            live = pd.concat(live_frames, ignore_index=True)
            releases = merge_release_updates(releases, live)
            write_release_store(releases, storage_path)

    model_releases = calculate_release_surprises(select_point_in_time_releases(releases))
    first_date = pd.to_datetime(model_releases["ReleaseDate"], errors="coerce").min()
    history_start = max(pd.Timestamp("2010-01-01"), first_date if pd.notna(first_date) else pd.Timestamp("2010-01-01"))
    weekly_dates = pd.date_range(history_start, as_of_ts, freq="W-FRI")
    history = build_surprise_history(model_releases, weekly_dates, business_history)
    current_frame = build_surprise_history(model_releases, pd.DatetimeIndex([as_of_ts]), business_history)
    current = current_frame.iloc[-1].to_dict() if not current_frame.empty else {}
    current["Persistence"] = current_persistence(current.get("TurningSignal"), history)
    current["ModelVersion"] = MACRO_SURPRISES_MODEL_VERSION
    latest = build_latest_releases(model_releases, as_of_ts, current)

    cpi_definition = infer_cpi_definition(model_releases)
    diagnostics = build_macro_diagnostics(seed_meta, cpi_definition, source_status, releases)
    return MacroSurprisesSnapshot(releases, history, current, latest, diagnostics, source_status)


def load_or_initialize_releases(
    *,
    seed_path: str | Path = MACRO_SURPRISES_SEED_PATH,
    storage_path: str | Path = MACRO_SURPRISES_RELEASES_PATH,
) -> tuple[pd.DataFrame, bool, dict[str, Any]]:
    seed, seed_meta = load_seed_releases(seed_path)
    path = Path(storage_path)
    if path.exists():
        try:
            stored = pd.read_parquet(path)
            if not stored.empty:
                return normalize_release_schema(stored), False, seed_meta
        except Exception:
            pass
    write_release_store(seed, path)
    return seed, True, seed_meta


def load_seed_releases(path: str | Path = MACRO_SURPRISES_SEED_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return empty_release_frame(), {"SeedStatus": "MISSING", "RejectedRows": 0, "ValidRows": 0}
    raw = pd.read_excel(source, sheet_name="PMI")
    frames: list[pd.DataFrame] = []
    rejected = 0
    for indicator, (start, stop) in SEED_BLOCKS.items():
        block = raw.iloc[:, start:stop].copy()
        block.columns = ["ReleaseDate", "ReleaseTime", "Actual", "Forecast", "Previous"]
        normalized, rejected_count = normalize_release_block(
            block,
            indicator=indicator,
            definition=seed_definition(indicator),
            source_name=source.name,
            source_url="",
        )
        rejected += rejected_count
        frames.append(normalized)
    releases = pd.concat(frames, ignore_index=True) if frames else empty_release_frame()
    releases = normalize_release_schema(releases).sort_values(["ReleaseDate", "Indicator"]).reset_index(drop=True)
    return releases, {
        "SeedStatus": "LOADED",
        "SeedFile": source.name,
        "RejectedRows": rejected,
        "ValidRows": len(releases),
    }


def fetch_investing_releases(indicator: str, url: str, definition: str) -> pd.DataFrame:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(request, timeout=45) as response:
        payload = response.read().decode("utf-8", errors="replace")
    tables = pd.read_html(StringIO(payload), flavor="lxml")
    release_table = next((table for table in tables if _is_release_table(table)), None)
    if release_table is None:
        raise ValueError("Investing release table not found")
    release_table = release_table.iloc[:, :5].copy()
    release_table.columns = ["ReleaseDate", "ReleaseTime", "Actual", "Forecast", "Previous"]
    normalized, _ = normalize_release_block(
        release_table,
        indicator=indicator,
        definition=definition,
        source_name="Investing.com",
        source_url=url,
    )
    if normalized.empty:
        raise ValueError("Investing release table has no completed observations")
    return normalized


def normalize_release_block(
    block: pd.DataFrame,
    *,
    indicator: str,
    definition: str,
    source_name: str,
    source_url: str,
) -> tuple[pd.DataFrame, int]:
    data = block.copy()
    data["ReleaseDate"] = data["ReleaseDate"].map(parse_release_date)
    data = data.loc[data["ReleaseDate"].notna()].copy()
    data["ReleaseTime"] = data["ReleaseTime"].map(format_release_time)
    for column in ["Actual", "Forecast", "Previous"]:
        data[column] = data[column].map(lambda value: parse_release_value(value, indicator))
    invalid = data["ReleaseDate"].isna() | data["Actual"].isna() | data["Forecast"].isna()
    rejected = int(invalid.sum())
    data = data.loc[~invalid].copy()
    data["Indicator"] = indicator
    data["SeriesDefinition"] = definition
    data["ReleaseTimestamp"] = [combine_release_timestamp(date, time) for date, time in zip(data["ReleaseDate"], data["ReleaseTime"])]
    data["Source"] = source_name
    data["SourceURL"] = source_url
    data["EventKey"] = data["Indicator"] + "|" + data["ReleaseDate"].dt.strftime("%Y-%m-%d")
    data["ReleaseKey"] = data["EventKey"] + "|" + data["ReleaseTime"].fillna("00:00:00")
    data["RevisionNumber"] = 0
    data["RevisionDetected"] = False
    data["RetrievedAt"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return normalize_release_schema(data), rejected


def normalize_release_schema(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in RELEASE_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    out["ReleaseDate"] = pd.to_datetime(out["ReleaseDate"], errors="coerce").dt.normalize()
    out["ReleaseTimestamp"] = pd.to_datetime(out["ReleaseTimestamp"], errors="coerce")
    out["RetrievedAt"] = pd.to_datetime(out["RetrievedAt"], errors="coerce")
    for column in ["Actual", "Forecast", "Previous"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["RevisionNumber"] = pd.to_numeric(out["RevisionNumber"], errors="coerce").fillna(0).astype(int)
    out["RevisionDetected"] = out["RevisionDetected"].fillna(False).astype(bool)
    out = out.dropna(subset=["Indicator", "ReleaseDate", "Actual", "Forecast"])
    return out[RELEASE_COLUMNS].sort_values(["ReleaseDate", "Indicator", "RevisionNumber"]).reset_index(drop=True)


def merge_release_updates(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    out = normalize_release_schema(existing)
    updates = normalize_release_schema(incoming)
    additions: list[pd.Series] = []
    for _, row in updates.iterrows():
        matches = out.loc[out["EventKey"].eq(row["EventKey"])]
        if matches.empty:
            additions.append(row)
            continue
        latest = matches.sort_values("RevisionNumber").iloc[-1]
        same_values = all(_same_number(latest.get(col), row.get(col)) for col in ["Actual", "Forecast", "Previous"])
        if same_values:
            continue
        revised = row.copy()
        revised["RevisionNumber"] = int(matches["RevisionNumber"].max()) + 1
        revised["RevisionDetected"] = True
        revised["ReleaseKey"] = f"{row['ReleaseKey']}|R{revised['RevisionNumber']}"
        additions.append(revised)
    if additions:
        out = pd.concat([out, pd.DataFrame(additions)], ignore_index=True)
    return normalize_release_schema(out)


def calculate_release_surprises(releases: pd.DataFrame) -> pd.DataFrame:
    out = normalize_release_schema(releases)
    out["RawSurprise"] = np.nan
    out["ZSurprise"] = np.nan
    out["HistoryCount"] = 0
    for indicator, rows in out.groupby("Indicator", sort=False):
        indices = rows.sort_values(["ReleaseDate", "RevisionNumber"]).index
        actual = pd.to_numeric(out.loc[indices, "Actual"], errors="coerce")
        forecast = pd.to_numeric(out.loc[indices, "Forecast"], errors="coerce")
        raw = actual - forecast
        if indicator == "Initial Jobless Claims":
            raw = -raw
        minimum = MIN_HISTORY.get(indicator, 24)
        prior = raw.shift(1)
        prior_mean = prior.expanding(min_periods=minimum).mean()
        prior_std = prior.expanding(min_periods=minimum).std(ddof=1).replace(0, np.nan)
        zscore = ((raw - prior_mean) / prior_std).clip(-3.0, 3.0)
        out.loc[indices, "RawSurprise"] = raw.to_numpy()
        out.loc[indices, "ZSurprise"] = zscore.to_numpy()
        out.loc[indices, "HistoryCount"] = np.arange(len(indices))
    return out


def select_point_in_time_releases(releases: pd.DataFrame) -> pd.DataFrame:
    if releases.empty:
        return releases.copy()
    # The first stored observation remains the backtest value. Later revisions are
    # retained for audit but cannot rewrite the historical surprise distribution.
    ordered = releases.sort_values(["EventKey", "RevisionNumber", "RetrievedAt"])
    return ordered.drop_duplicates("EventKey", keep="first").sort_values(["ReleaseDate", "Indicator"]).reset_index(drop=True)


def build_surprise_history(
    releases: pd.DataFrame,
    dates: pd.DatetimeIndex,
    business_history: pd.DataFrame,
) -> pd.DataFrame:
    base = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce")}).dropna().sort_values("date")
    if base.empty:
        return base
    for indicator in HALF_LIVES:
        slug = indicator_slug(indicator)
        rows = releases.loc[releases["Indicator"].eq(indicator)].copy()
        rows = rows.sort_values("ReleaseDate")
        if rows.empty:
            base[f"{slug}ReleaseDate"] = pd.NaT
            base[f"{slug}ZSurprise"] = np.nan
            base[f"{slug}Contribution"] = np.nan
            continue
        right = rows[["ReleaseDate", "ZSurprise"]].rename(columns={"ReleaseDate": f"{slug}ReleaseDate", "ZSurprise": f"{slug}ZSurprise"})
        base = pd.merge_asof(
            base.sort_values("date"),
            right.sort_values(f"{slug}ReleaseDate"),
            left_on="date",
            right_on=f"{slug}ReleaseDate",
            direction="backward",
        )
        age = (base["date"] - base[f"{slug}ReleaseDate"]).dt.days.clip(lower=0)
        base[f"{slug}Contribution"] = pd.to_numeric(base[f"{slug}ZSurprise"], errors="coerce") * np.power(2.0, -age / HALF_LIVES[indicator])

    weighted_sum = pd.Series(0.0, index=base.index)
    available_weight = pd.Series(0.0, index=base.index)
    for indicator, weight in GROWTH_WEIGHTS.items():
        contribution = pd.to_numeric(base[f"{indicator_slug(indicator)}Contribution"], errors="coerce")
        available = contribution.notna()
        weighted_sum = weighted_sum.add(contribution.fillna(0.0) * weight)
        available_weight = available_weight.add(available.astype(float) * weight)
    base["GrowthAvailableWeight"] = available_weight
    base["GrowthSurpriseScore"] = (weighted_sum / available_weight.replace(0, np.nan)).where(available_weight >= 0.50)
    base["GrowthState"] = base["GrowthSurpriseScore"].map(classify_growth_state)
    base["GrowthBreadth"] = base.apply(calculate_growth_breadth, axis=1)

    context = normalize_business_context(business_history)
    if context.empty:
        base["BusinessCycleDirection"] = "DATA INCOMPLETE"
        base["InflationState"] = "DATA INCOMPLETE"
    else:
        base = pd.merge_asof(base.sort_values("date"), context, on="date", direction="backward")
        base["BusinessCycleDirection"] = base["BusinessCycleDirection"].fillna("DATA INCOMPLETE")
        base["InflationState"] = base["InflationState"].fillna("DATA INCOMPLETE")
    base["TurningSignal"] = [
        classify_turning_signal(direction, score)
        for direction, score in zip(base["BusinessCycleDirection"], base["GrowthSurpriseScore"])
    ]
    base["Persistence"] = classify_persistence(base["TurningSignal"])
    base["CPIContribution"] = pd.to_numeric(base.get("CPIContribution"), errors="coerce")
    base["CPISurpriseState"] = base["CPIContribution"].map(classify_cpi_state)
    base["InflationSignal"] = [
        classify_inflation_signal(state, score)
        for state, score in zip(base["InflationState"], base["CPIContribution"])
    ]
    return base


def normalize_business_context(history: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "BusinessCycleDirection", "InflationState"]
    if history is None or history.empty or any(column not in history.columns for column in required):
        return pd.DataFrame(columns=required)
    out = history[required].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")


def build_latest_releases(releases: pd.DataFrame, as_of: pd.Timestamp, current: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for indicator in ["PMI", "Retail Sales", "Initial Jobless Claims", "CPI"]:
        eligible = releases.loc[releases["Indicator"].eq(indicator) & releases["ReleaseDate"].le(as_of)].sort_values("ReleaseDate")
        if eligible.empty:
            continue
        release = eligible.iloc[-1]
        slug = indicator_slug(indicator)
        contribution = current.get(f"{slug}Contribution", np.nan)
        if indicator == "CPI":
            signal = classify_cpi_state(contribution)
        else:
            signal = classify_growth_state(contribution)
        rows.append(
            {
                "Indicator": indicator,
                "Release Date": release["ReleaseDate"],
                "Actual": release["Actual"],
                "Forecast": release["Forecast"],
                "Previous": release["Previous"],
                "Raw Surprise": release.get("RawSurprise"),
                "Z Surprise": release.get("ZSurprise"),
                "Age": int(max(0, (as_of - release["ReleaseDate"]).days)),
                "Current Contribution": contribution,
                "Signal": signal,
            }
        )
    return pd.DataFrame(rows).sort_values("Release Date", ascending=False).reset_index(drop=True)


def build_macro_diagnostics(
    seed_meta: dict[str, Any],
    cpi_definition: str,
    source_status: dict[str, str],
    releases: pd.DataFrame,
) -> pd.DataFrame:
    revisions = int(pd.to_numeric(releases.get("RevisionDetected"), errors="coerce").fillna(0).astype(bool).sum())
    rows = [
        ("ModelVersion", MACRO_SURPRISES_MODEL_VERSION),
        ("Model Role", "Early-warning / confirmation overlay"),
        ("Seed File", seed_meta.get("SeedFile", "N/A")),
        ("Valid Historical Releases", seed_meta.get("ValidRows", 0)),
        ("Rejected Incomplete Rows", seed_meta.get("RejectedRows", 0)),
        ("Historical CPI Definition", cpi_definition),
        ("Live CPI Definition", "CPI YoY (matching Investing series)"),
        ("CPI Consistency", "MATCHED" if cpi_definition == "CPI YoY" else "HISTORICAL/LIVE SERIES MISMATCH"),
        ("Requested CPI MoM Source", "NOT SPLICED INTO CPI YOY HISTORY"),
        ("PMI Consistency", "ISM Manufacturing PMI / MATCHED"),
        ("Growth Weights", "PMI 50% / Retail Sales 30% / Claims 20%"),
        ("Half-lives", "PMI 28d / Retail Sales 28d / Claims 14d / CPI 28d"),
        ("Threshold", "+/-0.20"),
        ("Normalization", "Expanding point-in-time Z-score"),
        ("Winsorization", "[-3, +3]"),
        ("Historical Revisions Detected", revisions),
    ]
    rows.extend((f"Source: {indicator}", status) for indicator, status in source_status.items())
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def classify_growth_state(value: Any) -> str:
    number = _finite(value)
    if number is None:
        return "INSUFFICIENT DATA"
    if number > SURPRISE_THRESHOLD:
        return "POSITIVE"
    if number < -SURPRISE_THRESHOLD:
        return "NEGATIVE"
    return "NEUTRAL"


def classify_turning_signal(direction: Any, score: Any) -> str:
    number = _finite(score)
    direction_text = str(direction).upper()
    if number is None or direction_text not in {"IMPROVING", "DETERIORATING"}:
        return "DATA INCOMPLETE"
    if -SURPRISE_THRESHOLD <= number <= SURPRISE_THRESHOLD:
        return "NEUTRAL"
    if direction_text == "IMPROVING":
        return "CONFIRMED IMPROVEMENT" if number > SURPRISE_THRESHOLD else "SLOWDOWN WARNING"
    return "POTENTIAL BOTTOMING" if number > SURPRISE_THRESHOLD else "CONFIRMED DETERIORATION"


def classify_cpi_state(value: Any) -> str:
    number = _finite(value)
    if number is None:
        return "INSUFFICIENT DATA"
    if number > SURPRISE_THRESHOLD:
        return "HOTTER"
    if number < -SURPRISE_THRESHOLD:
        return "COOLER"
    return "NEUTRAL"


def classify_inflation_signal(inflation_state: Any, value: Any) -> str:
    number = _finite(value)
    state = str(inflation_state).upper()
    if number is None or state not in {"RISING", "FALLING"}:
        return "DATA INCOMPLETE"
    if -SURPRISE_THRESHOLD <= number <= SURPRISE_THRESHOLD:
        return "NEUTRAL"
    if state == "RISING":
        return "INFLATION PRESSURE CONFIRMED" if number > SURPRISE_THRESHOLD else "DISINFLATION CHALLENGE"
    return "REACCELERATION CHALLENGE" if number > SURPRISE_THRESHOLD else "DISINFLATION CONFIRMED"


def classify_persistence(signals: pd.Series) -> pd.Series:
    current = signals.astype(str)
    persistent = current.eq(current.shift(1)) & ~current.isin(["NEUTRAL", "DATA INCOMPLETE"])
    return pd.Series(np.where(current.eq("NEUTRAL"), "NEUTRAL", np.where(persistent, "PERSISTENT", "NEW")), index=signals.index)


def current_persistence(signal: Any, history: pd.DataFrame) -> str:
    text = str(signal)
    if text == "NEUTRAL":
        return "NEUTRAL"
    if text in {"DATA INCOMPLETE", "None", "nan"}:
        return "NEW"
    recent = history.get("TurningSignal", pd.Series(dtype="object")).dropna().tail(2).astype(str)
    return "PERSISTENT" if len(recent) == 2 and recent.eq(text).all() else "NEW"


def calculate_growth_breadth(row: pd.Series) -> float:
    score = _finite(row.get("GrowthSurpriseScore"))
    if score is None or score == 0:
        return np.nan
    available = 0.0
    aligned = 0.0
    sign = np.sign(score)
    for indicator, weight in GROWTH_WEIGHTS.items():
        contribution = _finite(row.get(f"{indicator_slug(indicator)}Contribution"))
        if contribution is None:
            continue
        available += weight
        if np.sign(contribution) == sign:
            aligned += weight
    return aligned / available if available else np.nan


def infer_cpi_definition(releases: pd.DataFrame) -> str:
    values = pd.to_numeric(releases.loc[releases["Indicator"].eq("CPI"), "Actual"], errors="coerce").abs().dropna()
    if values.empty:
        return "UNKNOWN"
    return "CPI YoY" if values.median() >= 0.01 else "CPI MoM"


def parse_release_date(value: Any) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    text = re.sub(r"\s*\([^)]*\)\s*$", "", str(value)).strip()
    return pd.to_datetime(text, errors="coerce").normalize()


def format_release_time(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "00:00:00"
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M:%S")
    text = str(value).strip()
    parsed = pd.to_datetime(text, errors="coerce")
    return parsed.strftime("%H:%M:%S") if pd.notna(parsed) else "00:00:00"


def parse_release_value(value: Any, indicator: str) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "--", "N/A", "nan"}:
        return np.nan
    multiplier = 1.0
    if text.endswith("K"):
        multiplier, text = 1_000.0, text[:-1]
    elif text.endswith("M"):
        multiplier, text = 1_000_000.0, text[:-1]
    elif text.endswith("B"):
        multiplier, text = 1_000_000_000.0, text[:-1]
    percent = text.endswith("%")
    text = text.rstrip("%").strip()
    try:
        number = float(text) * multiplier
    except ValueError:
        return np.nan
    if percent and indicator in {"Retail Sales", "CPI"}:
        number /= 100.0
    return number


def combine_release_timestamp(date: Any, time: str) -> pd.Timestamp:
    date_value = pd.to_datetime(date, errors="coerce")
    if pd.isna(date_value):
        return pd.NaT
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*", str(time or "00:00:00"))
    if match is None:
        return date_value.normalize()
    hours, minutes, seconds = (int(value or 0) for value in match.groups())
    return date_value.normalize() + timedelta(hours=hours, minutes=minutes, seconds=seconds)


def seed_definition(indicator: str) -> str:
    return {
        "PMI": "ISM Manufacturing PMI",
        "Retail Sales": "Retail Sales MoM",
        "Initial Jobless Claims": "Initial Jobless Claims",
        "CPI": "CPI YoY",
    }[indicator]


def indicator_slug(indicator: str) -> str:
    return {
        "PMI": "PMI",
        "Retail Sales": "RetailSales",
        "Initial Jobless Claims": "Claims",
        "CPI": "CPI",
    }[indicator]


def write_release_store(frame: pd.DataFrame, path: str | Path = MACRO_SURPRISES_RELEASES_PATH) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp.parquet")
    normalize_release_schema(frame).to_parquet(temp, index=False)
    temp.replace(target)


def empty_release_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=RELEASE_COLUMNS)


def _is_release_table(table: pd.DataFrame) -> bool:
    columns = [str(value).strip().lower() for value in table.columns]
    return all(any(token in column for column in columns) for token in ["release date", "actual", "forecast", "previous"])


def _normalize_date(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()


def _same_number(left: Any, right: Any) -> bool:
    a, b = _finite(left), _finite(right)
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return bool(np.isclose(a, b, rtol=0, atol=1e-12))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
