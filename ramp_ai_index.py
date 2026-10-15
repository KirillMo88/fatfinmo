from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAMP_AI_INDEX_URL = "https://ramp.com/data/ai-index?metric=token-volume&detail=maker&mode=share"
SEED_PATH = Path(__file__).resolve().parent / "data" / "ramp_ai_index_seed.xlsx"
SNAPSHOT_PATH = Path("persistent") / "ramp_ai_index" / "snapshot.json"

DATASET_KEYS = (
    "adoption_models",
    "adoption_sectors",
    "spend_per_employee",
    "share_of_business_spend",
    "token_volume_maker",
    "token_volume_maker_share",
    "token_spend_maker",
    "token_price",
)

MAKER_LABELS = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "xai": "xAI",
    "google": "Google",
    "cursor": "Cursor",
    "moonshot": "Moonshot",
    "deepseek": "DeepSeek",
    "zhipu": "Z.ai",
    "alibaba": "Alibaba",
    "typesafe": "Typesafe",
    "minimax": "MiniMax",
    "nvidia": "NVIDIA",
    "meta": "Meta",
    "thinking-machines-lab": "Thinking Machines Lab",
}
MAKER_ORDER = tuple(MAKER_LABELS)

ADOPTION_SECTOR_LABELS = {
    "Accommodation & Food": "Accommodation and food services",
    "Construction": "Construction",
    "Finance & Insurance": "Finance and insurance",
    "Healthcare": "Health care",
    "Information": "Technology and media",
    "Manufacturing": "Manufacturing",
    "Retail Trade": "Retail",
}

SPEND_SECTOR_LABELS = {
    "Accommodation and Food Services": "Accommodation and food services",
    "Finance and Insurance": "Finance and insurance",
    "Health Care and Social Assistance": "Health care",
    "Information": "Technology and media",
    "Professional, Scientific, and Technical Services": "Professional, scientific, and technical services",
    "Retail Trade": "Retail",
}


@dataclass(frozen=True)
class RampAIIndexSnapshot:
    datasets: dict[str, pd.DataFrame]
    status: str
    source: str
    retrieved_at: str
    latest_dates: dict[str, str]
    error: str | None = None


def load_ramp_ai_index_snapshot() -> RampAIIndexSnapshot:
    errors: list[str] = []
    for path, status, source in (
        (SNAPSHOT_PATH, "CURRENT", "Ramp AI Index public page"),
        (SEED_PATH, "HISTORICAL_FALLBACK", "Ramp Historical data workbook"),
    ):
        try:
            if path.exists():
                if path.suffix.lower() == ".xlsx":
                    snapshot = snapshot_from_seed_workbook(path, status=status, source=source)
                else:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    snapshot = snapshot_from_payload(payload, default_status=status, default_source=source)
                validate_datasets(snapshot.datasets)
                return snapshot
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    return RampAIIndexSnapshot(
        datasets={key: pd.DataFrame() for key in DATASET_KEYS},
        status="UNAVAILABLE",
        source="Ramp AI Index",
        retrieved_at="",
        latest_dates={},
        error="; ".join(errors) or "No Ramp AI Index snapshot is available.",
    )


def snapshot_from_seed_workbook(
    path: Path,
    status: str = "HISTORICAL_FALLBACK",
    source: str = "Ramp Historical data workbook",
) -> RampAIIndexSnapshot:
    sheet_map = {
        "Adoption + Models": "adoption_models",
        "Adoption + Sectors": "adoption_sectors",
        "AI spend by employee": "spend_per_employee",
        "AI share of business spend": "share_of_business_spend",
        "Token volume by maker": "token_volume_maker",
        "Token volume by maker share": "token_volume_maker_share",
        "Token spend by maker": "token_spend_maker",
        "Token Price": "token_price",
    }
    sheets = pd.read_excel(path, sheet_name=list(sheet_map), engine="openpyxl")
    datasets = {key: sheets[sheet].copy() for sheet, key in sheet_map.items()}
    token_volume = datasets["token_volume_maker"].reset_index(drop=True)
    token_share_dates = datasets["token_volume_maker_share"]["Date"].reset_index(drop=True)
    if len(token_volume) == len(token_share_dates):
        token_volume["Date"] = token_volume["Date"].where(token_volume["Date"].notna(), token_share_dates)
    datasets["token_volume_maker"] = token_volume
    datasets = {key: normalize_dates(frame) for key, frame in datasets.items()}
    latest_dates = {
        key: frame["Date"].max().date().isoformat()
        for key, frame in datasets.items()
        if not frame.empty
    }
    retrieved_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return RampAIIndexSnapshot(
        datasets=datasets,
        status=status,
        source=source,
        retrieved_at=retrieved_at,
        latest_dates=latest_dates,
    )


def refresh_ramp_ai_index_snapshot(url: str = RAMP_AI_INDEX_URL) -> RampAIIndexSnapshot:
    import httpx

    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
    }
    with httpx.Client(timeout=120.0, follow_redirects=True, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
    props = extract_ramp_page_props(response.text)
    datasets = build_datasets_from_props(props)
    validate_datasets(datasets)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    payload = payload_from_datasets(
        datasets,
        status="CURRENT",
        source="Ramp AI Index public page",
        retrieved_at=retrieved_at,
    )
    atomic_write_json(SNAPSHOT_PATH, payload)
    return snapshot_from_payload(payload)


def extract_ramp_page_props(html: str) -> dict[str, Any]:
    chunks: list[str] = []
    pattern = re.compile(r"<script[^>]*>\s*self\.__next_f\.push\((\[[\s\S]*?\])\)\s*</script>")
    for match in pattern.finditer(html):
        try:
            item = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(item, list) and len(item) > 1 and isinstance(item[1], str):
            chunks.append(item[1])
    rsc = "".join(chunks)
    marker = rsc.find('"tokenPrices"')
    if marker < 0:
        raise ValueError("Ramp page payload does not contain tokenPrices.")
    start = rsc.rfind("{", 0, marker)
    if start < 0:
        raise ValueError("Ramp page payload object could not be located.")
    props, _ = json.JSONDecoder().raw_decode(rsc[start:])
    required = {
        "tokenPrices",
        "tokenVolumes",
        "adoptionOverall",
        "adoptionVendor",
        "adoptionIndustry",
        "spendPerEmployee",
        "spendShareData",
    }
    missing = sorted(required.difference(props))
    if missing:
        raise ValueError(f"Ramp page payload is missing: {', '.join(missing)}")
    return props


def build_datasets_from_props(props: dict[str, Any]) -> dict[str, pd.DataFrame]:
    datasets = {
        "adoption_models": build_adoption_models(props),
        "adoption_sectors": build_adoption_sectors(props),
        "spend_per_employee": build_spend_per_employee(props),
        "share_of_business_spend": build_share_of_business_spend(props),
        "token_volume_maker": build_token_maker(props, "token_volume", "index"),
        "token_volume_maker_share": build_token_maker(props, "token_volume", "share"),
        "token_spend_maker": build_token_maker(props, "token_spend", "index"),
        "token_price": build_token_price(props),
    }
    return {key: normalize_dates(frame) for key, frame in datasets.items()}


def build_adoption_models(props: dict[str, Any]) -> pd.DataFrame:
    overall = pd.DataFrame(props.get("adoptionOverall", [])).rename(
        columns={
            "date_month": "Date",
            "adoption_rate_pct": "Adoption rate (%)",
            "mom_change_pp": "Monthly change (pp)",
            "yoy_change_pp": "Yearly change (pp)",
        }
    )
    overall["Series"] = "Ramp Overall"
    vendor = pd.DataFrame(props.get("adoptionVendor", [])).rename(
        columns={
            "date_month": "Date",
            "vendor": "Series",
            "adoption_rate_pct": "Adoption rate (%)",
            "mom_change_pp": "Monthly change (pp)",
        }
    )
    vendor["Yearly change (pp)"] = np.nan
    columns = ["Date", "Series", "Adoption rate (%)", "Monthly change (pp)", "Yearly change (pp)"]
    return pd.concat([overall.reindex(columns=columns), vendor.reindex(columns=columns)], ignore_index=True)


def build_adoption_sectors(props: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(props.get("adoptionIndustry", [])).rename(
        columns={
            "date_month": "Date",
            "naics_sector": "Sector",
            "adoption_rate_pct": "Adoption rate (%)",
            "mom_change_pp": "Monthly change (pp)",
        }
    )
    frame["Sector"] = frame["Sector"].replace(ADOPTION_SECTOR_LABELS)
    return frame[["Date", "Sector", "Adoption rate (%)", "Monthly change (pp)"]]


def build_spend_per_employee(props: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(props.get("spendPerEmployee", [])).rename(
        columns={
            "date_month": "Date",
            "median_pepm": "Median (USD / employee / month)",
            "top_10_percent_median_pepm": "Top 10% (USD / employee / month)",
            "top_1_percent_median_pepm": "Top 1% (USD / employee / month)",
        }
    )
    columns = [
        "Date",
        "Median (USD / employee / month)",
        "Top 10% (USD / employee / month)",
        "Top 1% (USD / employee / month)",
    ]
    return frame.reindex(columns=columns)


def build_share_of_business_spend(props: dict[str, Any]) -> pd.DataFrame:
    data = props.get("spendShareData") or {}
    overall = pd.DataFrame(data.get("overall", [])).rename(
        columns={"my_date": "Date", "ai_share_of_tpv": "AI share of business spend (%)"}
    )
    if not overall.empty:
        overall["AI share of business spend (%)"] = pd.to_numeric(
            overall["AI share of business spend (%)"], errors="coerce"
        ) * 100.0
    sector = pd.DataFrame(data.get("sector", []))
    if sector.empty:
        return overall
    sector = sector[sector["naics_sector"].ne("Unknown")].copy()
    sector["naics_sector"] = sector["naics_sector"].replace(SPEND_SECTOR_LABELS)
    sector["ai_share_of_tpv"] = pd.to_numeric(sector["ai_share_of_tpv"], errors="coerce") * 100.0
    wide = sector.pivot_table(index="my_date", columns="naics_sector", values="ai_share_of_tpv", aggfunc="last")
    wide.columns = [f"{column} (%)" for column in wide.columns]
    wide = wide.reset_index().rename(columns={"my_date": "Date"})
    return overall.merge(wide, on="Date", how="outer").sort_values("Date")


def build_token_maker(props: dict[str, Any], metric_key: str, mode: str) -> pd.DataFrame:
    frame = pd.DataFrame(props.get("tokenVolumes", []))
    frame = frame[frame["metric_key"].eq(metric_key)].copy()
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    grouped = frame.groupby(["period_end", "model_maker"], as_index=False)["metric_value"].sum()
    wide = grouped.pivot(index="period_end", columns="model_maker", values="metric_value").fillna(0.0)
    positive = [maker for maker in MAKER_ORDER if maker in wide.columns and float(wide[maker].max()) > 0.0]
    wide = wide.reindex(columns=positive)
    if mode == "share":
        total = wide.sum(axis=1).replace(0.0, np.nan)
        wide = wide.div(total, axis=0)
    elif mode == "index":
        maximum = float(wide.max().max()) if not wide.empty else np.nan
        wide = wide / maximum * 100.0 if np.isfinite(maximum) and maximum > 0.0 else wide * np.nan
    else:
        raise ValueError(f"Unsupported token maker mode: {mode}")
    wide = wide.rename(columns=MAKER_LABELS).reset_index().rename(columns={"period_end": "Date"})
    return wide


def build_token_price(props: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(props.get("tokenPrices", []))
    frame = frame[
        frame["token_type"].eq("all_tokens") & frame["model_maker"].isin(["openai", "anthropic"])
    ].copy()
    for column in ("token_cost_usd_7d", "token_count_7d"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["price"] = frame["token_cost_usd_7d"] / frame["token_count_7d"] * 1_000_000.0
    makers = frame.pivot_table(index="usage_date", columns="model_maker", values="price", aggfunc="last")
    combined = frame.groupby("usage_date")[["token_cost_usd_7d", "token_count_7d"]].sum()
    combined["OpenAI & Anthropic"] = (
        combined["token_cost_usd_7d"] / combined["token_count_7d"] * 1_000_000.0
    )
    out = combined[["OpenAI & Anthropic"]].join(makers, how="outer").reset_index()
    return out.rename(columns={"usage_date": "Date", "openai": "OpenAI", "anthropic": "Anthropic"})


def normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
        out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return out


def validate_datasets(datasets: dict[str, pd.DataFrame]) -> None:
    missing = [key for key in DATASET_KEYS if key not in datasets or datasets[key].empty]
    if missing:
        raise ValueError(f"Ramp AI Index datasets are missing or empty: {', '.join(missing)}")
    for key, frame in datasets.items():
        if "Date" not in frame.columns or pd.to_datetime(frame["Date"], errors="coerce").isna().any():
            raise ValueError(f"Dataset {key} contains invalid dates.")
    share = datasets["token_volume_maker_share"].drop(columns=["Date"], errors="ignore")
    totals = share.apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    if totals.empty or float((totals - 1.0).abs().max()) > 1e-6:
        raise ValueError("Token volume maker shares do not sum to 100%.")


def payload_from_datasets(
    datasets: dict[str, pd.DataFrame],
    status: str,
    source: str,
    retrieved_at: str,
) -> dict[str, Any]:
    latest_dates = {
        key: pd.to_datetime(frame["Date"], errors="coerce").max().date().isoformat()
        for key, frame in datasets.items()
    }
    records: dict[str, list[dict[str, Any]]] = {}
    for key, frame in datasets.items():
        clean = frame.copy()
        clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        clean = clean.astype(object).where(pd.notna(clean), None)
        records[key] = clean.to_dict(orient="records")
    return {
        "status": status,
        "source": source,
        "source_url": RAMP_AI_INDEX_URL,
        "retrieved_at": retrieved_at,
        "latest_dates": latest_dates,
        "datasets": records,
    }


def snapshot_from_payload(
    payload: dict[str, Any],
    default_status: str = "CURRENT",
    default_source: str = "Ramp AI Index public page",
) -> RampAIIndexSnapshot:
    datasets = {
        key: normalize_dates(pd.DataFrame((payload.get("datasets") or {}).get(key, [])))
        for key in DATASET_KEYS
    }
    latest_dates = payload.get("latest_dates") or {
        key: frame["Date"].max().date().isoformat()
        for key, frame in datasets.items()
        if not frame.empty and "Date" in frame
    }
    return RampAIIndexSnapshot(
        datasets=datasets,
        status=str(payload.get("status") or default_status),
        source=str(payload.get("source") or default_source),
        retrieved_at=str(payload.get("retrieved_at") or ""),
        latest_dates={str(key): str(value) for key, value in latest_dates.items()},
        error=None,
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temp.replace(path)
