from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import numpy as np
import pandas as pd


TRADINGVIEW_MCP_URL = "https://mcp.tradingview.com/mcp"
TRADINGVIEW_AUTH_METADATA_URL = "https://www.tradingview.com/.well-known/oauth-authorization-server"
TRADINGVIEW_RESOURCE_METADATA_URL = "https://mcp.tradingview.com/.well-known/oauth-protected-resource/mcp"
TRADINGVIEW_MCP_RESOURCE = "https://mcp.tradingview.com/mcp"
TRADINGVIEW_MCP_PROTOCOL_VERSION = "2025-06-18"
DEFAULT_STORAGE_DIR = Path(
    os.environ.get(
        "TRADINGVIEW_MCP_STORAGE_DIR",
        str(Path(__file__).with_name("persistent") / "tradingview_mcp"),
    )
)
CLIENT_PATH = DEFAULT_STORAGE_DIR / "client.json"
TOKEN_PATH = DEFAULT_STORAGE_DIR / "token.json"
PENDING_AUTH_PATH = DEFAULT_STORAGE_DIR / "pending_auth.json"
ECONOMIC_CACHE_PATH = DEFAULT_STORAGE_DIR / "economic_history.csv"
TOOLS_CACHE_PATH = DEFAULT_STORAGE_DIR / "tools.json"
ECONOMIC_HISTORY_TTL_SECONDS = 86400


@dataclass(frozen=True)
class TradingViewMcpStatus:
    connected: bool
    oauth_status: str
    detail: str


@dataclass(frozen=True)
class TradingViewEconomicResult:
    symbol: str
    frame: pd.DataFrame
    description: str
    unit: str
    scale: str
    frequency: str
    data_status: str
    source_mode: str
    notes: str


class TradingViewMcpError(RuntimeError):
    pass


def mcp_enabled() -> bool:
    value = os.environ.get("TRADINGVIEW_MCP_ENABLED", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def ensure_storage_dir() -> None:
    DEFAULT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_storage_dir()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    try:
        path.chmod(0o600)
    except Exception:
        pass


def now_ts() -> float:
    return time.time()


def oauth_metadata() -> dict[str, Any]:
    response = httpx.get(TRADINGVIEW_AUTH_METADATA_URL, timeout=20)
    response.raise_for_status()
    return response.json()


def resource_metadata() -> dict[str, Any]:
    response = httpx.get(TRADINGVIEW_RESOURCE_METADATA_URL, timeout=20)
    response.raise_for_status()
    return response.json()


def register_client(redirect_uri: str) -> dict[str, Any]:
    metadata = oauth_metadata()
    endpoint = metadata.get("registration_endpoint")
    if not endpoint:
        raise TradingViewMcpError("TradingView OAuth registration endpoint is not available.")
    payload = {
        "client_name": "Screener TradingView MCP",
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "scope": "mcp:read mcp:tools",
    }
    response = httpx.post(endpoint, json=payload, timeout=20)
    response.raise_for_status()
    client = response.json()
    if not client.get("client_id"):
        raise TradingViewMcpError("TradingView OAuth registration did not return client_id.")
    client["redirect_uri"] = redirect_uri
    write_json(CLIENT_PATH, client)
    return client


def get_or_register_client(redirect_uri: str) -> dict[str, Any]:
    client = read_json(CLIENT_PATH)
    if client.get("client_id") and client.get("redirect_uri") == redirect_uri:
        return client
    return register_client(redirect_uri)


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    return verifier, challenge


def build_authorization_url(redirect_uri: str) -> str:
    metadata = oauth_metadata()
    client = get_or_register_client(redirect_uri)
    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(32)
    write_json(
        PENDING_AUTH_PATH,
        {
            "state": state,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
            "created_at": now_ts(),
        },
    )
    params = {
        "response_type": "code",
        "client_id": client["client_id"],
        "redirect_uri": redirect_uri,
        "scope": "mcp:read mcp:tools",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "resource": TRADINGVIEW_MCP_RESOURCE,
    }
    return f"{metadata['authorization_endpoint']}?{urlencode(params)}"


def exchange_code(code: str, state: str) -> None:
    pending = read_json(PENDING_AUTH_PATH)
    if not pending or pending.get("state") != state:
        raise TradingViewMcpError("OAuth state mismatch.")
    client = read_json(CLIENT_PATH)
    if not client.get("client_id"):
        raise TradingViewMcpError("OAuth client is not registered.")
    metadata = oauth_metadata()
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": pending["redirect_uri"],
        "client_id": client["client_id"],
        "code_verifier": pending["code_verifier"],
        "resource": TRADINGVIEW_MCP_RESOURCE,
    }
    response = httpx.post(metadata["token_endpoint"], data=payload, timeout=20)
    response.raise_for_status()
    token = response.json()
    token["obtained_at"] = now_ts()
    token["expires_at"] = now_ts() + float(token.get("expires_in", 3600)) - 60.0
    write_json(TOKEN_PATH, token)
    try:
        PENDING_AUTH_PATH.unlink()
    except Exception:
        pass


def token_payload() -> dict[str, Any]:
    token = read_json(TOKEN_PATH)
    if not token.get("access_token"):
        raise TradingViewMcpError("TradingView MCP is not authorized.")
    if float(token.get("expires_at", 0)) <= now_ts():
        token = refresh_token(token)
    return token


def refresh_token(token: dict[str, Any] | None = None) -> dict[str, Any]:
    token = token or read_json(TOKEN_PATH)
    refresh = token.get("refresh_token")
    if not refresh:
        raise TradingViewMcpError("TradingView MCP refresh token is not available.")
    client = read_json(CLIENT_PATH)
    metadata = oauth_metadata()
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh,
        "client_id": client.get("client_id", ""),
        "resource": TRADINGVIEW_MCP_RESOURCE,
    }
    response = httpx.post(metadata["token_endpoint"], data=payload, timeout=20)
    response.raise_for_status()
    updated = response.json()
    updated["obtained_at"] = now_ts()
    updated["expires_at"] = now_ts() + float(updated.get("expires_in", 3600)) - 60.0
    if "refresh_token" not in updated and refresh:
        updated["refresh_token"] = refresh
    write_json(TOKEN_PATH, updated)
    return updated


def connection_status() -> TradingViewMcpStatus:
    if not mcp_enabled():
        return TradingViewMcpStatus(False, "DISABLED", "TRADINGVIEW_MCP_ENABLED is false.")
    try:
        resource_metadata()
    except Exception as exc:
        return TradingViewMcpStatus(False, "ENDPOINT_ERROR", str(exc))
    try:
        token_payload()
    except Exception as exc:
        return TradingViewMcpStatus(False, "AUTH_ERROR", str(exc))
    try:
        tools = discover_tools(force=False)
    except Exception as exc:
        return TradingViewMcpStatus(False, "TOOLS_ERROR", str(exc))
    return TradingViewMcpStatus(True, "OK", f"{len(tools)} tools discovered.")


def mcp_headers() -> dict[str, str]:
    token = token_payload()
    return {
        "Authorization": f"Bearer {token['access_token']}",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "MCP-Protocol-Version": TRADINGVIEW_MCP_PROTOCOL_VERSION,
    }


def mcp_request(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": secrets.randbelow(1_000_000_000),
        "method": method,
        "params": params or {},
    }
    response = httpx.post(TRADINGVIEW_MCP_URL, json=payload, headers=mcp_headers(), timeout=60)
    if response.status_code == 401:
        refresh_token()
        response = httpx.post(TRADINGVIEW_MCP_URL, json=payload, headers=mcp_headers(), timeout=60)
    response.raise_for_status()
    message = parse_mcp_response(response)
    if "error" in message:
        raise TradingViewMcpError(str(message["error"]))
    return message.get("result", {})


def parse_mcp_response(response: httpx.Response) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    text = response.text
    if "text/event-stream" in content_type or text.lstrip().startswith("event:") or text.lstrip().startswith("data:"):
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data and data != "[DONE]":
                return json.loads(data)
        raise TradingViewMcpError("MCP SSE response did not include JSON data.")
    return response.json()


def discover_tools(force: bool = False) -> list[dict[str, Any]]:
    if not force and TOOLS_CACHE_PATH.exists() and now_ts() - TOOLS_CACHE_PATH.stat().st_mtime < ECONOMIC_HISTORY_TTL_SECONDS:
        cached = read_json(TOOLS_CACHE_PATH)
        if isinstance(cached.get("tools"), list):
            return cached["tools"]
    result = mcp_request("tools/list")
    tools = result.get("tools", [])
    if not isinstance(tools, list):
        tools = []
    write_json(TOOLS_CACHE_PATH, {"tools": tools, "updated_at": now_ts()})
    return tools


def resolve_tool_name(preferred_name: str) -> str | None:
    tools = discover_tools(force=False)
    names = [str(tool.get("name", "")) for tool in tools]
    if preferred_name in names:
        return preferred_name
    suffix_matches = [name for name in names if name.endswith(f".{preferred_name}") or name.endswith(f"/{preferred_name}")]
    if suffix_matches:
        return suffix_matches[0]
    return None


def call_tool(preferred_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool_name = resolve_tool_name(preferred_name)
    if not tool_name:
        raise TradingViewMcpError(f"TradingView MCP tool not discovered: {preferred_name}")
    result = mcp_request("tools/call", {"name": tool_name, "arguments": arguments})
    return normalize_tool_result(result)


def normalize_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                    return {"data": parsed}
                except Exception:
                    return {"text": text}
    if isinstance(result, dict):
        return result
    return {"data": result}


def get_economic_data(symbol: str, date_from: str = "2010-01-01", date_to: str | None = None, force: bool = False) -> TradingViewEconomicResult:
    if not mcp_enabled():
        raise TradingViewMcpError("TradingView MCP is disabled.")
    date_to = date_to or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    cached = read_cached_economic(symbol)
    if not force and cached is not None:
        return cached
    payload = call_tool("get_economic_data", {"symbol": symbol, "date_from": date_from, "date_to": date_to})
    result = parse_economic_payload(symbol, payload)
    if result.frame.empty:
        raise TradingViewMcpError(f"{symbol} returned no economic observations.")
    append_cached_economic(result)
    return result


def read_cached_economic(symbol: str) -> TradingViewEconomicResult | None:
    if not ECONOMIC_CACHE_PATH.exists() or now_ts() - ECONOMIC_CACHE_PATH.stat().st_mtime > ECONOMIC_HISTORY_TTL_SECONDS:
        return None
    try:
        frame = pd.read_csv(ECONOMIC_CACHE_PATH)
    except Exception:
        return None
    if frame.empty or "symbol" not in frame.columns:
        return None
    rows = frame[frame["symbol"].eq(symbol)].copy()
    if rows.empty:
        return None
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    values = rows.dropna(subset=["date", "value"]).sort_values("date")
    if values.empty:
        return None
    meta = values.iloc[-1]
    output_columns = ["date", "value"]
    if "release_date" in values.columns:
        values["release_date"] = pd.to_datetime(values["release_date"], errors="coerce")
        output_columns.append("release_date")
    return TradingViewEconomicResult(
        symbol=symbol,
        frame=values[output_columns].copy(),
        description=str(meta.get("description", "")),
        unit=str(meta.get("unit", "")),
        scale=str(meta.get("scale", "")),
        frequency=str(meta.get("frequency", "")),
        data_status=str(meta.get("data_status", "OK")),
        source_mode="MCP_PRIMARY",
        notes=str(meta.get("notes", "")),
    )


def append_cached_economic(result: TradingViewEconomicResult) -> None:
    ensure_storage_dir()
    frame = result.frame.copy()
    frame["symbol"] = result.symbol
    frame["description"] = result.description
    frame["unit"] = result.unit
    frame["scale"] = result.scale
    frame["frequency"] = result.frequency
    frame["data_status"] = result.data_status
    frame["notes"] = result.notes
    existing = pd.read_csv(ECONOMIC_CACHE_PATH) if ECONOMIC_CACHE_PATH.exists() else pd.DataFrame()
    combined = pd.concat([existing, frame], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = (
        combined.dropna(subset=["date", "symbol"])
        .sort_values(["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
    )
    combined.to_csv(ECONOMIC_CACHE_PATH, index=False)


def parse_economic_payload(symbol: str, payload: dict[str, Any]) -> TradingViewEconomicResult:
    data = first_list(payload, ["data", "values", "observations", "series", "history"])
    metadata = first_dict(payload, ["metadata", "meta", "symbol"])
    rows: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        date_source = first_present(item, ["date", "time", "t"])
        value_source = first_present(item, ["value", "actual", "actualRaw", "c"])
        release_source = first_present(item, ["release_date", "releaseDate", "released_at", "release_time"])
        date = pd.to_datetime(date_source, unit="s" if isinstance(date_source, (int, float)) else None, errors="coerce")
        value = pd.to_numeric(value_source, errors="coerce")
        release_date = pd.to_datetime(
            release_source,
            unit="s" if isinstance(release_source, (int, float)) else None,
            errors="coerce",
        )
        if pd.isna(date) or not np.isfinite(value):
            continue
        clean_date = pd.Timestamp(date).tz_localize(None) if pd.Timestamp(date).tzinfo else pd.Timestamp(date)
        clean_release = (
            pd.Timestamp(release_date).tz_localize(None)
            if pd.notna(release_date) and pd.Timestamp(release_date).tzinfo
            else pd.Timestamp(release_date)
            if pd.notna(release_date)
            else pd.NaT
        )
        rows.append({"date": clean_date, "value": float(value), "release_date": clean_release})
    frame = pd.DataFrame(rows, columns=["date", "value", "release_date"]).dropna(subset=["date", "value"])
    if not frame.empty:
        frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    unit = str(payload.get("unit") or metadata.get("unit") or "")
    scale = str(payload.get("scale") or metadata.get("scale") or "")
    frequency = str(payload.get("frequency") or metadata.get("frequency") or "")
    description = str(payload.get("description") or metadata.get("description") or symbol)
    status = "OK" if not frame.empty else "MISSING"
    duplicate_count = len(rows) - len(frame)
    notes = f"TradingView MCP economic data; duplicate_date_count={duplicate_count}"
    if frame.empty or "release_date" not in frame.columns or frame["release_date"].isna().all():
        notes += "; RELEASE_DATE_UNKNOWN"
    return TradingViewEconomicResult(symbol, frame, description, unit, scale, frequency, status, "MCP_PRIMARY", notes)


def first_present(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def first_list(payload: Any, keys: list[str]) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    for value in payload.values():
        nested = first_list(value, keys)
        if nested:
            return nested
    return []


def first_dict(payload: Any, keys: list[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def normalize_cny_to_100mn(value: float, unit: str = "", scale: str = "", expected: str = "") -> tuple[float, str]:
    text = f"{unit} {scale}".strip().lower()
    if not np.isfinite(value):
        return math.nan, "invalid numeric value"
    if "100 million" in text or "hundred million" in text:
        return value, "unit=CNY 100 million"
    if "trillion" in text or " tn" in text or "t cny" in text:
        return value * 10_000.0, "unit=CNY trillion"
    if "billion" in text or " bn" in text:
        return value * 10.0, "unit=CNY billion"
    if "million" in text:
        return value / 100.0, "unit=CNY million"
    if "yuan" in text or "cny" in text or "rmb" in text:
        return value / 100_000_000.0, "unit=CNY/yuan"
    if expected == "cnm2" and 100.0 <= value <= 500.0:
        return value * 10_000.0, "inferred CNY trillion from CNM2 magnitude"
    if expected == "cnm2" and 100_000.0 <= value <= 5_000_000.0:
        return value, "inferred CNY 100 million from CNM2 magnitude"
    if expected == "cncbbs" and 10.0 <= value <= 100.0:
        return value * 10_000.0, "inferred CNY trillion from CNCBBS magnitude"
    if expected == "cncbbs" and 100_000.0 <= value <= 900_000.0:
        return value, "inferred CNY 100 million from CNCBBS magnitude"
    return math.nan, f"unit validation failed; unit={unit}; scale={scale}; value={value}"


def validate_economic_result(result: TradingViewEconomicResult, min_observations: int = 24, max_stale_days: int = 95) -> tuple[bool, str]:
    frame = result.frame.copy()
    if frame.empty:
        return False, "MISSING"
    if len(frame) < min_observations:
        return False, "PARTIAL"
    duplicate_count = int(pd.to_datetime(frame["date"], errors="coerce").duplicated().sum())
    if duplicate_count:
        return False, "DUPLICATE_DATES"
    latest_date = pd.to_datetime(frame["date"], errors="coerce").max()
    if pd.isna(latest_date):
        return False, "MISSING"
    if (pd.Timestamp.utcnow().tz_localize(None) - pd.Timestamp(latest_date)).days > max_stale_days:
        return False, "STALE"
    null_count = int(pd.to_numeric(frame["value"], errors="coerce").isna().sum())
    if null_count:
        return False, "PARTIAL"
    return True, "OK"


def validation_summary(result: TradingViewEconomicResult) -> dict[str, Any]:
    frame = result.frame.copy()
    dates = pd.to_datetime(frame["date"], errors="coerce") if not frame.empty else pd.Series(dtype="datetime64[ns]")
    values = pd.to_numeric(frame["value"], errors="coerce") if not frame.empty else pd.Series(dtype="float64")
    return {
        "Symbol": result.symbol,
        "Description": result.description or result.symbol,
        "Source": "TradingView MCP",
        "Unit": result.unit or "n/a",
        "Scale": result.scale or "n/a",
        "Frequency": result.frequency or "n/a",
        "First available date": dates.min().strftime("%Y-%m-%d") if not dates.empty and pd.notna(dates.min()) else "n/a",
        "Last available date": dates.max().strftime("%Y-%m-%d") if not dates.empty and pd.notna(dates.max()) else "n/a",
        "Latest value": values.dropna().iloc[-1] if not values.dropna().empty else np.nan,
        "Number of observations": int(len(frame)),
        "Null count": int(values.isna().sum()) if len(values) else 0,
        "Duplicate date count": int(dates.duplicated().sum()) if len(dates) else 0,
        "Data Status": result.data_status,
    }
