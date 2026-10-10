"""Independent positioning state machines displayed by the Portfolio tab."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


BTC_ZERO_PHASES = {"BEAR_DELEVERAGING", "BOTTOMING_TRANSITION"}


def _flag(value: Any) -> bool:
    return bool(value) if pd.notna(value) else False


def _dated_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Date"])
    out = frame.copy()
    if "Date" in out:
        dates = pd.to_datetime(out["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    else:
        dates = pd.to_datetime(out.index, errors="coerce", utc=True).tz_localize(None)
    out.index.name = None
    out["Date"] = dates
    return out.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")


def build_spx_positioning_history(
    current_risk_daily: pd.DataFrame,
    correction_daily: pd.DataFrame,
    initial_position: int = 100,
) -> pd.DataFrame:
    """Replay SPX risk reduction and Correction Bottom rebuilding events.

    Correction Bottom events have same-day precedence. Level 2 is a subset of
    classic capitulation, so its minimum target is 66% even from a 0% state.
    Positions persist until another defined event occurs.
    """
    risk = _dated_frame(current_risk_daily)
    correction = _dated_frame(correction_daily)
    if risk.empty and correction.empty:
        return pd.DataFrame(columns=["Date", "position_pct", "trigger", "signal_source"])

    risk_columns = [
        column for column in (
            "Date", "CurrentRiskEventMarker", "CurrentRiskEventMarkerClass",
            "CurrentRiskEventMarkerReason",
        ) if column in risk
    ]
    correction_columns = [
        column for column in (
            "Date", "ExtremeEpisodeDisplayMarker", "ExtremeCapitulation2EpisodeMarker",
            "DurableBottomConfirmed", "CorrectionEventType",
        ) if column in correction
    ]
    dates = pd.DataFrame({
        "Date": pd.DatetimeIndex(sorted(set(risk["Date"].tolist()) | set(correction["Date"].tolist())))
    })
    combined = dates.merge(risk[risk_columns], on="Date", how="left") if risk_columns else dates.copy()
    if correction_columns:
        combined = combined.merge(correction[correction_columns], on="Date", how="left")

    position = int(initial_position)
    rows: list[dict[str, Any]] = []
    last_signal_date = pd.NaT
    last_trigger = "INITIAL 100%"
    for _, row in combined.sort_values("Date").iterrows():
        previous = position
        trigger = ""
        source = ""
        durable = _flag(row.get("DurableBottomConfirmed", False))
        extreme_2 = _flag(row.get("ExtremeCapitulation2EpisodeMarker", False))
        extreme_1 = _flag(row.get("ExtremeEpisodeDisplayMarker", False))
        risk_marker = _flag(row.get("CurrentRiskEventMarker", False))
        risk_class = str(row.get("CurrentRiskEventMarkerClass", "")).strip().upper()

        # Same-day Correction Bottom events override Current Risk events.
        if durable:
            position, trigger, source = 100, "DURABLE BOTTOM CONFIRMED", "Correction Bottom Indicator"
        elif extreme_2:
            position, trigger, source = max(position, 66), "EXTREME CAPITULATION 2", "Correction Bottom Indicator"
        elif extreme_1:
            position, trigger, source = max(position, 33), "EXTREME CAPITULATION", "Correction Bottom Indicator"
        elif risk_marker and risk_class == "RED FLAG + CREDIT CONFIRMATION":
            position, trigger, source = 0, risk_class, "Current Risk Signals"
        elif risk_marker and "RED FLAG" in risk_class:
            position, trigger, source = min(position, 50), risk_class, "Current Risk Signals"

        changed = position != previous
        if trigger:
            last_signal_date = pd.Timestamp(row["Date"])
            last_trigger = trigger
        rows.append({
            "Date": pd.Timestamp(row["Date"]),
            "position_pct": int(position),
            "previous_position_pct": int(previous),
            "position_changed": bool(changed),
            "trigger": trigger,
            "signal_source": source,
            "last_signal_date": last_signal_date,
            "last_trigger": last_trigger,
        })
    return pd.DataFrame(rows)


def gold_positioning(regime: Any) -> dict[str, Any]:
    normalized = str(regime or "DATA_INCOMPLETE").strip().upper()
    if normalized == "CASH":
        return {"position_pct": 0, "instrument": "Cash", "regime": normalized, "status": "READY"}
    if normalized in {"GOLD", "SILVER", "PLATINUM"}:
        return {
            "position_pct": 100,
            "instrument": normalized.title(),
            "regime": normalized,
            "status": "READY",
        }
    return {"position_pct": np.nan, "instrument": "N/A", "regime": normalized, "status": "DATA_INCOMPLETE"}


def build_btc_positioning_history(btc_gold_history: pd.DataFrame) -> pd.DataFrame:
    """Replay BTC exposure from BTC/Gold switches and halving-phase transitions."""
    history = _dated_frame(btc_gold_history)
    required = {"position", "halving_phase"}
    if history.empty or not required.issubset(history.columns):
        return pd.DataFrame(columns=["Date", "position_pct", "trigger", "signal_source"])

    exposure = 0
    previous_engine: str | None = None
    previous_phase: str | None = None
    last_signal_date = pd.NaT
    last_trigger = "INITIAL 0%"
    rows: list[dict[str, Any]] = []
    for _, row in history.iterrows():
        engine = str(row.get("position") or "UNINITIALIZED")
        phase = str(row.get("halving_phase") or "DATA_INCOMPLETE").upper()
        previous_exposure = exposure
        trigger = ""

        entered_btc = engine == "BTC" and previous_engine != "BTC"
        entered_gold = engine == "Gold" and previous_engine != "Gold"
        phase_changed = previous_phase is not None and phase != previous_phase

        if engine == "Gold":
            exposure = 0
            if entered_gold:
                trigger = "BTC / GOLD: BTC → GOLD"
        elif engine == "BTC":
            if entered_btc:
                exposure = 75
                trigger = "BTC / GOLD: GOLD → BTC"
            elif phase_changed:
                if phase in BTC_ZERO_PHASES:
                    exposure = 0
                    trigger = f"HALVING PHASE: {phase}"
                else:
                    exposure = 100
                    trigger = f"HALVING PHASE: {phase}"
        else:
            exposure = 0

        if trigger:
            last_signal_date = pd.Timestamp(row["Date"])
            last_trigger = trigger
        rows.append({
            "Date": pd.Timestamp(row["Date"]),
            "position_pct": int(exposure),
            "previous_position_pct": int(previous_exposure),
            "position_changed": bool(exposure != previous_exposure),
            "trigger": trigger,
            "signal_source": "BTC / Gold + Halving Cycle" if trigger else "",
            "btc_gold_position": engine,
            "halving_phase": phase,
            "phase_changed": bool(phase_changed),
            "last_signal_date": last_signal_date,
            "last_trigger": last_trigger,
        })
        previous_engine = engine
        previous_phase = phase
    return pd.DataFrame(rows)


def positioning_snapshot(
    spx_history: pd.DataFrame,
    gold_regime: Any,
    btc_history: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "SPX": {} if spx_history is None or spx_history.empty else spx_history.iloc[-1].to_dict(),
        "GOLD": gold_positioning(gold_regime),
        "BTC": {} if btc_history is None or btc_history.empty else btc_history.iloc[-1].to_dict(),
    }
