from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


COMMODITY_SECTORS = {
    "Energy": ("WTI", "Natural Gas", "RBOB"),
    "Metals": ("Copper", "Aluminum"),
    "Agriculture": ("Corn", "Wheat", "Soybeans"),
}
CORE_SERIES = ("Petroleum / Energy", "Metals", "Agriculture")
CONFIRMATION_SERIES = (
    "Chemicals", "Lumber", "Hardware / Plumbing", "Machinery", "Electrical / Electronics"
)


def midrank_percentile(value: float, reference: Sequence[float]) -> float:
    """Mid-rank percentile in [0, 100], ignoring unavailable reference values."""
    if pd.isna(value):
        return np.nan
    values = pd.to_numeric(pd.Series(reference), errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return np.nan
    return float((np.count_nonzero(values < value) + 0.5 * np.count_nonzero(values == value)) / len(values) * 100)


def calculate_stress(history: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    """Build monthly rolling and calendar-month stress percentiles, without look-ahead."""
    out = pd.DataFrame(index=history.index)
    dates = pd.to_datetime(history.index)
    for column in history.columns:
        values = pd.to_numeric(history[column], errors="coerce")
        rolling, seasonal = [], []
        for i, (date, value) in enumerate(zip(dates, values)):
            prior = values.iloc[max(0, i - window):i]
            rolling.append(100 - midrank_percentile(value, prior) if len(prior.dropna()) >= window else np.nan)
            same_month = values.iloc[:i]
            same_month = same_month.loc[
                (pd.to_datetime(same_month.index).month == date.month)
                & (pd.to_datetime(same_month.index) >= date - pd.DateOffset(years=10))
            ]
            seasonal.append(100 - midrank_percentile(value, same_month) if len(same_month.dropna()) >= 10 else np.nan)
        out[f"{column} Rolling Stress"] = rolling
        out[f"{column} Seasonal Stress"] = seasonal
    return out


def classify_delta_stress(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if value > 15:
        return "Extreme Tightening"
    if value > 10:
        return "Strong Tightening"
    if value > 5:
        return "Moderate Tightening"
    if value >= -5:
        return "Stable"
    if value >= -10:
        return "Moderate Easing"
    if value >= -15:
        return "Strong Easing"
    return "Extreme Easing"


def classify_commodity_price_momentum(r3m: float, r6m: float, r12m: float) -> str:
    if any(pd.isna(v) for v in (r3m, r6m, r12m)):
        return "N/A"
    if r3m > 0 and r6m > 0 and r12m > 0:
        return "Strong Bullish"
    if r6m > 0 and r12m > 0:
        return "Bullish"
    if r3m < 0 and r6m < 0 and r12m < 0:
        return "Strong Bearish"
    if r6m < 0 and r12m < 0:
        return "Bearish"
    return "Neutral"


def classify_sector_price_state(price_states: Sequence[str], sector: str | None = None) -> str:
    states = list(price_states)
    if not states or any(s in {"N/A", "DATA INCOMPLETE"} for s in states):
        return "DATA INCOMPLETE"
    bullish = sum(s in {"Bullish", "Strong Bullish"} for s in states)
    bearish = sum(s in {"Bearish", "Strong Bearish"} for s in states)
    n = len(states)
    if n == 2:
        if bullish == 2:
            return "Strong Bullish"
        if bullish == 1 and bearish == 0:
            return "Bullish"
        if bearish == 2:
            return "Strong Bearish"
        if bearish == 1 and bullish == 0:
            return "Bearish"
        return "Mixed / Neutral"
    if n == 3:
        if bullish == 3:
            return "Strong Bullish"
        if bullish == 2:
            return "Bullish"
        if bearish == 3:
            return "Strong Bearish"
        if bearish == 2:
            return "Bearish"
        return "Mixed / Neutral"
    return "DATA INCOMPLETE"


def classify_seasonal_curve(percentile: float) -> str:
    if pd.isna(percentile):
        return "N/A"
    if percentile < 10:
        return "Extreme Loose"
    if percentile < 25:
        return "Strong Loose"
    if percentile < 40:
        return "Loose"
    if percentile < 60:
        return "Neutral"
    if percentile < 75:
        return "Tight"
    if percentile <= 90:
        return "Strong Tightness"
    return "Extreme Tightness"


def classify_cftc_relative_state(percentile: float) -> str:
    if pd.isna(percentile):
        return "N/A"
    if percentile < 10:
        return "Extreme Low"
    if percentile < 25:
        return "Low"
    if percentile < 75:
        return "Neutral"
    if percentile <= 90:
        return "High"
    return "Extreme High"


def resolve_price_curve_market_state(price_state: str, curve_state: str) -> str:
    price_group = (
        "Bullish" if price_state in {"Bullish", "Strong Bullish"}
        else "Bearish" if price_state in {"Bearish", "Strong Bearish"}
        else "Neutral" if price_state == "Mixed / Neutral"
        else None
    )
    curve_group = (
        "Tight" if curve_state in {"Tight", "Strong Tightness", "Extreme Tightness"}
        else "Loose" if curve_state in {"Loose", "Strong Loose", "Extreme Loose"}
        else "Neutral" if curve_state == "Neutral"
        else None
    )
    if price_group is None or curve_group is None:
        return "N/A"
    matrix = {
        ("Bullish", "Tight"): "Bullish Confirmation",
        ("Bullish", "Neutral"): "Price-Led / Unconfirmed",
        ("Bullish", "Loose"): "Mixed / Divergent",
        ("Neutral", "Tight"): "Physical Tightness / Price Lag",
        ("Neutral", "Neutral"): "Mixed / Divergent",
        ("Neutral", "Loose"): "Mixed / Divergent",
        ("Bearish", "Tight"): "Physical Tightness / Price Lag",
        ("Bearish", "Neutral"): "Mixed / Divergent",
        ("Bearish", "Loose"): "Confirmed Weakness",
    }
    result = matrix[(price_group, curve_group)]
    if result == "Bullish Confirmation" and (
        price_state == "Strong Bullish" or curve_state in {"Strong Tightness", "Extreme Tightness"}
    ):
        return "Strong Physical Confirmation"
    return result


def resolve_cftc_qualifier(percentiles: Sequence[float], net_positions: Sequence[float]) -> str:
    values = np.asarray(percentiles, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return "N/A"
    pctl = float(np.median(values[valid]))
    net = np.asarray(net_positions, dtype=float)
    net = net[np.isfinite(net)]
    if not len(net):
        return "N/A"
    majority_long = np.count_nonzero(net > 0) > len(net) / 2
    majority_short = np.count_nonzero(net < 0) > len(net) / 2
    if pctl < 25:
        base = "Short / Contrarian"
    elif pctl < 75:
        base = "Not Crowded"
    elif pctl <= 90 and majority_long:
        base = "Crowded"
    elif pctl > 90 and majority_long:
        base = "Extremely Crowded"
    elif pctl >= 75 and majority_short:
        base = "High Relative Positioning / Still Net Short"
    else:
        base = "High Relative Positioning"
    if valid.sum() >= 2 and float(np.max(values[valid]) - np.min(values[valid])) >= 50:
        return f"{base} / High Dispersion"
    return base


def classify_capex_direction(change: float) -> str:
    if pd.isna(change):
        return "N/A"
    if change < -0.02:
        return "Falling"
    if change <= 0.02:
        return "Stable"
    return "Rising"


def calculate_capex_current_percentile(values: pd.Series, minimum: int = 20) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    observed = numeric.dropna()
    out = pd.Series(np.nan, index=numeric.index, dtype=float)
    if len(observed) >= minimum:
        out.loc[observed.index] = [midrank_percentile(v, observed.to_numpy()) for v in observed]
    return out


def calculate_capex_expanding_percentile(values: pd.Series, minimum: int = 20) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=numeric.index, dtype=float)
    seen: list[float] = []
    for index, value in numeric.items():
        if pd.notna(value):
            seen.append(float(value))
            if len(seen) >= minimum:
                out.loc[index] = midrank_percentile(float(value), seen)
    return out


def classify_capex_state(percentile: float, direction: str) -> str:
    if pd.isna(percentile) or direction not in {"Falling", "Stable", "Rising"}:
        return "N/A"
    band = 0 if percentile < 20 else 1 if percentile < 40 else 2 if percentile < 60 else 3 if percentile < 80 else 4
    states = (
        ("Extreme Vulnerability", "Extreme Vulnerability", "High Vulnerability"),
        ("High Vulnerability", "High Vulnerability", "Moderate-High Vulnerability"),
        ("Moderate-High Vulnerability", "Neutral", "Neutral"),
        ("Neutral / Deteriorating", "Low Vulnerability", "Low Vulnerability / Supply Response Building"),
        ("Low Vulnerability / Deteriorating", "Very Low Vulnerability", "Strong Supply Response Risk"),
    )
    return states[band][{"Falling": 0, "Stable": 1, "Rising": 2}[direction]]


def calculate_capex(history: pd.Series) -> pd.DataFrame:
    values = pd.to_numeric(history, errors="coerce")
    full_history = calculate_capex_current_percentile(values)
    expanding = calculate_capex_expanding_percentile(values)
    change = values.div(values.shift(8)).sub(1)
    directions = change.map(classify_capex_direction)
    state = [classify_capex_state(p, d) for p, d in zip(full_history, directions)]
    return pd.DataFrame({
        "CAPEX Intensity": values,
        "CAPEX Current Percentile": full_history,
        "CAPEX Expanding Percentile": expanding,
        "CAPEX Vulnerability RT": 100 - expanding,
        "CAPEX 24M Change": change,
        "CAPEX Direction": directions,
        "CAPEX State": state,
    }, index=values.index)


def _classify_raw_core(row: pd.Series) -> dict[str, bool]:
    s5, s10 = row.get("Seasonal Tightening 5"), row.get("Seasonal Tightening 10")
    r5, r10 = row.get("Rolling Tightening 5"), row.get("Rolling Tightening 10")
    e5, e10 = row.get("Seasonal Easing 5"), row.get("Rolling Easing 10")
    conf_t10, conf_net10 = row.get("Confirmation Tightening 10"), row.get("Confirmation Net 10")
    conf_ease5, conf_net5 = row.get("Confirmation Easing 5"), row.get("Confirmation Net 5")
    core_stress = row.get("Core Median Stress")
    systemic = all(pd.notna(v) for v in (s10, r10, conf_t10, conf_net10)) and s10 >= 2 and r10 >= 2 and conf_t10 >= 2 and conf_net10 >= 0.2
    early_broad = pd.notna(s5) and s5 >= 2
    confirmed_broad = early_broad and pd.notna(r5) and r5 >= 2
    route_a = pd.notna(e5) and e5 >= 2
    veto = pd.notna(core_stress) and core_stress >= 75 and ((pd.notna(s10) and s10 >= 1) or (pd.notna(r10) and r10 >= 1))
    route_a = route_a and not veto
    route_b = all(pd.notna(v) for v in (row.get("Seasonal Easing 5"), conf_ease5, conf_net5, core_stress)) and row["Seasonal Easing 5"] >= 1 and conf_ease5 >= 3 and conf_net5 <= -0.4 and core_stress < 60
    early_easing = route_a or route_b
    confirmed_easing = False  # Persistence is evaluated with the trailing 3-month window below.
    return {"systemic": systemic, "confirmed_broad": confirmed_broad, "early_broad": early_broad,
            "early_easing": early_easing, "confirmed_easing": confirmed_easing}


def resolve_core_state_transition(
    previous: str,
    raw: dict[str, bool],
    *,
    mature: bool = False,
    data_complete: bool = True,
) -> str:
    """Apply the deterministic priority and sticky-exit rules for one month."""
    if not data_complete:
        return "DATA INCOMPLETE"
    if raw.get("confirmed_easing", False):
        return "Confirmed Easing"
    if previous == "Early Easing":
        return "Systemic Broadening" if raw.get("systemic", False) else "Early Easing"
    if previous == "Confirmed Easing":
        if raw.get("systemic", False):
            return "Systemic Broadening"
        if raw.get("confirmed_broad", False):
            return "Confirmed Broadening"
        if raw.get("early_broad", False):
            return "Early Broadening"
        return "Confirmed Easing"
    if raw.get("early_easing", False):
        return "Early Easing"
    if raw.get("systemic", False):
        return "Systemic Broadening"
    if previous == "Systemic Broadening":
        return "Mature" if mature else "Systemic Broadening"
    if previous == "Mature":
        return "Mature"
    if previous == "Early Easing":
        return "Early Easing"
    if raw.get("confirmed_broad", False):
        return "Confirmed Broadening"
    if raw.get("early_broad", False):
        return "Early Broadening"
    return "Neutral"


def update_systemic_wave(state: str, previous: str, previous_duration: int, previous_max_stress: float, stress: float) -> tuple[int, float]:
    """Advance/reset duration and maximum stress for an uninterrupted systemic run."""
    if state != "Systemic Broadening":
        return 0, np.nan
    if previous == "Systemic Broadening":
        values = [v for v in (previous_max_stress, stress) if pd.notna(v)]
        return previous_duration + 1, max(values) if values else np.nan
    return 1, float(stress) if pd.notna(stress) else np.nan


def is_confirmed_easing_persistent(last_three_rolling_easing_counts: Sequence[float]) -> bool:
    values = pd.to_numeric(pd.Series(last_three_rolling_easing_counts), errors="coerce").tail(3)
    return bool(len(values) == 3 and values.notna().all() and values.ge(2).sum() >= 2)


def calculate_core_state(history: pd.DataFrame) -> pd.DataFrame:
    """Calculate raw breadth and apply the explicit sticky monthly state machine."""
    out = pd.DataFrame(index=history.index)
    out["Core Median Stress"] = history[[f"{name} Rolling Stress" for name in CORE_SERIES]].median(axis=1)
    for kind in ("Seasonal", "Rolling"):
        delta_cols = []
        for name in CORE_SERIES + CONFIRMATION_SERIES:
            stress_col = f"{name} {kind} Stress"
            delta_col = f"{name} {kind} Delta 6M"
            out[delta_col] = history[stress_col].diff(6)
            delta_cols.append(delta_col)
        for threshold in (5, 10):
            tight_core = out[[f"{name} {kind} Delta 6M" for name in CORE_SERIES]].gt(threshold).sum(axis=1)
            ease_core = out[[f"{name} {kind} Delta 6M" for name in CORE_SERIES]].lt(-threshold).sum(axis=1)
            out[f"{kind} Tightening {threshold}"] = tight_core
            out[f"{kind} Easing {threshold}"] = ease_core
            tight_conf = out[[f"{name} {kind} Delta 6M" for name in CONFIRMATION_SERIES]].gt(threshold).sum(axis=1)
            ease_conf = out[[f"{name} {kind} Delta 6M" for name in CONFIRMATION_SERIES]].lt(-threshold).sum(axis=1)
            out[f"Confirmation Tightening {threshold}"] = tight_conf
            out[f"Confirmation Easing {threshold}"] = ease_conf
            out[f"Confirmation Net {threshold}"] = (tight_conf - ease_conf) / 5
    raw = [_classify_raw_core(row) for _, row in out.iterrows()]
    states: list[str] = []
    system_duration: list[int] = []
    wave_max: list[float] = []
    recent_mature: list[bool] = []
    transition_diagnostic: list[str] = []
    current_duration, current_max, last_mature_position = 0, np.nan, -10_000
    for pos, ((_, row), flags) in enumerate(zip(out.iterrows(), raw)):
        required_breadth = ["Seasonal Tightening 5", "Seasonal Tightening 10", "Rolling Tightening 5",
                            "Rolling Tightening 10", "Seasonal Easing 5", "Seasonal Easing 10",
                            "Rolling Easing 10", "Confirmation Tightening 10", "Confirmation Net 10",
                            "Confirmation Easing 5", "Confirmation Net 5", "Core Median Stress"]
        previous = states[-1] if states else "Neutral"
        recent_mature_flag = pos - last_mature_position <= 6
        systemic_active = previous == "Systemic Broadening"
        mature_eligible = (current_duration >= 6 if systemic_active else False) or recent_mature_flag
        max_stress = current_max if systemic_active else np.nan
        loss_seasonal = out["Seasonal Tightening 10"].iloc[max(0, pos - 2):pos + 1].le(1).sum() >= 2
        net_peak = out["Confirmation Net 10"].iloc[max(0, pos - 6):pos + 1].max()
        net_now = row.get("Confirmation Net 10")
        net_decline = pd.notna(net_peak) and pd.notna(net_now) and net_peak - net_now >= 0.4
        mature_raw = mature_eligible and pd.notna(max_stress) and max_stress >= 60 and loss_seasonal and (
            (pd.notna(row.get("Rolling Tightening 10")) and row["Rolling Tightening 10"] <= 1) or net_decline
        )
        data_complete = not row[required_breadth].isna().any()
        flags = dict(flags)
        flags["confirmed_easing"] = is_confirmed_easing_persistent(
            out["Rolling Easing 10"].iloc[max(0, pos - 2):pos + 1]
        )
        candidate = resolve_core_state_transition(previous, flags, mature=mature_raw, data_complete=data_complete)
        transition_diagnostic.append("Re-Broadening" if previous == "Mature" and candidate == "Systemic Broadening" else "")

        current_duration, current_max = update_systemic_wave(
            candidate, previous, current_duration, current_max, row.get("Core Median Stress", np.nan)
        )
        if candidate == "Mature":
            last_mature_position = pos
        states.append(candidate)
        system_duration.append(current_duration)
        wave_max.append(current_max)
        recent_mature.append(pos - last_mature_position <= 6)
    out["Core State"] = states
    out["Systemic State Duration"] = system_duration
    out["Current Systemic Wave Max Stress"] = wave_max
    out["Recent Mature Within 6M"] = recent_mature
    out["Transition Diagnostic"] = transition_diagnostic
    return out


def classify_ppi_level(ppi12m: float) -> str:
    if pd.isna(ppi12m):
        return "N/A"
    if ppi12m < 0.04:
        return "Low"
    if ppi12m < 0.06:
        return "Moderate"
    if ppi12m < 0.08:
        return "Aggressive"
    if ppi12m <= 0.10:
        return "Severe"
    return "Extreme"


def classify_ppi_direction(impulse: float) -> str:
    if pd.isna(impulse):
        return "N/A"
    if impulse > 1:
        return "Accelerating"
    if impulse < -1:
        return "Decelerating"
    return "Stable"


def resolve_final_state(core_state: str, ppi12m: float, impulse_pp: float) -> str:
    if core_state in {"DATA INCOMPLETE", "N/A", "nan"} or pd.isna(ppi12m) or pd.isna(impulse_pp):
        return "N/A"
    accelerating = impulse_pp > 1
    decelerating = impulse_pp < -1
    if accelerating:
        return "Reflation / Early Inflation" if ppi12m < 0.06 else "Inflation Expansion"
    if decelerating:
        if core_state == "Confirmed Easing":
            return "Confirmed Disinflation"
        if core_state == "Early Easing":
            return "Disinflation Transition"
        if core_state == "Neutral" and ppi12m < 0.04:
            return "Low Inflation / Neutral"
        return "Late Cycle / Peak Risk"
    if core_state == "Neutral":
        return "Low Inflation / Neutral" if ppi12m < 0.04 else "Late Cycle / Peak Risk"
    if core_state in {"Early Broadening", "Confirmed Broadening"}:
        return "Reflation / Early Inflation"
    if core_state == "Systemic Broadening":
        return "Inflation Expansion" if ppi12m >= 0.06 else "Reflation / Early Inflation"
    if core_state == "Mature":
        return "Late Cycle / Peak Risk"
    if core_state == "Early Easing":
        return "Disinflation Transition"
    if core_state == "Confirmed Easing":
        return "Low Inflation / Neutral" if ppi12m < 0.04 else "Late Cycle / Peak Risk"
    return "N/A"


def calculate_ppi_states(ppiaco: pd.Series, core_states: pd.Series) -> pd.DataFrame:
    values = pd.to_numeric(ppiaco, errors="coerce")
    ppi12m = values.div(values.shift(12)).sub(1) * 100
    impulse = ppi12m - ppi12m.shift(3)
    direction = impulse.map(classify_ppi_direction)
    final = [resolve_final_state(state, level / 100 if pd.notna(level) else np.nan, change)
             for state, level, change in zip(core_states, ppi12m, impulse)]
    final2 = ["Broad Inflation" if state in {"Reflation / Early Inflation", "Inflation Expansion", "Late Cycle / Peak Risk"}
              else state for state in final]
    return pd.DataFrame({
        "PPIACO": values,
        "PPI12M": ppi12m,
        "PPI Impulse 3M pp": impulse,
        "PPI Level": ppi12m.map(lambda x: classify_ppi_level(x / 100 if pd.notna(x) else np.nan)),
        "PPI Direction": direction,
        "PPI Confirmation": [f"{level} / {direction_}" for level, direction_ in zip(ppi12m.map(lambda x: classify_ppi_level(x / 100 if pd.notna(x) else np.nan)), direction)],
        "Final State": final,
        "Final State 2": final2,
    }, index=values.index)
