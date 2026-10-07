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
    # Percentiles are displayed to two decimals, while upstream calculations may
    # land microscopically below an exact threshold (for example 59.999999999).
    # Normalize numerical noise before applying the centralized half-open bands.
    percentile = round(float(percentile), 8)
    if percentile < 10:
        return "Extreme Loose vs Seasonal"
    if percentile < 25:
        return "Strong Loose vs Seasonal"
    if percentile < 40:
        return "Mild Loose vs Seasonal"
    if percentile < 60:
        return "Neutral"
    if percentile < 75:
        return "Mild Tight vs Seasonal"
    if percentile <= 90:
        return "Strong Tight vs Seasonal"
    return "Extreme Tight vs Seasonal"


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


def classify_net_direction(net_pct_oi: float, tolerance: float = 1e-9) -> str:
    if pd.isna(net_pct_oi):
        return "N/A"
    if float(net_pct_oi) > tolerance:
        return "Net Long"
    if float(net_pct_oi) < -tolerance:
        return "Net Short"
    return "Neutral"


def resolve_price_curve_market_state(price_state: str, curve_state: str) -> str:
    price_group = (
        "Bullish" if price_state in {"Bullish", "Strong Bullish"}
        else "Bearish" if price_state in {"Bearish", "Strong Bearish"}
        else "Neutral" if price_state == "Mixed / Neutral"
        else None
    )
    curve_group = (
        "Tight" if curve_state in {
            "Mild Tight vs Seasonal", "Strong Tight vs Seasonal", "Extreme Tight vs Seasonal"
        }
        else "Loose" if curve_state in {
            "Mild Loose vs Seasonal", "Strong Loose vs Seasonal", "Extreme Loose vs Seasonal"
        }
        else "Neutral" if curve_state == "Neutral"
        else None
    )
    if price_group is None or curve_group is None:
        return "N/A"
    matrix = {
        ("Bullish", "Tight"): "Bullish Confirmation",
        ("Bullish", "Neutral"): "Price-Led / Unconfirmed",
        ("Bullish", "Loose"): "Mixed / Divergent",
        ("Neutral", "Tight"): "Seasonal Tightness / Price Lag",
        ("Neutral", "Neutral"): "Mixed / Divergent",
        ("Neutral", "Loose"): "Mixed / Divergent",
        ("Bearish", "Tight"): "Seasonal Tightness / Price Lag",
        ("Bearish", "Neutral"): "Mixed / Divergent",
        ("Bearish", "Loose"): "Confirmed Weakness",
    }
    result = matrix[(price_group, curve_group)]
    if result == "Bullish Confirmation" and (
        price_state == "Strong Bullish"
        or curve_state in {"Strong Tight vs Seasonal", "Extreme Tight vs Seasonal"}
    ):
        return "Strong Seasonal Confirmation"
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
    direction_value = float(np.median(net))
    direction = classify_net_direction(direction_value)
    relative = classify_cftc_relative_state(pctl)
    if direction == "N/A" or relative == "N/A":
        return "N/A"
    base = f"{direction} / {relative} Relative Positioning"
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
    e5 = row.get("Seasonal Easing 5")
    conf_t10 = row.get("Seasonal Confirmation Tightening 10", row.get("Confirmation Tightening 10"))
    conf_net10 = row.get("Seasonal Confirmation Net 10", row.get("Confirmation Net 10"))
    conf_ease5 = row.get("Seasonal Confirmation Easing 5", row.get("Confirmation Easing 5"))
    conf_net5 = row.get("Seasonal Confirmation Net 5", row.get("Confirmation Net 5"))
    core_stress = row.get("Core Median Stress")
    systemic = all(pd.notna(v) for v in (s10, r10, conf_t10, conf_net10)) and s10 >= 2 and r10 >= 2 and conf_t10 >= 2 and conf_net10 >= 0.2
    early_broad = pd.notna(s5) and s5 >= 2
    confirmed_broad = early_broad and pd.notna(r5) and r5 >= 2
    route_a = pd.notna(e5) and e5 >= 2
    veto = pd.notna(core_stress) and core_stress >= 75 and ((pd.notna(s10) and s10 >= 1) or (pd.notna(r10) and r10 >= 1))
    route_a = route_a and not veto
    route_b = all(pd.notna(v) for v in (e5, conf_ease5, conf_net5, core_stress)) and e5 >= 1 and conf_ease5 >= 3 and conf_net5 <= -0.4 and core_stress < 60
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
    """Resolve one V1.2 lifecycle transition from the prior valid state and raw signals."""
    if not data_complete:
        return "DATA INCOMPLETE"
    previous = previous if previous != "DATA INCOMPLETE" else "Neutral"

    def first_true(rules: tuple[tuple[str, str], ...], default: str) -> str:
        return next((state for signal, state in rules if raw.get(signal, False)), default)

    broadening_rules = (
        ("systemic", "Systemic Broadening"),
        ("confirmed_broad", "Confirmed Broadening"),
        ("early_broad", "Early Broadening"),
    )
    if previous == "Systemic Broadening":
        return "Mature" if mature else "Systemic Broadening"
    if previous == "Mature":
        if raw.get("systemic", False):
            return "Systemic Broadening"
        return "Early Easing" if raw.get("early_easing", False) else "Mature"
    if previous == "Early Easing":
        # Early Easing is the hand-off stage from Mature toward Confirmed Easing.
        # It remains active until that transition confirms; unrelated raw
        # broadening/easing fluctuations do not restart the cycle.
        return "Confirmed Easing" if raw.get("confirmed_easing", False) else "Early Easing"
    if previous == "Confirmed Easing":
        # A new broadening signal can end easing; otherwise confirmed easing
        # remains only while its rolling persistence condition is active.
        return first_true(broadening_rules, "Confirmed Easing" if raw.get("confirmed_easing", False)
                          else "Early Easing" if raw.get("early_easing", False) else "Neutral")
    if previous in {"Early Broadening", "Confirmed Broadening"}:
        return first_true(broadening_rules, "Neutral")
    return first_true(broadening_rules, "Neutral")


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
    """Calculate strict raw breadth and apply the V1.2 lifecycle state machine."""
    out = pd.DataFrame(index=history.index)
    core_rolling_stress = history[[f"{name} Rolling Stress" for name in CORE_SERIES]]
    out["Core Median Stress"] = core_rolling_stress.median(axis=1, skipna=False)
    for kind in ("Seasonal", "Rolling"):
        delta_cols = []
        for name in CORE_SERIES + CONFIRMATION_SERIES:
            stress_col = f"{name} {kind} Stress"
            delta_col = f"{name} {kind} Delta 6M"
            out[delta_col] = history[stress_col].diff(6)
            delta_cols.append(delta_col)
        for threshold in (5, 10):
            core_delta = out[[f"{name} {kind} Delta 6M" for name in CORE_SERIES]]
            core_complete = core_delta.notna().all(axis=1)
            tight_core = core_delta.gt(threshold).sum(axis=1).where(core_complete)
            ease_core = core_delta.lt(-threshold).sum(axis=1).where(core_complete)
            out[f"{kind} Tightening {threshold}"] = tight_core
            out[f"{kind} Easing {threshold}"] = ease_core
            confirmation_delta = out[[f"{name} {kind} Delta 6M" for name in CONFIRMATION_SERIES]]
            confirmation_complete = confirmation_delta.notna().all(axis=1)
            tight_conf = confirmation_delta.gt(threshold).sum(axis=1).where(confirmation_complete)
            ease_conf = confirmation_delta.lt(-threshold).sum(axis=1).where(confirmation_complete)
            net_conf = ((tight_conf - ease_conf) / 5).where(confirmation_complete)
            out[f"{kind} Confirmation Tightening {threshold}"] = tight_conf
            out[f"{kind} Confirmation Easing {threshold}"] = ease_conf
            out[f"{kind} Confirmation Net {threshold}"] = net_conf
            if kind == "Seasonal":
                # Keep the historical field names, now explicitly mapped to the V1.2 source.
                out[f"Confirmation Tightening {threshold}"] = tight_conf
                out[f"Confirmation Easing {threshold}"] = ease_conf
                out[f"Confirmation Net {threshold}"] = net_conf

    required_deltas = (
        [f"{name} Seasonal Delta 6M" for name in CORE_SERIES]
        + [f"{name} Rolling Delta 6M" for name in CORE_SERIES]
        + [f"{name} Seasonal Delta 6M" for name in CONFIRMATION_SERIES]
    )
    required_stress = [f"{name} Rolling Stress" for name in CORE_SERIES]
    required_inputs = pd.concat([out[required_deltas], history[required_stress]], axis=1)
    current_complete = required_inputs.notna().all(axis=1)
    out["Data Complete"] = current_complete
    out["Missing Required Inputs"] = required_inputs.isna().apply(
        lambda row: ", ".join(row.index[row].tolist()), axis=1
    )

    states: list[str] = []
    system_duration: list[int] = []
    wave_max: list[float] = []
    recent_mature: list[bool] = []
    mature_eligibility: list[bool] = []
    mature_reference_max: list[float] = []
    raw_early_broadening: list[Any] = []
    raw_confirmed_broadening: list[Any] = []
    raw_systemic: list[Any] = []
    raw_mature: list[Any] = []
    raw_early_easing: list[Any] = []
    raw_confirmed_easing: list[Any] = []
    transition_diagnostic: list[str] = []
    current_duration, current_max = 0, np.nan
    last_mature_position = -10_000
    last_mature_wave_max = np.nan
    previous_valid_state = "Neutral"
    for pos, (_, row) in enumerate(out.iterrows()):
        previous = previous_valid_state
        recent_mature_flag = pos - last_mature_position <= 6
        systemic_active = previous == "Systemic Broadening"
        mature_eligible = (current_duration >= 6 if systemic_active else False) or recent_mature_flag

        easing_window = out["Rolling Easing 10"].iloc[max(0, pos - 2):pos + 1]
        easing_window_complete = len(easing_window) == 3 and easing_window.notna().all()
        confirmed_easing = easing_window_complete and is_confirmed_easing_persistent(easing_window)

        seasonal_mature_window = out["Seasonal Tightening 10"].iloc[max(0, pos - 2):pos + 1]
        confirmation_peak_window = out["Seasonal Confirmation Net 10"].iloc[max(0, pos - 6):pos + 1]
        mature_windows_complete = (
            len(seasonal_mature_window) == 3 and seasonal_mature_window.notna().all()
            and len(confirmation_peak_window) == 7 and confirmation_peak_window.notna().all()
        )
        if systemic_active and current_duration >= 6:
            mature_reference_stress = current_max
        elif recent_mature_flag:
            mature_reference_stress = last_mature_wave_max
        elif systemic_active:
            mature_reference_stress = current_max
        else:
            mature_reference_stress = np.nan
        loss_seasonal = bool(seasonal_mature_window.le(1).sum() >= 2) if seasonal_mature_window.notna().all() else False
        net_peak = confirmation_peak_window.max() if confirmation_peak_window.notna().all() else np.nan
        net_now = row.get("Seasonal Confirmation Net 10")
        net_decline = pd.notna(net_peak) and pd.notna(net_now) and net_peak - net_now >= 0.4
        mature_signal = (
            mature_eligible and mature_windows_complete and pd.notna(mature_reference_stress)
            and mature_reference_stress >= 60 and loss_seasonal
            and (row.get("Rolling Tightening 10") <= 1 or net_decline)
        )
        data_complete = bool(current_complete.iloc[pos])
        if previous in {"Early Easing", "Confirmed Easing"} and not easing_window_complete:
            data_complete = False
        if mature_eligible and not mature_windows_complete:
            data_complete = False

        flags = _classify_raw_core(row)
        flags["confirmed_easing"] = bool(confirmed_easing)
        candidate = resolve_core_state_transition(previous, flags, mature=bool(mature_signal), data_complete=data_complete)

        for values, key in (
            (raw_early_broadening, "early_broad"),
            (raw_confirmed_broadening, "confirmed_broad"),
            (raw_systemic, "systemic"),
            (raw_early_easing, "early_easing"),
        ):
            values.append(bool(flags[key]) if data_complete else np.nan)
        raw_confirmed_easing.append(bool(confirmed_easing) if data_complete else np.nan)
        raw_mature.append(bool(mature_signal) if data_complete else np.nan)
        mature_eligibility.append(bool(mature_eligible) if data_complete else np.nan)
        mature_reference_max.append(float(mature_reference_stress) if pd.notna(mature_reference_stress) else np.nan)
        transition_diagnostic.append("Re-Broadening" if previous == "Mature" and candidate == "Systemic Broadening" else "")

        if data_complete:
            if candidate == "Mature":
                if systemic_active:
                    last_mature_wave_max = current_max
                last_mature_position = pos
            current_duration, current_max = update_systemic_wave(
                candidate, previous, current_duration, current_max, row.get("Core Median Stress", np.nan)
            )
            previous_valid_state = candidate
        states.append(candidate)
        system_duration.append(current_duration)
        wave_max.append(current_max)
        recent_mature.append(pos - last_mature_position <= 6)

    out["Core State"] = states
    out["Systemic State Duration"] = system_duration
    out["Current Systemic Wave Max Stress"] = wave_max
    out["Recent Mature Within 6M"] = recent_mature
    out["Mature Eligibility"] = mature_eligibility
    out["Mature Reference Wave Max Stress"] = mature_reference_max
    out["Raw Early Broadening"] = raw_early_broadening
    out["Raw Confirmed Broadening"] = raw_confirmed_broadening
    out["Raw Systemic"] = raw_systemic
    out["Raw Mature"] = raw_mature
    out["Raw Early Easing"] = raw_early_easing
    out["Raw Confirmed Easing"] = raw_confirmed_easing
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
