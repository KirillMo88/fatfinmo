from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from elliott_waves.config import AssetSpec
from elliott_waves.data import aggregate_daily_bars, validate_bars

from .config import CONFIG, MODEL_VERSION


def build_timeframe_bars(daily: pd.DataFrame, timeframe: str, spec: AssetSpec) -> pd.DataFrame:
    values = validate_bars(daily, spec)
    bars = aggregate_daily_bars(values, timeframe, spec)
    return bars.loc[bars["is_closed"].fillna(False)].reset_index(drop=True)


def calculate_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy().reset_index(drop=True)
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    previous = close.shift(1)
    true_range = pd.concat([(high - low).abs(), (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
    frame["atr14"] = true_range.rolling(14, min_periods=14).mean()
    slope_window = int(CONFIG["ma"]["slope_window"])
    for window in CONFIG["ma"]["windows"]:
        column = f"sma{window}"
        frame[column] = close.rolling(window, min_periods=window).mean()
        frame[f"{column}_slope"] = frame[column].pct_change(slope_window) * 100.0 / slope_window
        frame[f"price_vs_{column}"] = np.where(frame[column].gt(0), (close / frame[column] - 1.0) * 100.0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    rsi = rsi.mask(loss.eq(0.0) & gain.gt(0.0), 100.0)
    rsi = rsi.mask(loss.eq(0.0) & gain.eq(0.0), 50.0)
    frame["rsi14"] = rsi

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    frame["macd"] = ema12 - ema26
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    frame["macd_hist"] = frame["macd"] - frame["macd_signal"]
    frame["macd_hist_delta"] = frame["macd_hist"].diff()
    frame["roc12"] = close.pct_change(12) * 100.0
    frame["roc12_delta"] = frame["roc12"].diff()
    frame["extension200"] = frame["price_vs_sma200"]
    frame["drawdown_ath"] = (close / close.cummax() - 1.0) * 100.0
    frame["volatility12"] = close.pct_change().rolling(12, min_periods=12).std() * math.sqrt(12) * 100.0
    volume = pd.to_numeric(frame.get("volume", pd.Series(np.nan, index=frame.index)), errors="coerce")
    frame["volume_ma20"] = volume.rolling(20, min_periods=10).mean()
    frame["volume_ratio"] = np.where(frame["volume_ma20"].gt(0), volume / frame["volume_ma20"], np.nan)
    return frame


def detect_pivots(
    frame: pd.DataFrame,
    timeframe: str,
    degree: str | None = None,
    *,
    atr_multiplier: float | None = None,
    minimum_reversal_pct: float | None = None,
) -> list[dict[str, Any]]:
    """Streaming ATR/percentage ZigZag with causal confirmation.

    A pivot is timestamped at its extreme but is only CONFIRMED on the later
    bar whose reversal crosses the configured threshold.  The final extreme is
    retained as POTENTIAL and therefore cannot leak into historical snapshots
    as a confirmed endpoint.
    """
    degree_name = degree or ("MAJOR" if timeframe == "MONTHLY" else "INTERMEDIATE")
    defaults = CONFIG["pivot"]["degrees"][degree_name]
    multiplier = float(atr_multiplier if atr_multiplier is not None else defaults["atr_multiplier"])
    minimum_pct = float(minimum_reversal_pct if minimum_reversal_pct is not None else defaults["minimum_reversal_pct"]) / 100.0
    if frame.empty:
        return []
    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    atrs = pd.to_numeric(frame["atr14"], errors="coerce").to_numpy(dtype=float)
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce").tolist()

    def threshold(index: int, price: float) -> float:
        atr_value = atrs[index] if 0 <= index < len(atrs) and np.isfinite(atrs[index]) else abs(price) * minimum_pct
        return max(float(atr_value) * multiplier, abs(price) * minimum_pct)

    def pivot(kind: str, index: int, status: str, confirmation_index: int | None, reversal: float | None) -> dict[str, Any]:
        price = highs[index] if kind == "HIGH" else lows[index]
        return {
            "pivot_id": f"{degree_name}:{kind}:{pd.Timestamp(timestamps[index]).isoformat()}",
            "kind": kind,
            "price": float(price),
            "bar_index": int(index),
            "pivot_time": pd.Timestamp(timestamps[index]).isoformat(),
            "confirmation_time": pd.Timestamp(timestamps[confirmation_index]).isoformat() if confirmation_index is not None else None,
            "confirmed_at": pd.Timestamp(timestamps[confirmation_index]).isoformat() if confirmation_index is not None else None,
            "timeframe": timeframe,
            "degree": degree_name,
            "status": status,
            "reversal_magnitude": finite(reversal),
            "atr_reference": finite(atrs[index]),
            "threshold_used": float(threshold(index, float(price))),
            "atr_multiplier": multiplier,
            "minimum_reversal_pct": minimum_pct * 100.0,
        }

    confirmed: list[dict[str, Any]] = []
    high_index = 0
    low_index = 0
    direction = 0  # 1 seeks a high after a low; -1 seeks a low after a high.
    for index in range(1, len(frame)):
        if highs[index] >= highs[high_index]:
            high_index = index
        if lows[index] <= lows[low_index]:
            low_index = index
        if direction == 0:
            if low_index < high_index and highs[high_index] - lows[low_index] >= threshold(low_index, lows[low_index]):
                confirmed.append(pivot("LOW", low_index, "CONFIRMED", high_index, highs[high_index] - lows[low_index]))
                direction = 1
            elif high_index < low_index and highs[high_index] - lows[low_index] >= threshold(high_index, highs[high_index]):
                confirmed.append(pivot("HIGH", high_index, "CONFIRMED", low_index, highs[high_index] - lows[low_index]))
                direction = -1
            continue
        if direction == 1:
            if highs[index] >= highs[high_index]:
                high_index = index
            reversal = highs[high_index] - lows[index]
            if reversal >= threshold(high_index, highs[high_index]):
                confirmed.append(pivot("HIGH", high_index, "CONFIRMED", index, reversal))
                low_index = index
                direction = -1
        else:
            if lows[index] <= lows[low_index]:
                low_index = index
            reversal = highs[index] - lows[low_index]
            if reversal >= threshold(low_index, lows[low_index]):
                confirmed.append(pivot("LOW", low_index, "CONFIRMED", index, reversal))
                high_index = index
                direction = 1

    potential_kind = "HIGH" if direction == 1 else "LOW" if direction == -1 else ("HIGH" if high_index > low_index else "LOW")
    potential_index = high_index if potential_kind == "HIGH" else low_index
    if not confirmed or confirmed[-1]["bar_index"] != potential_index:
        confirmed.append(pivot(potential_kind, potential_index, "POTENTIAL", None, None))
    return confirmed


def classify_structure(frame: pd.DataFrame, pivots: list[dict[str, Any]]) -> dict[str, Any]:
    confirmed = [pivot for pivot in pivots if pivot.get("status") == "CONFIRMED"]
    highs = [pivot for pivot in confirmed if pivot["kind"] == "HIGH"]
    lows = [pivot for pivot in confirmed if pivot["kind"] == "LOW"]
    sequence: list[str] = []
    state = "TRANSITION"
    if len(highs) >= 2 and len(lows) >= 2:
        high_label = "HH" if highs[-1]["price"] > highs[-2]["price"] else "LH"
        low_label = "HL" if lows[-1]["price"] > lows[-2]["price"] else "LL"
        sequence = [high_label, low_label]
        if sequence == ["HH", "HL"]:
            state = "BULL"
        elif sequence == ["LH", "LL"]:
            state = "BEAR"
        else:
            state = "RANGE"
    atr = pd.to_numeric(frame.get("atr14"), errors="coerce").dropna()
    volatility_state = "NORMAL"
    if len(atr) >= 24 and atr.iloc[-24:].mean() > 0:
        ratio = float(atr.iloc[-6:].mean() / atr.iloc[-24:].mean())
        volatility_state = "COMPRESSION" if ratio < 0.82 else "EXPANSION" if ratio > 1.20 else "NORMAL"
        if state == "RANGE" and volatility_state == "EXPANSION":
            state = "TRANSITION"
    return {
        "state": state,
        "sequence": "-".join(sequence) if sequence else "INSUFFICIENT_PIVOTS",
        "volatility_state": volatility_state,
        "last_high": highs[-1] if highs else None,
        "last_low": lows[-1] if lows else None,
        "explanation": _structure_explanation(state, sequence, volatility_state),
    }


def moving_average_state(frame: pd.DataFrame) -> dict[str, Any]:
    row = frame.iloc[-1]
    price = finite(row.get("close"))
    values = {f"sma{window}": finite(row.get(f"sma{window}")) for window in CONFIG["ma"]["windows"]}
    slopes = {f"sma{window}": finite(row.get(f"sma{window}_slope")) for window in CONFIG["ma"]["windows"]}
    distances = {f"sma{window}": finite(row.get(f"price_vs_sma{window}")) for window in CONFIG["ma"]["windows"]}
    available = [values[f"sma{window}"] is not None for window in CONFIG["ma"]["windows"]]
    state = "INSUFFICIENT_HISTORY"
    ordering = "Unavailable"
    if price is not None and all(available):
        sma50, sma100, sma200 = values["sma50"], values["sma100"], values["sma200"]
        assert sma50 is not None and sma100 is not None and sma200 is not None
        positive = sum((slopes[key] or 0.0) > 0 for key in slopes)
        negative = sum((slopes[key] or 0.0) < 0 for key in slopes)
        if price > sma50 > sma100 > sma200 and positive >= 2:
            state, ordering = "STRONG_BULL", "Price > SMA50 > SMA100 > SMA200"
        elif price < sma50 and sma50 > sma100 > sma200:
            state, ordering = "BULL_CORRECTION", "Price < SMA50; SMA50 > SMA100 > SMA200"
        elif price < sma50 < sma100 < sma200 and negative >= 2:
            state, ordering = "STRONG_BEAR", "Price < SMA50 < SMA100 < SMA200"
        elif price > sma200 and sma50 > sma100:
            state, ordering = "BULL", "Price > SMA200; SMA50 > SMA100"
        elif price < sma200 and sma50 < sma100:
            state, ordering = "BEAR", "Price < SMA200; SMA50 < SMA100"
        else:
            state, ordering = "TRANSITION", "Moving averages are compressed or crossing"
    elif price is not None:
        finite_pairs = [(window, values[f"sma{window}"]) for window in CONFIG["ma"]["windows"] if values[f"sma{window}"] is not None]
        if finite_pairs:
            above = sum(price > value for _, value in finite_pairs if value is not None)
            state = "BULL" if above == len(finite_pairs) else "BEAR" if above == 0 else "TRANSITION"
            ordering = f"{above}/{len(finite_pairs)} available MAs below price"
    return {"state": state, "ordering": ordering, "values": values, "slopes": slopes, "distances_pct": distances}


def extension_state(value: Any) -> str:
    number = finite(value)
    if number is None:
        return "NOT_AVAILABLE"
    magnitude = abs(number)
    thresholds = CONFIG["extension_thresholds"]
    if magnitude < thresholds["low"]:
        return "LOW"
    if magnitude < thresholds["normal"]:
        return "NORMAL"
    if magnitude < thresholds["high"]:
        return "HIGH"
    return "EXTREME"


def momentum_state(frame: pd.DataFrame) -> dict[str, Any]:
    row = frame.iloc[-1]
    rsi = finite(row.get("rsi14"))
    macd = finite(row.get("macd"))
    signal = finite(row.get("macd_signal"))
    hist = finite(row.get("macd_hist"))
    hist_delta = finite(row.get("macd_hist_delta"))
    roc = finite(row.get("roc12"))
    roc_delta = finite(row.get("roc12_delta"))
    score_parts = []
    if rsi is not None:
        score_parts.append(np.clip((rsi - 50.0) / 20.0, -1.0, 1.0) * 35.0)
    if hist is not None:
        scale = max(abs(macd or 0.0), abs(signal or 0.0), 1e-9)
        score_parts.append(np.clip(hist / scale, -1.0, 1.0) * 35.0)
    if roc is not None:
        score_parts.append(np.clip(roc / 15.0, -1.0, 1.0) * 30.0)
    score = float(sum(score_parts) / max(1, len(score_parts)) * 3.0) if score_parts else 0.0
    score = float(np.clip(score, -100.0, 100.0))
    classification = (
        "STRONG_POSITIVE" if score >= 60 else "POSITIVE" if score >= 20 else
        "STRONG_NEGATIVE" if score <= -60 else "NEGATIVE" if score <= -20 else "NEUTRAL"
    )
    trajectory_raw = sum(value for value in [hist_delta, roc_delta] if value is not None)
    if abs(trajectory_raw) < 0.05:
        trajectory = "STABLE"
    elif (score >= 0 and trajectory_raw > 0) or (score < 0 and trajectory_raw < 0):
        trajectory = "ACCELERATING"
    else:
        trajectory = "DECELERATING"
    return {
        "classification": classification,
        "trajectory": trajectory,
        "score": round(score, 2),
        "rsi14": rsi,
        "rsi_regime": "ABOVE_50" if rsi is not None and rsi >= 50 else "BELOW_50" if rsi is not None else "NOT_AVAILABLE",
        "macd": macd,
        "macd_signal": signal,
        "macd_hist": hist,
        "macd_zero_state": "ABOVE_ZERO" if (macd or 0.0) >= 0 else "BELOW_ZERO",
        "macd_crossover": "BULLISH" if macd is not None and signal is not None and macd >= signal else "BEARISH",
        "macd_histogram": "EXPANDING" if hist is not None and hist_delta is not None and hist * hist_delta > 0 else "CONTRACTING",
        "roc12": roc,
        "roc_state": _roc_state(roc, roc_delta),
        "extension200": finite(row.get("extension200")),
        "extension_state": extension_state(row.get("extension200")),
    }


def detect_divergences(frame: pd.DataFrame, pivots: list[dict[str, Any]], timeframe: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for indicator in ("rsi14", "macd"):
        for kind, direction in (("HIGH", "BEARISH"), ("LOW", "BULLISH")):
            candidates = [pivot for pivot in pivots if pivot["kind"] == kind]
            if len(candidates) < 2:
                continue
            first, second = candidates[-2], candidates[-1]
            first_value = finite(frame.iloc[int(first["bar_index"])].get(indicator))
            second_value = finite(frame.iloc[int(second["bar_index"])].get(indicator))
            if first_value is None or second_value is None:
                continue
            price_condition = second["price"] > first["price"] if kind == "HIGH" else second["price"] < first["price"]
            indicator_condition = second_value < first_value if kind == "HIGH" else second_value > first_value
            if not (price_condition and indicator_condition):
                continue
            magnitude = abs(second_value - first_value) / max(abs(first_value), 1e-9) * 100.0
            results.append({
                "type": f"{direction}_{indicator.upper()}",
                "indicator": indicator.upper(),
                "direction": direction,
                "timeframe": timeframe,
                "start_pivot": first,
                "end_pivot": second,
                "magnitude": round(float(magnitude), 2),
                "active": len(frame) - 1 - int(second["bar_index"]) <= int(CONFIG["divergence"]["active_bars"]),
                "detected_at": second["confirmed_at"],
            })
    return results


def volume_state(frame: pd.DataFrame) -> dict[str, Any]:
    row = frame.iloc[-1]
    ratio = finite(row.get("volume_ratio"))
    volume = pd.to_numeric(frame.get("volume"), errors="coerce")
    if ratio is None or volume.notna().sum() < 20 or float(volume.fillna(0).abs().sum()) == 0.0:
        return {"status": "NOT_AVAILABLE", "score": None, "current_vs_average": None, "trend": "NOT_AVAILABLE", "event": "NOT_AVAILABLE"}
    close = pd.to_numeric(frame["close"], errors="coerce")
    return_ = float(close.pct_change().iloc[-1])
    recent_high = float(pd.to_numeric(frame["high"], errors="coerce").shift(1).tail(20).max())
    recent_low = float(pd.to_numeric(frame["low"], errors="coerce").shift(1).tail(20).min())
    event = "NORMAL"
    if float(row["close"]) > recent_high and ratio >= 1.25:
        event = "BREAKOUT_CONFIRMED"
    elif float(row["close"]) < recent_low and ratio >= 1.25:
        event = "BREAKDOWN_CONFIRMED"
    elif return_ < -0.04 and ratio >= 1.8:
        event = "POTENTIAL_CAPITULATION"
    elif return_ < 0 and ratio < 0.8:
        event = "DECLINING_CORRECTION_VOLUME"
    trend = "EXPANSION" if ratio >= 1.20 else "CONTRACTION" if ratio <= 0.80 else "NORMAL"
    score = 70.0 if event == "BREAKOUT_CONFIRMED" else 30.0 if event == "BREAKDOWN_CONFIRMED" else 50.0
    return {"status": "AVAILABLE", "score": score, "current_vs_average": ratio, "trend": trend, "event": event}


def volume_profile(frame: pd.DataFrame) -> dict[str, Any]:
    cfg = CONFIG["volume_profile"]
    values = frame.tail(int(cfg["lookback_bars"])).copy()
    volume = pd.to_numeric(values.get("volume"), errors="coerce")
    price = (pd.to_numeric(values["high"], errors="coerce") + pd.to_numeric(values["low"], errors="coerce") + pd.to_numeric(values["close"], errors="coerce")) / 3.0
    valid = price.notna() & volume.notna() & volume.gt(0)
    if valid.sum() < 20:
        return {"status": "NOT_AVAILABLE", "poc": None, "hvns": [], "lvns": [], "value_area": None, "lookback_bars": int(len(values))}
    bins = int(cfg["bins"])
    histogram, edges = np.histogram(price.loc[valid], bins=bins, weights=volume.loc[valid])
    centers = (edges[:-1] + edges[1:]) / 2.0
    poc_index = int(np.argmax(histogram))
    order = list(np.argsort(histogram)[::-1])
    total = float(histogram.sum())
    selected: list[int] = []
    running = 0.0
    for index in order:
        selected.append(int(index))
        running += float(histogram[index])
        if total > 0 and running / total >= float(cfg["value_area"]):
            break
    positive = histogram[histogram > 0]
    high_cutoff = float(np.quantile(positive, 0.80)) if len(positive) else np.inf
    low_cutoff = float(np.quantile(positive, 0.20)) if len(positive) else -np.inf
    hvns = [float(centers[index]) for index in np.where(histogram >= high_cutoff)[0] if index != poc_index][:4]
    lvns = [float(centers[index]) for index in np.where((histogram > 0) & (histogram <= low_cutoff))[0]][:4]
    return {
        "status": "AVAILABLE",
        "poc": float(centers[poc_index]),
        "hvns": hvns,
        "lvns": lvns,
        "value_area": {"low": float(edges[min(selected)]), "high": float(edges[max(selected) + 1])},
        "lookback_bars": int(len(values)),
        "range": {"low": float(edges[0]), "high": float(edges[-1])},
    }


def build_support_resistance(
    frame: pd.DataFrame,
    pivots: list[dict[str, Any]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    row = frame.iloc[-1]
    current = float(row["close"])
    atr = finite(row.get("atr14")) or current * 0.02
    weights = CONFIG["support_resistance"]["weights"]
    levels: list[dict[str, Any]] = []
    for pivot in pivots[-10:]:
        levels.append({"price": float(pivot["price"]), "source": "major_swing", "weight": weights["major_swing"], "timeframe": pivot["timeframe"]})
    for window in CONFIG["ma"]["windows"]:
        value = finite(row.get(f"sma{window}"))
        if value is not None:
            levels.append({"price": value, "source": f"sma{window}", "weight": weights[f"sma{window}"], "timeframe": "WEEKLY"})
    if profile.get("status") == "AVAILABLE":
        levels.append({"price": float(profile["poc"]), "source": "poc", "weight": weights["poc"], "timeframe": "WEEKLY"})
        for value in profile.get("hvns", []):
            levels.append({"price": float(value), "source": "hvn", "weight": weights["hvn"], "timeframe": "WEEKLY"})
        for value in profile.get("lvns", []):
            levels.append({"price": float(value), "source": "lvn", "weight": weights["lvn"], "timeframe": "WEEKLY"})
    if len(pivots) >= 2:
        first, second = pivots[-2], pivots[-1]
        low, high = sorted([float(first["price"]), float(second["price"])])
        for ratio in (0.382, 0.5, 0.618, 1.0, 1.618):
            price = high - (high - low) * ratio if ratio <= 1 else high + (high - low) * (ratio - 1)
            levels.append({"price": price, "source": "fibonacci", "weight": weights["fibonacci"], "timeframe": "WEEKLY"})
    magnitude = 10 ** max(0, int(math.floor(math.log10(max(abs(current), 1.0)))) - 1)
    for multiple in range(-2, 3):
        rounded = round(current / magnitude + multiple) * magnitude
        levels.append({"price": rounded, "source": "round_number", "weight": weights["round_number"], "timeframe": "WEEKLY"})

    threshold = max(current * float(CONFIG["support_resistance"]["cluster_price_fraction"]), atr * float(CONFIG["support_resistance"]["cluster_atr_multiplier"]))
    clusters: list[list[dict[str, Any]]] = []
    for level in sorted(levels, key=lambda item: item["price"]):
        if not clusters or abs(level["price"] - np.mean([item["price"] for item in clusters[-1]])) > threshold:
            clusters.append([level])
        else:
            clusters[-1].append(level)
    zones = []
    for cluster in clusters:
        center = float(np.average([item["price"] for item in cluster], weights=[item["weight"] for item in cluster]))
        score = float(sum(item["weight"] for item in cluster))
        label = "VERY_HIGH" if score >= 8 else "HIGH" if score >= 5 else "MEDIUM" if score >= 2.5 else "LOW"
        zones.append({
            "low": float(min(item["price"] for item in cluster) - threshold * 0.20),
            "high": float(max(item["price"] for item in cluster) + threshold * 0.20),
            "center": center,
            "role": "SUPPORT" if center <= current else "RESISTANCE",
            "sources": sorted({item["source"] for item in cluster}),
            "timeframes": sorted({item["timeframe"] for item in cluster}),
            "confluence_score": round(score, 2),
            "confluence": label,
            "active": True,
            "broken": False,
        })
    return sorted(zones, key=lambda zone: (abs(zone["center"] - current), -zone["confluence_score"]))[:10]


def historical_analogs(frame: pd.DataFrame) -> dict[str, Any]:
    feature_columns = [
        "price_vs_sma50", "price_vs_sma100", "price_vs_sma200", "sma50_slope", "sma100_slope",
        "sma200_slope", "rsi14", "macd_hist", "roc12", "extension200", "drawdown_ath", "volatility12", "volume_ratio",
    ]
    available = [column for column in feature_columns if column in frame and frame[column].notna().sum() >= 30]
    if len(frame) < 260 or len(available) < 6:
        return {"sample_size": 0, "warning": "Insufficient history for reliable causal analogs", "periods": [], "returns": {}, "drawdown_probabilities": {}}
    latest_index = len(frame) - 1
    outcome_horizons = {"3M": 13, "6M": 26, "12M": 52}
    latest_valid = frame.iloc[latest_index][available]
    history = frame.iloc[: latest_index - max(outcome_horizons.values()) + 1].copy()
    history = history.dropna(subset=available)
    if history.empty or latest_valid.isna().any():
        return {"sample_size": 0, "warning": "Insufficient complete feature history", "periods": [], "returns": {}, "drawdown_probabilities": {}}
    scale = history[available].std().replace(0.0, 1.0)
    distances = ((history[available] - latest_valid) / scale).pow(2).mean(axis=1).pow(0.5)
    candidates = sorted(((int(index), float(distance)) for index, distance in distances.items()), key=lambda item: item[1])
    selected: list[tuple[int, float]] = []
    spacing = int(CONFIG["analogs"]["minimum_spacing_weeks"])
    for index, distance in candidates:
        if all(abs(index - previous_index) >= spacing for previous_index, _ in selected):
            selected.append((index, distance))
        if len(selected) >= int(CONFIG["analogs"]["neighbors"]):
            break
    close = pd.to_numeric(frame["close"], errors="coerce")
    returns: dict[str, dict[str, float | None]] = {}
    for label, horizon in outcome_horizons.items():
        outcomes = [(float(close.iloc[index + horizon]) / float(close.iloc[index]) - 1.0) * 100.0 for index, _ in selected if index + horizon <= latest_index]
        returns[label] = {"mean": finite(np.mean(outcomes)) if outcomes else None, "median": finite(np.median(outcomes)) if outcomes else None}
    drawdowns = []
    for index, _ in selected:
        future = close.iloc[index + 1 : index + 53]
        if len(future):
            drawdowns.append(float((future / float(close.iloc[index]) - 1.0).min() * 100.0))
    probabilities = {f">{threshold}%": round(sum(value <= -threshold for value in drawdowns) / len(drawdowns) * 100.0, 1) if drawdowns else None for threshold in (10, 15, 25)}
    minimum = int(CONFIG["analogs"]["minimum_sample"])
    return {
        "sample_size": len(selected),
        "warning": "Weak sample; statistics are indicative only" if len(selected) < minimum else None,
        "sampling": f"Nearest causal weekly states, minimum spacing {spacing} weeks; outcomes end before current as-of date",
        "periods": [{"date": pd.Timestamp(frame.iloc[index]["timestamp"]).date().isoformat(), "distance": round(distance, 4)} for index, distance in selected],
        "returns": returns,
        "drawdown_probabilities": probabilities,
    }


def scenario_probabilities(components: dict[str, float | None], confidence: str) -> dict[str, int]:
    weights = CONFIG["scenario_weights"]
    weighted = 0.0
    used = 0.0
    for key, weight in weights.items():
        value = components.get(key)
        if value is None:
            continue
        weighted += float(value) * float(weight)
        used += float(weight)
    direction = weighted / used if used else 0.0
    uncertainty = {"HIGH": 0.6, "MEDIUM": 1.0, "LOW": 1.5}.get(confidence, 1.5)
    logits = np.array([direction / 32.0, uncertainty - abs(direction) / 75.0, -direction / 32.0], dtype=float)
    logits -= logits.max()
    raw = np.exp(logits)
    percentages = raw / raw.sum() * 100.0
    floors = np.floor(percentages).astype(int)
    for index in np.argsort(percentages - floors)[::-1][: 100 - int(floors.sum())]:
        floors[index] += 1
    return {"BULLISH": int(floors[0]), "NEUTRAL": int(floors[1]), "BEARISH": int(floors[2])}


def build_scenarios(
    current: float,
    zones: list[dict[str, Any]],
    probabilities: dict[str, int],
    momentum: dict[str, Any],
) -> list[dict[str, Any]]:
    supports = sorted((zone for zone in zones if zone["role"] == "SUPPORT"), key=lambda zone: abs(zone["center"] - current))
    resistances = sorted((zone for zone in zones if zone["role"] == "RESISTANCE"), key=lambda zone: abs(zone["center"] - current))
    support1 = supports[0] if supports else _synthetic_zone(current * 0.92, "SUPPORT")
    support2 = supports[1] if len(supports) > 1 else _synthetic_zone(current * 0.82, "SUPPORT")
    resistance1 = resistances[0] if resistances else _synthetic_zone(current * 1.08, "RESISTANCE")
    resistance2 = resistances[1] if len(resistances) > 1 else _synthetic_zone(current * 1.18, "RESISTANCE")
    return [
        {
            "scenario": "BULLISH", "probability": probabilities["BULLISH"],
            "trigger": f"Weekly close > {resistance1['high']:.2f}",
            "confirmation": f"Momentum {momentum['classification']} with ROC acceleration or improving breadth",
            "expected_path": [current, resistance1["center"], resistance2["center"]],
            "target_zone": [resistance2["low"], resistance2["high"]],
            "invalidation": f"Weekly close < {support1['low']:.2f}",
        },
        {
            "scenario": "NEUTRAL", "probability": probabilities["NEUTRAL"],
            "trigger": f"Price remains between {support1['low']:.2f} and {resistance1['high']:.2f}",
            "confirmation": "Momentum remains mixed and weekly structure does not resolve",
            "expected_path": [current, support1["center"], resistance1["center"]],
            "target_zone": [support1["low"], resistance1["high"]],
            "invalidation": f"Weekly close outside {support1['low']:.2f}–{resistance1['high']:.2f}",
        },
        {
            "scenario": "BEARISH", "probability": probabilities["BEARISH"],
            "trigger": f"Weekly close < {support1['low']:.2f}",
            "confirmation": "Negative momentum expands and support fails on volume",
            "expected_path": [current, support1["center"], support2["center"]],
            "target_zone": [support2["low"], support2["high"]],
            "invalidation": f"Weekly close > {resistance1['high']:.2f}",
        },
    ]


def confirmation_matrix(scenarios: list[dict[str, Any]], zones: list[dict[str, Any]], current: float) -> list[dict[str, str]]:
    bullish = next(item for item in scenarios if item["scenario"] == "BULLISH")
    bearish = next(item for item in scenarios if item["scenario"] == "BEARISH")
    supports = sorted((zone for zone in zones if zone["role"] == "SUPPORT"), key=lambda zone: abs(zone["center"] - current))
    matrix = [
        {"event": bullish["trigger"], "interpretation": "Bull scenario confirmed"},
        {"event": bearish["trigger"], "interpretation": "Bear scenario activated"},
    ]
    if len(supports) > 1:
        matrix.append({"event": f"Weekly close < {supports[1]['low']:.2f}", "interpretation": "Structural trend deterioration"})
    matrix.append({"event": bullish["invalidation"], "interpretation": "Bull scenario invalidated"})
    return matrix


def deterministic_narrative(snapshot: dict[str, Any]) -> str:
    final = snapshot["final_state"]
    weekly = snapshot["weekly_structure"]
    daily = snapshot["daily_structure"]
    return (
        f"{snapshot['ticker']} is {final['structural_trend'].lower()} on the weekly structural frame and "
        f"{final['medium_term_trend'].lower()} on the daily frame. Weekly structure is {weekly['state']} "
        f"({weekly['sequence']}); daily structure is {daily['state']} ({daily['sequence']}). "
        f"Momentum is {final['momentum'].lower()} and {final['momentum_trajectory'].lower()}. "
        f"Daily Extension200 is {final['extension'].lower()}. Elliott Structure is assigned separately by the LLM. "
        f"The deterministic six-month bias is {final['six_month_bias'].lower()}."
    )


def final_confidence(weekly: dict[str, Any], daily: dict[str, Any], momentum: dict[str, Any], volume: dict[str, Any], analogs: dict[str, Any]) -> str:
    agreements = 0
    comparable = 0
    for state in (weekly["state"], daily["state"]):
        if state in {"BULL", "BEAR"}:
            comparable += 1
    if weekly["state"] == daily["state"] and weekly["state"] in {"BULL", "BEAR"}:
        agreements += 2
    if (daily["state"] == "BULL" and momentum["score"] > 0) or (daily["state"] == "BEAR" and momentum["score"] < 0):
        agreements += 1
    if volume.get("status") == "AVAILABLE":
        agreements += 1
    if analogs.get("sample_size", 0) >= int(CONFIG["analogs"]["minimum_sample"]):
        agreements += 1
    return "HIGH" if agreements >= 4 and comparable == 2 else "MEDIUM" if agreements >= 2 else "LOW"


def data_version(frame: pd.DataFrame) -> str:
    columns = [column for column in ["timestamp", "open", "high", "low", "close", "volume"] if column in frame]
    raw = frame[columns].tail(1000).to_json(date_format="iso", orient="split")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def snapshot_id(ticker: str, as_of: str, created_at: str, version: str) -> str:
    raw = json.dumps({"ticker": ticker, "as_of": as_of, "created_at": created_at, "data_version": version, "model": MODEL_VERSION}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except Exception:
        return None


def _structure_explanation(state: str, sequence: list[str], volatility: str) -> str:
    detail = "-".join(sequence) if sequence else "insufficient confirmed swings"
    return f"{state}: {detail}; volatility regime {volatility}."


def _roc_state(roc: float | None, delta: float | None) -> str:
    if roc is None or delta is None:
        return "NOT_AVAILABLE"
    if roc >= 0 and delta >= 0:
        return "BULLISH_ACCELERATION"
    if roc >= 0:
        return "BULLISH_DECELERATION"
    if delta < 0:
        return "BEARISH_ACCELERATION"
    return "BEARISH_EXHAUSTION_RECOVERY"


def _synthetic_zone(center: float, role: str) -> dict[str, Any]:
    return {"low": center * 0.99, "high": center * 1.01, "center": center, "role": role, "sources": ["volatility_projection"], "confluence": "LOW", "confluence_score": 0.0}
