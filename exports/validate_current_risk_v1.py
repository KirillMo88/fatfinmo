from __future__ import annotations

import pandas as pd

from market_cycle import build_market_cycle_snapshot
from market_cycle_tab import (
    build_current_risk_components_fig,
    build_current_risk_indicators_fig,
    build_current_risk_signals_fig,
)


def main() -> None:
    snapshot = build_market_cycle_snapshot()
    daily = snapshot.daily.copy()
    daily["Date"] = pd.to_datetime(daily.index, errors="coerce")
    history = daily.loc[daily["Date"].ge(pd.Timestamp("2015-01-01"))].copy()
    current = snapshot.current

    component_keys = [
        "CurrentRiskDrawdownRisk",
        "CurrentRiskPriceCycleVulnerabilityRisk",
        "CurrentRiskBreadthRisk",
        "CurrentRiskRSIDivergenceRisk",
        "CurrentRiskVIXRisk",
        "CurrentRiskHighBetaRisk",
        "CurrentRiskHYRisk",
    ]
    inactive_score_violations = int(
        (
            ~history["CurrentRiskActivation"].fillna(False).astype(bool)
            & pd.to_numeric(history["CurrentRiskEscalationScoreV2"], errors="coerce").fillna(0).ne(0)
        ).sum()
    )
    active_score_violations = int(
        (
            history["CurrentRiskActivation"].fillna(False).astype(bool)
            & ~pd.to_numeric(history["CurrentRiskEscalationScoreV2"], errors="coerce").between(0, 6)
        ).sum()
    )
    signal_without_activation = int(
        (
            history["CurrentRiskSignalClass"].ne("INACTIVE")
            & ~history["CurrentRiskActivation"].fillna(False).astype(bool)
        ).sum()
    )

    component_fig = build_current_risk_components_fig(current)
    signal_fig = build_current_risk_signals_fig(history, pd.Timestamp("2015-01-01"), history["Date"].max())
    indicator_fig = build_current_risk_indicators_fig(history, pd.Timestamp("2015-01-01"), history["Date"].max())
    lower_names = [str(trace.name) for trace in indicator_fig.data]

    print("MODEL", current.get("CurrentRiskModelVersion"))
    print("ROWS", len(history), history["Date"].min(), history["Date"].max())
    print("STATE", current.get("CurrentMarketRiskState"))
    print("ACTIVE_DAYS", int(history["CurrentRiskActivation"].fillna(False).sum()))
    print("NEW_EVENTS", int(history["CurrentRiskNewEvent"].fillna(False).sum()))
    print(
        "EVENT_CLASSES",
        history.loc[history["CurrentRiskNewEvent"].fillna(False), "CurrentRiskSignalClass"].value_counts().to_dict(),
    )
    validation = history.loc[history["Date"].eq(pd.Timestamp("2017-10-24"))]
    if not validation.empty:
        row = validation.iloc[-1]
        print(
            "VALIDATION_2017_10_24",
            {
                "spy_raw_close": row.get("CurrentRiskSPYRawClose"),
                "sma200w_raw": row.get("CurrentRiskSPYSMA200WRaw"),
                "extension_pct": 100.0 * row.get("CurrentRiskSPYExtension200W"),
                "extension_condition": bool(row.get("PVC_ExtensionCondition")),
            },
        )
    print("COMPONENTS", {key: current.get(key) for key in component_keys})
    print("HY_SOURCE", history.get("HYOASSourceFrequency", pd.Series(dtype="object")).value_counts().to_dict())
    print(
        "VIOLATIONS",
        {
            "inactive_nonzero_escalation": inactive_score_violations,
            "active_score_out_of_range": active_score_violations,
            "signal_without_activation": signal_without_activation,
            "lower_panel_spy_trace": int(any(name.upper() == "SPY" for name in lower_names)),
            "lower_panel_btc_trace": int(any("BTC" in name.upper() for name in lower_names)),
        },
    )
    print("UI", len(component_fig.data), len(signal_fig.data), len(indicator_fig.data), lower_names)


if __name__ == "__main__":
    main()
