from __future__ import annotations

import pandas as pd

from market_cycle import build_market_cycle_snapshot
from market_cycle_tab import build_correction_primary_fig, correction_component_table, correction_validation_tables_v2


def main() -> None:
    snapshot = build_market_cycle_snapshot()
    history = snapshot.correction_daily.copy()
    print("ROWS", len(history), history["Date"].min(), history["Date"].max())
    print("BREADTH", history["BreadthSourceFrequency"].iloc[-1])
    print("HY_COVERAGE", round(float(history["HY_OAS"].notna().mean()), 4))
    print("HY_SOURCE", history["HYOASSourceFrequency"].value_counts(dropna=False).to_dict())
    print("EXTREME_EPISODES", int(history["ExtremeEpisodeMarker"].sum()))
    print("EXTREME_2_EPISODES", int(history["ExtremeCapitulation2EpisodeMarker"].sum()))
    print("DURABLE_SIGNALS", int(history["DurableBottomConfirmed"].sum()))
    print("MASTER_CYCLES", history["CorrectionCycleID"].nunique())
    print("WAVES", history["CorrectionWaveID"].nunique())

    for year in [2008, 2009, 2018, 2019, 2020, 2022, 2025]:
        rows = history.loc[
            history["Date"].dt.year.eq(year)
            & (
                history["ExtremeEpisodeDisplayMarker"]
                | history["ExtremeCapitulation2EpisodeMarker"]
                | history["DurableBottomConfirmed"]
            ),
            [
                "Date",
                "CorrectionCycleID",
                "CorrectionWaveID",
                "ExtremeEpisodeMarker",
                "ExtremeCapitulation2EpisodeMarker",
                "DurableBottomConfirmed",
                "DurableRoute",
                "BearRiskScore",
                "BearRiskCategory",
                "CorrectionCurrentDrawdown",
            ],
        ]
        print(f"YEAR_{year}")
        print(rows.to_string(index=False))

    violations = {
        "tactical_without_correction": int((history["TacticalInternalEvent"] & ~history["CorrectionActive"]).sum()),
        "durable_without_tactical": int((history["DurableBottomConfirmed"] & ~history["TacticalInternalEvent"]).sum()),
        "missing_breadth_as_zero": int(
            (
                history["SPXAboveSMA50D"].isna()
                & history["Breadth50_Stress"].eq(0)
            ).sum()
        ),
        "missing_hy_as_zero": int((history["HY_OAS"].isna() & history["BearRiskScore"].eq(0)).sum()),
    }
    print("VIOLATIONS", violations)
    figure = build_correction_primary_fig(history)
    extreme_validation, durable_validation, events = correction_validation_tables_v2(history)
    components = correction_component_table(snapshot.current)
    print("UI", len(figure.data), len(figure.layout.shapes), len(extreme_validation), len(durable_validation), len(events), len(components))


if __name__ == "__main__":
    main()
