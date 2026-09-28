from __future__ import annotations

from business_cycle import (
    economy_regime_assets,
    economy_regime_composition,
    economy_regime_label,
)


def test_economy_regime_assets_use_the_active_regime_profile():
    assert economy_regime_assets("GOLDILOCKS") == "Growth, SMID, EM, HY, Cyclicals"
    assert economy_regime_assets("DISINFLATIONARY SLOWDOWN") == "Long Treasuries, Growth, Large Cap, IG, Gold"
    assert economy_regime_label("DISINFLATIONARY SLOWDOWN") == "DEFLATION"


def test_asset_composition_contains_all_regime_columns_and_factors():
    table = economy_regime_composition()
    assert list(table.columns) == ["Factor", "Goldilocks", "Reflation", "Stagflation / Inflation", "Deflation"]
    assert list(table["Factor"]) == [
        "General", "Beta", "Cyclicality", "Style", "Market Cap", "Regional",
        "Geography", "Fixed Income", "Treasury Curve", "Credit", "Commodities", "Currencies",
    ]
    assert table.loc[table["Factor"].eq("Credit"), "Deflation"].iloc[0] == "IG"


if __name__ == "__main__":
    test_economy_regime_assets_use_the_active_regime_profile()
    test_asset_composition_contains_all_regime_columns_and_factors()
    print("PASS business cycle composition tests")
