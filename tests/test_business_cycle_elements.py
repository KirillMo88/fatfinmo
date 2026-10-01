import numpy as np
import pandas as pd
import pytest

from business_cycle_tab import (
    build_business_cycle_elements,
    build_demand_income_score_fig,
    build_labor_score_fig,
    build_production_score_fig,
    build_survey_score_fig,
    describe_business_cycle_element,
)


def test_element_interpretation_detects_improvement_that_stalled() -> None:
    text = describe_business_cycle_element(level=0.02, change_13w=0.01, change_52w=0.35)

    assert "улучшение за 52 недели" in text
    assert "рост остановился" in text
    assert "около исторической нормы" in text


def test_element_interpretation_detects_recent_upward_turn() -> None:
    text = describe_business_cycle_element(level=0.40, change_13w=0.12, change_52w=-0.30)

    assert "ухудшение за 52 недели" in text
    assert "разворот вверх" in text
    assert "умеренно выше исторической нормы" in text


def test_business_cycle_elements_use_exact_13w_and_52w_changes() -> None:
    history = pd.DataFrame(
        {
            "SurveyScore": np.arange(60, dtype=float) / 10.0,
            "ProductionScore": np.arange(60, dtype=float) / 20.0,
            "DemandIncomeScore": np.arange(60, dtype=float) / -10.0,
            "LaborScore": np.arange(60, dtype=float) / -20.0,
        }
    )

    rows = {row["Pillar"]: row for row in build_business_cycle_elements(history)}

    assert rows["Survey"]["Change13W"] == pytest.approx(1.3)
    assert rows["Survey"]["Change52W"] == pytest.approx(5.2)
    assert rows["Labor"]["Change13W"] == pytest.approx(-0.65)
    assert rows["Labor"]["Change52W"] == pytest.approx(-2.6)


def test_regime_dynamics_pillar_charts_include_score_and_components() -> None:
    d = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-02", periods=4, freq="W-FRI"),
            "SurveyScore": [0.1, 0.2, 0.3, 0.4],
            "ISM_Z": [0.0, 0.1, 0.2, 0.3],
            "CFNAI_Z": [0.2, 0.3, 0.4, 0.5],
            "ProductionScore": [0.1, 0.0, -0.1, 0.0],
            "IndustrialProduction_Z": [0.1, 0.0, -0.1, 0.0],
            "DemandIncomeScore": [0.0, 0.1, 0.1, 0.2],
            "RetailSales_Z": [0.1, 0.2, 0.2, 0.3],
            "RealPCE_Z": [0.0, 0.1, 0.1, 0.2],
            "RealPersonalIncome_Z": [-0.1, 0.0, 0.0, 0.1],
            "LaborScore": [0.4, 0.3, 0.3, 0.4],
            "InitialClaims_Z_INV": [0.5, 0.4, 0.4, 0.5],
            "ContinuingClaims_Z_INV": [0.4, 0.3, 0.3, 0.4],
            "Unemployment_Z_INV": [0.3, 0.2, 0.2, 0.3],
            "Payrolls_Z": [0.4, 0.3, 0.3, 0.4],
        }
    )

    survey = build_survey_score_fig(d)
    production = build_production_score_fig(d)
    demand = build_demand_income_score_fig(d)
    labor = build_labor_score_fig(d)

    assert survey.layout.title.text == "Survey Score = 50% × ISM+50% × CFNAI"
    assert [trace.name for trace in survey.data] == ["Survey Score", "ISM", "CFNAI"]
    assert production.layout.title.text == "Production Score = Industrial Production"
    assert [trace.name for trace in production.data] == ["Production Score", "Industrial Production"]
    assert demand.layout.title.text == "Demand & Income Score = Retail Sales, Real PCE и Real Income"
    assert [trace.name for trace in demand.data] == ["Demand & Income Score", "Retail Sales", "Real PCE", "Real Income"]
    assert labor.layout.title.text == (
        "Labor Score = 35% × Initial Claims + 15% × Continuing Claims +  + 25% × Unemployment +  + 25% × Payrolls"
    )
    assert [trace.name for trace in labor.data] == ["Labor Score", "Initial Claims", "Continuing Claims", "Unemployment", "Payrolls"]
