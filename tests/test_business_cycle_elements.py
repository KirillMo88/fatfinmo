import numpy as np
import pandas as pd
import pytest

from business_cycle_tab import build_business_cycle_elements, describe_business_cycle_element


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
