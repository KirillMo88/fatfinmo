import numpy as np
import pandas as pd
import pytest

import app


def test_ai_group_fast_performance_averages_component_returns(monkeypatch):
    dates = pd.to_datetime(["2026-09-11", "2026-09-14"])
    frames = {
        "AAA": pd.DataFrame({"Close": [100.0, 110.0]}, index=dates),
        "BBB": pd.DataFrame({"Close": [200.0, 190.0]}, index=dates),
    }

    monkeypatch.setitem(app.AI_UNIVERSE, "Memory", ["AAA", "BBB"])
    monkeypatch.setattr(app, "download_performance_ohlcv", lambda ticker, **_: frames[ticker])

    result = app.get_ai_group_performance_metrics("Memory")

    assert result is not None
    assert result[0] == pytest.approx(2.5)
    assert np.isnan(result[1])
