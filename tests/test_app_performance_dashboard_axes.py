from contextlib import nullcontext

import pandas as pd
import pytest

import app


@pytest.mark.parametrize(
    ("renderer", "chart_key"),
    [
        (app._render_performance_rotation_map, "performance_rotation_map"),
        (app._render_performance_directional_trend_map, "performance_directional_trend_map"),
        (app._render_momentum_rsi_map, "momentum_rsi_map"),
        (app._render_momentum_extension_risk_map, "momentum_extension_risk_map"),
    ],
)
def test_performance_maps_start_with_autoscaled_axes(monkeypatch, renderer, chart_key):
    data = pd.DataFrame(
        {
            "Entity_ID": ["etf:SPY", "etf:QQQ"],
            "Entity_Label": ["SPY", "QQQ"],
            "Instrument_Name": ["S&P 500", "Nasdaq 100"],
            "Group_2": ["Broad Market", "Technology"],
            "Perf_1W_%": [1.0, -2.0],
            "Perf_1M_%": [3.0, -4.0],
            "Perf_3M_%": [8.0, -6.0],
            "Perf_6M_%": [12.0, -7.0],
            "Signed_ADX_DI_14D": [20.0, -15.0],
            "Signed_ADX_DI_14W": [25.0, -10.0],
            "Momentum_Score": [65.0, 35.0],
            "Short_Term_Momentum_Score": [60.0, 30.0],
            "RSI_14": [62.0, 38.0],
            "RSI_14W": [58.0, 42.0],
            "Perf_12M_Percentile": [75.0, 25.0],
            "Risk_Breadth_%": [40.0, 10.0],
            "Extension_Risk_Flag": ["Elevated", "Not Triggered"],
            "Constituent_Count": [1, 1],
            "Valid_Constituent_Count": [1, 1],
            "Risk_Valid_Constituent_Count": [1, 1],
        }
    )
    figures = []
    monkeypatch.setattr(app.st, "markdown", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app.st, "columns", lambda *_args, **_kwargs: (nullcontext(), nullcontext()))
    monkeypatch.setattr(app.st, "radio", lambda _label, options, **kwargs: options[kwargs["index"]])
    monkeypatch.setattr(app.st, "selectbox", lambda _label, options, **kwargs: options[kwargs["index"]])
    monkeypatch.setattr(app, "_performance_dashboard_render_plotly", lambda fig, key: figures.append((fig, key)))

    renderer(data, "ETF")

    assert len(figures) == 1
    figure, rendered_key = figures[0]
    assert rendered_key == chart_key
    assert figure.layout.xaxis.autorange is True
    assert figure.layout.yaxis.autorange is True
    assert figure.layout.xaxis.range is None
    assert figure.layout.yaxis.range is None
