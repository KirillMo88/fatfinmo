import unittest

import numpy as np
import pandas as pd

from technical_outlook_simple_v3.chart import build_reference_chart
from technical_outlook_simple_v3.volume_profile import build_volume_profile


def chart_fixture():
    dates = pd.date_range("2020-01-03", periods=300, freq="W-FRI")
    x = np.arange(300)
    close = 300 + x * 1.4 + np.sin(x / 15) * 45
    frame = pd.DataFrame(dict(timestamp=dates, open=close + np.sin(x) * 5,
                              close=close, high=close + 8, low=close - 8,
                              volume=1_000_000 + (1 + np.cos(x / 40)) * 400_000,
                              atr14=np.full(300, 12)))
    for window in (50, 100, 200):
        frame[f"sma{window}"] = frame.close.rolling(window, min_periods=1).mean()
    pivots = [dict(pivot_time=dates[i].isoformat(), price=float(close[i]),
                   status="CONFIRMED", kind="LOW") for i in range(10, 270, 30)]
    fib = dict(ath=dict(date=dates[-10].isoformat(), price=float(close[-10])),
               strategic_anchor=pivots[0], tactical_anchor=pivots[-1],
               strategic=dict(levels=[dict(type="0.382", price=610),
                                     dict(type="0.500_0.618", low=520, high=560)]),
               tactical=dict(levels=[dict(type="0.382", price=690),
                                    dict(type="0.500_0.618", low=650, high=670)]))
    return frame, dict(primary=None, alternative=None, timeframe="WEEKLY", ticker="SPY",
                       pivots=pivots, fibonacci=fib, profile=build_volume_profile(frame, timeframe="WEEKLY"))


class ReferenceChartTests(unittest.TestCase):
    def test_volume_uses_full_horizontal_bars_and_narrow_domain(self):
        bars, options = chart_fixture()
        before = bars.copy(deep=True)
        chart = build_reference_chart(bars, [], **options)
        chart.to_json()  # Validate the complete Plotly schema, not only Python syntax.
        volume = next(trace for trace in chart.data if trace.type == "bar")
        self.assertEqual(volume.orientation, "h")
        self.assertEqual(volume.base, 0)
        self.assertEqual(len(volume.x), 24)
        self.assertEqual(chart.layout.xaxis.domain, (0, 0.875))
        self.assertEqual(chart.layout.xaxis2.domain, (0.89, 1))
        self.assertTrue(chart.layout.xaxis.fixedrange)
        self.assertTrue(chart.layout.yaxis.fixedrange)
        self.assertFalse(chart.layout.xaxis.rangeslider.visible)
        self.assertEqual(chart.layout.height, 480)
        pd.testing.assert_frame_equal(bars, before)

    def test_volume_off_restores_full_width_and_removes_volume_lines(self):
        bars, options = chart_fixture()
        options["enabled_sources"] = {"SWING_STRUCTURE", "FIBONACCI", "MOVING_AVERAGE"}
        chart = build_reference_chart(bars, [], **options)
        self.assertEqual(chart.layout.xaxis.domain, (0, 1))
        self.assertFalse(any(trace.type == "bar" for trace in chart.data))
        self.assertFalse(any((shape.name or "").startswith("POC") for shape in chart.layout.shapes))

    def test_legend_has_filled_bands_and_respects_suppressed_tactical(self):
        bars, options = chart_fixture()
        options["fibonacci"]["tactical_suppressed"] = True
        chart = build_reference_chart(bars, [], **options)
        named = {shape.name: shape for shape in chart.layout.shapes if shape.showlegend}
        band = named["Strategic Fibonacci Zone 0.500–0.618"]
        self.assertEqual(band.type, "rect")
        self.assertEqual(band.fillcolor, "#ad8cff")
        self.assertFalse(any("Tactical" in name for name in named))
        self.assertFalse(any("Tactical" in trace.name for trace in chart.data))
        self.assertEqual(chart.layout.paper_bgcolor, "#ffffff")


if __name__ == "__main__":
    unittest.main()
