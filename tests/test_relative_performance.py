import unittest

import pandas as pd

from relative_performance import relative_price_series


class RelativePerformanceTests(unittest.TestCase):
    def test_builds_asset_to_benchmark_ratio_on_aligned_dates(self):
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        asset = pd.Series([100.0, 120.0, 150.0], index=dates)
        benchmark = pd.Series([50.0, 60.0, 75.0], index=dates)

        result = relative_price_series(asset, benchmark)

        self.assertEqual(result.tolist(), [2.0, 2.0, 2.0])

    def test_btc_against_itself_is_constant_one(self):
        dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
        btc = pd.Series([90_000.0, 95_000.0], index=dates)

        result = relative_price_series(btc, btc)

        self.assertEqual(result.tolist(), [1.0, 1.0])

    def test_drops_missing_and_zero_benchmark_values(self):
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        asset = pd.Series([100.0, None, 150.0], index=dates)
        benchmark = pd.Series([50.0, 60.0, 0.0], index=dates)

        result = relative_price_series(asset, benchmark)

        self.assertEqual(result.tolist(), [2.0])


if __name__ == "__main__":
    unittest.main()
