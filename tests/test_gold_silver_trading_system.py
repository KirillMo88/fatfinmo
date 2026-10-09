from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from gold_regime.trading_system import (
    build_daily_observations,
    expanding_regression,
    instrument_for_regime,
    monthly_ratio_series,
    prepare_usd_ohlcv,
    ratio_close_series,
    run_backtest,
    run_state_machine,
    silver_platinum_deviation,
)


def ohlc(index: pd.DatetimeIndex, close: list[float] | np.ndarray) -> pd.DataFrame:
    values = np.asarray(close, dtype=float)
    return pd.DataFrame({"open": values, "high": values, "low": values, "close": values}, index=index)


class TradingSystemRegressionTests(unittest.TestCase):
    def test_ratio_uses_only_exact_shared_dates(self) -> None:
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        gold = ohlc(dates, [2, 4, 6, 8])
        silver = ohlc(dates.delete(1), [1, 2, 4])
        ratio = ratio_close_series(gold, silver)
        self.assertEqual(list(ratio.index), list(dates.delete(1)))
        self.assertEqual(ratio.tolist(), [2.0, 3.0, 2.0])

    def test_monthly_ratio_excludes_current_incomplete_month(self) -> None:
        dates = pd.date_range("2020-01-31", periods=4, freq="ME")
        gold = pd.Series([2, 4, 6, 8], index=dates)
        silver = pd.Series([1, 2, 3, 4], index=dates)
        result = monthly_ratio_series(gold, silver, as_of="2020-04-15", start="2020-01")
        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[-1], 2.0)

    def test_expanding_regression_uses_120_observations_and_raw_ratio(self) -> None:
        dates = pd.date_range("1998-01-31", periods=125, freq="ME")
        x = np.arange(125, dtype=float)
        ratio = pd.Series(40 + 0.08 * x + np.sin(x / 5) * 2, index=dates)
        result = expanding_regression(ratio)
        self.assertEqual(len(result), 6)
        self.assertEqual(result.iloc[0]["observations"], 120)
        fit_x = x[:120]
        fit_y = ratio.to_numpy()[:120]
        design = np.column_stack([np.ones(120), fit_x])
        coefficients = np.linalg.lstsq(design, fit_y, rcond=None)[0]
        residual = fit_y - design @ coefficients
        self.assertAlmostEqual(result.iloc[0]["sigma"], np.sqrt(np.sum(residual**2) / 118), places=10)
        self.assertAlmostEqual(result.iloc[0]["mean"], coefficients[0] + coefficients[1] * 119, places=10)

    def test_historical_fit_does_not_use_future_months(self) -> None:
        dates = pd.date_range("1998-01-31", periods=130, freq="ME")
        values = pd.Series(np.linspace(40, 60, 130) + np.sin(np.arange(130)), index=dates)
        original = expanding_regression(values)
        changed = values.copy()
        changed.iloc[120:] *= 10
        refit = expanding_regression(changed)
        pd.testing.assert_series_equal(original.iloc[0], refit.iloc[0])

    def test_fitted_line_is_evaluated_at_each_daily_month_coordinate(self) -> None:
        dates = pd.date_range("2021-01-01", periods=10, freq="B")
        ratio = ohlc(dates, np.linspace(50, 51, 10))
        monthly_model = pd.DataFrame(
            [{"month_index": 1.0, "ratio": 50.0, "intercept": 40.0, "slope": 2.0, "sigma": 1.0, "mean": 42.0,
              "upper_1": 43.0, "upper_2": 44.0, "lower_1": 41.0, "lower_2": 40.0, "observations": 120.0}],
            index=pd.DatetimeIndex(["2020-12-31"], name="date"),
        )
        sp = pd.Series([0.0], index=pd.DatetimeIndex(["2020-12-31"]))
        observations = build_daily_observations(ratio, ratio, ratio, monthly_model, sp)
        expected = 40.0 + 2.0 * (pd.Period("2021-01", freq="M").ordinal - pd.Period("1998-01", freq="M").ordinal)
        self.assertEqual(observations.iloc[0]["mean"], expected)
        self.assertEqual(observations.iloc[-1]["mean"], expected)

    def test_sp_deviation_uses_completed_month_sma(self) -> None:
        dates = pd.date_range("2015-01-31", periods=60, freq="ME")
        silver = pd.Series(np.linspace(10, 20, 60), index=dates)
        platinum = pd.Series(10.0, index=dates)
        deviation = silver_platinum_deviation(silver, platinum, as_of="2020-01-15")
        self.assertTrue(deviation.notna().any())
        self.assertLess(deviation.index.max(), pd.Timestamp("2020-01-31"))

    def test_sp_sma_requires_50_consecutive_completed_months(self) -> None:
        dates = pd.date_range("2015-01-31", periods=60, freq="ME").delete(35)
        silver = pd.Series(np.linspace(10, 20, len(dates)), index=dates)
        platinum = pd.Series(10.0, index=dates)
        deviation = silver_platinum_deviation(silver, platinum, as_of="2020-02-15")
        self.assertFalse(deviation.notna().any())


class TradingSystemStateTests(unittest.TestCase):
    def _observations(self, dates: pd.DatetimeIndex, ratios: list[float], sp: list[float] | None = None) -> pd.DataFrame:
        count = len(dates)
        return pd.DataFrame(
            {
                "gold_silver_ratio": ratios,
                "mean": [100.0] * count,
                "sigma": [10.0] * count,
                "upper_1": [110.0] * count,
                "upper_2": [120.0] * count,
                "lower_1": [90.0] * count,
                "lower_2": [80.0] * count,
                "zscore": [(value - 100) / 10 for value in ratios],
                "sp_deviation_pct": sp or [0.0] * count,
                "model_ready": [True] * count,
            },
            index=dates,
        )

    def test_cash_gold_late_metal_and_direct_cash_exit(self) -> None:
        dates = pd.to_datetime(["2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29", "2024-02-05"])
        observations = self._observations(dates, [100, 111, 121, 100, 79], [0, 0, 35, 35, 35])
        _, events, _ = run_state_machine(observations, as_of="2024-02-15")
        self.assertEqual(events["to_regime"].tolist(), ["GOLD", "PLATINUM", "CASH"])
        self.assertEqual(events.iloc[-1]["from_regime"], "PLATINUM")
        self.assertEqual(events.iloc[-1]["threshold"], "−2σ")

    def test_minus_one_returns_to_gold_but_not_cash(self) -> None:
        dates = pd.to_datetime(["2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29"])
        observations = self._observations(dates, [100, 111, 121, 89], [0, 0, 35, 35])
        history, events, _ = run_state_machine(observations, as_of="2024-02-10")
        self.assertEqual(events["to_regime"].tolist(), ["GOLD", "PLATINUM", "GOLD"])
        self.assertEqual(history.iloc[-1]["regime"], "GOLD")

    def test_sp_selector_hysteresis_and_reentry(self) -> None:
        dates = pd.to_datetime(["2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29", "2024-02-05", "2024-02-12"])
        observations = self._observations(dates, [100, 111, 121, 121, 121, 121], [0, 0, 35, 10, 4, 35])
        history, events, _ = run_state_machine(observations, as_of="2024-02-20")
        self.assertEqual(events["to_regime"].tolist(), ["GOLD", "PLATINUM", "SILVER"])
        self.assertEqual(history.iloc[-1]["regime"], "SILVER")
        reentry_dates = pd.to_datetime(["2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29", "2024-02-05"])
        reentry = self._observations(reentry_dates, [100, 111, 121, 79, 121], [0, 0, 35, 35, 35])
        _, reentry_events, _ = run_state_machine(reentry, as_of="2024-02-12")
        self.assertEqual(reentry_events["to_regime"].tolist(), ["GOLD", "PLATINUM", "CASH", "GOLD", "PLATINUM"])

    def test_incomplete_current_week_is_not_processed(self) -> None:
        dates = pd.to_datetime(["2024-01-08", "2024-01-15", "2024-01-22", "2024-02-07"])
        observations = self._observations(dates, [100, 111, 121, 79], [0, 0, 35, 35])
        history, events, _ = run_state_machine(observations, as_of="2024-02-08")
        self.assertNotIn(pd.Timestamp("2024-02-07"), history.index)
        self.assertEqual(events["to_regime"].tolist(), ["GOLD", "PLATINUM"])

    def test_weekly_summary_uses_only_observed_daily_closes(self) -> None:
        dates = pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-15"])
        observations = self._observations(dates, [100, 112, 106, 111])
        _, _, weeks = run_state_machine(observations, as_of="2024-01-25")
        self.assertEqual(weeks.iloc[0]["ratio_high_observed"], 112)
        self.assertEqual(weeks.iloc[0]["ratio_low_observed"], 100)

    def test_3x_mapping_ignores_platinum(self) -> None:
        self.assertEqual(instrument_for_regime("PLATINUM", "1x"), "PLATINUM")
        self.assertEqual(instrument_for_regime("PLATINUM", "3x"), "3SIL.L")
        self.assertEqual(instrument_for_regime("GOLD", "3x"), "3GOL.L")
        self.assertEqual(instrument_for_regime("CASH", "3x"), "CASH")

    def test_currency_conversion_preserves_usd_and_converts_pence_without_fill(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        source = ohlc(dates, [100, 100, 100])
        fx = pd.Series([1.25, np.nan, 1.30], index=dates)
        converted = prepare_usd_ohlcv(source, "GBp", fx, fx)
        self.assertEqual(converted["close"].tolist(), [1.25, 1.3])
        usd = prepare_usd_ohlcv(source, "USD")
        pd.testing.assert_frame_equal(source, usd, check_names=False)


class TradingSystemPerformanceTests(unittest.TestCase):
    def test_next_week_open_execution_and_actual_instrument_returns(self) -> None:
        dates = pd.bdate_range("2020-01-06", periods=20)
        price = np.arange(100, 120, dtype=float)
        prices = {name: pd.DataFrame({"open": price, "high": price + 1, "low": price - 1, "close": price + 1}, index=dates) for name in ("GOLD", "SILVER", "PLATINUM")}
        regimes = pd.DataFrame(
            {"model_ready": True, "regime": ["NOT_READY"] * 4 + ["GOLD"] * 16},
            index=dates,
        )
        events = pd.DataFrame(
            [{"signal_date": dates[6], "confirmed_date": dates[9], "from_regime": "GOLD", "to_regime": "CASH"}]
        )
        result = run_backtest(prices, regimes, events, "1x", as_of="2020-02-15")
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["initial_capital"], 1000.0)
        self.assertEqual(result["start_date"], dates[5])
        self.assertEqual(result["curve"].iloc[0]["equity"], 1000.0)
        self.assertEqual(result["executed_events"].iloc[0]["execution_date"], dates[10])
        self.assertEqual(result["number_of_trades"], 1)
        closes = result["curve"].loc[result["curve"]["phase"].eq("Close")].set_index("date")
        self.assertAlmostEqual(closes.loc[dates[5], "equity"], 1000.0 * 106.0 / 105.0)
        self.assertAlmostEqual(closes.loc[dates[10], "equity"], closes.loc[dates[9], "equity"])
        self.assertAlmostEqual(result["curve"].iloc[0]["equity"], result["initial_capital"])
        self.assertLessEqual(result["max_drawdown_pct"], 0)

    def test_missing_dates_are_not_filled_and_annual_return_reconciles(self) -> None:
        dates = pd.bdate_range("2019-12-23", periods=30)
        close = np.linspace(100, 130, len(dates))
        prices = {name: ohlc(dates, close) for name in ("GOLD", "SILVER", "PLATINUM")}
        prices["SILVER"] = prices["SILVER"].drop(index=dates[7])
        aligned = prices["GOLD"].index.intersection(prices["SILVER"].index).intersection(prices["PLATINUM"].index)
        history = pd.DataFrame({"model_ready": True, "regime": "CASH"}, index=dates)
        result = run_backtest(prices, history, pd.DataFrame(), "1x", as_of="2020-02-15")
        self.assertEqual(len(result["curve"].loc[result["curve"]["phase"].eq("Close")]), len(aligned[aligned >= dates[5]]))
        if not result["annual"].empty:
            compounded = np.prod(1 + result["annual"]["Strategy Return"].to_numpy() / 100.0) - 1
            self.assertAlmostEqual(compounded * 100.0, result["total_return_pct"], places=6)


if __name__ == "__main__":
    unittest.main()
