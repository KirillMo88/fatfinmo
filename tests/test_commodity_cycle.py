import numpy as np
import pandas as pd
from datetime import date

from commodity_cycle.model import (
    calculate_capex_current_percentile,
    calculate_capex_expanding_percentile,
    classify_capex_direction,
    classify_capex_state,
    classify_commodity_price_momentum,
    classify_cftc_relative_state,
    classify_sector_price_state,
    classify_seasonal_curve,
    calculate_stress,
    is_confirmed_easing_persistent,
    resolve_cftc_qualifier,
    resolve_core_state_transition,
    resolve_price_curve_market_state,
    update_systemic_wave,
)
from commodity_cycle.term_structure import TermStructureStore, candidate_contracts, contract_symbol, make_rows, quote_is_fresh, seasonal_percentile
from commodity_cycle.term_structure_pipeline import (
    TermStructureStore as PersistentCurveStore,
    TermStructureContractProvider,
    _active_front,
    fetch_current_term_structure,
    _completed_eod_frame,
    _current_month_daily,
    expected_contracts,
    percentile_with_history,
    seasonal_history_frame,
)


def test_price_momentum_priority_and_edges():
    assert classify_commodity_price_momentum(0.01, 0.02, 0.03) == "Strong Bullish"
    assert classify_commodity_price_momentum(-0.01, 0.02, 0.03) == "Bullish"
    assert classify_commodity_price_momentum(-0.01, -0.02, -0.03) == "Strong Bearish"
    assert classify_commodity_price_momentum(0.01, -0.02, -0.03) == "Bearish"
    assert classify_commodity_price_momentum(0.0, 0.02, 0.03) == "Bullish"
    assert classify_commodity_price_momentum(0.01, 0.0, 0.03) == "Neutral"


def test_inventory_sales_stress_inverts_ratio_percentile_without_lookahead():
    idx = pd.date_range("2000-01-01", periods=121, freq="MS")
    rising = pd.DataFrame({"series": np.arange(1.0, 122.0)}, index=idx)
    stress = calculate_stress(rising)
    assert np.isnan(stress["series Rolling Stress"].iloc[119])
    assert stress["series Rolling Stress"].iloc[-1] == 0
    falling = rising.copy()
    falling.iloc[-1, 0] = -1
    falling_stress = calculate_stress(falling)
    assert falling_stress["series Rolling Stress"].iloc[-1] == 100


def test_sector_price_state_uses_both_directions():
    assert classify_sector_price_state(["Bullish", "Strong Bullish", "Neutral"]) == "Bullish"
    assert classify_sector_price_state(["Bullish", "Bearish", "Bearish"]) == "Bearish"
    assert classify_sector_price_state(["Bullish", "Bearish", "Neutral"]) == "Mixed / Neutral"
    assert classify_sector_price_state(["Bullish", "Neutral"]) == "Bullish"
    assert classify_sector_price_state(["Bullish", "Bearish"]) == "Mixed / Neutral"
    assert classify_sector_price_state(["Bearish", "Neutral"]) == "Bearish"


def test_seasonal_curve_half_open_boundaries():
    expected = {9.999: "Extreme Loose", 10: "Strong Loose", 25: "Loose", 40: "Neutral",
                60: "Tight", 75: "Strong Tightness", 90: "Strong Tightness", 90.001: "Extreme Tightness"}
    assert {value: classify_seasonal_curve(value) for value in expected} == expected


def test_price_curve_matrix_and_strong_physical_upgrade():
    assert resolve_price_curve_market_state("Bullish", "Tight") == "Bullish Confirmation"
    assert resolve_price_curve_market_state("Strong Bullish", "Tight") == "Strong Physical Confirmation"
    assert resolve_price_curve_market_state("Bullish", "Strong Tightness") == "Strong Physical Confirmation"
    assert resolve_price_curve_market_state("Mixed / Neutral", "Tight") == "Physical Tightness / Price Lag"
    assert resolve_price_curve_market_state("Strong Bearish", "Extreme Loose") == "Confirmed Weakness"


def test_cftc_relative_state_half_open_boundaries():
    expected = {9.999: "Extreme Low", 10: "Low", 25: "Neutral", 75: "High", 90: "High", 90.001: "Extreme High"}
    assert {value: classify_cftc_relative_state(value) for value in expected} == expected


def test_sector_cftc_qualifier_exact_boundaries_and_absolute_direction():
    def qualifier(pctl, net):
        return resolve_cftc_qualifier([pctl, pctl], [net, net])
    assert qualifier(10, -1) == "Short / Contrarian"
    assert qualifier(25, -1) == "Not Crowded"
    assert qualifier(75, 1) == "Crowded"
    assert qualifier(90, 1) == "Crowded"
    assert qualifier(90.01, 1) == "Extremely Crowded"
    assert qualifier(80, -1) == "High Relative Positioning / Still Net Short"


def test_yahoo_contract_symbols_are_generated_for_rolling_years():
    assert contract_symbol("CL", 11, 2026, "NYM") == "CLX26.NYM"
    candidates = candidate_contracts("WTI", today=date(2027, 1, 10))
    assert candidates[0][0] == "CLG27.NYM"
    assert candidates[-1][0].endswith(".NYM")


def test_agriculture_uses_calendar_comparable_delivery_pairs():
    today = date(2026, 10, 7)
    def quote(symbol):
        return 100.0, pd.Timestamp(today)
    class Response:
        text = "<table><tr><th>date</th><th>LME Cash-Settlement</th><th>LME 3-month</th></tr><tr><td>7 October 2026</td><td>100.00</td><td>101.00</td></tr></table>"
        def raise_for_status(self):
            return None
    rows = make_rows(today=today, quote_fetcher=quote, request_get=lambda *args, **kwargs: Response())
    corn = rows.loc[rows["asset"].eq("Corn")]
    beans = rows.loc[rows["asset"].eq("Soybeans")]
    assert corn.iloc[0]["structure"] == "Dec/Mar"
    assert "ZCZ26.CBT" == corn.iloc[0]["leg1_contract"]
    assert "ZCH27.CBT" == corn.iloc[0]["leg2_contract"]
    assert beans.iloc[0]["structure"] == "Nov/Jan"
    assert beans.iloc[0]["leg1_contract"] == "ZSX26.CBT"
    assert beans.iloc[0]["leg2_contract"] == "ZSF27.CBT"


def test_term_structure_store_records_current_and_month_end_separately(tmp_path):
    store = TermStructureStore(tmp_path / "term.sqlite3")
    row = pd.DataFrame([{
        "asset": "WTI", "structure": "F1/F3", "as_of": pd.Timestamp("2026-09-30"),
        "leg1_contract": "CLV26.NYM", "leg2_contract": "CLZ26.NYM", "leg1": 70.0,
        "leg2": 71.0, "spread": 70 / 71 - 1, "source": "Yahoo Finance individual futures",
        "status": "CURRENT",
    }])
    store.save_current(row, now=pd.Timestamp("2026-10-01 12:00").to_pydatetime())
    assert len(store.latest_snapshots()) == 1
    month_end = store.observations()
    assert len(month_end) == 1
    assert month_end.iloc[0]["source"] == "Yahoo Finance individual futures"
    assert month_end.iloc[0]["leg1_contract"] == "CLV26.NYM"


def test_term_curve_freshness_uses_business_days():
    assert quote_is_fresh("2026-10-02", today=date(2026, 10, 7), max_business_days=3)
    assert not quote_is_fresh("2026-10-01", today=date(2026, 10, 7), max_business_days=3)


def test_current_term_contract_calendar_is_dynamic_and_seasonally_comparable():
    today = date(2026, 10, 7)
    corn = expected_contracts("Corn", today)
    beans = expected_contracts("Soybeans", today)
    assert [c.symbol for c in corn[:2]] == ["ZCZ26.CBT", "ZCH27.CBT"]
    assert [c.symbol for c in beans[:2]] == ["ZSX26.CBT", "ZSF27.CBT"]
    assert corn[-1].delivery_year > 2026
    assert len(expected_contracts("WTI", today)) >= 18


def test_only_completed_common_eod_quotes_can_form_current_spread():
    today = date(2026, 10, 7)
    quotes = pd.DataFrame({"price": [10.0, 99.0], "volume": [10.0, 100.0]},
                          index=pd.to_datetime(["2026-10-06", "2026-10-07"]))
    completed = _completed_eod_frame(quotes, today)
    assert completed.index.tolist() == [pd.Timestamp("2026-10-06")]
    assert completed.iloc[-1]["price"] == 10.0
    assert _current_month_daily([{"as_of": pd.Timestamp("2026-09-30")},
                                 {"as_of": pd.Timestamp("2026-10-06")}], today) == [{"as_of": pd.Timestamp("2026-10-06")}]


def test_front_roll_requires_two_completed_volume_crossover_sessions(tmp_path):
    today = date(2026, 10, 7)
    contracts = expected_contracts("WTI", today)[:3]
    store = PersistentCurveStore(tmp_path / "roll.sqlite3")
    store.save_roll("WTI", contracts[0].symbol, None, "CALENDAR_FALLBACK", today, 10)
    dates = pd.to_datetime(["2026-10-05", "2026-10-06"])
    histories = {
        contracts[0].symbol: pd.DataFrame({"volume": [100.0, 90.0]}, index=dates),
        contracts[1].symbol: pd.DataFrame({"volume": [110.0, 100.0]}, index=dates),
    }
    metrics = {item.symbol: {"price_date": pd.Timestamp("2026-10-06"), "status": "LIQUID"} for item in contracts}
    index, method, state = _active_front("WTI", contracts, metrics, histories, store, today)
    assert index == 1
    assert method == "VOLUME_CROSSOVER"
    assert state["active_symbol"] == contracts[1].symbol
    assert state["previous_symbol"] == contracts[0].symbol


def test_daily_curve_finalizer_averages_daily_spreads_and_keeps_mean_legs(tmp_path):
    store = PersistentCurveStore(tmp_path / "curve.sqlite3")
    prices = [90.0, 100.0, 90.0, 100.0, 95.0]
    rows = pd.DataFrame([{
        "asset": "WTI", "pair_key": "WTI_F1/F3", "as_of": pd.Timestamp("2026-09-01") + pd.Timedelta(days=i),
        "leg1_symbol": "CLV26.NYM", "leg2_symbol": "CLZ26.NYM", "leg1_month": 10, "leg2_month": 12,
        "leg1_price": price, "leg2_price": 100.0, "spread": price / 100.0 - 1,
        "quality": "HIGH", "source": "Yahoo", "observed_at": f"t{i}",
    } for i, price in enumerate(prices)])
    store.save_daily_curves(rows)
    store.finalize_closed_months(pd.Period("2026-10", freq="M"), "finalized")
    month = store.monthly_curves().iloc[0]
    assert month["monthly_spread"] == -0.05
    assert month["avg_leg1_price"] == 95.0
    assert month["avg_leg2_price"] == 100.0
    assert month["observation_count"] == 5


def test_seasonal_percentile_requires_full_same_month_same_pair_history():
    history = pd.DataFrame({
        "Asset": ["Corn"] * 6,
        "Date": pd.to_datetime([f"{year}-10-31" for year in range(2020, 2026)]),
        "Month": [10] * 6,
        "Spread": np.arange(1, 7, dtype=float),
        "PairKey": ["Corn_Z_H"] * 6,
    })
    incomplete = percentile_with_history(7.0, history, "Corn", 10, "Corn_Z_H", 10,
                                         before=pd.Timestamp("2026-10-01"))
    complete = percentile_with_history(7.0, history, "Corn", 10, "Corn_Z_H", 5,
                                       before=pd.Timestamp("2026-10-01"))
    wrong_pair = percentile_with_history(7.0, history, "Corn", 10, "Corn_H_K", 5,
                                         before=pd.Timestamp("2026-10-01"))
    assert np.isnan(incomplete["percentile"])
    assert complete["percentile"] == 100
    assert complete["history_n"] == 5
    assert np.isnan(wrong_pair["percentile"])


def test_agriculture_baseline_pair_names_map_to_exact_contract_pair_keys():
    baseline = pd.DataFrame({
        "Asset": ["Corn", "Wheat", "Soybeans"],
        "Date": pd.to_datetime(["2025-10-31"] * 3),
        "Month": [10] * 3,
        "Spread %": [0.01, 0.02, 0.03],
        "Structure": ["Dec/Mar", "Dec/Mar", "Nov/Jan"],
    })
    normalized = seasonal_history_frame(baseline, pd.DataFrame())
    assert normalized["PairKey"].tolist() == ["Corn_Z_H", "Wheat_Z_H", "Soybeans_X_F"]


def test_current_curve_refresh_builds_synchronized_energy_ag_and_lme_rows(tmp_path):
    today = date(2026, 10, 7)
    dates = pd.bdate_range(end="2026-10-06", periods=12)

    class FakeYahoo(TermStructureContractProvider):
        name = "Fake Yahoo"

        def fetch_contracts(self, contracts):
            grouped = {}
            for contract in contracts:
                grouped.setdefault(contract.asset, []).append(contract)
            result = {}
            for group in grouped.values():
                ordered = sorted(group, key=lambda item: item.expiry)
                for rank, contract in enumerate(ordered):
                    result[contract.symbol] = pd.DataFrame({
                        "price": np.linspace(70 + rank, 71 + rank, len(dates)),
                        "volume": np.repeat(10_000 / (rank + 1), len(dates)),
                        "open_interest": np.repeat(100_000 / (rank + 1), len(dates)),
                        "source": self.name,
                    }, index=dates)
            return result

    class FakeTV(TermStructureContractProvider):
        name = "Fake TradingView"

        def fetch_contracts(self, contracts):
            return {}

    class FakeLME:
        name = "Fake Westmetall"

        def fetch(self, asset):
            return pd.DataFrame({
                "leg1_price": np.repeat(100.0, len(dates)),
                "leg2_price": np.repeat(101.0, len(dates)),
            }, index=dates)

    bundle = fetch_current_term_structure(
        PersistentCurveStore(tmp_path / "integration.sqlite3"),
        pd.DataFrame(columns=["Asset", "Date", "Month", "Spread %", "Structure"]),
        today=today, yahoo_provider=FakeYahoo(), fallback_provider=FakeTV(), lme_provider=FakeLME(),
    )
    assert len(bundle["current"]) == 11, bundle["current"].to_dict("records")
    assert set(bundle["current"]["quality"]) == {"HIGH"}
    assert bundle["current"]["as_of"].notna().all()
    assert bundle["current"]["source"].str.contains("Fake Yahoo|Fake Westmetall").all()
    assert set(bundle["current"].loc[bundle["current"]["asset"].isin(["Corn", "Wheat"]), "pair_key"]) == {"Corn_Z_H", "Wheat_Z_H"}
    assert set(bundle["current"].loc[bundle["current"]["asset"].isin(["Corn", "Wheat"]), "structure"]) == {"Dec/Mar"}
    assert len(bundle["daily"]) > 0


def test_seasonal_percentile_uses_calendar_month_and_configuration():
    baseline = pd.DataFrame({
        "Asset": ["WTI"] * 10,
        "Date": pd.to_datetime([f"{year}-10-01" for year in range(2016, 2026)]),
        "Spread %": np.arange(1, 11, dtype=float) / 100,
    })
    stored = pd.DataFrame([{
        "asset": "WTI", "as_of": "2025-10-31", "raw_spread": 99.0,
        "structure": "F1/F6",
    }])
    assert seasonal_percentile("WTI", 0.20, pd.Timestamp("2026-10-07"), "F1/F3", baseline, stored, 10) == 100
    assert seasonal_percentile("WTI", 0.20, pd.Timestamp("2026-10-07"), "F1/F3", baseline, stored, 5) == 100


def test_capex_direction_noise_band_boundaries():
    assert classify_capex_direction(-0.020001) == "Falling"
    assert classify_capex_direction(-0.02) == "Stable"
    assert classify_capex_direction(0.02) == "Stable"
    assert classify_capex_direction(0.020001) == "Rising"


def test_capex_percentile_bands_boundaries():
    assert classify_capex_state(19.999, "Stable") == "Extreme Vulnerability"
    assert classify_capex_state(20, "Stable") == "High Vulnerability"
    assert classify_capex_state(40, "Falling") == "Moderate-High Vulnerability"
    assert classify_capex_state(60, "Falling") == "Neutral / Deteriorating"
    assert classify_capex_state(80, "Rising") == "Strong Supply Response Risk"


def test_capex_expanding_percentile_is_no_lookahead_and_minimum_20():
    series = pd.Series(np.arange(1.0, 23.0))
    result = calculate_capex_expanding_percentile(series)
    assert result.iloc[:19].isna().all()
    assert result.iloc[19] == 97.5
    # Future observations do not change an earlier point-in-time percentile.
    extended = calculate_capex_expanding_percentile(pd.concat([series, pd.Series([1000.0])], ignore_index=True))
    assert extended.iloc[19] == result.iloc[19]


def test_capex_current_percentile_uses_full_history():
    values = pd.Series(np.arange(1.0, 22.0))
    result = calculate_capex_current_percentile(values)
    assert result.iloc[-1] == 97.61904761904762


def test_sticky_core_transitions_and_systemic_wave_reset_rules():
    no_raw = {"confirmed_easing": False, "early_easing": False, "systemic": False,
              "confirmed_broad": False, "early_broad": False}
    assert resolve_core_state_transition("Systemic Broadening", no_raw) == "Systemic Broadening"
    assert resolve_core_state_transition("Mature", no_raw) == "Mature"
    assert resolve_core_state_transition("Early Easing", no_raw) == "Early Easing"
    assert resolve_core_state_transition("Confirmed Easing", no_raw) == "Confirmed Easing"
    assert resolve_core_state_transition("Confirmed Easing", {**no_raw, "early_easing": True}) == "Confirmed Easing"
    assert resolve_core_state_transition("Confirmed Easing", {**no_raw, "early_broad": True}) == "Early Broadening"
    assert resolve_core_state_transition("Mature", {**no_raw, "systemic": True}) == "Systemic Broadening"
    assert resolve_core_state_transition("Systemic Broadening", {**no_raw, "early_easing": True}) == "Early Easing"
    assert resolve_core_state_transition("Systemic Broadening", no_raw, mature=True) == "Mature"
    assert resolve_core_state_transition("Systemic Broadening", no_raw, data_complete=False) == "DATA INCOMPLETE"
    assert resolve_core_state_transition("Early Easing", {**no_raw, "early_easing": True, "systemic": True}) == "Systemic Broadening"
    assert update_systemic_wave("Systemic Broadening", "Mature", 0, np.nan, 70) == (1, 70)
    reset_duration, reset_max = update_systemic_wave("Mature", "Systemic Broadening", 8, 85, 50)
    assert reset_duration == 0 and np.isnan(reset_max)
    assert is_confirmed_easing_persistent([2, 1, 2])
    assert not is_confirmed_easing_persistent([2, 1, 1])
    assert not is_confirmed_easing_persistent([2, 2])
