from datetime import date

import numpy as np
import pandas as pd

from fund_flows import FundFlowObservation
from gold_regime.config import gold_regime_config
from gold_regime.cot import calculate_cot_momentum_score, download_cftc_cot, extract_comex_gold_cot, load_comex_gold_cot_from_positioning, normalize_cot_columns
from gold_regime.etf_flows import aggregate_gold_etf_flows, load_gold_etf_flows
from gold_regime.macro import calculate_gold_macro_history
from gold_regime.regime import calculate_gold_tactical_flow, determine_flow_flags, determine_gold_regime
from gold_regime.service import calculate_freshness, carry_forward_cot_history
from gold_regime.utils import rolling_percentile_rank


def weekly(values):
    return pd.Series(values, index=pd.date_range("2020-01-03", periods=len(values), freq="W-FRI"))


def test_point_in_time_percentile_does_not_use_future_values():
    values = pd.Series([0.0] * 104 + [100.0, -100.0], index=pd.date_range("2020-01-03", periods=106, freq="W-FRI"))

    rank = rolling_percentile_rank(values, window=156, min_periods=104)

    assert rank.iloc[104] > 99.0
    assert rank.iloc[105] < rank.iloc[104]


def test_gold_macro_scores_reward_dxy_and_real_yield_decreases():
    dates = pd.date_range("2020-01-03", periods=180, freq="W-FRI")
    dxy_rising = pd.Series(np.r_[np.full(160, 100.0), np.linspace(100.0, 120.0, 20)], index=dates)
    dxy_falling = pd.Series(np.r_[np.full(160, 100.0), np.linspace(100.0, 85.0, 20)], index=dates)
    real_yield_rising = pd.Series(np.r_[np.full(160, 1.0), np.linspace(1.0, 2.5, 20)], index=dates)
    real_yield_falling = pd.Series(np.r_[np.full(160, 1.0), np.linspace(1.0, -0.5, 20)], index=dates)
    us2y = pd.Series(np.linspace(1.0, 3.0, 180), index=dates)
    wti = pd.Series(np.linspace(70.0, 80.0, 180), index=dates)

    rising = calculate_gold_macro_history(dxy_rising, real_yield_rising, us2y, wti).iloc[-1]
    falling = calculate_gold_macro_history(dxy_falling, real_yield_falling, us2y, wti).iloc[-1]

    assert falling["dxy_score"] > rising["dxy_score"]
    assert falling["real_yield_score"] > rising["real_yield_score"]


def test_gold_forward_risk_increases_with_us2y_and_wti_momentum():
    dates = pd.date_range("2020-01-03", periods=180, freq="W-FRI")
    dxy = pd.Series(np.linspace(100.0, 102.0, 180), index=dates)
    real_yield = pd.Series(np.linspace(1.0, 1.2, 180), index=dates)
    flat_us2y = pd.Series(np.full(180, 2.0), index=dates)
    rising_us2y = pd.Series(np.r_[np.full(160, 2.0), np.linspace(2.0, 5.0, 20)], index=dates)
    flat_wti = pd.Series(np.full(180, 70.0), index=dates)
    rising_wti = pd.Series(np.r_[np.full(154, 70.0), np.linspace(70.0, 110.0, 26)], index=dates)

    flat = calculate_gold_macro_history(dxy, real_yield, flat_us2y, flat_wti).iloc[-1]
    rising = calculate_gold_macro_history(dxy, real_yield, rising_us2y, rising_wti).iloc[-1]

    assert rising["us2y_risk_score"] > flat["us2y_risk_score"]
    assert rising["wti_risk_score"] > flat["wti_risk_score"]
    assert rising["forward_macro_risk"] > flat["forward_macro_risk"]


def test_gold_etf_flows_aggregate_available_etfs_without_zero_filling_missing():
    cfg = gold_regime_config()
    tickers = cfg["etf_tickers"]
    dates = pd.date_range("2022-01-07", periods=240, freq="W-FRI")
    rows = []
    for ticker in tickers[:-1]:
        for i, dt in enumerate(dates):
            rows.append({"date": dt, "ticker": ticker, "net_flow": 1_000_000.0 + i * 10_000.0})
    daily = pd.DataFrame(rows)

    history = aggregate_gold_etf_flows(daily, tickers, cfg)

    assert history["etf_coverage_count"].iloc[-1] == 7
    assert history["etf_coverage_total"].iloc[-1] == 8
    assert "BAR" in history["etf_unavailable_tickers"].iloc[-1]
    assert history["etf_flow_1w"].iloc[-1] == 7 * (1_000_000.0 + 239 * 10_000.0)
    assert np.isfinite(history["etf_flow_score"].dropna().iloc[-1])


def test_load_gold_etf_flows_reports_unavailable_tickers():
    cfg = gold_regime_config()

    def fake_fetcher(ticker, start_date, end_date):
        if ticker == "BAR":
            return []
        return [FundFlowObservation(ticker, date(2026, 1, 2), net_flow=1_000_000.0, aum=None)]

    daily, available, unavailable = load_gold_etf_flows(date(2026, 1, 1), date(2026, 1, 3), cfg, fake_fetcher)

    assert len(available) == 7
    assert unavailable == ["BAR"]
    assert "BAR" not in set(daily["ticker"])


def test_cot_managed_money_net_and_publication_availability(tmp_path):
    rows = []
    dates = pd.date_range("2020-01-07", periods=120, freq="W-TUE")
    for i, dt in enumerate(dates):
        rows.append(
            {
                "report_date_as_yyyy_mm_dd": dt.strftime("%Y-%m-%d"),
                "market_and_exchange_names": "COMMODITY EXCHANGE INC.",
                "contract_market_name": "GOLD - COMMODITY EXCHANGE INC.",
                "commodity_name": "GOLD",
                "open_interest_all": "1,000",
                "m_money_positions_long_all": 500 + i,
                "m_money_positions_short_all": 300,
                "m_money_positions_spread_all": 0,
            }
        )
    cot, contract = extract_comex_gold_cot(pd.DataFrame(rows), cache_dir=tmp_path)
    scored = calculate_cot_momentum_score(cot)

    assert contract == "GOLD - COMMODITY EXCHANGE INC."
    assert scored["cot_mm_net"].iloc[0] == 200
    assert scored["cot_mm_net_pct_oi"].iloc[0] == 0.2
    assert scored["date"].iloc[0].day_name() == "Friday"
    assert scored["cot_report_date"].iloc[0].strftime("%Y-%m-%d") == "2020-01-07"
    assert np.isclose(scored["cot_change_4w"].iloc[4], 0.004)


def test_gold_cot_can_load_from_unified_positioning(monkeypatch):
    import positioning

    dates = pd.date_range("2020-01-07", periods=120, freq="W-TUE")
    master = pd.DataFrame(
        {
            "Date": dates,
            "Canonical_Asset": ["GOLD"] * len(dates),
            "Preferred_For_Dashboard": [True] * len(dates),
            "Participant_Category": ["Managed Money"] * len(dates),
            "Raw_Contract_Name": ["GOLD - COMMODITY EXCHANGE INC."] * len(dates),
            "Exchange": ["COMMODITY EXCHANGE INC."] * len(dates),
            "Open_Interest": [1000] * len(dates),
            "Long": np.arange(500, 620),
            "Short": [300] * len(dates),
            "Spreading": [0] * len(dates),
            "Net": np.arange(200, 320),
            "NetPctOI": np.arange(20.0, 32.0, 0.1),
            "NetPctOI_3Y_Percentile": np.linspace(10.0, 90.0, len(dates)),
            "NetPctOI_4W_Change": [np.nan] * 4 + [0.4] * (len(dates) - 4),
        }
    )
    monkeypatch.setattr(positioning, "read_processed", lambda name: master if name == "cftc_master" else pd.DataFrame())

    cot, contract = load_comex_gold_cot_from_positioning()

    assert contract == "GOLD - COMMODITY EXCHANGE INC."
    assert cot["date"].iloc[0].day_name() == "Friday"
    assert np.isclose(cot["cot_mm_net_pct_oi"].iloc[0], 0.20)
    assert np.isclose(cot["cot_change_4w"].iloc[-1], 0.004)
    assert np.isfinite(cot["cot_momentum_score"].dropna().iloc[-1])


def test_cot_normalization_drops_duplicate_columns():
    frame = pd.DataFrame([[1, 2, 3]], columns=["Commodity Name", "commodity_name", "Open Interest All"])

    normalized = normalize_cot_columns(frame)

    assert normalized.columns.tolist() == ["commodity_name", "open_interest_all"]


def test_cot_download_uses_cached_data_when_refresh_fails(tmp_path):
    cache_path = tmp_path / "cftc_disaggregated_futures_only.csv"
    cache_path.write_text("report_date_as_yyyy_mm_dd,commodity_name\n2026-09-01,GOLD\n")
    cfg = gold_regime_config()
    cfg["cot"]["url"] = str(tmp_path / "missing.csv")
    cfg["cot"]["cache_ttl_seconds"] = -1

    frame = download_cftc_cot(cfg, tmp_path)

    assert frame.iloc[0]["report_date_as_yyyy_mm_dd"] == "2026-09-01"


def test_cot_values_carry_forward_until_next_weekly_update():
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-04", "2026-09-11"]),
            "cot_report_date": [date(2026, 9, 1), np.nan],
            "cot_momentum_score": [29.0, np.nan],
            "cot_mm_net_pct_oi": [0.32, np.nan],
            "etf_flow_score": [80.0, 75.0],
        }
    )

    carried = carry_forward_cot_history(history)

    assert carried.iloc[-1]["cot_report_date"] == date(2026, 9, 1)
    assert carried.iloc[-1]["cot_momentum_score"] == 29.0
    assert carried.iloc[-1]["cot_mm_net_pct_oi"] == 0.32


def test_tactical_flow_requires_both_etf_and_cot_components():
    score, state = calculate_gold_tactical_flow(80.0, np.nan)

    assert np.isnan(score)
    assert state == "PARTIAL_DATA"


def test_tactical_flow_uses_etf_led_85_15_weighting():
    score, state = calculate_gold_tactical_flow(80.0, 20.0)

    assert np.isclose(score, 71.0)
    assert state == "BULLISH_FLOW"


def test_gold_regime_priority_is_explicit():
    assert determine_gold_regime(55.0, 30.0, 85.0, 20.0) == "HIGH_RISK"
    assert determine_gold_regime(75.0, 30.0, 85.0, 75.0) == "STRONG_TREND_WITH_FLOW_SUPPORT"
    assert determine_gold_regime(75.0, 80.0, 20.0, 20.0) == "FLOW_DIVERGENCE_WARNING"
    assert determine_gold_regime(75.0, 65.0, 30.0, 70.0) == "HIGH_CONVICTION_LONG"
    assert determine_gold_regime(45.0, 30.0, 50.0, 30.0) == "BEARISH"


def test_gold_divergence_flags_are_informational():
    flags = determine_flow_flags(
        alpha=75.0,
        tactical_flow=76.0,
        etf_score=84.0,
        cot_score=35.0,
        structural_macro=29.0,
        forward_macro_risk=80.0,
        long_liquidity_cycle="DECELERATING_EXPANSION",
    )

    assert flags["ETF_LED_BULLISH_DIVERGENCE"] is True
    assert flags["MACRO_FLOW_CONFLICT"] is True
    assert flags["LONG_CYCLE_HEADWIND"] is True
    assert "ETF_LED_BULLISH_DIVERGENCE" in flags["ACTIVE_DIVERGENCE_FLAGS"]


def test_gold_freshness_uses_latest_available_cot_when_current_week_is_missing():
    cfg = gold_regime_config()
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-04", "2026-09-11"]),
            "gold_price": [300.0, 305.0],
            "structural_macro_score": [40.0, 42.0],
            "etf_flow_1w": [1_000_000.0, 2_000_000.0],
            "cot_report_date": [date(2026, 9, 1), np.nan],
        }
    )
    current = history.iloc[-1].to_dict()
    current["structural_demand_last_updated"] = None

    freshness = calculate_freshness(current, history, cfg)

    assert freshness["COT"].last_updated == date(2026, 9, 1)
    assert freshness["COT"].status != "ERROR"
    assert freshness["Structural Demand"].status == "NOT_CONFIGURED"
