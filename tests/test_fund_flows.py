from datetime import date

import numpy as np

from fund_flows import (
    FundFlowObservation,
    calculate_fund_flow_metrics,
    get_fund_flow_metrics,
    normalize_ticker_for_etf_com,
    parse_etf_com_fund_flow_payload,
)


def test_normalize_ticker_for_etf_com_rejects_unsupported_symbols():
    assert normalize_ticker_for_etf_com("spy") == "SPY"
    assert normalize_ticker_for_etf_com("BRK.B") == "BRK-B"
    assert normalize_ticker_for_etf_com("BTC-USD") is None
    assert normalize_ticker_for_etf_com("^VIX") is None


def test_parse_etf_com_payload_accepts_nested_rows_and_money_suffixes():
    payload = {
        "data": {
            "results": {
                "data": [
                    {"date": "2026-01-02", "netFlow": "$12.5M", "aum": "$500M"},
                    {"date": "2026-01-03", "netFlow": "1.0", "assetsUnderManagement": "501"},
                ]
            }
        }
    }

    observations = parse_etf_com_fund_flow_payload("SPY", payload)

    assert len(observations) == 2
    assert observations[0].net_flow == 12_500_000.0
    assert observations[0].aum == 500_000_000.0
    assert observations[1].net_flow == 1_000_000.0
    assert observations[1].aum == 501_000_000.0


def test_calculate_fund_flow_metrics_uses_reference_aum():
    observations = [
        FundFlowObservation("SPY", date(2026, 1, 1), net_flow=0.0, aum=1_000_000_000.0),
        FundFlowObservation("SPY", date(2026, 2, 1), net_flow=10_000_000.0, aum=1_100_000_000.0),
        FundFlowObservation("SPY", date(2026, 3, 1), net_flow=20_000_000.0, aum=1_200_000_000.0),
        FundFlowObservation("SPY", date(2026, 4, 1), net_flow=30_000_000.0, aum=1_300_000_000.0),
    ]

    metrics = calculate_fund_flow_metrics(observations, latest_date=date(2026, 4, 1))

    assert metrics is not None
    assert metrics.flow_1m_pct == 2.5
    assert metrics.flow_3m_pct == 6.0


def test_calculate_fund_flow_metrics_can_use_fallback_aum():
    observations = [
        FundFlowObservation("SPY", date(2026, 2, 1), net_flow=10_000_000.0, aum=None),
        FundFlowObservation("SPY", date(2026, 3, 1), net_flow=20_000_000.0, aum=None),
        FundFlowObservation("SPY", date(2026, 4, 1), net_flow=30_000_000.0, aum=None),
    ]

    metrics = calculate_fund_flow_metrics(
        observations,
        latest_date=date(2026, 4, 1),
        fallback_aum=1_000_000_000.0,
    )

    assert metrics is not None
    assert metrics.flow_1m_pct == 3.0
    assert metrics.flow_3m_pct == 6.0
    assert metrics.method == "latest_aum_fallback"


def test_get_fund_flow_metrics_fetches_and_caches_observations(tmp_path):
    calls = []

    def fake_fetcher(ticker, start_date, end_date):
        calls.append((ticker, start_date, end_date))
        return [
            FundFlowObservation(ticker, date(2026, 1, 1), net_flow=0.0, aum=1_000_000_000.0),
            FundFlowObservation(ticker, date(2026, 2, 1), net_flow=10_000_000.0, aum=1_000_000_000.0),
            FundFlowObservation(ticker, date(2026, 3, 1), net_flow=20_000_000.0, aum=1_000_000_000.0),
            FundFlowObservation(ticker, date(2026, 4, 1), net_flow=30_000_000.0, aum=1_000_000_000.0),
        ]

    cache_path = tmp_path / "fund_flows.sqlite"

    first = get_fund_flow_metrics("spy", cache_path=cache_path, today=date(2026, 4, 1), fetcher=fake_fetcher)
    second = get_fund_flow_metrics("spy", cache_path=cache_path, today=date(2026, 4, 1), fetcher=fake_fetcher)

    assert first is not None
    assert second is not None
    assert np.isclose(first.flow_3m_pct, 6.0)
    assert np.isclose(second.flow_3m_pct, 6.0)
    assert len(calls) == 1


def test_get_fund_flow_metrics_uses_aum_fetcher_when_payload_has_no_aum(tmp_path):
    def fake_fetcher(ticker, start_date, end_date):
        return [
            FundFlowObservation(ticker, date(2026, 2, 1), net_flow=10_000_000.0, aum=None),
            FundFlowObservation(ticker, date(2026, 3, 1), net_flow=20_000_000.0, aum=None),
            FundFlowObservation(ticker, date(2026, 4, 1), net_flow=30_000_000.0, aum=None),
        ]

    metrics = get_fund_flow_metrics(
        "SPY",
        cache_path=tmp_path / "fund_flows.sqlite",
        today=date(2026, 4, 1),
        fetcher=fake_fetcher,
        aum_fetcher=lambda ticker: 1_000_000_000.0,
    )

    assert metrics is not None
    assert np.isclose(metrics.flow_3m_pct, 6.0)
