import json

import cio_view


def test_parse_json_text_accepts_fenced_json():
    result = cio_view.parse_json_text('```json\n{"as_of_date":"2026-09-14","x":1}\n```')

    assert result["x"] == 1


def test_deterministic_cio_view_has_required_top_level_sections():
    snapshot = {
        "as_of_date": "2026-09-14T00:00:00+00:00",
        "market_regime": {"structural_regime": "BULL", "final_state": "BULL_LIQUIDITY_WARNING"},
        "global_liquidity": {"global_liquidity_score": 44, "global_liquidity_direction": "DETERIORATING_FAST"},
        "global_macro": {"items": []},
        "gold_regime": {"tactical_flow": 70, "forward_macro_risk": 65},
        "btc_regime": {"cycle_bottom_status": "BOTTOMING_WATCH"},
        "asset_market_data": {},
        "data_quality": {"notes": []},
    }

    result = cio_view.deterministic_cio_view(snapshot, "TEST")

    assert result["overall_system_state"]["label"] == "RISK_ON_WITH_WARNING"
    assert set(result["asset_outlook_3m"]) == {"SPY", "QQQ", "GLD", "BTC-USD"}
    assert "scenarios" in result
    assert "what_would_change_view" in result
