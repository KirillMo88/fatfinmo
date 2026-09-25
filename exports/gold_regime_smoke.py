from __future__ import annotations

import traceback
import sys

sys.path.insert(0, "/app")

from gold_regime.service import build_gold_regime_snapshot


try:
    snapshot = build_gold_regime_snapshot(gold_alpha=75.0)
    current = snapshot.current
    print("rows", len(snapshot.history))
    print("regime", current.get("gold_regime"))
    print("alpha", current.get("gold_alpha"))
    print("structural", current.get("structural_macro_score"), current.get("structural_macro_state"))
    print("forward", current.get("forward_macro_risk"), current.get("forward_macro_risk_state"))
    print("tactical", current.get("tactical_flow_score"), current.get("flow_state"))
    print("etf", current.get("etf_flow_score"), current.get("etf_coverage_count"), current.get("etf_coverage_total"))
    print("cot", current.get("cot_momentum_score"), current.get("cot_report_date"))
    print("contract", snapshot.cot_contract_market_name)
    print("unavailable", snapshot.etf_unavailable_tickers)
    print("explanation", current.get("explanation"))
except Exception:
    traceback.print_exc()
    raise
