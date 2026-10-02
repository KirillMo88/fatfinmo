from pathlib import Path

import streamlit as st


KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"
HOWELL_TEXT_PATH = KNOWLEDGE_BASE_DIR / "michael_howell_debt_liquidity_cycle.md"
HOWELL_INFOGRAPHIC_PATH = KNOWLEDGE_BASE_DIR / "howell_debt_liquidity_cycle.png"


def render_knowledge_base_tab() -> None:
    """Render the source interview summary and its accompanying infographic."""
    st.image(
        str(HOWELL_INFOGRAPHIC_PATH),
        caption="The Howell Model — Debt–Liquidity Cycle",
        use_container_width=True,
    )
    content = HOWELL_TEXT_PATH.read_text(encoding="utf-8")
    st.markdown(content)
