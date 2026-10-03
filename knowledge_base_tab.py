from pathlib import Path

import streamlit as st


KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"
HOWELL_TEXT_PATH = KNOWLEDGE_BASE_DIR / "michael_howell_debt_liquidity_cycle.md"
HOWELL_INFOGRAPHIC_PATH = KNOWLEDGE_BASE_DIR / "howell_debt_liquidity_cycle.png"
HOWELL_PRESENTATION_DIR = KNOWLEDGE_BASE_DIR / "presentation"


def render_knowledge_base_tab() -> None:
    """Render the source interview summary and its accompanying infographic."""
    infographic_column, _ = st.columns(2)
    with infographic_column:
        st.image(
            str(HOWELL_INFOGRAPHIC_PATH),
            caption="The Howell Model — Debt–Liquidity Cycle",
            use_container_width=True,
        )

    slide_paths = sorted(HOWELL_PRESENTATION_DIR.glob("slide-*.jpg"))
    if slide_paths:
        st.subheader("Эпоха рефинансирования: глобальная ликвидность и рынки капитала")
        for row_start in range(0, len(slide_paths), 2):
            columns = st.columns(2)
            row_slides = slide_paths[row_start : row_start + 2]
            for column_offset, (column, slide_path) in enumerate(zip(columns, row_slides)):
                slide_number = row_start + column_offset + 1
                with column:
                    st.image(
                        str(slide_path),
                        caption=f"Слайд {slide_number}",
                        use_container_width=True,
                    )

    content = HOWELL_TEXT_PATH.read_text(encoding="utf-8")
    st.markdown(content)
