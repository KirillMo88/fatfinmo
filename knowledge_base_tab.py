from pathlib import Path

import streamlit as st
from knowledge_base.presentation_layout import group_slides


KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"
HOWELL_TEXT_PATH = KNOWLEDGE_BASE_DIR / "michael_howell_debt_liquidity_cycle.md"
HOWELL_INFOGRAPHIC_PATH = KNOWLEDGE_BASE_DIR / "howell_debt_liquidity_cycle.png"
HOWELL_PRESENTATION_DIR = KNOWLEDGE_BASE_DIR / "presentation"


def render_knowledge_base_tab() -> None:
    """Render the source interview summary and its accompanying infographic."""
    st.image(
        str(HOWELL_INFOGRAPHIC_PATH),
        caption="The Howell Model — Debt–Liquidity Cycle",
        use_container_width=True,
    )

    slide_paths = sorted(HOWELL_PRESENTATION_DIR.glob("slide-*.jpg"))
    if slide_paths:
        slide_groups = group_slides(slide_paths, slides_per_screen=4)
        st.subheader("Эпоха рефинансирования: глобальная ликвидность и рынки капитала")
        selected_group = st.selectbox(
            "Слайды презентации",
            options=range(len(slide_groups)),
            format_func=lambda index: (
                f"Слайды {index * 4 + 1}–{min((index + 1) * 4, len(slide_paths))}"
                f" из {len(slide_paths)}"
            ),
            key="howell_presentation_screen",
        )
        selected_slides = slide_groups[selected_group]
        for row_start in range(0, len(selected_slides), 2):
            columns = st.columns(2)
            row_slides = selected_slides[row_start : row_start + 2]
            for column_offset, (column, slide_path) in enumerate(zip(columns, row_slides)):
                slide_number = row_start + column_offset + 1 + selected_group * 4
                with column:
                    st.image(
                        str(slide_path),
                        caption=f"Слайд {slide_number}",
                        use_container_width=True,
                    )

    content = HOWELL_TEXT_PATH.read_text(encoding="utf-8")
    st.markdown(content)
