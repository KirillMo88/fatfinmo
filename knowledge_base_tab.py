from pathlib import Path

import streamlit as st


KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"
HOWELL_TEXT_PATH = KNOWLEDGE_BASE_DIR / "michael_howell_debt_liquidity_cycle.md"
HOWELL_INFOGRAPHIC_PATH = KNOWLEDGE_BASE_DIR / "howell_debt_liquidity_cycle.png"
HOWELL_PRESENTATION_DIR = KNOWLEDGE_BASE_DIR / "presentation"
LUKE_GROMEN_INFOGRAPHICS = (
    (
        KNOWLEDGE_BASE_DIR / "luke_gromen_macro_analysis_framework.png",
        "Luke Gromen — Macro Analysis Framework",
    ),
    (
        KNOWLEDGE_BASE_DIR / "luke_gromen_gold_model.png",
        "Золото в модели Luke Gromen",
    ),
    (
        KNOWLEDGE_BASE_DIR / "luke_gromen_debt_debasement.png",
        "Долговая математика и дебасмент доллара",
    ),
)
LYN_ALDEN_INFOGRAPHIC_PATH = KNOWLEDGE_BASE_DIR / "lyn_alden_financial_framework.png"
BRENT_JOHNSON_INFOGRAPHIC_PATH = (
    KNOWLEDGE_BASE_DIR / "brent_johnson_dollar_milkshake_model.png"
)
JEFF_SNYDER_INFOGRAPHICS = (
    (
        KNOWLEDGE_BASE_DIR / "jeff_snyder_global_dollar_system.png",
        "Глобальная долларовая система по Jeff Snyder",
    ),
    (
        KNOWLEDGE_BASE_DIR / "jeff_snyder_eurodollar_crisis_2007_2008.png",
        "Евродолларовый кризис 2007–2008 годов",
    ),
)
HELICOPTER_VIEW_INFOGRAPHIC_PATH = (
    KNOWLEDGE_BASE_DIR / "helicopter_view_global_liquidity.png"
)


def render_knowledge_base_tab() -> None:
    """Render Michael Howell materials as one knowledge-base block."""
    with st.container(border=True):
        st.subheader("Michael Howell")

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
        with st.expander("Подробное текстовое саммари", expanded=False):
            st.markdown(content)

    with st.container(border=True):
        st.subheader("Luke Gromen")
        for infographic_path, caption in LUKE_GROMEN_INFOGRAPHICS:
            st.image(
                str(infographic_path),
                caption=caption,
                use_container_width=True,
            )

    with st.container(border=True):
        st.subheader("Lyn Alden")
        st.image(
            str(LYN_ALDEN_INFOGRAPHIC_PATH),
            caption="Модель Lyn Alden — финансовый инфографический маршрут",
            use_container_width=True,
        )

    with st.container(border=True):
        st.subheader("Brent Johnson")
        st.image(
            str(BRENT_JOHNSON_INFOGRAPHIC_PATH),
            caption="Brent Johnson — Dollar Milkshake Model",
            use_container_width=True,
        )

    with st.container(border=True):
        st.subheader("Jeff Snyder")
        for infographic_path, caption in JEFF_SNYDER_INFOGRAPHICS:
            st.image(
                str(infographic_path),
                caption=caption,
                use_container_width=True,
            )

    with st.container(border=True):
        st.subheader("Helicopter View")
        st.image(
            str(HELICOPTER_VIEW_INFOGRAPHIC_PATH),
            caption="Единый механизм глобальной ликвидности",
            use_container_width=True,
        )
