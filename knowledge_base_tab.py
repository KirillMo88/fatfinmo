from pathlib import Path

import streamlit as st


KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"
HOWELL_TEXT_PATH = KNOWLEDGE_BASE_DIR / "michael_howell_debt_liquidity_cycle.md"
HOWELL_INFOGRAPHIC_PATH = KNOWLEDGE_BASE_DIR / "howell_debt_liquidity_cycle.png"
HOWELL_ASSET_ALLOCATION_PATH = KNOWLEDGE_BASE_DIR / "howell_asset_allocation_cycle.png"
HOWELL_PRESENTATION_DIR = KNOWLEDGE_BASE_DIR / "presentation"
LUKE_GROMEN_INFOGRAPHICS = (
    (
        KNOWLEDGE_BASE_DIR / "luke_gromen_debt_debasement.png",
        "Долговая математика и дебасмент доллара",
    ),
    (
        KNOWLEDGE_BASE_DIR / "luke_gromen_gold_model.png",
        "Золото в модели Luke Gromen",
    ),
    (
        KNOWLEDGE_BASE_DIR / "luke_gromen_macro_analysis_framework.png",
        "Luke Gromen — Macro Analysis Framework",
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


def _render_infographics(infographics: tuple[tuple[Path, str], ...]) -> None:
    """Render infographics two per row at the same width as Howell's image."""
    for row_start in range(0, len(infographics), 2):
        columns = st.columns(2)
        row_infographics = infographics[row_start : row_start + 2]
        for column, (infographic_path, caption) in zip(columns, row_infographics):
            with column:
                st.image(
                    str(infographic_path),
                    caption=caption,
                    use_container_width=True,
                )


def render_knowledge_base_tab() -> None:
    """Render Michael Howell materials as one knowledge-base block."""
    with st.container(border=True):
        st.subheader("Michael Howell")
        _render_infographics(
            ((HOWELL_INFOGRAPHIC_PATH, "The Howell Model — Debt–Liquidity Cycle"),)
        )
        _render_infographics(
            ((HOWELL_ASSET_ALLOCATION_PATH, "Asset Allocation Cycle"),)
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
        _render_infographics(LUKE_GROMEN_INFOGRAPHICS)

    with st.container(border=True):
        st.subheader("Lyn Alden")
        _render_infographics(
            (
                (
                    LYN_ALDEN_INFOGRAPHIC_PATH,
                    "Модель Lyn Alden — финансовый инфографический маршрут",
                ),
            )
        )

    with st.container(border=True):
        st.subheader("Brent Johnson")
        _render_infographics(
            ((BRENT_JOHNSON_INFOGRAPHIC_PATH, "Brent Johnson — Dollar Milkshake Model"),)
        )

    with st.container(border=True):
        st.subheader("Jeff Snyder")
        for infographic in JEFF_SNYDER_INFOGRAPHICS:
            _render_infographics((infographic,))

    with st.container(border=True):
        st.subheader("Helicopter View")
        _render_infographics(
            ((HELICOPTER_VIEW_INFOGRAPHIC_PATH, "Единый механизм глобальной ликвидности"),)
        )
