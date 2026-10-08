from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE = ROOT / "knowledge_base" / "michael_howell_debt_liquidity_cycle.md"
HOWELL_ASSET_ALLOCATION = ROOT / "knowledge_base" / "howell_asset_allocation_cycle.png"
APP_SOURCE = ROOT / "app.py"
TAB_SOURCE = ROOT / "knowledge_base_tab.py"
PRESENTATION_DIR = ROOT / "knowledge_base" / "presentation"
LUKE_GROMEN_INFOGRAPHICS = (
    ROOT / "knowledge_base" / "luke_gromen_debt_debasement.png",
    ROOT / "knowledge_base" / "luke_gromen_gold_model.png",
    ROOT / "knowledge_base" / "luke_gromen_macro_analysis_framework.png",
)
LYN_ALDEN_INFOGRAPHIC = ROOT / "knowledge_base" / "lyn_alden_financial_framework.png"
BRENT_JOHNSON_INFOGRAPHIC = (
    ROOT / "knowledge_base" / "brent_johnson_dollar_milkshake_model.png"
)
JEFF_SNYDER_INFOGRAPHICS = (
    ROOT / "knowledge_base" / "jeff_snyder_global_dollar_system.png",
    ROOT / "knowledge_base" / "jeff_snyder_eurodollar_crisis_2007_2008.png",
)
HELICOPTER_VIEW_INFOGRAPHIC = (
    ROOT / "knowledge_base" / "helicopter_view_global_liquidity.png"
)


class KnowledgeBaseContentTests(unittest.TestCase):
    def test_howell_summary_is_present_and_display_equations_are_converted(self):
        content = KNOWLEDGE_BASE.read_text(encoding="utf-8")

        self.assertIn("Главный тезис интервью", content)
        self.assertIn("## 57.", content)
        self.assertIn("$$", content)
        self.assertNotRegex(content, re.compile(r"\\\[|\\\]"))
        self.assertIn("| Macro | Nominal GDP |", content)
        section_numbers = [int(value) for value in re.findall(r"^## (\d+)\.", content, re.MULTILINE)]
        self.assertEqual(section_numbers, list(range(1, 58)))
        self.assertIn("## 22. PBoC может влиять даже на Bitcoin", content)
        self.assertNotIn("## 2. MOVE Index", content)
        self.assertIn(
            "| Layer | Indicator | Что показывает |\n"
            "| --- | --- | --- |\n"
            "| Macro | Nominal GDP | Fundamental pressure на yields |\n"
            "| Debt | Debt maturities / refinancing wall | Будущий спрос на liquidity |",
            content,
        )

    def test_knowledge_base_tab_follows_description(self):
        app = APP_SOURCE.read_text(encoding="utf-8")
        options = re.search(r"view_options\s*=\s*\[(.*?)\n\s*\]", app, re.DOTALL)
        self.assertIsNotNone(options)
        labels = re.findall(r'^\s*"([^"]+)"\s*,?$', options.group(1), re.MULTILINE)
        self.assertEqual(labels.index("Knowledge Base"), labels.index("Description") + 1)
        self.assertIn('elif active_view == "Knowledge Base":\n        render_knowledge_base_tab()', app)

    def test_infographic_is_included(self):
        self.assertTrue((ROOT / "knowledge_base" / "howell_debt_liquidity_cycle.png").is_file())

    def test_howell_asset_allocation_infographic_is_compact_and_included(self):
        tab = TAB_SOURCE.read_text(encoding="utf-8")

        self.assertTrue(HOWELL_ASSET_ALLOCATION.is_file())
        self.assertIn(
            '((HOWELL_ASSET_ALLOCATION_PATH, "Asset Allocation Cycle"),)',
            tab,
        )

    def test_infographic_is_half_width_and_all_presentation_slides_render_without_a_selector(self):
        tab = TAB_SOURCE.read_text(encoding="utf-8")
        self.assertIn("def _render_infographics", tab)
        self.assertIn("columns = st.columns(2)", tab)
        self.assertIn("for row_start in range(0, len(slide_paths), 2):", tab)
        self.assertNotIn("st.selectbox", tab)

        slides = sorted(PRESENTATION_DIR.glob("slide-*.jpg"))
        self.assertEqual(len(slides), 15)
        self.assertIn('HOWELL_PRESENTATION_DIR.glob("slide-*.jpg")', tab)

    def test_howell_content_is_grouped_and_text_starts_collapsed(self):
        tab = TAB_SOURCE.read_text(encoding="utf-8")

        self.assertIn('with st.container(border=True):', tab)
        self.assertIn('st.subheader("Michael Howell")', tab)
        self.assertIn(
            'with st.expander("Подробное текстовое саммари", expanded=False):',
            tab,
        )
        self.assertLess(tab.index("st.image("), tab.index("with st.expander("))
        self.assertLess(
            tab.index("for row_start in range(0, len(slide_paths), 2):"),
            tab.index("with st.expander("),
        )

    def test_luke_gromen_block_includes_all_infographics(self):
        tab = TAB_SOURCE.read_text(encoding="utf-8")

        self.assertIn('st.subheader("Luke Gromen")', tab)
        self.assertIn("_render_infographics(LUKE_GROMEN_INFOGRAPHICS)", tab)
        self.assertEqual(
            LUKE_GROMEN_INFOGRAPHICS[0].name,
            "luke_gromen_debt_debasement.png",
        )
        self.assertEqual(
            LUKE_GROMEN_INFOGRAPHICS[-1].name,
            "luke_gromen_macro_analysis_framework.png",
        )
        for infographic_path in LUKE_GROMEN_INFOGRAPHICS:
            self.assertTrue(infographic_path.is_file(), infographic_path)

    def test_lyn_alden_block_includes_infographic(self):
        tab = TAB_SOURCE.read_text(encoding="utf-8")

        self.assertIn('st.subheader("Lyn Alden")', tab)
        self.assertIn("LYN_ALDEN_INFOGRAPHIC_PATH,", tab)
        self.assertTrue(LYN_ALDEN_INFOGRAPHIC.is_file())

    def test_brent_johnson_block_includes_infographic(self):
        tab = TAB_SOURCE.read_text(encoding="utf-8")

        self.assertIn('st.subheader("Brent Johnson")', tab)
        self.assertIn("BRENT_JOHNSON_INFOGRAPHIC_PATH,", tab)
        self.assertTrue(BRENT_JOHNSON_INFOGRAPHIC.is_file())

    def test_jeff_snyder_block_includes_all_infographics(self):
        tab = TAB_SOURCE.read_text(encoding="utf-8")

        self.assertIn('st.subheader("Jeff Snyder")', tab)
        self.assertIn("for infographic in JEFF_SNYDER_INFOGRAPHICS:", tab)
        self.assertIn("_render_infographics((infographic,))", tab)
        for infographic_path in JEFF_SNYDER_INFOGRAPHICS:
            self.assertTrue(infographic_path.is_file(), infographic_path)

    def test_helicopter_view_block_includes_infographic(self):
        tab = TAB_SOURCE.read_text(encoding="utf-8")

        self.assertIn('st.subheader("Helicopter View")', tab)
        self.assertIn("HELICOPTER_VIEW_INFOGRAPHIC_PATH,", tab)
        self.assertTrue(HELICOPTER_VIEW_INFOGRAPHIC.is_file())

    def test_all_infographic_blocks_use_the_shared_half_width_renderer(self):
        tab = TAB_SOURCE.read_text(encoding="utf-8")

        self.assertIn("((HOWELL_INFOGRAPHIC_PATH,", tab)
        self.assertIn("_render_infographics(LUKE_GROMEN_INFOGRAPHICS)", tab)
        self.assertIn("LYN_ALDEN_INFOGRAPHIC_PATH,", tab)
        self.assertIn("((BRENT_JOHNSON_INFOGRAPHIC_PATH,", tab)
        self.assertIn("_render_infographics((infographic,))", tab)
        self.assertIn("((HELICOPTER_VIEW_INFOGRAPHIC_PATH,", tab)


if __name__ == "__main__":
    unittest.main()
