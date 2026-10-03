from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE = ROOT / "knowledge_base" / "michael_howell_debt_liquidity_cycle.md"
APP_SOURCE = ROOT / "app.py"
TAB_SOURCE = ROOT / "knowledge_base_tab.py"
PRESENTATION_DIR = ROOT / "knowledge_base" / "presentation"


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

    def test_infographic_is_half_width_and_all_presentation_slides_render_without_a_selector(self):
        tab = TAB_SOURCE.read_text(encoding="utf-8")
        self.assertIn("infographic_column, _ = st.columns(2)", tab)
        self.assertIn("for row_start in range(0, len(slide_paths), 2):", tab)
        self.assertNotIn("st.selectbox", tab)

        slides = sorted(PRESENTATION_DIR.glob("slide-*.jpg"))
        self.assertEqual(len(slides), 15)
        self.assertIn('HOWELL_PRESENTATION_DIR.glob("slide-*.jpg")', tab)


if __name__ == "__main__":
    unittest.main()
