import pandas as pd

from table_export import dataframe_to_excel_xls_bytes


def test_table_xls_export_drops_internal_row_id_and_escapes_values():
    df = pd.DataFrame(
        {
            "__row_id__": [2],
            "Ticker": ["AAA<script>"],
            "Perf\n1D %": [1.25],
        }
    )

    content = dataframe_to_excel_xls_bytes(df).decode("utf-8")

    assert "__row_id__" not in content
    assert "AAA&lt;script&gt;" in content
    assert "Perf\n1D %" in content
    assert "urn:schemas-microsoft-com:office:excel" in content
