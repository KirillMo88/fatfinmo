from html import escape

import pandas as pd


def dataframe_to_excel_xls_bytes(df: pd.DataFrame, sheet_name: str = "Table") -> bytes:
    export_df = df.drop(columns=["__row_id__"], errors="ignore")
    safe_sheet_name = escape(sheet_name[:31] or "Table")
    html_table = export_df.to_html(index=False, border=1, na_rep="", escape=True)
    workbook = f"""<html xmlns:o="urn:schemas-microsoft-com:office:office"
xmlns:x="urn:schemas-microsoft-com:office:excel"
xmlns="http://www.w3.org/TR/REC-html40">
<head>
<meta charset="utf-8">
<!--[if gte mso 9]><xml>
<x:ExcelWorkbook>
<x:ExcelWorksheets>
<x:ExcelWorksheet>
<x:Name>{safe_sheet_name}</x:Name>
<x:WorksheetOptions><x:DisplayGridlines/></x:WorksheetOptions>
</x:ExcelWorksheet>
</x:ExcelWorksheets>
</x:ExcelWorkbook>
</xml><![endif]-->
</head>
<body>
{html_table}
</body>
</html>"""
    return workbook.encode("utf-8")
