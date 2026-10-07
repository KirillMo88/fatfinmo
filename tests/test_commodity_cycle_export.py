from io import BytesIO

import pandas as pd
from openpyxl import load_workbook

from commodity_cycle.export import commodity_tables_to_xlsx


def test_commodity_tables_export_to_two_formatted_xlsx_sheets():
    primary = pd.DataFrame({
        "Sector": ["Energy", "Energy"],
        "Commodity": ["WTI", "Natural Gas"],
        "Price": [90.421, 3.026],
        "Return 3M": [0.30123, -0.07601],
        "Raw Curve State": ["Backwardation", "Contango"],
        "COT 5Y Percentile": [20.3846, 10.0],
        "Spread 5Y Seasonal Percentile": [20.0, 80.0],
        "Price Date": pd.to_datetime(["2026-09-30", "2026-09-30"]),
    })
    diagnostics = pd.DataFrame({
        "Sector": ["Energy"], "Commodity": ["WTI"], "COT 3Y Percentile": [10.1234],
        "Spread 10Y Seasonal Percentile": [30.0],
    })

    workbook_bytes = commodity_tables_to_xlsx(primary, diagnostics)
    workbook = load_workbook(BytesIO(workbook_bytes))

    assert workbook.sheetnames == ["Confirmation", "Diagnostics"]
    confirmation = workbook["Confirmation"]
    diagnostics_sheet = workbook["Diagnostics"]
    assert [cell.value for cell in confirmation[1]] == primary.columns.tolist()
    assert [cell.value for cell in diagnostics_sheet[1]] == diagnostics.columns.tolist()
    assert confirmation["C2"].number_format == "0.00"
    assert confirmation["D2"].number_format == "0.00%"
    assert confirmation["H2"].number_format == "yyyy-mm-dd"
    formatting = list(confirmation.conditional_formatting)
    assert len(formatting) == 2
    assert sorted(len(item.rules) for item in formatting) == [1, 2]


def test_export_includes_engine_seasonal_history_and_term_audit_when_supplied():
    primary = pd.DataFrame({"Commodity": ["Corn"], "Price": [500.0]})
    diagnostics = pd.DataFrame({"Commodity": ["Corn"], "Seasonal Percentile Status": ["CURRENT_MTD"]})
    seasonal_history = pd.DataFrame({
        "Asset": ["Corn"], "Date": [pd.Timestamp("2025-11-15")], "Spread %": [-0.02],
        "Structure": ["Corn_Z_H"], "Source": ["TradingView MCP daily close"],
        "History Data Quality": ["OK"],
    })
    audit = pd.DataFrame({
        "Diagnostic Type": ["Agriculture Seasonal Structure"], "Asset": ["Corn"],
        "Data Quality": ["OK"], "Median Seasonal Spread": [-0.02],
    })

    workbook_bytes = commodity_tables_to_xlsx(primary, diagnostics, seasonal_history, audit)
    workbook = load_workbook(BytesIO(workbook_bytes), data_only=False)

    assert workbook.sheetnames == ["Confirmation", "Diagnostics", "Seasonal History", "Term Structure Audit"]
    seasonal = workbook["Seasonal History"]
    assert [cell.value for cell in seasonal[1]] == seasonal_history.columns.tolist()
    assert seasonal["A2"].value == "Corn"
    assert "Missing historical" not in " ".join(str(cell.value) for row in seasonal.iter_rows() for cell in row)
