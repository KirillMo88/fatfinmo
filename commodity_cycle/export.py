from __future__ import annotations

from datetime import date, datetime
from io import BytesIO
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
import xlsxwriter


PERCENT_COLUMNS = {
    "Return 1M", "Return 3M", "Return 6M", "Return 12M", "Curve Spread", "Annualized Curve Spread", "MTD Average Spread",
}


def commodity_tables_to_xlsx(
    primary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    seasonal_history: pd.DataFrame | None = None,
    term_structure_audit: pd.DataFrame | None = None,
) -> bytes:
    """Export current tables plus the seasonal history actually used by the engine."""
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {"in_memory": True, "nan_inf_to_errors": True})
    formats = {
        "number": workbook.add_format({"num_format": "0.00"}),
        "percent": workbook.add_format({"num_format": "0.00%"}),
        "date": workbook.add_format({"num_format": "yyyy-mm-dd"}),
        "contango": workbook.add_format({"bg_color": "#14532D", "font_color": "#FFFFFF"}),
        "backwardation": workbook.add_format({"bg_color": "#7F1D1D", "font_color": "#FFFFFF"}),
    }
    _write_table_sheet(workbook, "Confirmation", "CommodityConfirmation", primary, formats, highlight=True)
    _write_table_sheet(workbook, "Diagnostics", "CommodityDiagnostics", diagnostics, formats, highlight=False)
    if seasonal_history is not None and not seasonal_history.empty:
        _write_table_sheet(
            workbook,
            "Seasonal History",
            "CommoditySeasonalHistory",
            seasonal_history,
            formats,
            highlight=False,
        )
    if term_structure_audit is not None and not term_structure_audit.empty:
        _write_table_sheet(
            workbook,
            "Term Structure Audit",
            "CommodityTermStructureAudit",
            term_structure_audit,
            formats,
            highlight=False,
        )
    workbook.close()
    return output.getvalue()


def _write_table_sheet(
    workbook: xlsxwriter.Workbook,
    sheet_name: str,
    table_name: str,
    frame: pd.DataFrame,
    formats: dict[str, Any],
    *,
    highlight: bool,
) -> None:
    worksheet = workbook.add_worksheet(sheet_name)
    worksheet.hide_gridlines(2)
    worksheet.freeze_panes(1, min(2, len(frame.columns)))
    columns = frame.columns.tolist()
    for column_index, column in enumerate(columns):
        worksheet.write(0, column_index, column)
        for row_index, value in enumerate(frame[column], start=1):
            _write_excel_value(worksheet, row_index, column_index, value, column, formats)
        rendered = [str(column)] + ["" if pd.isna(value) else str(value) for value in frame[column]]
        width = min(max(len(value) for value in rendered) + 2, 34)
        worksheet.set_column(column_index, column_index, max(width, 11))

    if columns:
        last_row = max(len(frame), 1)
        worksheet.add_table(0, 0, last_row, len(columns) - 1, {
            "name": table_name,
            "style": "Table Style Medium 2",
            "columns": [{"header": column} for column in columns],
        })
    if not highlight or frame.empty:
        return

    for column in ("Return 1M", "Return 3M", "Return 6M", "Return 12M"):
        if column not in columns:
            continue
        index = columns.index(column)
        worksheet.conditional_format(1, index, len(frame), index, {
            "type": "3_color_scale",
            "min_color": "#7F1D1D",
            "mid_color": "#854D0E",
            "max_color": "#14532D",
        })
    if "Raw Curve State" in columns:
        index = columns.index("Raw Curve State")
        worksheet.conditional_format(1, index, len(frame), index, {
            "type": "text", "criteria": "containing", "value": "Contango", "format": formats["contango"],
        })
        worksheet.conditional_format(1, index, len(frame), index, {
            "type": "text", "criteria": "containing", "value": "Backwardation", "format": formats["backwardation"],
        })


def _write_excel_value(
    worksheet: xlsxwriter.worksheet.Worksheet,
    row: int,
    column: int,
    value: Any,
    column_name: str,
    formats: dict[str, Any],
) -> None:
    if value is None or value is pd.NaT or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        worksheet.write_blank(row, column, None)
    elif isinstance(value, (pd.Timestamp, datetime, date)):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        worksheet.write_datetime(row, column, timestamp.to_pydatetime(), formats["date"])
    elif isinstance(value, (Real, np.number)) and not isinstance(value, (bool, np.bool_)):
        cell_format = formats["percent"] if column_name in PERCENT_COLUMNS else formats["number"]
        worksheet.write_number(row, column, float(value), cell_format)
    elif isinstance(value, (bool, np.bool_)):
        worksheet.write_boolean(row, column, bool(value))
    else:
        worksheet.write(row, column, str(value))
