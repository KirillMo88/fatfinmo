from __future__ import annotations

from io import BytesIO

import pandas as pd


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA})
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def read_uploaded_file(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    warnings: list[str] = []

    if name.endswith(".csv"):
        # 1) Fast path: strict default parser.
        try:
            return pd.read_csv(BytesIO(data)), warnings
        except Exception:
            pass

        # 2) Fallback: python engine + delimiter inference.
        try:
            return pd.read_csv(BytesIO(data), engine="python", sep=None), warnings
        except Exception:
            pass

        # 3) Last-resort parse: skip malformed lines.
        try:
            df = pd.read_csv(BytesIO(data), engine="python", sep=None, on_bad_lines="skip")
            total_lines = sum(1 for line in data.decode("utf-8", errors="ignore").splitlines() if line.strip())
            # Approximate skipped data lines (excludes header).
            skipped = max(total_lines - 1 - len(df), 0)
            if skipped > 0:
                warnings.append(
                    f"CSV contained malformed rows. Skipped approximately {skipped} line(s) to load the file."
                )
            return df, warnings
        except Exception as exc:
            raise ValueError(f"Unable to parse CSV file: {exc}") from exc
    if name.endswith(".xlsx"):
        return pd.read_excel(BytesIO(data), engine="openpyxl"), warnings
    raise ValueError("Unsupported file type. Please upload .csv or .xlsx.")


def detect_date_candidates(df: pd.DataFrame, threshold: float = 0.8) -> list[str]:
    candidates: list[str] = []
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        ratio = parsed.notna().mean() if len(parsed) else 0.0
        if ratio >= threshold:
            candidates.append(col)
    return candidates


def prepare_timeseries_df(df_raw: pd.DataFrame, date_col: str) -> tuple[pd.DataFrame, list[str]]:
    if date_col not in df_raw.columns:
        raise ValueError(f"Selected date column '{date_col}' is not present in the uploaded file.")

    df = df_raw.copy()
    parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    invalid = int(parsed.isna().sum())
    if invalid > 0:
        raise ValueError(
            f"Invalid date formats detected in column '{date_col}' ({invalid} rows). "
            "Please correct the source data and upload again."
        )

    df[date_col] = parsed
    df = df.sort_values(date_col)

    dup_count = int(df.duplicated(subset=[date_col]).sum())
    warnings = []
    if dup_count > 0:
        warnings.append(f"Dropped {dup_count} duplicate date rows (kept first occurrence).")
        df = df.drop_duplicates(subset=[date_col], keep="first")

    df = df.set_index(date_col)
    df.index.name = "date"
    return df, warnings


def find_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols: list[str] = []
    for col in df.columns:
        converted = _coerce_numeric(df[col])
        non_null = df[col].notna().sum()
        if non_null == 0:
            continue
        valid_ratio = converted[df[col].notna()].notna().mean()
        # Treat column as numeric if most non-null values are numeric-like.
        if valid_ratio >= 0.8:
            numeric_cols.append(col)
    return numeric_cols


def ensure_columns_numeric(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    warnings: list[str] = []
    for col in columns:
        if col not in out.columns:
            raise ValueError(f"Missing selected column: {col}")
        converted = _coerce_numeric(out[col])
        original_non_null = out[col].notna()
        invalid_mask = original_non_null & converted.isna()
        valid_ratio = converted[original_non_null].notna().mean() if original_non_null.any() else 0.0
        if valid_ratio < 0.8:
            raise ValueError(f"Column '{col}' contains non-numeric values but is used in numeric logic.")
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            warnings.append(
                f"Column '{col}' has {invalid_count} non-numeric row(s); coerced to NaN."
            )
        out[col] = converted.astype(float)
    return out, warnings
