import pandas as pd

import global_liquidity as gl


def _raw(date, series, value):
    row = {column: None for column in gl.RAW_COLUMNS}
    row.update(
        {
            "observation_date": pd.Timestamp(date),
            "source": "TEST",
            "series_id": series,
            "raw_value": value,
            "data_status": "CURRENT",
        }
    )
    return row


def test_archive_raw_wins_before_2026_and_live_wins_after(monkeypatch):
    archive = pd.DataFrame([_raw("2025-12-01", "M2SL", 100.0)], columns=gl.RAW_COLUMNS)
    live = pd.DataFrame(
        [
            _raw("2025-12-01", "M2SL", 999.0),
            _raw("2026-01-01", "M2SL", 110.0),
        ],
        columns=gl.RAW_COLUMNS,
    )
    monkeypatch.setattr(gl, "read_frame", lambda path, columns: archive if path == gl.GLOBAL_LIQUIDITY_ARCHIVE_RAW_PATH else pd.DataFrame(columns=columns))

    merged = gl.merge_global_liquidity_archive_raw(live)

    assert merged["raw_value"].tolist() == [100.0, 110.0]
    assert merged["observation_date"].tolist() == [pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-01")]


def test_archive_layer_keeps_frozen_history(monkeypatch):
    archive = pd.DataFrame(
        [{"date": pd.Timestamp("2025-12-01"), "global_m2_usd_bn": 100.0}],
        columns=["date", "global_m2_usd_bn"],
    )
    live = pd.DataFrame(
        [
            {"date": pd.Timestamp("2025-12-01"), "global_m2_usd_bn": 999.0},
            {"date": pd.Timestamp("2026-01-01"), "global_m2_usd_bn": 110.0},
        ]
    )
    merged = gl.merge_global_liquidity_archive_layer(live, archive, gl.MONTHLY_COLUMNS)

    assert merged[["date", "global_m2_usd_bn"]].to_dict("records") == [
        {"date": pd.Timestamp("2025-12-01"), "global_m2_usd_bn": 100.0},
        {"date": pd.Timestamp("2026-01-01"), "global_m2_usd_bn": 110.0},
    ]
