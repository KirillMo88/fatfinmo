import numpy as np
import pandas as pd

import global_liquidity as gl


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def raw_row(date, series_id, value, source="TEST", region="US", metric="M2", frequency="monthly", currency="USD", unit="native"):
    return {
        "observation_date": pd.Timestamp(date),
        "release_date": pd.NaT,
        "source": source,
        "source_name": source,
        "source_url": "",
        "series_id": series_id,
        "region": region,
        "metric": metric,
        "frequency": frequency,
        "currency": currency,
        "unit": unit,
        "raw_value": value,
        "download_timestamp": "2026-09-11 00:00 UTC",
        "data_status": "CURRENT",
        "notes": "",
    }


def test_global_m2_requires_china_and_does_not_zero_fill():
    raw = pd.DataFrame(
        [
            raw_row("2020-01-01", "M2SL", 1000.0),
            raw_row("2020-01-01", gl.GLOBAL_LIQUIDITY_CONFIG["ecb_m2_key"], 2000.0, region="Euro Area", currency="EUR", unit="EUR millions"),
            raw_row("2020-01-01", "MD02'MAM1NAM2M2MO", 30000.0, region="Japan", currency="JPY", unit="JPY 100 million"),
            raw_row("2020-01-03", "DEXUSEU", 1.2, region="FX", metric="EURUSD", frequency="daily", currency="USD per EUR", unit="rate"),
            raw_row("2020-01-03", "DEXJPUS", 100.0, region="FX", metric="USDJPY", frequency="daily", currency="JPY per USD", unit="rate"),
            raw_row("2020-01-03", "DEXCHUS", 7.0, region="FX", metric="USDCNY", frequency="daily", currency="CNY per USD", unit="rate"),
        ],
        columns=gl.RAW_COLUMNS,
    )

    monthly = gl.build_monthly_layer(raw)

    assert monthly["global_m2_usd_bn"].isna().all()
    assert np.isclose(monthly["global_m2_partial_usd_bn"].iloc[-1], 1032.4)
    assert set(monthly["data_status"]) == {"PARTIAL_DATA"}


def test_monthly_layer_does_not_create_fx_only_observations():
    raw = pd.DataFrame(
        [
            raw_row("2020-01-01", "M2SL", 1000.0),
            raw_row("2020-02-03", "DEXUSEU", 1.2, region="FX", metric="EURUSD", frequency="daily", currency="USD per EUR", unit="rate"),
            raw_row("2020-02-03", "DEXJPUS", 100.0, region="FX", metric="USDJPY", frequency="daily", currency="JPY per USD", unit="rate"),
            raw_row("2020-02-03", "DEXCHUS", 7.0, region="FX", metric="USDCNY", frequency="daily", currency="CNY per USD", unit="rate"),
        ],
        columns=gl.RAW_COLUMNS,
    )

    monthly = gl.build_monthly_layer(raw)

    assert monthly["date"].dt.strftime("%Y-%m").tolist() == ["2020-01"]


def test_monthly_layer_excludes_current_incomplete_month(monkeypatch):
    monkeypatch.setattr(gl, "now_utc", lambda: pd.Timestamp("2026-09-13 12:00:00"))
    raw = pd.DataFrame(
        [
            raw_row("2026-08-01", "M2SL", 1000.0),
            raw_row("2026-09-01", "M2SL", 1001.0),
        ],
        columns=gl.RAW_COLUMNS,
    )

    monthly = gl.build_monthly_layer(raw)

    assert monthly["date"].dt.strftime("%Y-%m").tolist() == ["2026-08"]


def test_fx_neutral_m2_does_not_forward_fill_stale_m2_components():
    raw = pd.DataFrame(
        [
            raw_row("2019-12-01", "Money & Quasi-money (M2)", 40000.0, region="China", currency="CNY", unit="CNY 100 million"),
            raw_row("2020-01-01", "M2SL", 1000.0),
            raw_row("2020-01-01", gl.GLOBAL_LIQUIDITY_CONFIG["ecb_m2_key"], 2000.0, region="Euro Area", currency="EUR", unit="EUR millions"),
            raw_row("2020-01-01", "MD02'MAM1NAM2M2MO", 30000.0, region="Japan", currency="JPY", unit="JPY 100 million"),
            raw_row("2019-12-03", "DEXUSEU", 1.2, region="FX", metric="EURUSD", frequency="daily", currency="USD per EUR", unit="rate"),
            raw_row("2019-12-03", "DEXJPUS", 100.0, region="FX", metric="USDJPY", frequency="daily", currency="JPY per USD", unit="rate"),
            raw_row("2019-12-03", "DEXCHUS", 7.0, region="FX", metric="USDCNY", frequency="daily", currency="CNY per USD", unit="rate"),
            raw_row("2020-01-03", "DEXUSEU", 1.2, region="FX", metric="EURUSD", frequency="daily", currency="USD per EUR", unit="rate"),
            raw_row("2020-01-03", "DEXJPUS", 100.0, region="FX", metric="USDJPY", frequency="daily", currency="JPY per USD", unit="rate"),
            raw_row("2020-01-03", "DEXCHUS", 7.0, region="FX", metric="USDCNY", frequency="daily", currency="CNY per USD", unit="rate"),
        ],
        columns=gl.RAW_COLUMNS,
    )

    monthly = gl.build_monthly_layer(raw)
    january = monthly[monthly["date"].eq(pd.Timestamp("2020-01-01"))].iloc[0]

    assert np.isnan(january["global_m2_usd_bn"])
    assert np.isnan(january["global_m2_fx_neutral_bn"])
    assert np.isclose(january["global_m2_fx_neutral_partial_bn"], 1032.4)


def test_reconstruct_china_m2_from_yoy_recursively_to_feb_2021_control_point():
    fred_dates = pd.date_range("2018-09-01", "2019-08-01", freq="MS")
    fred_values = [2_000_000.0] * len(fred_dates)
    fred_values[5] = 2_000_000.0
    fred_frame = pd.DataFrame({"Date": fred_dates, "Value": fred_values})
    yoy_dates = pd.date_range("2019-09-01", "2021-02-01", freq="MS")
    yoy_values = [0.0] * len(yoy_dates)
    yoy_values[yoy_dates.get_loc(pd.Timestamp("2020-02-01"))] = 1.0
    yoy_values[yoy_dates.get_loc(pd.Timestamp("2021-02-01"))] = 10.693069306930693
    yoy = pd.DataFrame(
        {
            "observation_date": yoy_dates,
            "release_date": yoy_dates + pd.DateOffset(days=10),
            "series_id": gl.TRADINGVIEW_CNM2_YOY_SERIES,
            "yoy_pct": yoy_values,
        }
    )

    reconstructed = gl.reconstruct_china_m2_from_yoy(fred_frame, yoy)
    feb_2021 = reconstructed[reconstructed["observation_date"].eq(pd.Timestamp("2021-02-01"))].iloc[0]

    assert np.isclose(feb_2021["raw_value"], 2_236_000.0)


def test_china_m2_fred_tradingview_fallback_converts_fred_units(monkeypatch):
    fred_frame = pd.DataFrame(
        {
            "Series_ID": [gl.FRED_CHINA_M2_LEGACY_SERIES],
            "Date": [pd.Timestamp("2019-08-01")],
            "Value": [193_549_242_773_720.0],
        }
    )
    yoy = pd.DataFrame(columns=["observation_date", "release_date", "series_id", "yoy_pct"])
    monkeypatch.setattr(gl, "download_fred_series_batch", lambda *args, **kwargs: fred_frame)
    monkeypatch.setattr(gl, "tradingview_china_m2_yoy_raw", lambda *args, **kwargs: yoy)

    fallback = gl.china_m2_fred_tradingview_fallback_raw(api_key="x", official_error="test")

    assert fallback.iloc[-1]["source"] == "FRED_IMF"
    assert fallback.iloc[-1]["series_id"] == gl.CHINA_M2_SERIES_ID
    assert np.isclose(fallback.iloc[-1]["raw_value"], 1_935_492.4277372)


def test_monthly_fx_conversion_uses_source_units():
    raw = pd.DataFrame(
        [
            raw_row("2020-01-01", "M2SL", 1000.0, unit="USD billions"),
            raw_row("2020-01-01", gl.GLOBAL_LIQUIDITY_CONFIG["ecb_m2_key"], 2000.0, region="Euro Area", currency="EUR", unit="EUR millions"),
            raw_row("2020-01-01", "MD02'MAM1NAM2M2MO", 30000.0, region="Japan", currency="JPY", unit="JPY 100 million"),
            raw_row("2020-01-01", "Money & Quasi-money (M2)", 40000.0, region="China", currency="CNY", unit="CNY 100 million"),
            raw_row("2020-01-03", "DEXUSEU", 1.2, region="FX", metric="EURUSD", frequency="daily", currency="USD per EUR", unit="rate"),
            raw_row("2020-01-17", "DEXUSEU", 1.4, region="FX", metric="EURUSD", frequency="daily", currency="USD per EUR", unit="rate"),
            raw_row("2020-01-03", "DEXJPUS", 100.0, region="FX", metric="USDJPY", frequency="daily", currency="JPY per USD", unit="rate"),
            raw_row("2020-01-03", "DEXCHUS", 7.0, region="FX", metric="USDCNY", frequency="daily", currency="CNY per USD", unit="rate"),
        ],
        columns=gl.RAW_COLUMNS,
    )

    monthly = gl.build_monthly_layer(raw)
    row = monthly.iloc[-1]

    assert np.isclose(row["us_m2_usd_bn"], 1000.0)
    assert np.isclose(row["ea_m2_usd_bn"], 2.6)
    assert np.isclose(row["japan_m2_usd_bn"], 30.0)
    assert np.isclose(row["china_m2_usd_bn"], 571.4285714285714)
    assert np.isclose(row["global_m2_usd_bn"], 1604.0285714285715)


def test_monthly_global_cb_assets_uses_all_official_components_and_fx():
    raw = pd.DataFrame(
        [
            raw_row("2020-01-03", "WALCL", 8_000_000.0, metric="Fed Total Assets", frequency="weekly", unit="USD millions"),
            raw_row("2020-01-03", gl.ECB_TOTAL_ASSETS_SERIES_ID, 7_000_000.0, source="ECB", region="Euro Area", metric="ECB Total Assets", frequency="weekly", currency="EUR", unit="EUR millions"),
            raw_row("2020-01-01", "BS01'MABJMTA", 7_500_000.0, source="BOJ", region="Japan", metric="BoJ Total Assets", frequency="monthly", currency="JPY", unit="100 million yen"),
            raw_row("2020-01-01", gl.PBOC_TOTAL_ASSETS_SERIES_ID, 450_000.0, source="PBOC", region="China", metric="PBoC Total Assets", frequency="monthly", currency="CNY", unit="CNY 100 million"),
            raw_row("2020-01-03", "DEXUSEU", 1.2, region="FX", metric="EURUSD", frequency="daily", currency="USD per EUR", unit="rate"),
            raw_row("2020-01-03", "DEXJPUS", 100.0, region="FX", metric="USDJPY", frequency="daily", currency="JPY per USD", unit="rate"),
            raw_row("2020-01-03", "DEXCHUS", 7.0, region="FX", metric="USDCNY", frequency="daily", currency="CNY per USD", unit="rate"),
        ],
        columns=gl.RAW_COLUMNS,
    )

    monthly = gl.build_monthly_layer(raw)
    row = monthly.iloc[-1]

    assert np.isclose(row["fed_assets_usd_bn"], 8000.0)
    assert np.isclose(row["ecb_assets_usd_bn"], 8400.0)
    assert np.isclose(row["boj_assets_usd_bn"], 7500.0)
    assert np.isclose(row["pboc_assets_usd_bn"], 6428.571428571428)
    assert np.isclose(row["global_cb_assets_usd_bn"], 30328.571428571428)


def test_pboc_total_assets_table_raw_converts_trillion_cny(tmp_path):
    table_path = tmp_path / "pboc_total_assets.csv"
    pd.DataFrame(
        [
            {
                "observation_date": "2026-07-01",
                "pboc_total_assets_cny_trn": 50.21,
                "source_note": "sample",
            }
        ]
    ).to_csv(table_path, index=False)

    raw = gl.pboc_total_assets_table_raw(table_path)

    row = raw.iloc[0]
    assert row["source"] == "PBOC_LOCAL_TABLE"
    assert row["series_id"] == gl.PBOC_TOTAL_ASSETS_SERIES_ID
    assert row["unit"] == "CNY 100 million"
    assert np.isclose(row["raw_value"], 502_100.0)


def test_pboc_total_assets_raw_prefers_local_table(monkeypatch, tmp_path):
    table_path = tmp_path / "pboc_total_assets.csv"
    pd.DataFrame(
        [
            {
                "observation_date": "2026-07-01",
                "raw_value_cny_100mn": 502_100.0,
            }
        ]
    ).to_csv(table_path, index=False)
    monkeypatch.setenv("PBOC_TOTAL_ASSETS_TABLE_PATH", str(table_path))
    monkeypatch.setattr(gl, "PBOC_TOTAL_ASSETS_STORAGE_PATH", tmp_path / "storage_pboc_total_assets.csv")
    monkeypatch.setattr(gl, "PBOC_TOTAL_ASSETS_BUNDLED_PATH", tmp_path / "bundled_pboc_total_assets.csv")
    monkeypatch.setattr(gl, "tradingview_mcp_pboc_total_assets_raw", lambda: pd.DataFrame(columns=gl.RAW_COLUMNS))
    monkeypatch.setattr(gl, "tradingview_pboc_total_assets_latest_raw", lambda: pd.DataFrame(columns=gl.RAW_COLUMNS))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("network parser should not be called when local table exists")

    monkeypatch.setattr(gl, "discover_pboc_balance_sheet_links", fail_if_called)
    raw = gl.pboc_total_assets_raw()

    assert len(raw) == 1
    assert raw.iloc[0]["source"] == "PBOC_LOCAL_TABLE"
    assert np.isclose(raw.iloc[0]["raw_value"], 502_100.0)


def test_pboc_total_assets_raw_uses_tradingview_latest_over_seed(monkeypatch, tmp_path):
    table_path = tmp_path / "pboc_total_assets.csv"
    pd.DataFrame(
        [
            {
                "observation_date": "2026-07-01",
                "raw_value_cny_100mn": 502_100.0,
            }
        ]
    ).to_csv(table_path, index=False)
    monkeypatch.setenv("PBOC_TOTAL_ASSETS_TABLE_PATH", str(table_path))
    monkeypatch.setattr(gl, "PBOC_TOTAL_ASSETS_STORAGE_PATH", tmp_path / "storage_pboc_total_assets.csv")
    monkeypatch.setattr(gl, "PBOC_TOTAL_ASSETS_BUNDLED_PATH", tmp_path / "bundled_pboc_total_assets.csv")
    monkeypatch.setattr(gl, "tradingview_mcp_pboc_total_assets_raw", lambda: pd.DataFrame(columns=gl.RAW_COLUMNS))
    monkeypatch.setattr(
        gl,
        "tradingview_pboc_total_assets_latest_raw",
        lambda: pd.DataFrame(
            [
                raw_row(
                    "2026-07-01",
                    gl.PBOC_TOTAL_ASSETS_SERIES_ID,
                    502_068.47,
                    source="TRADINGVIEW",
                    region="China",
                    metric="PBoC Total Assets",
                    frequency="monthly",
                    currency="CNY",
                    unit="CNY 100 million",
                )
            ],
            columns=gl.RAW_COLUMNS,
        ),
    )

    raw = gl.pboc_total_assets_raw()

    assert len(raw) == 1
    assert raw.iloc[0]["source"] == "TRADINGVIEW"
    assert np.isclose(raw.iloc[0]["raw_value"], 502_068.47)


def test_pboc_total_assets_raw_prefers_valid_mcp(monkeypatch, tmp_path):
    monkeypatch.setenv("PBOC_TOTAL_ASSETS_TABLE_PATH", str(tmp_path / "missing.csv"))
    monkeypatch.setattr(gl, "PBOC_TOTAL_ASSETS_STORAGE_PATH", tmp_path / "storage_pboc_total_assets.csv")
    monkeypatch.setattr(gl, "PBOC_TOTAL_ASSETS_BUNDLED_PATH", tmp_path / "bundled_pboc_total_assets.csv")
    monkeypatch.setattr(
        gl,
        "tradingview_mcp_pboc_total_assets_raw",
        lambda: pd.DataFrame(
            [
                {
                    **raw_row(
                        "2026-07-01",
                        gl.PBOC_TOTAL_ASSETS_SERIES_ID,
                        502_100.0,
                        source="TRADINGVIEW_MCP",
                        region="China",
                        metric="PBoC Total Assets",
                        frequency="monthly",
                        currency="CNY",
                        unit="CNY 100 million",
                    ),
                    "source_mode": "MCP_PRIMARY",
                }
            ],
            columns=gl.RAW_COLUMNS,
        ),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fallback should not be called when MCP is valid")

    monkeypatch.setattr(gl, "tradingview_pboc_total_assets_latest_raw", fail_if_called)
    monkeypatch.setattr(gl, "discover_pboc_balance_sheet_links", fail_if_called)

    raw = gl.pboc_total_assets_raw()

    assert len(raw) == 1
    assert raw.iloc[0]["source"] == "TRADINGVIEW_MCP"
    assert raw.iloc[0]["source_mode"] == "MCP_PRIMARY"


def test_parse_tradingview_observation_month():
    html = "<div>Observation period</div><span>Jul 2026</span>"

    assert gl.parse_tradingview_observation_month(html) == pd.Timestamp("2026-07-01")


def test_weekly_fast_cb_layer_uses_fed_and_ecb_only(monkeypatch):
    monkeypatch.setattr(gl, "weekly_dxy", lambda: pd.Series([100.0], index=[pd.Timestamp("2020-01-03")]))
    raw = pd.DataFrame(
        [
            raw_row("2020-01-03", "WALCL", 8_000_000.0, metric="Fed Total Assets", frequency="weekly", unit="USD millions"),
            raw_row("2020-01-03", "WTREGEN", 1_000_000.0, metric="Treasury General Account", frequency="weekly", unit="USD millions"),
            raw_row("2020-01-03", "RRPONTSYD", 500.0, metric="Overnight Reverse Repo", frequency="daily", unit="USD billions"),
            raw_row("2020-01-03", gl.ECB_TOTAL_ASSETS_SERIES_ID, 7_000_000.0, source="ECB", region="Euro Area", metric="ECB Total Assets", frequency="weekly", currency="EUR", unit="EUR millions"),
            raw_row("2020-01-03", "DEXUSEU", 1.2, region="FX", metric="EURUSD", frequency="daily", currency="USD per EUR", unit="rate"),
        ],
        columns=gl.RAW_COLUMNS,
    )

    weekly = gl.build_weekly_layer(raw)
    row = weekly.iloc[-1]

    assert np.isclose(row["fed_assets_usd_bn"], 8000.0)
    assert np.isclose(row["ecb_assets_usd_bn"], 8400.0)
    assert np.isnan(row["boj_assets_usd_bn"])
    assert np.isnan(row["pboc_assets_usd_bn"])
    assert np.isclose(row["global_cb_assets_usd_bn"], 16400.0)


def test_parse_ecb_week_period_returns_week_friday():
    assert gl.parse_ecb_week_period("2020-W01") == pd.Timestamp("2020-01-03")


def test_boj_total_assets_code_is_validated_from_metadata(monkeypatch):
    payload = {
        "RESULTSET": [
            {
                "SERIES_CODE": "MABJMTA",
                "NAME_OF_TIME_SERIES": "Bank of Japan Accounts/Assets/Total (Assets, or Liabilities and Net Assets) (s)",
                "UNIT": "100 million yen",
                "FREQUENCY": "MONTHLY",
            }
        ]
    }
    monkeypatch.setattr(gl, "get_with_retries", lambda *args, **kwargs: FakeResponse(payload))

    assert gl.validate_boj_total_assets_code() == "MABJMTA"


def test_us_net_liquidity_formula_uses_normalized_units(monkeypatch):
    monkeypatch.setattr(gl, "weekly_dxy", lambda: pd.Series([100.0], index=[pd.Timestamp("2020-01-03")]))
    raw = pd.DataFrame(
        [
            raw_row("2020-01-03", "WALCL", 8_000_000.0, metric="Fed Total Assets", frequency="weekly", unit="USD millions"),
            raw_row("2020-01-03", "WTREGEN", 1_000_000.0, metric="Treasury General Account", frequency="weekly", unit="USD millions"),
            raw_row("2020-01-03", "RRPONTSYD", 500.0, metric="Overnight Reverse Repo", frequency="daily", unit="USD billions"),
        ],
        columns=gl.RAW_COLUMNS,
    )

    weekly = gl.build_weekly_layer(raw)

    assert pd.notna(weekly["date"].iloc[-1])
    assert np.isclose(weekly["fed_assets_usd_bn"].iloc[-1], 8000.0)
    assert np.isclose(weekly["tga_usd_bn"].iloc[-1], 1000.0)
    assert np.isclose(weekly["rrp_usd_bn"].iloc[-1], 500.0)
    assert np.isclose(weekly["us_net_liquidity_usd_bn"].iloc[-1], 6500.0)


def test_impulse_state_labels():
    states = gl.impulse_state(pd.Series([np.nan, 10.0, 25.0, 45.0, 65.0, 85.0]))
    assert states.tolist() == [
        "DATA_INCOMPLETE",
        "STRONG_CONTRACTION",
        "CONTRACTION",
        "NEUTRAL",
        "EXPANSION",
        "STRONG_EXPANSION",
    ]
