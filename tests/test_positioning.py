from io import BytesIO

import numpy as np
import pandas as pd

import positioning
from positioning import (
    calculate_aaii_metrics,
    calculate_cftc_positioning_metrics,
    calculate_naaim_metrics,
    cftc_asset_series,
    cftc_dashboard_frame,
    export_positioning_xlsx,
    normalize_cftc,
    parse_aaii_live_results_html,
    read_aaii_historical_workbook,
    resolve_canonical_contracts,
    validate_cftc_master,
)


def test_normalize_disaggregated_gold_and_calculate_metrics():
    dates = pd.date_range("2020-01-07", periods=300, freq="W-TUE")
    raw = pd.DataFrame(
        {
            "Report_Date_as_YYYY-MM-DD": dates,
            "Market_and_Exchange_Names": ["GOLD - COMMODITY EXCHANGE INC."] * len(dates),
            "Contract_Market_Name": ["GOLD"] * len(dates),
            "CFTC_Contract_Market_Code": ["088691"] * len(dates),
            "Open_Interest_All": [1000] * len(dates),
            "M_Money_Positions_Long_All": np.arange(300, 600),
            "M_Money_Positions_Short_All": np.arange(100, 400),
            "M_Money_Positions_Spread_All": [50] * len(dates),
        }
    )

    master = calculate_cftc_positioning_metrics(resolve_canonical_contracts(normalize_cftc(raw, pd.DataFrame())))
    gold = cftc_asset_series(master, "GOLD", "Managed Money")

    assert not gold.empty
    assert gold.iloc[-1]["Canonical_Asset"] == "GOLD"
    assert gold.iloc[-1]["Preferred_For_Dashboard"] is True or bool(gold.iloc[-1]["Preferred_For_Dashboard"])
    assert gold.iloc[-1]["Net"] == gold.iloc[-1]["Long"] - gold.iloc[-1]["Short"]
    assert np.isclose(gold.iloc[-1]["NetPctOI"], 20.0)
    assert np.isfinite(gold.iloc[-1]["NetPctOI_3Y_Percentile"])
    assert np.isfinite(gold.iloc[-1]["NetPctOI_5Y_Percentile"])
    assert not validate_cftc_master(master)


def test_wti_same_code_history_continues_across_name_change_without_ice_splice():
    dates = pd.date_range("2020-01-07", periods=300, freq="W-TUE")
    names = ["CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE"] * 200
    names += ["WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE"] * 100
    raw = pd.DataFrame(
        {
            "Report_Date_as_YYYY-MM-DD": dates.tolist() + dates.tolist(),
            "Market_and_Exchange_Names": names + ["WTI - ICE FUTURES EUROPE"] * len(dates),
            "Contract_Market_Name": ["CRUDE OIL, LIGHT SWEET"] * 200 + ["WTI-PHYSICAL"] * 100 + ["WTI"] * len(dates),
            "CFTC_Contract_Market_Code": ["067651"] * len(dates) + ["067411"] * len(dates),
            "Open_Interest_All": [1000] * (len(dates) * 2),
            "M_Money_Positions_Long_All": list(np.arange(300, 600)) * 2,
            "M_Money_Positions_Short_All": list(np.arange(100, 400)) * 2,
            "M_Money_Positions_Spread_All": [50] * (len(dates) * 2),
        }
    )

    master = calculate_cftc_positioning_metrics(resolve_canonical_contracts(normalize_cftc(raw, pd.DataFrame())))
    wti = cftc_asset_series(master, "WTI", "Managed Money")
    ice = master.loc[master["CFTC_Code"] == "067411"]

    assert len(wti) == len(dates)
    assert wti.iloc[199]["History_Weeks"] == 200
    assert wti.iloc[200]["History_Weeks"] == 201
    assert np.isfinite(wti.iloc[199]["NetPctOI_4W_Change"])
    assert np.isfinite(wti.iloc[-1]["NetPctOI_13W_Change"])
    assert np.isfinite(wti.iloc[-1]["NetPctOI_3Y_Percentile"])
    assert np.isfinite(wti.iloc[-1]["NetPctOI_5Y_Percentile"])
    assert not ice["Preferred_For_Dashboard"].any()
    assert set(wti["CFTC_Code"]) == {"067651"}


def test_expected_commodity_contracts_map_by_exact_cftc_code():
    contracts = [
        ("023651", "NAT GAS NYME", "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE", "Natural Gas"),
        ("111659", "GASOLINE RBOB", "GASOLINE RBOB - NEW YORK MERCANTILE EXCHANGE", "RBOB"),
        ("085692", "COPPER- #1", "COPPER - COMMODITY EXCHANGE INC.", "Copper"),
        ("191691", "ALUMINUM", "ALUMINUM - COMMODITY EXCHANGE INC.", "Aluminum"),
        ("002602", "CORN", "CORN - CHICAGO BOARD OF TRADE", "Corn"),
        ("005602", "SOYBEANS", "SOYBEANS - CHICAGO BOARD OF TRADE", "Soybeans"),
        ("001602", "WHEAT-SRW", "WHEAT-SRW - CHICAGO BOARD OF TRADE", "Wheat"),
    ]
    raw = pd.DataFrame(
        {
            "Report_Date_as_YYYY-MM-DD": [pd.Timestamp("2026-09-29")] * len(contracts),
            "Market_and_Exchange_Names": [item[2] for item in contracts],
            "Contract_Market_Name": [item[1] for item in contracts],
            "CFTC_Contract_Market_Code": [item[0] for item in contracts],
            "Open_Interest_All": [1000] * len(contracts),
            "M_Money_Positions_Long_All": [400] * len(contracts),
            "M_Money_Positions_Short_All": [200] * len(contracts),
            "M_Money_Positions_Spread_All": [50] * len(contracts),
        }
    )
    master = resolve_canonical_contracts(normalize_cftc(raw, pd.DataFrame()))
    mapped = master.loc[master["Preferred_For_Dashboard"]]

    assert set(zip(mapped["CFTC_Code"], mapped["Canonical_Asset"])) == {
        (item[0], item[3]) for item in contracts
    }


def test_aluminum_percentiles_require_full_156_and_260_week_history():
    dates = pd.date_range("2023-01-03", periods=143, freq="W-TUE")
    raw = pd.DataFrame(
        {
            "Report_Date_as_YYYY-MM-DD": dates,
            "Market_and_Exchange_Names": ["ALUMINUM - COMMODITY EXCHANGE INC."] * len(dates),
            "Contract_Market_Name": ["ALUMINUM"] * len(dates),
            "CFTC_Contract_Market_Code": ["191691"] * len(dates),
            "Open_Interest_All": [1000] * len(dates),
            "M_Money_Positions_Long_All": np.arange(300, 300 + len(dates)),
            "M_Money_Positions_Short_All": np.arange(100, 100 + len(dates)),
            "M_Money_Positions_Spread_All": [50] * len(dates),
        }
    )
    master = calculate_cftc_positioning_metrics(resolve_canonical_contracts(normalize_cftc(raw, pd.DataFrame())))
    aluminum = cftc_asset_series(master, "Aluminum", "Managed Money")

    assert len(aluminum) == 143
    assert np.isfinite(aluminum.iloc[-1]["NetPctOI_4W_Change"])
    assert np.isfinite(aluminum.iloc[-1]["NetPctOI_13W_Change"])
    assert pd.isna(aluminum.iloc[-1]["NetPctOI_3Y_Percentile"])
    assert pd.isna(aluminum.iloc[-1]["NetPctOI_5Y_Percentile"])

    full_dates = pd.date_range("2021-01-05", periods=260, freq="W-TUE")
    full_raw = raw.iloc[np.zeros(260, dtype=int)].copy().reset_index(drop=True)
    full_raw["Report_Date_as_YYYY-MM-DD"] = full_dates
    full_raw["M_Money_Positions_Long_All"] = np.arange(300, 560)
    full_raw["M_Money_Positions_Short_All"] = np.arange(100, 360)
    full_master = calculate_cftc_positioning_metrics(
        resolve_canonical_contracts(normalize_cftc(full_raw, pd.DataFrame()))
    )
    full_aluminum = cftc_asset_series(full_master, "Aluminum", "Managed Money")
    assert pd.isna(full_aluminum.iloc[154]["NetPctOI_3Y_Percentile"])
    assert np.isfinite(full_aluminum.iloc[155]["NetPctOI_3Y_Percentile"])
    assert pd.isna(full_aluminum.iloc[258]["NetPctOI_5Y_Percentile"])
    assert np.isfinite(full_aluminum.iloc[259]["NetPctOI_5Y_Percentile"])


def test_normalize_tff_sp500_keeps_participant_definitions():
    dates = pd.date_range("2020-01-07", periods=60, freq="W-TUE")
    raw = pd.DataFrame(
        {
            "Report_Date_as_YYYY-MM-DD": dates,
            "Market_and_Exchange_Names": ["E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE"] * len(dates),
            "Contract_Market_Name": ["E-MINI S&P 500"] * len(dates),
            "CFTC_Contract_Market_Code": ["138741"] * len(dates),
            "Open_Interest_All": [2000] * len(dates),
            "Asset_Mgr_Positions_Long_All": np.arange(800, 860),
            "Asset_Mgr_Positions_Short_All": np.arange(200, 260),
            "Asset_Mgr_Positions_Spread_All": [20] * len(dates),
            "Lev_Money_Positions_Long_All": np.arange(500, 560),
            "Lev_Money_Positions_Short_All": np.arange(400, 460),
            "Lev_Money_Positions_Spread_All": [10] * len(dates),
        }
    )

    master = calculate_cftc_positioning_metrics(resolve_canonical_contracts(normalize_cftc(pd.DataFrame(), raw)))
    dashboard = cftc_dashboard_frame(master)

    assert "Asset Manager" in set(master["Participant_Category"])
    assert "Leveraged Money" in set(master["Participant_Category"])
    assert "Managed Money" not in set(master["Participant_Category"])
    assert "S&P 500" in set(dashboard["Canonical_Asset"])


def test_aaii_and_naaim_metrics_are_point_in_time():
    dates = pd.date_range("2020-01-02", periods=60, freq="W-THU")
    aaii = calculate_aaii_metrics(
        pd.DataFrame(
            {
                "Date": dates,
                "Bullish": np.linspace(20, 60, len(dates)),
                "Neutral": [30] * len(dates),
                "Bearish": np.linspace(50, 10, len(dates)),
            }
        )
    )
    naaim = calculate_naaim_metrics(pd.DataFrame({"Date": dates, "NAAIM Exposure": np.linspace(-20, 120, len(dates))}))

    assert np.isclose(aaii.iloc[-1]["AAII_BullBearSpread"], 50.0)
    assert np.isfinite(aaii.iloc[-1]["AAII_BullBearSpread_3Y_Percentile"])
    assert np.isfinite(naaim.iloc[-1]["NAAIM_3Y_Percentile"])
    assert np.isclose(naaim.iloc[-1]["NAAIM_4W_Change"], naaim.iloc[-1]["NAAIM_Exposure"] - naaim.iloc[-5]["NAAIM_Exposure"])


def test_aaii_metrics_parse_embedded_excel_header():
    raw = pd.DataFrame(
        [
            [None, None, "AAII header"],
            ["Reported", None, None],
            ["Date", "Bullish", "Neutral", "Bearish"],
            [pd.Timestamp("2024-01-04"), 0.40, 0.30, 0.30],
            [pd.Timestamp("2024-01-11"), 0.45, 0.25, 0.30],
        ]
    )

    metrics = calculate_aaii_metrics(raw)

    assert list(metrics["Date"]) == [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-11")]
    assert np.isclose(metrics.iloc[-1]["AAII_Bullish"], 45.0)
    assert np.isclose(metrics.iloc[-1]["AAII_Bearish"], 30.0)


def test_aaii_live_results_html_parser():
    html = """
    <table>
      <thead><tr><th>Date</th><th>Bullish</th><th>Neutral</th><th>Bearish</th></tr></thead>
      <tbody><tr><td>09/11/2025</td><td>28.0%</td><td>35.0%</td><td>37.0%</td></tr></tbody>
    </table>
    """

    parsed = parse_aaii_live_results_html(html)
    metrics = calculate_aaii_metrics(parsed)

    assert pd.Timestamp("2025-09-11") in set(metrics["Date"])
    assert np.isclose(metrics.iloc[-1]["AAII_Bearish"], 37.0)


def test_read_aaii_historical_workbook_promotes_sentiment_sheet(tmp_path):
    path = tmp_path / "aaii_historical.xlsx"
    raw = pd.DataFrame(
        [
            [None, None, "AAII header"],
            ["Reported", None, None],
            ["Date", "Bullish", "Neutral", "Bearish"],
            [pd.Timestamp("2024-01-04"), 0.40, 0.30, 0.30],
        ]
    )
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        raw.to_excel(writer, sheet_name="SENTIMENT", index=False, header=False)

    parsed = read_aaii_historical_workbook(path)

    assert parsed.shape[0] == 1
    assert pd.to_datetime(parsed.iloc[0]["Date"]) == pd.Timestamp("2024-01-04")


def test_export_chunks_large_cftc_master(monkeypatch):
    monkeypatch.setattr(positioning, "EXCEL_MAX_ROWS", 2)
    master = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=5, freq="W"),
            "Canonical_Asset": ["GOLD"] * 5,
            "Preferred_For_Dashboard": [True] * 5,
        }
    )

    payload = export_positioning_xlsx(master, pd.DataFrame(), pd.DataFrame(), {})
    sheets = pd.ExcelFile(BytesIO(payload)).sheet_names

    assert "CFTC_Master" in sheets
    assert "CFTC_Master_2" in sheets
    assert "CFTC_Master_3" in sheets
