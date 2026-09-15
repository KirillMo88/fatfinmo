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
    resolve_canonical_contracts,
    validate_cftc_master,
)


def test_normalize_disaggregated_gold_and_calculate_metrics():
    dates = pd.date_range("2020-01-07", periods=60, freq="W-TUE")
    raw = pd.DataFrame(
        {
            "Report_Date_as_YYYY-MM-DD": dates,
            "Market_and_Exchange_Names": ["GOLD - COMMODITY EXCHANGE INC."] * len(dates),
            "Contract_Market_Name": ["GOLD"] * len(dates),
            "CFTC_Contract_Market_Code": ["088691"] * len(dates),
            "Open_Interest_All": [1000] * len(dates),
            "M_Money_Positions_Long_All": np.arange(300, 360),
            "M_Money_Positions_Short_All": np.arange(100, 160),
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
    assert not validate_cftc_master(master)


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
