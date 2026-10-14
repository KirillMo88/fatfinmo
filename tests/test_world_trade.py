from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from world_trade import ChinaExportGeographyService, GlobalTradeService, REGION_ORDER, TradeBalanceService
from world_trade_tab import CHINA_REGION_COLORS, _ranked_region_order, _render_china_stacked_columns


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Cp1252Response(_Response):
    def __init__(self, payload):
        super().__init__(payload)
        self.content = json.dumps(payload, ensure_ascii=False).encode("cp1252")

    def json(self):
        raise UnicodeDecodeError("utf-8", self.content, 0, 1, "test invalid byte")


class _WdiClient:
    def get(self, url, params=None, **kwargs):
        code = url.rsplit("/", 1)[-1]
        years = range(2000, datetime.now(timezone.utc).year)
        rows = []
        for offset, year in enumerate(years):
            if code == "NE.EXP.GNFS.KD":
                value = 100.0 * (1.1 ** offset)
            elif code == "NE.EXP.GNFS.CD":
                value = 1e12 + offset * 1e11
            elif code == "NY.GDP.MKTP.KD.ZG":
                value = 2.0
            else:
                value = 5e12 + offset * 2e11
            rows.append({"date": str(year), "value": value})
        return _Response([{"lastupdated": "2026-01-01"}, rows])


class _WtoClient:
    def __init__(self, response_type=_Response):
        self.calls = []
        self.response_type = response_type

    def get(self, url, params=None, headers=None, **kwargs):
        self.calls.append({"url": url, "params": params, "headers": headers})
        if url.endswith("/reporters"):
            return self.response_type([
                {"code": f"X{index:03d}", "iso3A": f"X{index:03d}", "name": f"Economy {index:03d}", "displayOrder": index}
                for index in range(159)
            ] + [{"code": "AAA", "iso3A": "AAA", "name": "Economy – A", "displayOrder": 159}])
        value_field = "exports" if params["i"] == TradeBalanceService.EXPORT_CODE else "imports"
        values = {"exports": (100.0, 100.0), "imports": (90.0, 90.0)}[value_field]
        return self.response_type([
            {
                "year": 2025, "reportingEconomy": "Economy – A", "reportingEconomyCode": "AAA",
                "partnerEconomyCode": "000", "productOrSectorCode": "TO", "value": values[0],
            },
            {
                "year": 2025, "reportingEconomy": "World", "reportingEconomyCode": "000",
                "partnerEconomyCode": "000", "productOrSectorCode": "TO", "value": values[1],
            },
        ])


def test_global_trade_calculations_exclude_current_year(tmp_path):
    service = GlobalTradeService(tmp_path, client=_WdiClient())
    result = service.load(force_refresh=True)
    frame = result["data"]
    assert frame["year"].max() == datetime.now(timezone.utc).year - 1
    assert frame["real_exports_growth_pct"].dropna().iloc[0] == pytest.approx(10.0)
    sixth = frame.loc[frame["year"].eq(2005)].iloc[0]
    assert sixth["real_exports_growth_5y_avg_pct"] == pytest.approx(10.0)
    assert sixth["exports_gdp_pct"] == pytest.approx(sixth["nominal_exports_usd"] / sixth["nominal_gdp_usd"] * 100)


def _wto_fixture(years=(2015, 2025), economies=160):
    rows = []
    for year in years:
        for index in range(economies):
            rows.append({
                "year": year, "economy": f"Economy {index:03d}", "economy_code": f"X{index:03d}",
                "exports_usd_mn": 10.0 + index, "imports_usd_mn": 9.0 + index,
            })
        rows.append({
            "year": year, "economy": "World", "economy_code": "000",
            "exports_usd_mn": sum(10.0 + index for index in range(economies)),
            "imports_usd_mn": sum(9.0 + index for index in range(economies)),
        })
    return pd.DataFrame(rows)


def test_trade_balance_validation_and_period_selection(tmp_path):
    service = TradeBalanceService(tmp_path)
    validated, metadata = service.validate(_wto_fixture())
    assert metadata["latest_valid_year"] == 2025
    assert metadata["ten_year_comparison_year"] == 2015
    assert len(validated.loc[validated["year"].eq(2025)]) == 160
    assert "World" not in set(validated["economy"])
    assert validated["trade_balance_usd_bn"].iloc[0] == pytest.approx(0.001)
    json.dumps(metadata)


def test_trade_balance_api_uses_official_data_endpoint_and_total_merchandise(tmp_path, monkeypatch):
    monkeypatch.setenv("WTO_API_KEY", "test-key")
    client = _WtoClient()
    frame = TradeBalanceService(tmp_path, client=client)._download_api()

    assert len(client.calls) == 3
    reporter_call = client.calls[0]
    assert reporter_call["url"].endswith("/timeseries/v1/reporters")
    assert reporter_call["params"]["ig"] == "individual"
    data_calls = [call for call in client.calls if call["url"].endswith("/timeseries/v1/data")]
    assert {call["params"]["i"] for call in data_calls} == {
        TradeBalanceService.EXPORT_CODE, TradeBalanceService.IMPORT_CODE,
    }
    for call in data_calls:
        assert call["params"]["r"] == "all"
        assert call["params"]["p"] == "000"
        assert call["params"]["pc"] == "TO"
        assert call["params"]["head"] == "M"
        assert call["params"]["max"] == 1_000_000
        assert call["headers"] == {"Ocp-Apim-Subscription-Key": "test-key"}
    assert frame.loc[frame["economy_code"].eq("AAA"), "exports_usd_mn"].iloc[0] == pytest.approx(100.0)
    assert frame.loc[frame["economy_code"].eq("AAA"), "imports_usd_mn"].iloc[0] == pytest.approx(90.0)
    assert frame.loc[frame["economy_code"].eq("0"), "economy"].iloc[0] == "World"
    assert not frame.loc[frame["economy_code"].eq("0"), "is_individual_economy"].iloc[0]


def test_trade_balance_api_accepts_wto_cp1252_json(tmp_path, monkeypatch):
    monkeypatch.setenv("WTO_API_KEY", "test-key")
    frame = TradeBalanceService(tmp_path, client=_WtoClient(_Cp1252Response))._download_api()
    assert frame.loc[frame["economy_code"].eq("AAA"), "economy"].iloc[0] == "Economy – A"


def test_trade_balance_missing_key_warning_is_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("WTO_API_KEY", raising=False)
    result = TradeBalanceService(tmp_path).load(force_refresh=True)
    assert result["status"] == "unavailable"
    assert "WTO_API_KEY" in result["warning"]


def test_trade_balance_rejects_incomplete_world_coverage(tmp_path):
    frame = _wto_fixture(economies=149)
    with pytest.raises(ValueError, match="No completed year"):
        TradeBalanceService(tmp_path).validate(frame)


def test_trade_balance_rejects_current_year_only(tmp_path):
    frame = _wto_fixture(years=(datetime.now(timezone.utc).year,), economies=160)
    with pytest.raises(ValueError, match="No completed year"):
        TradeBalanceService(tmp_path).validate(frame)


def test_trade_balance_top_five_each_side_is_dynamic():
    frame = pd.DataFrame({
        "year": [2025] * 14,
        "economy": [f"Country {index}" for index in range(14)],
        "exports_usd_mn": [10_000.0] * 14,
        "imports_usd_mn": [10_000.0] * 14,
        "trade_balance_usd_bn": [-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7],
    })
    selected = TradeBalanceService.top_balances(frame, 2025)
    assert len(selected) == 10
    assert set(selected["trade_balance_usd_bn"]) == {-7, -6, -5, -4, -3, 3, 4, 5, 6, 7}


def test_trade_balance_excludes_economic_groups_from_validation_and_rankings(tmp_path):
    frame = _wto_fixture()
    group_names = [
        "BRICS members", "BRIC members", "OPEC (Organization of the Petroleum Exporting Countries)",
        "Gulf Cooperation Council (GCC)", "South Asian Association for Regional Cooperation (SAARC)",
        "G-20", "United States-Mexico-Canada Agreement (USMCA/CUSMA/T-MEC)", "G-7 (Group of Seven)",
    ]
    groups = []
    for year in (2015, 2025):
        for index, name in enumerate(group_names):
            groups.append({
                "year": year,
                "economy": name,
                "economy_code": f"GROUP{index}",
                "exports_usd_mn": 2_000_000.0 + index * 100_000,
                "imports_usd_mn": 100_000.0,
            })
    frame = pd.concat([frame, pd.DataFrame(groups)], ignore_index=True)

    validated, metadata = TradeBalanceService(tmp_path).validate(frame)
    ranked = TradeBalanceService.top_balances(validated, metadata["latest_valid_year"])

    assert not set(group_names).intersection(validated["economy"])
    assert not set(group_names).intersection(ranked["economy"])
    assert len(validated.loc[validated["year"].eq(2025)]) == 160


def test_china_historical_seed_reconciles(tmp_path):
    service = ChinaExportGeographyService(tmp_path)
    service._refresh_from_raw = lambda: False
    result = service.load()
    history = result["data"]
    assert set(history["year"]) == {2000, 2010, 2020, 2025}
    assert result["status"] == "fallback"
    assert result["metadata"]["latest_valid_year"] == 2025
    assert result["metadata"]["latest_data_is_fallback"] is True
    assert list(history.loc[history["year"].eq(2000)].sort_values("region")["region"].sort_values()) == sorted(REGION_ORDER)
    for _, period in history.groupby("year"):
        assert period["share_pct"].sum() == pytest.approx(100.0, abs=0.05)


def test_china_2025_seed_fallback_matches_reference():
    seed = pd.read_csv("data/china_export_regions_2000_2010_2020_2025_full.csv")
    expected = {
        "ASEAN Plus": 31.5, "EU": 14.8, "US": 11.1, "Americas ex US": 9.2,
        "CIS": 4.9, "Middle East": 6.3, "Africa": 6.0, "Other": 7.3, "Hong Kong": 8.9,
    }
    actual = seed.loc[seed["Region"].ne("TOTAL")].set_index("Region")["2025 share %"]
    for region, reference in expected.items():
        assert actual[region] == pytest.approx(reference, abs=0.1)


def test_china_regions_are_ranked_from_largest_to_smallest_for_selected_period():
    seed = pd.read_csv("data/china_export_regions_2000_2010_2020_2025_full.csv")
    frame = pd.DataFrame({
        "year": 2025,
        "region": seed.loc[seed["Region"].ne("TOTAL"), "Region"],
        "share_pct": seed.loc[seed["Region"].ne("TOTAL"), "2025 share %"],
    })

    order = _ranked_region_order(frame, 2025, "share_pct")

    assert order == [
        "ASEAN Plus", "EU", "US", "Americas ex US", "Hong Kong",
        "Other", "Middle East", "Africa", "CIS",
    ]
    assert CHINA_REGION_COLORS["ASEAN Plus"] == "#5f8fae"


def test_china_historical_cache_migrates_to_2025_fallback(tmp_path):
    service = ChinaExportGeographyService(tmp_path)
    service.history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "year": [2000], "region": ["US"], "exports_usd_bn": [1.0],
        "share_pct": [1.0], "source": ["old cache"],
    }).to_csv(service.history_path, index=False)
    migrated = service._historical()
    assert set(migrated["year"]) == {2000, 2010, 2020, 2025}
    assert len(migrated.loc[migrated["year"].eq(2025)]) == len(REGION_ORDER)


@pytest.mark.parametrize(
    ("code", "name", "expected"),
    [
        ("USA", "", "US"), ("HKG", "", "Hong Kong"), ("DEU", "", "EU"),
        ("VNM", "", "ASEAN Plus"), ("RUS", "", "CIS"), ("ARE", "", "Middle East"),
        ("ZAF", "", "Africa"), ("BRA", "", "Americas ex US"), ("GBR", "", "Other"),
        ("UKR", "", "Other"), ("GEO", "", "Other"),
    ],
)
def test_china_destination_mapping(code, name, expected):
    assert ChinaExportGeographyService.classify_destination(code, name) == expected


def test_china_region_validation_has_no_double_counting(tmp_path):
    year = datetime.now(timezone.utc).year - 1
    frame = pd.DataFrame({
        "year": [year] * len(REGION_ORDER), "region": REGION_ORDER,
        "destination": [None] * len(REGION_ORDER), "iso3": [""] * len(REGION_ORDER),
        "exports_usd_bn": np.arange(1, len(REGION_ORDER) + 1, dtype=float),
    })
    validated, metadata = ChinaExportGeographyService(tmp_path).validate(frame)
    assert metadata["latest_valid_year"] == year
    assert validated["exports_usd_bn"].sum() == pytest.approx(frame["exports_usd_bn"].sum())
    assert validated["share_pct"].sum() == pytest.approx(100.0)


def test_gacc_parser_reads_year_from_filename_and_explicit_usd_units(tmp_path):
    source = pd.DataFrame({"Region": REGION_ORDER, "Exports USD mn": np.arange(1, 10) * 1_000.0})
    parsed = ChinaExportGeographyService(tmp_path)._normalise_gacc(source, source_name="gacc_country_2025.csv")
    assert set(parsed["year"]) == {2025}
    assert parsed["exports_usd_bn"].sum() == pytest.approx(45.0)


def test_gacc_cny_parser_reconciles_to_official_usd_total(tmp_path):
    source = pd.DataFrame({
        "Year": [2025] * len(REGION_ORDER), "Region": REGION_ORDER,
        "Exports CNY": np.arange(1, 10) * 100.0,
    })
    parsed = ChinaExportGeographyService(tmp_path)._normalise_gacc(
        source, source_name="gacc_country_2025.csv", annual_totals={2025: 3_771.9},
    )
    assert parsed["exports_usd_bn"].sum() == pytest.approx(3_771.9)


def test_china_stacked_columns_use_all_table_periods(tmp_path, monkeypatch):
    service = ChinaExportGeographyService(tmp_path)
    history = service._historical()
    captured = {}
    monkeypatch.setattr("world_trade_tab.st.plotly_chart", lambda figure, **_: captured.setdefault("figure", figure))
    monkeypatch.setattr("world_trade_tab.st.caption", lambda *_, **__: None)
    _render_china_stacked_columns(history, 2025, 2025, "Share %")
    figure = captured["figure"]
    assert figure.layout.barmode == "stack"
    assert len(figure.data) == len(REGION_ORDER)
    assert list(figure.data[0].x) == ["2000", "2010", "2020", "2025"]
    for column_index in range(4):
        assert sum(float(trace.y[column_index]) for trace in figure.data) == pytest.approx(100.0, abs=0.05)
