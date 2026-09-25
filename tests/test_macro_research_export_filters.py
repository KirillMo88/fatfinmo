import pandas as pd

from macro_research_export import CORE_CFTC_EXPORT, SeriesMeta, filter_export_components


def test_export_filters_keep_date_and_selected_layer_aligned():
    dataset = pd.DataFrame({"Date": pd.date_range("2024-01-05", periods=2, freq="W-FRI"), "Raw": [1, 2], "Derived": [3, 4]})
    metadata = [
        SeriesMeta("Date", "METADATA", "Dataset", "date", "date", "calendar", "Internal", "weekly", "Primary key"),
        SeriesMeta("Raw", "RAW", "Market", "raw", "number", "source", "Internal", "weekly", "Market"),
        SeriesMeta("Derived", "DERIVED", "Model", "derived", "number", "formula", "Internal", "weekly", "Business Cycle"),
    ]

    filtered, filtered_metadata = filter_export_components(dataset, metadata, layers=["RAW"])

    assert list(filtered.columns) == ["Date", "Raw"]
    assert [meta.column for meta in filtered_metadata] == ["Date", "Raw"]


def test_export_filters_support_explicit_all_and_model_usage():
    dataset = pd.DataFrame({"Date": pd.date_range("2024-01-05", periods=1, freq="W-FRI"), "Raw": [1], "Derived": [2]})
    metadata = [
        SeriesMeta("Date", "METADATA", "Dataset", "date", "date", "calendar", "Internal", "weekly", "Primary key"),
        SeriesMeta("Raw", "RAW", "Market", "raw", "number", "source", "Internal", "weekly", "Market"),
        SeriesMeta("Derived", "DERIVED", "Model", "derived", "number", "formula", "Internal", "weekly", "Business Cycle"),
    ]

    all_data, _ = filter_export_components(dataset, metadata, layers=["All"], model_usages=["All"])
    model_data, model_metadata = filter_export_components(dataset, metadata, model_usages=["Business Cycle"])

    assert list(all_data.columns) == ["Date", "Raw", "Derived"]
    assert list(model_data.columns) == ["Date", "Derived"]
    assert [meta.column for meta in model_metadata] == ["Date", "Derived"]


def test_export_filters_support_category():
    dataset = pd.DataFrame({"Date": pd.date_range("2024-01-05", periods=1, freq="W-FRI"), "Market": [1], "Liquidity": [2]})
    metadata = [
        SeriesMeta("Date", "METADATA", "Dataset", "date", "date", "calendar", "Internal", "weekly", "Primary key"),
        SeriesMeta("Market", "RAW", "Market OHLC", "market", "number", "source", "Internal", "weekly", "Market"),
        SeriesMeta("Liquidity", "RAW", "Global Liquidity", "liquidity", "number", "source", "Internal", "weekly", "Global Liquidity"),
    ]

    filtered, filtered_metadata = filter_export_components(dataset, metadata, categories=["Market OHLC"])

    assert list(filtered.columns) == ["Date", "Market"]
    assert [meta.column for meta in filtered_metadata] == ["Date", "Market"]


def test_cftc_participant_series_are_included_in_macro_export():
    expected = {
        ("S&P 500", "Leveraged Money", "SP500_LM"),
        ("NASDAQ-100", "Leveraged Money", "NASDAQ100_LM"),
        ("BTC", "Asset Manager", "BTC_AM"),
    }

    assert expected.issubset(set(CORE_CFTC_EXPORT))
