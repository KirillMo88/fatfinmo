from time_ranges import STANDARD_TIME_RANGE_OPTIONS, TIME_RANGE_YEARS, normalize_time_range_choice


def test_standard_time_range_contract() -> None:
    assert STANDARD_TIME_RANGE_OPTIONS == ("1Y", "3Y", "5Y", "10Y", "20Y", "MAX")
    assert TIME_RANGE_YEARS == {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "20Y": 20}


def test_legacy_full_history_values_normalize_to_max() -> None:
    assert normalize_time_range_choice("FULL") == "MAX"
    assert normalize_time_range_choice("Full") == "MAX"
    assert normalize_time_range_choice("2015 -> Latest", "20Y") == "20Y"
