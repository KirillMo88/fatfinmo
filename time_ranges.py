"""Shared dashboard time-range options."""

STANDARD_TIME_RANGE_OPTIONS = ("1Y", "3Y", "5Y", "10Y", "20Y", "MAX")
TIME_RANGE_YEARS = {
    "1Y": 1,
    "3Y": 3,
    "5Y": 5,
    "10Y": 10,
    "20Y": 20,
}


def normalize_time_range_choice(value: object, default: str = "5Y") -> str:
    """Map legacy full-history labels and invalid session values to the shared set."""
    if value in {"FULL", "Full"}:
        return "MAX"
    return str(value) if value in STANDARD_TIME_RANGE_OPTIONS else default

