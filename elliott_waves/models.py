from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Pivot:
    pivot_id: str
    pivot_time: str
    confirmed_at: str | None
    known_at: str
    price: float
    kind: str
    status: str
    source_timeframe: str
    k: float
    atr_reference: float | None
    evidence_bar_ids: list[str] = field(default_factory=list)
    bar_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuleCheck:
    rule_id: str
    rule_class: str
    result: str
    observed: Any = None
    expected: Any = None
    point_ids: list[str] = field(default_factory=list)
    interval: list[str] = field(default_factory=list)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Target:
    target_id: str
    node_id: str
    role: str
    coefficient: float
    price_low: float
    price_high: float
    center: float
    issued_at: str
    known_at: str
    anchor_ids: list[str]
    reference_wave_id: str | None = None
    status: str = "ACTIVE"
    target_time: None = None
    invalidation_scope: str | None = None
    exclusion_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WaveNode:
    node_id: str
    pattern_type: str
    subtype: str | None
    profile_id: str
    direction: int
    orientation_direction: int
    relative_degree: str
    start_point: dict[str, Any]
    end_point: dict[str, Any]
    internal_high: float
    internal_low: float
    extreme_times: dict[str, str]
    source_bar_range: list[str]
    source_timeframes: list[str]
    duration_bars: int
    duration_calendar: int
    endpoint_status: str
    geometry_status: str
    context_status: str
    subdivision_status: str
    verified_depth: int
    verification_coverage: float
    known_at: str
    first_observed_at: str
    last_updated_at: str
    version: int = 1
    scenario_ids: list[str] = field(default_factory=list)
    segmentation_evidence: list[dict[str, Any]] = field(default_factory=list)
    role_in_parent: str | None = None
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    labels: list[dict[str, Any]] = field(default_factory=list)
    ratios: list[dict[str, Any]] = field(default_factory=list)
    rule_checks: list[dict[str, Any]] = field(default_factory=list)
    unknown_requirements: list[str] = field(default_factory=list)
    invalidation: dict[str, Any] | None = None
    targets: list[dict[str, Any]] = field(default_factory=list)
    channels: list[dict[str, Any]] = field(default_factory=list)
    pivot_stream_k: float | None = None
    start_pivot_index: int | None = None
    end_pivot_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Scenario:
    scenario_id: str
    root_node_id: str
    status: str
    rank: int
    selection_label: str
    coverage: float
    forming_coverage: float
    unknown_check_count: int
    complexity_units: int
    exception_units: int
    fib_fit: float | None
    fib_fit_mask: list[str]
    reason_selected: str
    last_change_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
