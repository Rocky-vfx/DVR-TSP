from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NodeOut(BaseModel):
    id: str
    idx: int
    x: float
    y: float
    value: float
    corruption: float
    recoveryTime: float


class EdgeOut(BaseModel):
    a: str
    b: str
    w: float


class HubOut(BaseModel):
    x: float
    y: float


class ScenarioCreate(BaseModel):
    n_nodes: int = Field(default=9, ge=3, le=20)


class ScenarioOut(BaseModel):
    scenario_id: int
    hub: HubOut
    nodes: List[NodeOut]
    edges: List[EdgeOut]
    t_max: float
    speed: float
    source: Optional[str] = "synthetic"
    meta: Optional[Dict[str, Any]] = None


class RunCreate(BaseModel):
    algorithm: str


class RunStepOut(BaseModel):
    id: str
    reached: bool
    arrival: Optional[float] = None
    value: Optional[float] = None
    finish: Optional[float] = None


# Per-node timeline entry for recovery metrics
class NodeTimelineEntry(BaseModel):
    node_id: str
    arrival_time: float
    finish_time: float
    value_at_arrival: float
    bytes_recovered: float
    bytes_lost_to_decay: float


# Recovery metrics block (used in RunOut and summary)
class RecoveryMetrics(BaseModel):
    total_data_available: Optional[float] = None
    total_data_available_formatted: Optional[str] = None
    total_value_available: Optional[float] = None
    data_recovered: Optional[float] = None
    data_recovered_formatted: Optional[str] = None
    value_recovered: Optional[float] = None
    data_lost: Optional[float] = None
    lost_to_decay: Optional[float] = None
    lost_unvisited: Optional[float] = None
    recovery_percentage: Optional[float] = None
    time_budget: Optional[float] = None
    time_taken: Optional[float] = None
    time_remaining: Optional[float] = None
    time_utilization_percentage: Optional[float] = None
    travel_time: Optional[float] = None
    recovery_time: Optional[float] = None
    nodes_visited: Optional[int] = None
    nodes_total: Optional[int] = None
    per_node_timeline: Optional[List[NodeTimelineEntry]] = None


class RunOut(BaseModel):
    run_id: int
    scenario_id: int
    algorithm: str
    algorithm_name: str
    totalValue: float
    totalTime: float
    count: int
    order: List[str]
    steps: List[RunStepOut]
    # New recovery metrics (optional for backward compat)
    metrics: Optional[RecoveryMetrics] = None
    summary_text: Optional[str] = None


class RunSummary(BaseModel):
    run_id: int
    algorithm: str
    algorithm_name: str
    totalValue: float
    totalTime: float
    count: int
    created_at: Optional[str] = None
    # New recovery metrics (optional)
    recovery_percentage: Optional[float] = None
    data_recovered: Optional[float] = None
    nodes_visited: Optional[int] = None
    nodes_total: Optional[int] = None
    time_budget: Optional[float] = None


class AlgorithmInfo(BaseModel):
    key: str
    name: str


class BenchmarkRow(BaseModel):
    algorithm: str
    algorithm_name: str
    avg_value: float
    avg_time: float
    avg_compute_ms: float


# ---------------------------------------------------------------------------
# Backblaze Real-Data Mode Schemas
# ---------------------------------------------------------------------------

class IncidentDriveOut(BaseModel):
    serial_number: str
    model: str
    capacity_bytes: float
    failure: int
    smart_5_raw: float
    smart_187_raw: float
    smart_188_raw: float
    smart_197_raw: float
    smart_198_raw: float
    smart_9_raw: Optional[float] = None


class IncidentOut(BaseModel):
    incident_id: str
    date: str
    cluster_size: int
    degraded_count: int
    model_summary: List[str]
    drives: List[IncidentDriveOut]


class RealScenarioCreate(BaseModel):
    incident_id: Optional[str] = None
    date: Optional[str] = None
    serials: Optional[List[str]] = None
    cluster_size: int = Field(default=9, ge=3, le=20)
    topology: str = Field(default="random_geometric")
    seed: int = Field(default=42)


class ReplayRequest(BaseModel):
    start: str
    end: str
    step_days: int = Field(default=1, ge=1, le=30)
    algorithms: List[str] = Field(default_factory=lambda: ["dvrtsp", "nn", "dijkstra", "astar", "bfs"])
    cluster_size: int = Field(default=9, ge=3, le=20)
    topology: str = Field(default="random_geometric")
    seed: int = Field(default=42)


class ReplayDayStep(BaseModel):
    date: str
    scenario_id: int
    degraded_count: int
    results: Dict[str, Dict[str, Any]]


class ReplayOut(BaseModel):
    total_days: int
    steps: List[ReplayDayStep]


class ExperimentOut(BaseModel):
    status: str
    incidents_analyzed: int
    cluster_sizes: List[int]
    summary_table: List[Dict[str, Any]]
    wilcoxon_tests: Dict[str, Any]
    validation_analysis: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]
    artifacts_saved: List[str]


# ---------------------------------------------------------------------------
# Scenario Summary (new endpoint)
# ---------------------------------------------------------------------------

class ComparisonEntry(BaseModel):
    algorithm: str
    algorithm_name: str
    data_recovered: float
    data_recovered_formatted: Optional[str] = None
    recovery_percentage: float
    value_recovered: float
    time_taken: float
    time_budget: float
    time_remaining: Optional[float] = None
    time_utilization_percentage: Optional[float] = None
    nodes_visited: int
    nodes_total: int
    data_per_minute: float
    efficiency_data_per_min: Optional[float] = None


class ScenarioSummaryOut(BaseModel):
    scenario_id: int
    source: Optional[str] = "synthetic"
    total_data_available: Optional[float] = None
    total_data_available_formatted: Optional[str] = None
    total_value_available: Optional[float] = None
    time_budget: Optional[float] = None
    total_nodes: Optional[int] = None
    runs_count: int = 0
    comparison_table: List[ComparisonEntry] = []
    algorithms: Optional[List[ComparisonEntry]] = None
    best_by_data_recovered: Optional[ComparisonEntry] = None
    best_by_time_efficiency: Optional[ComparisonEntry] = None
    best_algorithm_by_data: Optional[str] = None
    best_algorithm_by_efficiency: Optional[str] = None
    overall_summary: str
    summary_text: Optional[str] = None

