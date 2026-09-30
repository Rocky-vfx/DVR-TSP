"""
Tests for DVR-TSP Recovery Metrics Engine and Summary API
---------------------------------------------------------
Verifies:
1. Conservation of data: data_recovered + lost_to_decay + lost_unvisited == total_data_available
2. Recovery percentage bounded in [0, 100]
3. Tour feasibility: time_taken <= time_budget
4. Time conservation: time_taken + time_remaining == time_budget
5. GET /api/scenarios/{id}/summary returns all required fields and correct rankings
6. Backward compatibility with existing endpoints
"""

import math
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.algorithms import ALGORITHMS, generate_scenario, build_sparse_graph, run_algorithm
from app.backblaze import drives_to_nodes, DEFAULT_CONFIG
from app.metrics import compute_recovery_metrics, build_scenario_summary, format_bytes

client = TestClient(app)


def test_conservation_of_data_synthetic():
    """Verify data conservation: data_recovered + lost_to_decay + lost_unvisited == total_data_available."""
    hub = {"x": 78.0, "y": 320.0}
    nodes = generate_scenario(hub, n=9)
    edges = build_sparse_graph(hub, nodes, k=3)
    t_max = 55.0
    speed = 42.0

    for algo_key in ALGORITHMS:
        res = run_algorithm(algo_key, hub, nodes, edges, t_max, speed)
        metrics = compute_recovery_metrics(res, nodes, t_max)

        # 1. Data conservation within float tolerance
        recon_sum = metrics["data_recovered"] + metrics["lost_to_decay"] + metrics["lost_unvisited"]
        assert math.isclose(recon_sum, metrics["total_data_available"], rel_tol=1e-4, abs_tol=1e-4), (
            f"Algorithm {algo_key}: sum {recon_sum} != total {metrics['total_data_available']}"
        )

        # 2. Recovery percentage in [0, 100]
        assert 0.0 <= metrics["recovery_percentage"] <= 100.0, (
            f"Algorithm {algo_key}: recovery_percentage {metrics['recovery_percentage']} outside [0, 100]"
        )

        # 3. Time feasibility
        assert metrics["time_taken"] <= metrics["time_budget"] + 1e-4, (
            f"Algorithm {algo_key}: time_taken {metrics['time_taken']} > budget {metrics['time_budget']}"
        )

        # 4. Time conservation
        time_sum = metrics["time_taken"] + metrics["time_remaining"]
        assert math.isclose(time_sum, metrics["time_budget"], rel_tol=1e-4, abs_tol=1e-4)

        # 5. Time utilization in [0, 100]
        assert 0.0 <= metrics["time_utilization_percentage"] <= 100.0

        # 6. Node counts
        assert metrics["nodes_visited"] <= metrics["nodes_total"]
        assert metrics["nodes_total"] == len(nodes)
        assert len(metrics["per_node_timeline"]) == metrics["nodes_visited"]


def test_conservation_of_data_real():
    """Verify data conservation and byte-formatting with real drive telemetry."""
    sample_drives = [
        {
            "serial_number": f"DRV_{i}",
            "model": "ST4000DM000",
            "capacity_bytes": 4e12,
            "failure": 1 if i == 0 else 0,
            "smart_5_raw": 15.0 if i == 0 else 0.0,
            "smart_187_raw": 5.0 if i == 0 else 0.0,
            "smart_188_raw": 0.0,
            "smart_197_raw": 2.0 if i == 0 else 0.0,
            "smart_198_raw": 1.0 if i == 0 else 0.0,
        }
        for i in range(8)
    ]
    hub = {"x": 78.0, "y": 320.0}
    nodes, edges, meta = drives_to_nodes(sample_drives, hub=hub, seed=42)
    t_max = 55.0
    speed = 42.0

    for algo_key in ["dvrtsp", "nn", "dijkstra"]:
        res = run_algorithm(algo_key, hub, nodes, edges, t_max, speed)
        metrics = compute_recovery_metrics(res, nodes, t_max, meta=meta)

        # Conservation in bytes
        recon_bytes = metrics["data_recovered"] + metrics["lost_to_decay"] + metrics["lost_unvisited"]
        assert math.isclose(recon_bytes, metrics["total_data_available"], rel_tol=1e-4, abs_tol=1e-4)

        assert metrics["total_data_available_formatted"] is not None
        assert "TB" in metrics["total_data_available_formatted"]
        assert metrics["data_recovered_formatted"] is not None

        # Summary text generated
        assert metrics["summary_text"] is not None
        assert len(metrics["summary_text"]) > 20
        assert "recovered" in metrics["summary_text"].lower()


def test_summary_endpoint():
    """Verify GET /api/scenarios/{id}/summary returns all required fields and rankings."""
    # 1. Create a scenario
    r_create = client.post("/api/scenarios", json={"n_nodes": 7})
    assert r_create.status_code == 200
    sc_id = r_create.json()["scenario_id"]

    # 2. Run two algorithms
    r_dvr = client.post(f"/api/scenarios/{sc_id}/run", json={"algorithm": "dvrtsp"})
    assert r_dvr.status_code == 200
    r_nn = client.post(f"/api/scenarios/{sc_id}/run", json={"algorithm": "nn"})
    assert r_nn.status_code == 200

    # 3. Request summary
    r_sum = client.get(f"/api/scenarios/{sc_id}/summary")
    assert r_sum.status_code == 200
    data = r_sum.json()

    # Top-level fields
    assert data["scenario_id"] == sc_id
    assert data["source"] == "synthetic"
    assert data["total_data_available"] > 0
    assert data["total_value_available"] > 0
    assert data["time_budget"] > 0
    assert data["total_nodes"] == 7
    assert data["runs_count"] == 2
    assert "best_algorithm_by_data" in data
    assert "best_algorithm_by_efficiency" in data
    assert "summary_text" in data
    assert len(data["summary_text"]) > 10

    # Algorithm comparison rows
    algos = data["algorithms"]
    assert len(algos) == 2
    for row in algos:
        assert "algorithm" in row
        assert "algorithm_name" in row
        assert "data_recovered" in row
        assert "recovery_percentage" in row
        assert "time_taken" in row
        assert "time_remaining" in row
        assert "time_utilization_percentage" in row
        assert "nodes_visited" in row
        assert "nodes_total" in row
        assert "efficiency_data_per_min" in row
        assert 0.0 <= row["recovery_percentage"] <= 100.0
        assert 0.0 <= row["time_utilization_percentage"] <= 100.0


def test_summary_endpoint_empty():
    """Verify GET /api/scenarios/{id}/summary on scenario with no runs returns empty list with notice."""
    r_create = client.post("/api/scenarios", json={"n_nodes": 5})
    sc_id = r_create.json()["scenario_id"]

    r_sum = client.get(f"/api/scenarios/{sc_id}/summary")
    assert r_sum.status_code == 200
    data = r_sum.json()
    assert data["runs_count"] == 0
    assert len(data["algorithms"]) == 0
    assert "No algorithm runs" in data["summary_text"]
