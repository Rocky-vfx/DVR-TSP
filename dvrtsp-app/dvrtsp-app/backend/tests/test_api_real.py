import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_existing_synthetic_endpoints_preserved():
    """Verify standard synthetic endpoints function completely unchanged."""
    r_create = client.post("/api/scenarios", json={"n_nodes": 7})
    assert r_create.status_code == 200
    sc_data = r_create.json()
    assert sc_data["source"] == "synthetic"
    assert len(sc_data["nodes"]) == 7
    sc_id = sc_data["scenario_id"]

    # Run algorithm
    r_run = client.post(f"/api/scenarios/{sc_id}/run", json={"algorithm": "dvrtsp"})
    assert r_run.status_code == 200
    assert r_run.json()["totalValue"] > 0

    # List runs
    r_runs = client.get(f"/api/scenarios/{sc_id}/runs")
    assert r_runs.status_code == 200
    assert len(r_runs.json()) == 1


def test_real_incidents_endpoint():
    """Verify querying real Backblaze incidents."""
    r = client.get("/api/real/incidents?cluster_size=9&min_degraded=2&seed=42")
    assert r.status_code == 200
    incidents = r.json()
    assert isinstance(incidents, list)
    assert len(incidents) > 0
    inc = incidents[0]
    assert "incident_id" in inc
    assert "date" in inc
    assert inc["cluster_size"] == 9
    assert len(inc["drives"]) == 9


def test_real_scenarios_endpoint():
    """Verify creating a DVR-TSP scenario from real Backblaze drive telemetry."""
    # 1. Fetch available incidents
    r_inc = client.get("/api/real/incidents?cluster_size=9")
    incidents = r_inc.json()
    target_inc_id = incidents[0]["incident_id"]

    # 2. Create scenario from incident
    r_create = client.post("/api/real/scenarios", json={
        "incident_id": target_inc_id,
        "topology": "ring",
        "seed": 42,
    })
    assert r_create.status_code == 200
    data = r_create.json()
    assert data["source"] == "backblaze"
    assert data["meta"]["topology_modeled"] is True
    assert data["meta"]["topology"] == "ring"
    assert len(data["nodes"]) == 9
    scenario_id = data["scenario_id"]

    # 3. Run DVR-TSP on real scenario using existing run endpoint
    r_run = client.post(f"/api/scenarios/{scenario_id}/run", json={"algorithm": "dvrtsp"})
    assert r_run.status_code == 200
    run_data = r_run.json()
    assert run_data["totalValue"] > 0
    assert run_data["count"] > 0

    # 4. Run baseline on same scenario
    r_nn = client.post(f"/api/scenarios/{scenario_id}/run", json={"algorithm": "nn"})
    assert r_nn.status_code == 200

    # 5. List runs for this real scenario
    r_runs = client.get(f"/api/scenarios/{scenario_id}/runs")
    assert r_runs.status_code == 200
    assert len(r_runs.json()) == 2


def test_real_replay_endpoint():
    """Verify multi-day sequential replay endpoint."""
    payload = {
        "start": "2024-01-01",
        "end": "2024-01-02",
        "step_days": 1,
        "algorithms": ["dvrtsp", "nn"],
        "cluster_size": 9,
    }
    r = client.post("/api/real/replay", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "total_days" in data
    assert len(data["steps"]) >= 1
    step0 = data["steps"][0]
    assert "results" in step0
    assert "dvrtsp" in step0["results"]


def test_real_stream_endpoint():
    """Verify Server-Sent Events stream endpoint."""
    r = client.get("/api/real/stream?start=2024-01-01&end=2024-01-02&algorithms=dvrtsp,nn&cluster_size=9")
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    content = r.text
    assert "data: " in content


def test_real_experiment_endpoint():
    """Verify research experiment endpoint returns statistical analysis."""
    r = client.get("/api/real/experiment?n_incidents=3&sizes=5,8&seed=42")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert "summary_table" in data
    assert "wilcoxon_tests" in data
    assert "validation_analysis" in data
