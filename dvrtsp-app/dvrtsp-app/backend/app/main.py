import asyncio
import json
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from . import algorithms, models, schemas, backblaze, experiment
from .database import Base, engine, get_db, run_migrations
from .metrics import compute_recovery_metrics, generate_summary_text, build_scenario_summary

Base.metadata.create_all(bind=engine)
run_migrations()

app = FastAPI(
    title="DVR-TSP Emergency Data Recovery API",
    description="Backend engine for the Dynamic Recoverable-Value TSP prototype with Synthetic & Real (Backblaze) modes.",
    version="2.0.0",
)

# Wide-open CORS: frontend can run from file:// or any local port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

HUB = {"x": 78.0, "y": 320.0}
T_MAX = 55.0
SPEED = 42.0
BACKBLAZE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "backblaze"


# ---------------------------------------------------------------------------
# Helper: compute and persist metrics for a run
# ---------------------------------------------------------------------------

def _compute_and_persist_metrics(
    run_obj: models.Run,
    result: Dict,
    nodes: List[Dict],
    t_max: float,
    meta: Optional[Dict] = None,
    db: Optional[Session] = None,
) -> Dict:
    """Compute recovery metrics, persist to DB, return metrics dict."""
    metrics = compute_recovery_metrics(result, nodes, t_max, meta=meta)
    is_real = bool(meta and meta.get("source") == "backblaze")
    algo_name = algorithms.ALGORITHMS.get(run_obj.algorithm, run_obj.algorithm)
    summary_text = generate_summary_text(algo_name, metrics, is_real=is_real)

    # Persist to Run model
    run_obj.total_data_available = metrics["total_data_available"]
    run_obj.data_recovered = metrics["data_recovered"]
    run_obj.data_lost = metrics["data_lost"]
    run_obj.lost_to_decay = metrics["lost_to_decay"]
    run_obj.lost_unvisited = metrics["lost_unvisited"]
    run_obj.recovery_percentage = metrics["recovery_percentage"]
    run_obj.time_budget = metrics["time_budget"]
    run_obj.time_remaining = metrics["time_remaining"]
    run_obj.time_utilization_percentage = metrics["time_utilization_percentage"]
    run_obj.travel_time = metrics.get("travel_time")
    run_obj.recovery_time_total = metrics.get("recovery_time")
    run_obj.nodes_visited = metrics["nodes_visited"]
    run_obj.nodes_total = metrics["nodes_total"]
    run_obj.total_value_available = metrics["total_value_available"]
    run_obj.value_recovered = metrics["value_recovered"]
    run_obj.per_node_timeline_json = json.dumps(metrics["per_node_timeline"])
    run_obj.summary_text = summary_text

    if db:
        db.commit()
        db.refresh(run_obj)

    metrics["summary_text"] = summary_text
    return metrics


def _metrics_to_schema(metrics: Dict) -> schemas.RecoveryMetrics:
    """Convert a metrics dict to the RecoveryMetrics Pydantic schema."""
    return schemas.RecoveryMetrics(
        total_data_available=metrics.get("total_data_available"),
        total_data_available_formatted=metrics.get("total_data_available_formatted"),
        total_value_available=metrics.get("total_value_available"),
        data_recovered=metrics.get("data_recovered"),
        data_recovered_formatted=metrics.get("data_recovered_formatted"),
        value_recovered=metrics.get("value_recovered"),
        data_lost=metrics.get("data_lost"),
        lost_to_decay=metrics.get("lost_to_decay"),
        lost_unvisited=metrics.get("lost_unvisited"),
        recovery_percentage=metrics.get("recovery_percentage"),
        time_budget=metrics.get("time_budget"),
        time_taken=metrics.get("time_taken"),
        time_remaining=metrics.get("time_remaining"),
        time_utilization_percentage=metrics.get("time_utilization_percentage"),
        travel_time=metrics.get("travel_time"),
        recovery_time=metrics.get("recovery_time"),
        nodes_visited=metrics.get("nodes_visited"),
        nodes_total=metrics.get("nodes_total"),
        per_node_timeline=metrics.get("per_node_timeline"),
    )


# ---------------------------------------------------------------------------
# Health & Algorithm Metadata
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/algorithms", response_model=List[schemas.AlgorithmInfo])
def list_algorithms():
    return [{"key": k, "name": v} for k, v in algorithms.ALGORITHMS.items()]


# ---------------------------------------------------------------------------
# Synthetic Scenario Routes (Preserved)
# ---------------------------------------------------------------------------

@app.post("/api/scenarios", response_model=schemas.ScenarioOut)
def create_scenario(payload: schemas.ScenarioCreate, db: Session = Depends(get_db)):
    """Generate a fresh distributed storage cluster (nodes + kNN routing graph)
    and persist it, so every algorithm run against it is comparable."""
    nodes = algorithms.generate_scenario(HUB, n=payload.n_nodes)
    edges = algorithms.build_sparse_graph(HUB, nodes, k=3)

    scenario = models.Scenario(
        t_max=T_MAX,
        speed=SPEED,
        hub_x=HUB["x"],
        hub_y=HUB["y"],
        nodes_json=json.dumps(nodes),
        edges_json=json.dumps(edges),
        source="synthetic",
        meta_json=json.dumps({"mode": "synthetic"}),
    )
    db.add(scenario)
    db.commit()
    db.refresh(scenario)

    return {
        "scenario_id": scenario.id,
        "hub": {"x": scenario.hub_x, "y": scenario.hub_y},
        "nodes": nodes,
        "edges": edges,
        "t_max": scenario.t_max,
        "speed": scenario.speed,
        "source": scenario.source or "synthetic",
        "meta": json.loads(scenario.meta_json) if scenario.meta_json else None,
    }


@app.get("/api/scenarios/{scenario_id}", response_model=schemas.ScenarioOut)
def get_scenario(scenario_id: int, db: Session = Depends(get_db)):
    scenario = db.get(models.Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {
        "scenario_id": scenario.id,
        "hub": {"x": scenario.hub_x, "y": scenario.hub_y},
        "nodes": json.loads(scenario.nodes_json),
        "edges": json.loads(scenario.edges_json),
        "t_max": scenario.t_max,
        "speed": scenario.speed,
        "source": scenario.source or "synthetic",
        "meta": json.loads(scenario.meta_json) if scenario.meta_json else None,
    }


@app.post("/api/scenarios/{scenario_id}/run", response_model=schemas.RunOut)
def run_recovery(scenario_id: int, payload: schemas.RunCreate, db: Session = Depends(get_db)):
    """Execute a recovery strategy against a stored scenario and persist the result."""
    scenario = db.get(models.Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    if payload.algorithm not in algorithms.ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm '{payload.algorithm}'")

    nodes = json.loads(scenario.nodes_json)
    edges = json.loads(scenario.edges_json)
    hub = {"x": scenario.hub_x, "y": scenario.hub_y}
    meta = json.loads(scenario.meta_json) if scenario.meta_json else None

    result = algorithms.run_algorithm(payload.algorithm, hub, nodes, edges, scenario.t_max, scenario.speed)

    run = models.Run(
        scenario_id=scenario.id,
        algorithm=payload.algorithm,
        total_value=result["totalValue"],
        total_time=result["totalTime"],
        count=result["count"],
        order_json=json.dumps(result["order"]),
        steps_json=json.dumps(result["steps"]),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Compute and persist recovery metrics
    metrics = _compute_and_persist_metrics(run, result, nodes, scenario.t_max, meta=meta, db=db)

    return {
        "run_id": run.id,
        "scenario_id": scenario.id,
        "algorithm": payload.algorithm,
        "algorithm_name": algorithms.ALGORITHMS[payload.algorithm],
        "totalValue": result["totalValue"],
        "totalTime": result["totalTime"],
        "count": result["count"],
        "order": result["order"],
        "steps": result["steps"],
        "metrics": _metrics_to_schema(metrics),
        "summary_text": metrics.get("summary_text"),
    }


@app.get("/api/scenarios/{scenario_id}/runs", response_model=List[schemas.RunSummary])
def list_runs(scenario_id: int, db: Session = Depends(get_db)):
    scenario = db.get(models.Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    runs = (
        db.query(models.Run)
        .filter(models.Run.scenario_id == scenario_id)
        .order_by(models.Run.total_value.desc())
        .all()
    )
    return [
        {
            "run_id": r.id,
            "algorithm": r.algorithm,
            "algorithm_name": algorithms.ALGORITHMS.get(r.algorithm, r.algorithm),
            "totalValue": r.total_value,
            "totalTime": r.total_time,
            "count": r.count,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "recovery_percentage": r.recovery_percentage,
            "data_recovered": r.data_recovered,
            "nodes_visited": r.nodes_visited,
            "nodes_total": r.nodes_total,
            "time_budget": r.time_budget,
        }
        for r in runs
    ]


@app.delete("/api/scenarios/{scenario_id}")
def delete_scenario(scenario_id: int, db: Session = Depends(get_db)):
    scenario = db.get(models.Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    db.delete(scenario)
    db.commit()
    return {"deleted": True}


@app.get("/api/benchmark")
def benchmark(sizes: str = "5,10,15,20", trials: int = 3):
    """Empirical complexity/quality comparison across synthetic cluster sizes."""
    size_list = [int(s) for s in sizes.split(",") if s.strip()]
    trials = max(1, min(trials, 20))
    results = {}

    for n in size_list:
        n = max(3, min(n, 40))
        stats = {k: {"values": [], "times": [], "compute_ms": []} for k in algorithms.ALGORITHMS}
        for _ in range(trials):
            nodes = algorithms.generate_scenario(HUB, n=n)
            edges = algorithms.build_sparse_graph(HUB, nodes, k=3)
            for key in algorithms.ALGORITHMS:
                t0 = time.perf_counter()
                res = algorithms.run_algorithm(key, HUB, nodes, edges, T_MAX, SPEED)
                t1 = time.perf_counter()
                stats[key]["values"].append(res["totalValue"])
                stats[key]["times"].append(res["totalTime"])
                stats[key]["compute_ms"].append((t1 - t0) * 1000)

        results[str(n)] = [
            {
                "algorithm": key,
                "algorithm_name": algorithms.ALGORITHMS[key],
                "avg_value": sum(s["values"]) / len(s["values"]),
                "avg_time": sum(s["times"]) / len(s["times"]),
                "avg_compute_ms": sum(s["compute_ms"]) / len(s["compute_ms"]),
            }
            for key, s in stats.items()
        ]

    return results


# ---------------------------------------------------------------------------
# Scenario Summary (NEW ENDPOINT)
# ---------------------------------------------------------------------------

@app.get("/api/scenarios/{scenario_id}/summary", response_model=schemas.ScenarioSummaryOut)
def get_scenario_summary(scenario_id: int, db: Session = Depends(get_db)):
    """Return a side-by-side comparison of all algorithm runs on a scenario,
    with best-algorithm picks and a plain-English overall summary."""
    scenario = db.get(models.Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    nodes = json.loads(scenario.nodes_json)
    meta = json.loads(scenario.meta_json) if scenario.meta_json else None
    is_real = bool(meta and meta.get("source") == "backblaze")

    runs = (
        db.query(models.Run)
        .filter(models.Run.scenario_id == scenario_id)
        .order_by(models.Run.total_value.desc())
        .all()
    )
    if not runs:
        return {
            "scenario_id": scenario_id,
            "source": scenario.source or "synthetic",
            "total_data_available": 0.0,
            "total_data_available_formatted": None,
            "total_value_available": sum(n.get("value", 0.0) for n in nodes),
            "time_budget": scenario.t_max,
            "total_nodes": len(nodes),
            "runs_count": 0,
            "comparison_table": [],
            "algorithms": [],
            "best_by_data_recovered": None,
            "best_by_time_efficiency": None,
            "best_algorithm_by_data": None,
            "best_algorithm_by_efficiency": None,
            "overall_summary": "No algorithm runs recorded for this scenario yet.",
            "summary_text": "No algorithm runs recorded for this scenario yet.",
        }

    runs_with_metrics = []
    for r in runs:
        # If metrics were already persisted, use them; otherwise re-compute
        if r.recovery_percentage is not None:
            data_rec_fmt = format_bytes(r.data_recovered) if (is_real and r.data_recovered is not None) else None
            data_avail_fmt = format_bytes(r.total_data_available) if (is_real and r.total_data_available is not None) else None
            m = {
                "total_data_available": r.total_data_available,
                "total_data_available_formatted": data_avail_fmt,
                "total_value_available": r.total_value_available,
                "data_recovered": r.data_recovered,
                "data_recovered_formatted": data_rec_fmt,
                "value_recovered": r.value_recovered,
                "data_lost": r.data_lost,
                "lost_to_decay": r.lost_to_decay,
                "lost_unvisited": r.lost_unvisited,
                "recovery_percentage": r.recovery_percentage,
                "time_budget": r.time_budget,
                "time_taken": r.total_time,
                "time_remaining": r.time_remaining,
                "time_utilization_percentage": r.time_utilization_percentage,
                "travel_time": r.travel_time,
                "recovery_time": r.recovery_time_total,
                "nodes_visited": r.nodes_visited,
                "nodes_total": r.nodes_total,
            }
        else:
            # Re-compute from stored simulation data
            sim_result = {
                "totalValue": r.total_value,
                "totalTime": r.total_time,
                "count": r.count,
                "steps": json.loads(r.steps_json),
                "order": json.loads(r.order_json),
            }
            m = compute_recovery_metrics(sim_result, nodes, scenario.t_max, meta=meta)

        runs_with_metrics.append({
            "algorithm": r.algorithm,
            "algorithm_name": algorithms.ALGORITHMS.get(r.algorithm, r.algorithm),
            "metrics": m,
        })

    summary = build_scenario_summary(runs_with_metrics, is_real=is_real)
    first_m = runs_with_metrics[0]["metrics"]

    summary["scenario_id"] = scenario_id
    summary["source"] = scenario.source or "synthetic"
    summary["total_data_available"] = first_m.get("total_data_available")
    summary["total_data_available_formatted"] = first_m.get("total_data_available_formatted")
    summary["total_value_available"] = first_m.get("total_value_available")
    summary["time_budget"] = scenario.t_max
    summary["total_nodes"] = len(nodes)
    summary["runs_count"] = len(runs)
    summary["algorithms"] = summary["comparison_table"]
    summary["best_algorithm_by_data"] = summary["best_by_data_recovered"]["algorithm_name"] if summary.get("best_by_data_recovered") else None
    summary["best_algorithm_by_efficiency"] = summary["best_by_time_efficiency"]["algorithm_name"] if summary.get("best_by_time_efficiency") else None
    summary["summary_text"] = summary["overall_summary"]
    return summary


# ---------------------------------------------------------------------------
# Real Data Mode (Backblaze Drive Stats)
# ---------------------------------------------------------------------------

@app.get("/api/real/incidents", response_model=List[schemas.IncidentOut])
def get_real_incidents(
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    min_degraded: int = Query(2, ge=1, le=10, description="Minimum degraded drives required in cluster"),
    cluster_size: int = Query(9, ge=3, le=20, description="Target cluster size"),
    seed: int = Query(42, description="Random seed for grouping"),
    model: Optional[str] = Query(None, description="Filter by specific drive model"),
):
    """Scan the Backblaze Drive Stats CSV files and return reproducible emergency incidents."""
    date_range = (start, end) if (start and end) else None
    incidents = backblaze.build_incidents(
        data_dir=BACKBLAZE_DATA_DIR,
        date_range=date_range,
        min_degraded=min_degraded,
        cluster_size=cluster_size,
        seed=seed,
        model=model,
    )
    return incidents


@app.post("/api/real/scenarios", response_model=schemas.ScenarioOut)
def create_real_scenario(payload: schemas.RealScenarioCreate, db: Session = Depends(get_db)):
    """Convert a Backblaze incident or drive set into a persisted DVR-TSP scenario."""
    # 1. Find or assemble drives
    drives = []
    if payload.incident_id:
        incidents = backblaze.build_incidents(
            data_dir=BACKBLAZE_DATA_DIR,
            cluster_size=payload.cluster_size,
            seed=payload.seed,
        )
        matched = next((inc for inc in incidents if inc["incident_id"] == payload.incident_id), None)
        if not matched:
            raise HTTPException(
                status_code=404,
                detail=f"Incident '{payload.incident_id}' not found. Please verify backend/data/backblaze CSVs.",
            )
        drives = matched["drives"]
    elif payload.date and payload.serials:
        # Load specific drives on a given date
        csv_files = backblaze.find_csv_files(BACKBLAZE_DATA_DIR)
        for fp in csv_files:
            for chunk in backblaze.read_backblaze_chunks(fp, filter_date_range=(payload.date, payload.date)):
                matched_rows = chunk[chunk["serial_number"].isin(payload.serials)]
                if not matched_rows.empty:
                    drives.extend(matched_rows.to_dict(orient="records"))
        if not drives:
            raise HTTPException(
                status_code=404,
                detail=f"Drives on date {payload.date} with serials {payload.serials} not found.",
            )
    else:
        # Pick the first available incident
        incidents = backblaze.build_incidents(
            data_dir=BACKBLAZE_DATA_DIR,
            cluster_size=payload.cluster_size,
            seed=payload.seed,
        )
        if not incidents:
            raise HTTPException(
                status_code=404,
                detail="No Backblaze incidents found. Please place CSV files in backend/data/backblaze/ (see README).",
            )
        drives = incidents[0]["drives"]

    # 2. Adapt drives to DVR-TSP graph
    nodes, edges, meta = backblaze.drives_to_nodes(
        drives=drives,
        hub=HUB,
        topology=payload.topology,
        seed=payload.seed,
    )

    scenario = models.Scenario(
        t_max=T_MAX,
        speed=SPEED,
        hub_x=HUB["x"],
        hub_y=HUB["y"],
        nodes_json=json.dumps(nodes),
        edges_json=json.dumps(edges),
        source="backblaze",
        meta_json=json.dumps(meta),
    )
    db.add(scenario)
    db.commit()
    db.refresh(scenario)

    return {
        "scenario_id": scenario.id,
        "hub": {"x": scenario.hub_x, "y": scenario.hub_y},
        "nodes": nodes,
        "edges": edges,
        "t_max": scenario.t_max,
        "speed": scenario.speed,
        "source": scenario.source,
        "meta": meta,
    }


@app.post("/api/real/replay", response_model=schemas.ReplayOut)
def replay_real_days(payload: schemas.ReplayRequest, db: Session = Depends(get_db)):
    """Replay consecutive days from Backblaze Drive Stats, re-evaluating recovery plans daily."""
    incidents = backblaze.build_incidents(
        data_dir=BACKBLAZE_DATA_DIR,
        date_range=(payload.start, payload.end),
        cluster_size=payload.cluster_size,
        seed=payload.seed,
    )
    if not incidents:
        raise HTTPException(
            status_code=404,
            detail=f"No incidents found between {payload.start} and {payload.end}.",
        )

    # Step through incidents
    step_incidents = incidents[::payload.step_days]
    steps = []

    for inc in step_incidents:
        nodes, edges, meta = backblaze.drives_to_nodes(
            drives=inc["drives"],
            hub=HUB,
            topology=payload.topology,
            seed=payload.seed,
        )
        scenario = models.Scenario(
            t_max=T_MAX,
            speed=SPEED,
            hub_x=HUB["x"],
            hub_y=HUB["y"],
            nodes_json=json.dumps(nodes),
            edges_json=json.dumps(edges),
            source="backblaze",
            meta_json=json.dumps(meta),
        )
        db.add(scenario)
        db.commit()
        db.refresh(scenario)

        results_by_algo = {}
        for algo in payload.algorithms:
            if algo in algorithms.ALGORITHMS:
                res = algorithms.run_algorithm(algo, HUB, nodes, edges, T_MAX, SPEED)
                # Compute metrics for replay results
                metrics = compute_recovery_metrics(res, nodes, T_MAX, meta=meta)
                results_by_algo[algo] = {
                    "totalValue": res["totalValue"],
                    "totalTime": res["totalTime"],
                    "count": res["count"],
                    "order": res["order"],
                    "recovery_percentage": metrics["recovery_percentage"],
                    "data_recovered": metrics["data_recovered"],
                    "nodes_visited": metrics["nodes_visited"],
                }

        steps.append({
            "date": inc["date"],
            "scenario_id": scenario.id,
            "degraded_count": inc["degraded_count"],
            "results": results_by_algo,
        })

    return {"total_days": len(steps), "steps": steps}


@app.get("/api/real/stream")
async def stream_real_replay(
    start: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end: str = Query(..., description="End date (YYYY-MM-DD)"),
    step_days: int = Query(1, ge=1, le=30),
    algorithms_param: str = Query("dvrtsp,nn", alias="algorithms"),
    cluster_size: int = Query(9, ge=3, le=20),
    topology: str = Query("random_geometric"),
    seed: int = Query(42),
    db: Session = Depends(get_db),
):
    """Server-Sent Events stream for live day-by-day telemetry and recovery plan animation."""
    algo_list = [a.strip() for a in algorithms_param.split(",") if a.strip()]

    incidents = backblaze.build_incidents(
        data_dir=BACKBLAZE_DATA_DIR,
        date_range=(start, end),
        cluster_size=cluster_size,
        seed=seed,
    )
    if not incidents:
        raise HTTPException(
            status_code=404,
            detail=f"No incidents found between {start} and {end} for SSE streaming.",
        )

    step_incidents = incidents[::step_days]

    async def event_generator() -> AsyncGenerator[str, None]:
        for inc in step_incidents:
            nodes, edges, meta = backblaze.drives_to_nodes(
                drives=inc["drives"],
                hub=HUB,
                topology=topology,
                seed=seed,
            )
            scenario = models.Scenario(
                t_max=T_MAX,
                speed=SPEED,
                hub_x=HUB["x"],
                hub_y=HUB["y"],
                nodes_json=json.dumps(nodes),
                edges_json=json.dumps(edges),
                source="backblaze",
                meta_json=json.dumps(meta),
            )
            db.add(scenario)
            db.commit()
            db.refresh(scenario)

            results_by_algo = {}
            for algo in algo_list:
                if algo in algorithms.ALGORITHMS:
                    res = algorithms.run_algorithm(algo, HUB, nodes, edges, T_MAX, SPEED)
                    # Compute metrics for SSE stream results
                    metrics = compute_recovery_metrics(res, nodes, T_MAX, meta=meta)
                    results_by_algo[algo] = {
                        "totalValue": res["totalValue"],
                        "totalTime": res["totalTime"],
                        "count": res["count"],
                        "order": res["order"],
                        "steps": res["steps"],
                        "recovery_percentage": metrics["recovery_percentage"],
                        "data_recovered": metrics["data_recovered"],
                        "time_taken": metrics["time_taken"],
                        "nodes_visited": metrics["nodes_visited"],
                    }

            payload = {
                "date": inc["date"],
                "scenario_id": scenario.id,
                "degraded_count": inc["degraded_count"],
                "nodes": nodes,
                "edges": edges,
                "meta": meta,
                "results": results_by_algo,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.5)  # Pace event stream for live frontend animation

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/real/experiment", response_model=schemas.ExperimentOut)
def run_real_experiment(
    n_incidents: int = Query(20, ge=1, le=500),
    sizes: str = Query("5,8,10", description="Comma-separated cluster sizes"),
    seed: int = Query(42),
    validation_days: int = Query(14),
):
    """Execute complete research experiment suite across real and synthetic Backblaze incidents."""
    size_list = [int(s.strip()) for s in sizes.split(",") if s.strip()]
    res = experiment.run_paper_experiment(
        data_dir=BACKBLAZE_DATA_DIR,
        n_incidents=n_incidents,
        sizes=size_list,
        seed=seed,
        out_dir=Path("results"),
        validation_days=validation_days,
    )
    return res
