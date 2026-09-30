"""
DVR-TSP Recovery Metrics Engine
-------------------------------
Computes detailed data-recovery and time-utilization metrics for every
algorithm run.  Designed to be called after simulate_order() returns,
enriching the result dict with:

    total_data_available, data_recovered, data_lost (decay + unvisited),
    recovery_percentage, time_budget, time_taken, time_remaining,
    time_utilization_percentage, nodes_visited, nodes_total,
    per_node_timeline, and a plain-English summary.

For **Real Mode (Backblaze)**, capacity_bytes is available in the
scenario metadata; effective bytes = capacity_bytes * 0.70.
For **Synthetic Mode**, we treat each node's V0 as its "data" unit.
"""

import math
from typing import Any, Dict, List, Optional


def format_bytes(b: float) -> str:
    """Human-friendly byte formatting: auto-select MB / GB / TB."""
    if b >= 1e12:
        return f"{b / 1e12:.2f} TB"
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.2f} MB"
    return f"{b:.0f} B"


def format_minutes(t: float) -> str:
    """Human-friendly time in minutes."""
    return f"{t:.1f} min"


def compute_recovery_metrics(
    sim_result: Dict[str, Any],
    nodes: List[Dict],
    t_max: float,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive recovery metrics from a simulate_order() result.

    Parameters
    ----------
    sim_result : dict
        Output of algorithms.simulate_order() — must contain keys:
        totalValue, totalTime, count, steps, order.
    nodes : list[dict]
        The scenario's node list (id, value, corruption, recoveryTime, …).
    t_max : float
        Time budget for the scenario.
    meta : dict or None
        Scenario metadata.  When source == "backblaze" the drive-level
        ``capacity_bytes`` is used to compute byte-level recovery metrics.

    Returns
    -------
    dict  — all the new metric fields; caller merges into the run result.
    """
    node_map = {n["id"]: n for n in nodes}
    steps = sim_result.get("steps", [])
    order = sim_result.get("order", [])

    # Determine if we have byte-level capacity info (real mode)
    drive_map: Dict[str, Dict] = {}
    utilization = 0.70  # default
    is_real = False
    if meta and meta.get("source") == "backblaze" and "drives" in meta:
        is_real = True
        if meta.get("config", {}).get("utilization"):
            utilization = meta["config"]["utilization"]
        for dm in meta["drives"]:
            drive_map[dm["node_id"]] = dm

    # ---- total_data_available & total_value_available ----
    total_value_available = 0.0
    total_data_available_bytes = 0.0

    for n in nodes:
        total_value_available += n["value"]
        if is_real and n["id"] in drive_map:
            cap = max(0.0, float(drive_map[n["id"]].get("capacity_tb", 0.0))) * 1e12
            total_data_available_bytes += cap * utilization
        else:
            # Synthetic: no byte semantics; use value as proxy
            total_data_available_bytes += n["value"]

    # ---- Build visited set and per-node timeline ----
    visited_ids = set()
    per_node_timeline: List[Dict[str, Any]] = []
    value_recovered = 0.0
    data_recovered_bytes = 0.0
    lost_to_decay_bytes = 0.0
    lost_to_decay_value = 0.0
    total_travel_time = 0.0
    total_recovery_time = 0.0

    prev_finish = 0.0  # track for travel time decomposition

    for step in steps:
        nid = step["id"]
        if not step.get("reached", False):
            continue

        visited_ids.add(nid)
        n = node_map.get(nid)
        if n is None:
            continue

        arrival = step.get("arrival", 0.0)
        finish = step.get("finish", arrival)
        decayed_val = step.get("value", 0.0)

        # Full (undecayed) value and data for this node
        full_value = n["value"]
        if is_real and nid in drive_map:
            cap = max(0.0, float(drive_map[nid].get("capacity_tb", 0.0))) * 1e12
            full_bytes = cap * utilization
        else:
            full_bytes = n["value"]

        # Decay fraction at arrival
        decay_fraction = math.exp(-n["corruption"] * arrival)
        recovered_bytes = full_bytes * decay_fraction
        lost_decay_bytes = full_bytes * (1.0 - decay_fraction)

        value_recovered += decayed_val
        data_recovered_bytes += recovered_bytes
        lost_to_decay_bytes += lost_decay_bytes
        lost_to_decay_value += full_value * (1.0 - decay_fraction)

        # Time decomposition
        travel_time_this = arrival - prev_finish if arrival > prev_finish else 0.0
        recovery_time_this = n["recoveryTime"]
        total_travel_time += travel_time_this
        total_recovery_time += recovery_time_this
        prev_finish = finish

        per_node_timeline.append({
            "node_id": nid,
            "arrival_time": round(arrival, 3),
            "finish_time": round(finish, 3),
            "value_at_arrival": round(decayed_val, 3),
            "bytes_recovered": round(recovered_bytes, 2),
            "bytes_lost_to_decay": round(lost_decay_bytes, 2),
        })

    # ---- Unvisited nodes ----
    unvisited_ids = [n["id"] for n in nodes if n["id"] not in visited_ids]
    lost_unvisited_bytes = 0.0
    lost_unvisited_value = 0.0
    for nid in unvisited_ids:
        n = node_map[nid]
        lost_unvisited_value += n["value"]
        if is_real and nid in drive_map:
            cap = max(0.0, float(drive_map[nid].get("capacity_tb", 0.0))) * 1e12
            lost_unvisited_bytes += cap * utilization
        else:
            lost_unvisited_bytes += n["value"]

    data_lost_bytes = lost_to_decay_bytes + lost_unvisited_bytes

    # ---- Percentages ----
    recovery_percentage = (
        (data_recovered_bytes / total_data_available_bytes * 100.0)
        if total_data_available_bytes > 0
        else 0.0
    )

    time_taken = sim_result.get("totalTime", 0.0)
    time_remaining = max(0.0, t_max - time_taken)
    time_utilization_pct = (time_taken / t_max * 100.0) if t_max > 0 else 0.0

    # ---- Build result ----
    metrics: Dict[str, Any] = {
        # Data metrics
        "total_data_available": round(total_data_available_bytes, 2),
        "total_data_available_formatted": format_bytes(total_data_available_bytes) if is_real else None,
        "total_value_available": round(total_value_available, 3),
        "data_recovered": round(data_recovered_bytes, 2),
        "data_recovered_formatted": format_bytes(data_recovered_bytes) if is_real else None,
        "value_recovered": round(value_recovered, 3),
        "data_lost": round(data_lost_bytes, 2),
        "lost_to_decay": round(lost_to_decay_bytes, 2),
        "lost_unvisited": round(lost_unvisited_bytes, 2),
        "recovery_percentage": round(recovery_percentage, 2),
        # Time metrics
        "time_budget": round(t_max, 3),
        "time_taken": round(time_taken, 3),
        "time_remaining": round(time_remaining, 3),
        "time_utilization_percentage": round(time_utilization_pct, 2),
        "travel_time": round(total_travel_time, 3),
        "recovery_time": round(total_recovery_time, 3),
        # Node counts
        "nodes_visited": len(visited_ids),
        "nodes_total": len(nodes),
        # Per-node timeline
        "per_node_timeline": per_node_timeline,
    }

    # Generate plain-English summary text for this run
    algo_name = sim_result.get("algorithm_name") or sim_result.get("algorithm") or "Tour"
    metrics["summary_text"] = generate_summary_text(algo_name, metrics, is_real=is_real)

    return metrics


def generate_summary_text(
    algorithm_name: str,
    metrics: Dict[str, Any],
    is_real: bool = False,
) -> str:
    """Generate a plain-English one-paragraph summary string."""
    nodes_total = metrics["nodes_total"]
    nodes_visited = metrics["nodes_visited"]
    recovery_pct = metrics["recovery_percentage"]
    time_taken = metrics["time_taken"]
    time_budget = metrics["time_budget"]
    time_util = metrics["time_utilization_percentage"]

    if is_real:
        total_avail = format_bytes(metrics["total_data_available"])
        data_rec = format_bytes(metrics["data_recovered"])
        decay_lost = format_bytes(metrics["lost_to_decay"])
        unvisited_lost = format_bytes(metrics["lost_unvisited"])
    else:
        total_avail = f"{metrics['total_value_available']:.1f} value units"
        data_rec = f"{metrics['value_recovered']:.1f} value units"
        decay_lost = f"{metrics['lost_to_decay']:.1f} value units"
        unvisited_lost = f"{metrics['lost_unvisited']:.1f} value units"

    parts = [
        f"Of {total_avail} available across {nodes_total} nodes,",
        f"{algorithm_name} recovered {data_rec} ({recovery_pct:.1f}%)",
        f"in {format_minutes(time_taken)} of {format_minutes(time_budget)}",
        f"({time_util:.0f}% of the time budget).",
    ]

    loss_parts = []
    if metrics["lost_to_decay"] > 0:
        loss_parts.append(f"{decay_lost} was lost to decay")
    if metrics["lost_unvisited"] > 0:
        loss_parts.append(f"{unvisited_lost} was unreachable")

    if loss_parts:
        parts.append(" and ".join(loss_parts) + ".")

    nodes_missed = nodes_total - nodes_visited
    if nodes_missed > 0:
        parts.append(f"{nodes_visited} of {nodes_total} nodes were visited.")

    return " ".join(parts)


def build_scenario_summary(
    runs_with_metrics: List[Dict[str, Any]],
    is_real: bool = False,
) -> Dict[str, Any]:
    """Build a cross-algorithm comparison summary for a scenario.

    Parameters
    ----------
    runs_with_metrics : list[dict]
        Each dict must contain at minimum:
        algorithm, algorithm_name, and the metrics fields
        (data_recovered, recovery_percentage, time_taken, nodes_visited,
        time_budget, value_recovered).

    Returns
    -------
    dict with comparison_table, best_by_data_recovered,
    best_by_time_efficiency, and overall_summary text.
    """
    if not runs_with_metrics:
        return {"comparison_table": [], "overall_summary": "No runs available for summary."}

    comparison_table = []
    best_data = None
    best_efficiency = None

    for run in runs_with_metrics:
        m = run.get("metrics", run)
        time_taken = m.get("time_taken", 0.0)
        data_rec = m.get("data_recovered", 0.0)
        eff = data_rec / max(time_taken, 0.001)  # data per minute

        time_budget = m.get("time_budget", 0.0)
        time_rem = m.get("time_remaining", max(0.0, time_budget - time_taken))
        time_util = m.get("time_utilization_percentage", round((time_taken / time_budget * 100.0) if time_budget > 0 else 0.0, 2))

        entry = {
            "algorithm": run.get("algorithm", ""),
            "algorithm_name": run.get("algorithm_name", ""),
            "data_recovered": data_rec,
            "data_recovered_formatted": m.get("data_recovered_formatted"),
            "recovery_percentage": m.get("recovery_percentage", 0.0),
            "value_recovered": m.get("value_recovered", 0.0),
            "time_taken": time_taken,
            "time_budget": time_budget,
            "time_remaining": time_rem,
            "time_utilization_percentage": time_util,
            "nodes_visited": m.get("nodes_visited", 0),
            "nodes_total": m.get("nodes_total", 0),
            "data_per_minute": round(eff, 3),
            "efficiency_data_per_min": round(eff, 3),
        }
        comparison_table.append(entry)

        if best_data is None or data_rec > best_data["data_recovered"]:
            best_data = entry
        if best_efficiency is None or eff > best_efficiency["data_per_minute"]:
            best_efficiency = entry

    # Overall summary text
    ref = runs_with_metrics[0]
    ref_m = ref.get("metrics", ref)

    if is_real and ref_m.get("total_data_available_formatted"):
        total_str = ref_m["total_data_available_formatted"]
    else:
        total_str = f"{ref_m.get('total_value_available', 0):.1f} value units"

    nodes_total = ref_m.get("nodes_total", 0)

    best_name = best_data["algorithm_name"] if best_data else "N/A"
    best_rec_str = (
        best_data.get("data_recovered_formatted") or f"{best_data['data_recovered']:.1f}"
    ) if best_data else "N/A"
    best_pct = best_data["recovery_percentage"] if best_data else 0

    eff_name = best_efficiency["algorithm_name"] if best_efficiency else "N/A"

    overall = (
        f"Across {len(runs_with_metrics)} algorithm(s) on {nodes_total} nodes "
        f"with {total_str} total data available: "
        f"{best_name} recovered the most data ({best_rec_str}, {best_pct:.1f}%). "
        f"{eff_name} was the most time-efficient (highest data recovered per minute)."
    )

    return {
        "comparison_table": comparison_table,
        "best_by_data_recovered": best_data,
        "best_by_time_efficiency": best_efficiency,
        "overall_summary": overall,
    }
