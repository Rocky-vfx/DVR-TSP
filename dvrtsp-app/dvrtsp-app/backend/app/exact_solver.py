"""
Exact Solver for DVR-TSP (Branch-and-Bound / Sub-Tour Search)
-------------------------------------------------------------
Solves the Dynamic Recoverable-Value Traveling Salesman Problem to optimality
for small instances (N <= 10) to establish ground-truth upper bounds and
compute the empirical Optimality Gap of the DVR-TSP greedy heuristic:

    Optimality Gap = (Value_exact - Value_heuristic) / Value_exact

Decay Model:
At arrival time t_arr = t_cur + dist(cur, n) / speed:
    V(t_arr) = V0 * exp(-k * t_arr)

Pruning Bounds:
1. Feasibility: If t_arr > T_max, the node cannot be reached.
2. Value Upper Bound: For any unvisited node j, max possible value is
   V_j * exp(-k_j * (t_cur + min_travel_j)).
   If (accumulated_value + sum(upper_bound(j))) <= best_value, prune branch.
3. Warm Start: Seeded with DVR-TSP heuristic result so branch-and-bound
   immediately prunes vast portions of the permutation tree.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

from .algorithms import dist, simulate_order, dvr_tsp_order, Point, Node


def solve_exact_dvrtsp(
    hub: Point,
    nodes: List[Node],
    t_max: float,
    speed: float,
    time_limit_sec: float = 30.0,
) -> Dict:
    """Find the exact optimal visiting order maximizing recovered value under decay.

    Applicable for N <= 10 instances. Returns simulated metrics matching simulate_order.
    """
    n_count = len(nodes)
    if n_count == 0:
        return {"order": [], "totalValue": 0.0, "totalTime": 0.0, "count": 0, "is_optimal": True}

    node_map = {n["id"]: n for n in nodes}
    node_ids = [n["id"] for n in nodes]

    # Precalculate pairwise distances and node specs
    all_pts = {"hub": hub}
    all_pts.update({n["id"]: {"x": n["x"], "y": n["y"]} for n in nodes})

    dist_matrix = {}
    for a in all_pts:
        for b in all_pts:
            dist_matrix[(a, b)] = dist(all_pts[a], all_pts[b])

    # 1. Warm start with DVR-TSP heuristic order
    heuristic_order = dvr_tsp_order(hub, nodes, t_max, speed)
    heuristic_sim = simulate_order(heuristic_order, hub, nodes, t_max, speed)

    best_value = heuristic_sim["totalValue"]
    best_order = list(heuristic_order)
    start_wall_time = time.perf_counter()
    nodes_explored = 0
    timeout_hit = False

    def search(
        cur_id: str,
        t_cur: float,
        acc_val: float,
        visited_mask: int,
        path: List[str],
    ):
        nonlocal best_value, best_order, nodes_explored, timeout_hit
        nodes_explored += 1

        if (nodes_explored & 0xFFF) == 0:
            if time.perf_counter() - start_wall_time > time_limit_sec:
                timeout_hit = True
                return

        # Pruning bound: upper bound on recoverable value from remaining unvisited nodes
        potential_val = 0.0
        for i, nid in enumerate(node_ids):
            if not (visited_mask & (1 << i)):
                n = node_map[nid]
                min_travel = dist_matrix[(cur_id, nid)] / speed
                earliest_arrival = t_cur + min_travel
                if earliest_arrival <= t_max:
                    potential_val += n["value"] * math.exp(-n["corruption"] * earliest_arrival)

        if acc_val + potential_val <= best_value:
            return

        # Try visiting unvisited candidates
        extended = False
        # Branch heuristic: order candidates by greedy value density from current position
        candidates = []
        for i, nid in enumerate(node_ids):
            if not (visited_mask & (1 << i)):
                n = node_map[nid]
                travel = dist_matrix[(cur_id, nid)] / speed
                arr = t_cur + travel
                if arr <= t_max:
                    decayed_v = n["value"] * math.exp(-n["corruption"] * arr)
                    score = decayed_v / max(travel + n["recoveryTime"], 0.001)
                    candidates.append((score, i, nid, n, travel, arr))

        candidates.sort(key=lambda c: c[0], reverse=True)

        for _, i, nid, n, travel, arr in candidates:
            decayed_v = n["value"] * math.exp(-n["corruption"] * arr)
            fin = arr + n["recoveryTime"]
            new_t = min(fin, t_max)
            new_acc = acc_val + decayed_v
            new_path = path + [nid]

            if new_acc > best_value:
                best_value = new_acc
                best_order = new_path

            extended = True
            if fin <= t_max:
                search(nid, new_t, new_acc, visited_mask | (1 << i), new_path)
                if timeout_hit:
                    return

    search("hub", 0.0, 0.0, 0, [])

    # Final simulation of the best discovered tour
    sim = simulate_order(best_order, hub, nodes, t_max, speed)
    sim["order"] = best_order
    sim["is_optimal"] = not timeout_hit
    sim["nodes_explored"] = nodes_explored
    sim["optimality_gap_vs_dvrtsp"] = (
        (sim["totalValue"] - heuristic_sim["totalValue"]) / max(sim["totalValue"], 0.001)
        if sim["totalValue"] > 0
        else 0.0
    )
    return sim
