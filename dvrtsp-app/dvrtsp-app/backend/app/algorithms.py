"""
DVR-TSP: Dynamic Recoverable-Value Traveling Salesman Problem
----------------------------------------------------------------
This module holds the actual algorithm engine for the emergency data
recovery prototype: scenario generation, the sparse k-NN routing graph,
the four value-blind baseline traversal orders (Nearest Neighbor,
Dijkstra, A*, BFS), and the DVR-TSP greedy value-density heuristic
itself, plus the shared simulator used to score any visiting order
under the same decaying-value conditions.

This is a direct, faithful port of the client-side prototype's logic —
now living server-side so it can be the single source of truth, be
benchmarked, and have every run persisted.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

Point = Dict[str, float]
Node = Dict
Edge = Dict


def dist(a: Point, b: Point) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def generate_scenario(
    hub: Point,
    n: int = 9,
    x_range: Tuple[float, float] = (230, 860),
    y_range: Tuple[float, float] = (40, 600),
    min_sep: float = 85.0,
    hub_exclusion: float = 140.0,
) -> List[Node]:
    """Rejection-sample n well-separated storage node positions, then assign
    each a recoverable value, a corruption (decay) rate, and a recovery time."""
    pts: List[Point] = []
    tries = 0
    while len(pts) < n and tries < 4000:
        tries += 1
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        candidate = {"x": x, "y": y}
        if any(dist(candidate, p) < min_sep for p in pts):
            continue
        if dist(candidate, hub) < hub_exclusion:
            continue
        pts.append(candidate)

    nodes = []
    for i, p in enumerate(pts):
        nodes.append({
            "id": f"N{i + 1}",
            "idx": i,
            "x": p["x"],
            "y": p["y"],
            "value": round(random.uniform(38, 100)),
            "corruption": round(random.uniform(0.012, 0.062), 3),
            "recoveryTime": round(random.uniform(1.6, 4.8), 1),
        })
    return nodes


def build_sparse_graph(hub: Point, nodes: List[Node], k: int = 3) -> List[Edge]:
    """k-nearest-neighbor routing graph over hub + all nodes (undirected, deduped)."""
    all_pts = [{"id": "hub", "x": hub["x"], "y": hub["y"]}] + [
        {"id": n["id"], "x": n["x"], "y": n["y"]} for n in nodes
    ]
    edge_set: Dict[Tuple[str, str], Edge] = {}
    for a in all_pts:
        neighbors = sorted(
            (b for b in all_pts if b["id"] != a["id"]),
            key=lambda b: dist(a, b),
        )[:k]
        for b in neighbors:
            key = tuple(sorted([a["id"], b["id"]]))
            if key not in edge_set:
                edge_set[key] = {"a": a["id"], "b": b["id"], "w": dist(a, b)}
    return list(edge_set.values())


def build_adjacency(edges: List[Edge]) -> Dict[str, List[Dict]]:
    adj: Dict[str, List[Dict]] = {}
    for e in edges:
        adj.setdefault(e["a"], []).append({"id": e["b"], "w": e["w"]})
        adj.setdefault(e["b"], []).append({"id": e["a"], "w": e["w"]})
    return adj


# ---------------------------------------------------------------------------
# Baseline traversal orders (value-blind)
# ---------------------------------------------------------------------------

def dijkstra_from(adj: Dict[str, List[Dict]], start_id: str) -> Tuple[List[str], Dict[str, float]]:
    dists = {k: math.inf for k in adj}
    dists[start_id] = 0.0
    visited = set()
    order: List[str] = []
    while True:
        u, best = None, math.inf
        for k, v in dists.items():
            if k not in visited and v < best:
                best, u = v, k
        if u is None:
            break
        visited.add(u)
        if u != start_id:
            order.append(u)
        for nb in adj.get(u, []):
            if dists[u] + nb["w"] < dists[nb["id"]]:
                dists[nb["id"]] = dists[u] + nb["w"]
    return order, dists


def bfs_from(adj: Dict[str, List[Dict]], start_id: str) -> List[str]:
    hops = {start_id: 0}
    q = [start_id]
    order: List[str] = []
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for nb in adj.get(u, []):
            if nb["id"] not in hops:
                hops[nb["id"]] = hops[u] + 1
                q.append(nb["id"])
                order.append(nb["id"])
    return order


def astar_from(adj: Dict[str, List[Dict]], start_id: str, node_map: Dict[str, Point], goal_id: str) -> List[str]:
    goal = node_map[goal_id]

    def h(node_id: str) -> float:
        return dist(node_map[node_id], goal)

    g = {k: math.inf for k in adj}
    g[start_id] = 0.0
    visited = set()
    order: List[str] = []
    while True:
        u, best = None, math.inf
        for k in g:
            if k in visited:
                continue
            f = g[k] + h(k)
            if f < best:
                best, u = f, k
        if u is None:
            break
        visited.add(u)
        if u != start_id:
            order.append(u)
        for nb in adj.get(u, []):
            if g[u] + nb["w"] < g[nb["id"]]:
                g[nb["id"]] = g[u] + nb["w"]
    return order


def nearest_neighbor_order(hub: Point, nodes: List[Node]) -> List[str]:
    remaining = list(nodes)
    order: List[str] = []
    cur = hub
    while remaining:
        bi, bd = 0, math.inf
        for i, n in enumerate(remaining):
            d = dist(cur, n)
            if d < bd:
                bd, bi = d, i
        chosen = remaining.pop(bi)
        order.append(chosen["id"])
        cur = chosen
    return order


# ---------------------------------------------------------------------------
# DVR-TSP: the value-aware, corruption-aware, time-aware greedy heuristic
# ---------------------------------------------------------------------------

def dvr_tsp_order(hub: Point, nodes: List[Node], t_max: float, speed: float) -> List[str]:
    """At every step, re-score every unvisited node by
    (decayed value at arrival) / (travel + recovery cost), and always
    take the best-scoring node that's still reachable within the window.
    This is a greedy orienteering-style heuristic: exact TSP-with-decay
    is NP-hard, and the decay means the score landscape changes every
    step anyway, so a static optimal tour would go stale immediately."""
    remaining = list(nodes)
    order: List[str] = []
    cur = hub
    t = 0.0
    while remaining:
        best = None
        best_score = -math.inf
        best_arrival = 0.0
        best_idx = -1
        for i, n in enumerate(remaining):
            travel = dist(cur, n) / speed
            arrival = t + travel
            finish_at = arrival + n["recoveryTime"]
            if finish_at > t_max:
                continue
            decayed_val = n["value"] * math.exp(-n["corruption"] * arrival)
            cost = travel + n["recoveryTime"]
            score = decayed_val / max(cost, 0.001)
            if score > best_score:
                best_score, best, best_idx, best_arrival = score, n, i, arrival
        if best is None:
            break
        remaining.pop(best_idx)
        order.append(best["id"])
        t = best_arrival + best["recoveryTime"]
        cur = best
    return order


# ---------------------------------------------------------------------------
# Shared simulator: score any visiting order under the same decay model
# ---------------------------------------------------------------------------

def simulate_order(order_ids: List[str], hub: Point, nodes: List[Node], t_max: float, speed: float) -> Dict:
    node_map = {n["id"]: n for n in nodes}
    cur = hub
    t = 0.0
    total_value = 0.0
    count = 0
    steps = []
    for nid in order_ids:
        n = node_map[nid]
        travel = dist(cur, n) / speed
        arrival = t + travel
        if arrival > t_max:
            steps.append({"id": nid, "reached": False})
            continue
        decayed_val = n["value"] * math.exp(-n["corruption"] * arrival)
        finish = arrival + n["recoveryTime"]
        total_value += decayed_val
        count += 1
        t = min(finish, t_max)
        cur = n
        steps.append({
            "id": nid, "reached": True,
            "arrival": arrival, "value": decayed_val, "finish": finish,
        })
        if finish > t_max:
            break
    return {"totalValue": total_value, "totalTime": t, "count": count, "steps": steps}


ALGORITHMS = {
    "dvrtsp": "DVR–TSP",
    "nn": "Nearest Neighbor",
    "dijkstra": "Dijkstra",
    "astar": "A*",
    "bfs": "BFS",
}


def run_algorithm(key: str, hub: Point, nodes: List[Node], edges: List[Edge], t_max: float, speed: float) -> Dict:
    if key not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{key}'")

    adj = build_adjacency(edges)
    node_map = {n["id"]: n for n in nodes}

    if key == "dvrtsp":
        order = dvr_tsp_order(hub, nodes, t_max, speed)
    elif key == "nn":
        order = nearest_neighbor_order(hub, nodes)
    elif key == "dijkstra":
        order, _ = dijkstra_from(adj, "hub")
    elif key == "astar":
        goal = max(nodes, key=lambda n: dist(hub, n))
        goal_map = dict(node_map)
        goal_map["hub"] = hub
        order = astar_from(adj, "hub", goal_map, goal["id"])
    elif key == "bfs":
        order = bfs_from(adj, "hub")
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unknown algorithm '{key}'")

    result = simulate_order(order, hub, nodes, t_max, speed)
    result["order"] = order
    return result
