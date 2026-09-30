import pytest

from app.algorithms import (
    ALGORITHMS,
    generate_scenario,
    build_sparse_graph,
    run_algorithm,
    simulate_order,
)
from app.exact_solver import solve_exact_dvrtsp

HUB = {"x": 78.0, "y": 320.0}
T_MAX = 55.0
SPEED = 42.0


def test_simulator_invariants():
    """Verify simulator invariants: total value <= sum of initial values, total time <= T_max."""
    nodes = generate_scenario(HUB, n=9)
    edges = build_sparse_graph(HUB, nodes, k=3)
    sum_v0 = sum(n["value"] for n in nodes)

    for key in ALGORITHMS:
        res = run_algorithm(key, HUB, nodes, edges, T_MAX, SPEED)
        assert res["totalValue"] <= sum_v0 + 1e-6
        assert res["totalTime"] <= T_MAX + 1e-6
        assert res["count"] <= len(nodes)

        # Permutation uniqueness invariant
        order = res["order"]
        assert len(order) == len(set(order)), f"Algorithm {key} produced duplicate nodes in order"
        valid_nids = {n["id"] for n in nodes}
        assert all(nid in valid_nids for nid in order), f"Algorithm {key} visited invalid node id"


def test_exact_solver_optimality():
    """Verify that exact solver achieves totalValue >= all heuristics on small instances (N <= 8)."""
    for n_count in [4, 6]:
        nodes = generate_scenario(HUB, n=n_count)
        edges = build_sparse_graph(HUB, nodes, k=3)

        exact_res = solve_exact_dvrtsp(HUB, nodes, T_MAX, SPEED)
        exact_val = exact_res["totalValue"]

        assert exact_res["is_optimal"] is True
        assert len(exact_res["order"]) == len(set(exact_res["order"]))

        for key in ALGORITHMS:
            heur_res = run_algorithm(key, HUB, nodes, edges, T_MAX, SPEED)
            heur_val = heur_res["totalValue"]
            assert exact_val >= heur_val - 1e-5, (
                f"Exact solver value ({exact_val}) was strictly less than heuristic {key} ({heur_val})"
            )
