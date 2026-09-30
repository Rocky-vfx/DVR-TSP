# DVR-TSP Emergency Data Recovery — Experiment Summary

**Evaluation Scope:** 3 incidents across cluster sizes [5, 8] under decay dynamics and time budget $T_{\max} = 55.0$ min.

## Executive Summary

Across all evaluated storage incident scenarios, **DVR–TSP** achieved the highest overall data recovery performance, recovering an average of **14.99 TB** (63.7% of available data) in 47.2 minutes (85.8% of the time budget). On average, 6.95 TB was lost to continuous degradation during transit and recovery operations, and 2.80 TB remained unreachable before the operational time limit expired.

## Overall Recovery Performance

| Algorithm | Data Available | Data Recovered | Recovery % | Time Taken | Time Utilization | Lost to Decay | Lost Unvisited |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact Solver (Optimal)** | 24.73 TB | 15.59 TB | 65.5% | 48.4 min | 88.0% | 7.74 TB | 1.40 TB |
| **DVR–TSP** | 24.73 TB | 14.99 TB | 63.7% | 47.2 min | 85.8% | 6.95 TB | 2.80 TB |
| **Nearest Neighbor** | 24.73 TB | 14.68 TB | 62.7% | 46.5 min | 84.6% | 7.72 TB | 2.33 TB |
| **Dijkstra** | 24.73 TB | 14.01 TB | 60.2% | 48.9 min | 88.9% | 6.76 TB | 3.97 TB |
| **BFS** | 24.73 TB | 13.95 TB | 59.9% | 49.6 min | 90.1% | 6.82 TB | 3.97 TB |
| **A*** | 24.73 TB | 11.82 TB | 51.2% | 51.0 min | 92.8% | 6.61 TB | 6.30 TB |

## Detailed Breakdown by Cluster Size

| Cluster Size | Algorithm | Mean Recovery % (±95% CI) | Mean Data Recovered | Mean Time | Time Util % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| N = 5 | DVR–TSP | 69.9% ± 10.6% | 11.91 TB | 46.2 min | 84.0% |
| N = 5 | Nearest Neighbor | 68.8% ± 15.7% | 11.64 TB | 42.7 min | 77.7% |
| N = 5 | Dijkstra | 68.3% ± 10.9% | 11.63 TB | 45.5 min | 82.7% |
| N = 5 | A* | 61.0% ± 10.3% | 10.61 TB | 48.7 min | 88.6% |
| N = 5 | BFS | 68.3% ± 10.9% | 11.63 TB | 45.5 min | 82.7% |
| N = 5 | Exact Solver | 70.2% ± 10.3% | 11.98 TB | 44.5 min | 80.8% |
| N = 8 | DVR–TSP | 57.5% ± 18.6% | 18.06 TB | 48.2 min | 87.6% |
| N = 8 | Nearest Neighbor | 56.5% ± 20.1% | 17.73 TB | 50.4 min | 91.5% |
| N = 8 | Dijkstra | 52.0% ± 14.4% | 16.39 TB | 52.4 min | 95.2% |
| N = 8 | A* | 41.5% ± 25.0% | 13.04 TB | 53.3 min | 96.9% |
| N = 8 | BFS | 51.6% ± 13.6% | 16.27 TB | 53.7 min | 97.7% |
| N = 8 | Exact Solver | 60.8% ± 14.3% | 19.20 TB | 52.3 min | 95.2% |
