# DVR-TSP
DVR-TSP (Dynamic Recoverable-Value Traveling Salesman Problem) finds the best order to visit failing storage nodes during emergency data recovery. Each node's data decays exponentially over time and needs recovery time, so a greedy value-per-time heuristic maximizes recovered data, tested on Backblaze drive data.
