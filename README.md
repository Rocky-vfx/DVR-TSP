# DVR-TSP
DVR-TSP (Dynamic Recoverable-Value Traveling Salesman Problem) finds the best order to visit failing storage nodes during emergency data recovery. Each node's data decays exponentially over time and needs recovery time, so a greedy value-per-time heuristic maximizes recovered data, tested on Backblaze drive data.
# DVR–TSP — Emergency Data Recovery (Full-Stack Research Prototype)

A full-stack implementation and empirical evaluation platform for the **Dynamic Recoverable-Value Traveling Salesman Problem (DVR–TSP)** in distributed storage environments.

The system features:
1. **Synthetic Mode**: Generates reproducible random geometric distributed storage clusters with sparse $k$-NN routing graphs ($k=3$).
2. **Real Data Mode (Backblaze Drive Stats)**: Ingests real-world drive telemetry and SMART failure indicators from the public Backblaze Drive Stats dataset, mapping telemetry to bounded recoverable values ($V_0$), decay rates ($k$), and recovery transfer times ($t_{\text{rec}}$).
3. **Exact Solver & Benchmarking**: Branch-and-bound solver for $N \le 10$ to measure empirical optimality gaps alongside paired **Wilcoxon signed-rank tests**.
4. **Interactive Illustrated UI**: Single-page HTML/SVG dashboard with live decay animation, telemetry tooltips, multi-day replay, and database-backed algorithm comparisons.

```
dvrtsp-app/
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI routes (REST + SSE streaming)
│   │   ├── algorithms.py       # DVR-TSP + 4 baselines (NN, Dijkstra, A*, BFS)
│   │   ├── backblaze.py        # Real-data ingestion, SMART risk scoring, adapter
│   │   ├── exact_solver.py     # Branch-and-bound exact solver & optimality gap
│   │   ├── experiment.py       # Paper experiment CLI runner & Wilcoxon tests
│   │   ├── models.py           # SQLAlchemy tables (Scenario, Run)
│   │   ├── schemas.py          # Pydantic schemas
│   │   └── database.py         # SQLite engine & non-destructive migration
│   ├── data/
│   │   └── backblaze/          # Place Backblaze Drive Stats daily CSVs here
│   ├── tests/
│   │   ├── fixtures/           # Validated Backblaze daily CSV test fixtures
│   │   ├── test_adapter.py     # Risk score monotonicity & numerical bounds
│   │   ├── test_incidents.py   # Incident builder determinism & edge cases
│   │   ├── test_simulator_and_exact.py # Simulator invariants & exact solver
│   │   └── test_api_real.py    # FastAPI TestClient API coverage
│   ├── dvrtsp.db               # SQLite database (auto-created on startup)
│   └── requirements.txt        # Pinned dependencies
├── frontend/
│   └── dvrtsp.html             # Illustrated dashboard with Mode toggle & tooltips
├── results/                    # Generated paper tables, statistical tests & plots
└── README.md
```

---

## 1. Quick Start

### Backend Startup

Requires Python 3.10+:

```bash
cd backend
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --port 8000 --reload
```

- **Health Check** → http://127.0.0.1:8000/api/health
- **Interactive Swagger Docs** → http://127.0.0.1:8000/docs
- **ReDoc API Documentation** → http://127.0.0.1:8000/redoc

### Frontend Dashboard

Open `frontend/dvrtsp.html` in any web browser (or serve it with `python -m http.server 3000`). It talks directly to `http://127.0.0.1:8000/api`.

---

## 2. Real Data Mode (Backblaze Drive Stats)

### Dataset Download & Placement

The Backblaze Drive Stats dataset provides daily drive telemetry including operational status, capacity, and raw SMART metrics.

1. Download daily CSV files from [Backblaze Hard Drive Data and Stats](https://www.backblaze.com/b2/hard-drive-test-data.html).
2. Place the unzipped CSV files into:
   ```
   backend/data/backblaze/
   ```
   For example:
   ```
   backend/data/backblaze/2024-01-01.csv
   backend/data/backblaze/2024-01-02.csv
   ```
3. Required columns validated by the loader:
   `date`, `serial_number`, `model`, `capacity_bytes`, `failure`, `smart_5_raw`, `smart_187_raw`, `smart_188_raw`, `smart_197_raw`, `smart_198_raw`. (Optional: `smart_9_raw`).

---

## 3. Mathematical Model & Telemetry Mapping

The adapter in [`app/backblaze.py`](backend/app/backblaze.py) transforms raw drive records into standard DVR-TSP storage nodes ($V_0$, $k$, $t_{\text{rec}}$):

### 1. Recoverable Value ($V_0$)
Derived from drive capacity under an assumed enterprise storage utilization factor ($u = 0.70$):
$$\text{EffCapacity} = \text{Capacity}_{\text{bytes}} \times u$$
Normalized linearly into the synthetic benchmark range $[38.0, 100.0]$:
$$V_0 = V_{\min} + (V_{\max} - V_{\min}) \cdot \frac{\text{EffCapacity} - \text{Cap}_{\min}}{\text{Cap}_{\max} - \text{Cap}_{\min}}$$

### 2. SMART Failure Risk Score & Corruption Rate ($k$)
Decay rate reflects impending drive failure probability using critical SMART indicators identified in literature (Pinheiro et al., 2007; Backblaze annual studies):
- **SMART 5**: Reallocated Sectors Count ($w_5 = 0.35$)
- **SMART 187**: Reported Uncorrectable Errors ($w_{187} = 0.25$)
- **SMART 188**: Command Timeouts ($w_{188} = 0.10$)
- **SMART 197**: Current Pending Sector Count ($w_{197} = 0.15$)
- **SMART 198**: Offline Uncorrectable Sector Count ($w_{198} = 0.15$)

The raw risk score incorporates logarithmic dampening on counters and a failure penalty:
$$R_{\text{raw}} = \sum_{i \in \{5, 187, 188, 197, 198\}} w_i \cdot \ln(1 + \text{SMART}_{i,\text{raw}}) + 5.0 \cdot \mathbb{I}(\text{failure} = 1)$$
Normalized monotonically via hyperbolic tangent:
$$R_{\text{norm}} = \tanh\left(\frac{R_{\text{raw}}}{8.0}\right) \in [0.0, 1.0)$$
The exponential corruption rate $k$ is mapped within bounded realistic limits ($k_{\min}=0.012, k_{\max}=0.062$):
$$k = k_{\min} + (k_{\max} - k_{\min}) \cdot R_{\text{norm}}$$

### 3. Recovery Time ($t_{\text{rec}}$)
Recovery duration models the time needed to pull essential data over the local interconnect at an assumed sustained network transfer rate ($100\text{ MB/s}$):
$$t_{\text{rec}} = t_{\min} + (t_{\max} - t_{\min}) \cdot \frac{\text{EffCapacity} - \text{Cap}_{\min}}{\text{Cap}_{\max} - \text{Cap}_{\min}} \in [1.6, 4.8]\text{ min}$$

### 4. Emergency Recovery Metrics & Formulas

For every algorithm execution, the platform computes comprehensive physical and value recovery metrics under exponential decay and time budget constraints ($T_{\max}$):

1. **Total Data Available**:
   $$\text{Total Data Available} = \sum_{i \in \text{Nodes}} \text{Data}_i$$
   * In **Real Mode (Backblaze)**: $\text{Data}_i = \text{Capacity}_{\text{bytes}, i} \times 0.70$ (formatted automatically in MB, GB, or TB).
   * In **Synthetic Mode**: $\text{Data}_i = V_{0, i}$.
   * Total value available is also tracked: $\text{Total Value Available} = \sum_{i} V_{0, i}$.

2. **Data Recovered**:
   Data recovered by the tour, accounting for continuous degradation during transit and extraction:
   $$\text{Data Recovered} = \sum_{i \in \text{Visited}} \text{Data}_i \cdot \exp(-k_i \cdot t_{\text{arr}, i})$$
   $$\text{Value Recovered} = \sum_{i \in \text{Visited}} V_{0, i} \cdot \exp(-k_i \cdot t_{\text{arr}, i})$$

3. **Data Lost (Decay vs. Unvisited)**:
   $$\text{Data Lost} = \text{Total Data Available} - \text{Data Recovered} = \text{Lost}_{\text{decay}} + \text{Lost}_{\text{unvisited}}$$
   * **Lost to Decay** (visited, but corrupted prior to rescue arrival):
     $$\text{Lost}_{\text{decay}} = \sum_{i \in \text{Visited}} \text{Data}_i \cdot \left(1 - \exp(-k_i \cdot t_{\text{arr}, i})\right)$$
   * **Lost Unvisited** (never reached before deadline $T_{\max}$):
     $$\text{Lost}_{\text{unvisited}} = \sum_{j \notin \text{Visited}} \text{Data}_j$$

4. **Recovery Percentage**:
   $$\text{Recovery Percentage} = \frac{\text{Data Recovered}}{\text{Total Data Available}} \times 100\% \in [0.0, 100.0]$$

5. **Time Budget Utilization & Conservation**:
   $$\text{Time Taken} = t_{\text{travel, total}} + t_{\text{recovery, total}} = \sum_{i \in \text{Visited}} \left(\frac{\text{dist}(\text{prev}, i)}{\text{speed}} + t_{\text{rec}, i}\right) \le T_{\max}$$
   $$\text{Time Remaining} = \max(0, T_{\max} - \text{Time Taken})$$
   $$\text{Time Utilization} = \frac{\text{Time Taken}}{T_{\max}} \times 100\%$$
   $$\text{Data Efficiency} = \frac{\text{Data Recovered}}{\max(\text{Time Taken}, 0.001)} \quad [\text{bytes/min or val/min}]$$

6. **Per-Node Timeline**:
   Full per-step audit record containing `node_id`, `arrival_time`, `finish_time`, `value_at_arrival`, `bytes_recovered`, and `bytes_lost_to_decay`.

### 5. Modeled Network Topology (Disclosure)
> [!IMPORTANT]
> **Topology Notice**: The Backblaze dataset contains drive telemetry without data-center network switch topologies. Physical coordinates $(x, y)$ and sparse $k$-NN edge graphs are generated deterministically under a modeled topology (`random_geometric`, `ring`, or `star`) and are clearly flagged as `topology_modeled: true`.

---

## 4. API Reference

### Core & Synthetic Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Service health status |
| `GET` | `/api/algorithms` | List available recovery strategies |
| `POST` | `/api/scenarios` | Create and store synthetic scenario |
| `GET` | `/api/scenarios/{id}` | Retrieve scenario with nodes, edges, metadata |
| `POST` | `/api/scenarios/{id}/run` | Execute recovery algorithm, persist run & recovery metrics |
| `GET` | `/api/scenarios/{id}/runs` | List persisted runs for scenario sorted by value with recovery metrics |
| `GET` | `/api/scenarios/{id}/summary` | **NEW**: Side-by-side scenario recovery comparison, best algorithm rankings, and plain-English narrative summary |
| `DELETE`| `/api/scenarios/{id}` | Delete scenario and historical runs |
| `GET` | `/api/benchmark` | Synthetic complexity and value benchmark |

### Real Data Mode Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/real/incidents` | Query reproducible incidents from Backblaze CSVs (`cluster_size`, `min_degraded`, `model`, `seed`) |
| `POST` | `/api/real/scenarios` | Convert incident drives into a persisted scenario (`topology`, `seed`) |
| `POST` | `/api/real/replay` | Multi-day sequential replay recalculating recovery strategies daily with recovery metrics |
| `GET` | `/api/real/stream` | Server-Sent Events (SSE) live streaming multi-day telemetry progression and recovery metrics |
| `GET` | `/api/real/experiment` | Execute paper statistical experiment suite on demand |

---

## 5. Research Experiment CLI & Paper Reproduction

To reproduce all paper results, statistical significance tables, and plots:

```bash
cd backend
python -m app.experiment \
  --data data/backblaze \
  --n-incidents 50 \
  --sizes 5,8,10,12 \
  --seed 42 \
  --out results/
```

### Generated Artifacts in `results/`:
- `summary_table.csv`: Mean, standard deviation, and 95% confidence intervals of recovered value, tour time, compute time, **recovery percentage**, **data recovered**, and **time utilization percentage** across cluster sizes.
- `detailed_runs.csv`: Full record of every algorithm execution per incident including all physical and time recovery metrics.
- `recovery_summary.md`: **NEW**: Automatic written summary report covering average data available, average data recovered per algorithm, average time taken, and the winning algorithm.
- `statistical_tests.json`: Paired **Wilcoxon signed-rank test** statistics and $p$-values comparing DVR-TSP against every baseline.
- `config.json`: Execution parameters, weights, and configuration for exact auditability.
- `plots/`:
  - `value_comparison.png`: Recoverable value distribution by algorithm.
  - `value_vs_cluster_size.png`: Scaling of recovered value as cluster size grows.
  - `recovery_percentage.png`: **NEW**: Scaling of data recovery percentage (%) vs cluster size across all algorithms.
  - `compute_time_scaling.png`: Empirical CPU execution time scaling.
  - `optimality_gap.png`: Heuristic performance vs Exact Solver ($N \le 10$).

---

## 6. Running Automated Tests

Run the full pytest suite (20 tests verifying data conservation, time feasibility, recovery percentages, scenario summary endpoint, adapter bounds, monotonicity, incident builder determinism, simulator invariants, exact solver optimality, and API endpoints):

```bash
cd backend
python -m pytest tests/ -v
```

---

## 7. Assumptions & Known Limitations

1. **Modeled Topology**: Drive stats provide telemetry but not network switches. Edge distances are modeled geometrically.
2. **Greedy Orienteering Heuristic**: DVR-TSP solves dynamic exponential decay via a greedy score heuristic ($\frac{V(t_{\text{arr}})}{t_{\text{travel}} + t_{\text{rec}}}$). While near-optimal for small $N$ ($\le 5\%$ optimality gap), it does not guarantee global optimality for large $N$.
3. **Derived Values**: Recoverable value and corruption rates are modeled from raw capacity and SMART error rates; real applications should adjust risk weights to match proprietary storage telemetry.
