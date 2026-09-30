"""
Backblaze Drive Stats Real-Data Adapter & Incident Builder
---------------------------------------------------------
This module provides a reproducible, statistically sound bridge between
the public Backblaze Drive Stats daily telemetry dataset and the DVR-TSP
(Dynamic Recoverable-Value Traveling Salesman Problem) algorithm engine.

Formulas & Assumptions (configurable via BackblazeConfig):
1. Value (V0):
   V0 = scale_value(capacity_bytes * utilization)
   Assumed utilization default = 0.70 (typical enterprise storage utilization).
   Mapped linearly to [38.0, 100.0] to mirror the synthetic benchmark range.

2. Risk & Corruption/Decay Rate (k):
   SMART indicators associated with imminent drive failure (Pinheiro et al., 2007;
   Backblaze studies):
   - SMART 5:   Reallocated Sectors Count (weight 0.35)
   - SMART 187: Reported Uncorrectable Errors (weight 0.25)
   - SMART 188: Command Timeout (weight 0.10)
   - SMART 197: Current Pending Sector Count (weight 0.15)
   - SMART 198: Offline Uncorrectable Sector Count (weight 0.15)
   Raw Risk Score:
     R_raw = sum(w_i * ln(1 + smart_i_raw)) + (5.0 if failure == 1 else 0.0)
   Normalized Risk Score:
     R_norm = tanh(R_raw / 8.0)  in [0.0, 1.0)
   Decay Rate:
     k = k_min + (k_max - k_min) * R_norm, where k_min=0.012, k_max=0.062.
   Monotonically non-decreasing in all error metrics.

3. Recovery Time (t_rec):
   Raw time: (capacity_bytes * utilization) / (transfer_rate_mb_s * 1e6 bytes/sec).
   Default transfer rate = 100 MB/s (typical sustained network transfer/rebuild).
   Mapped proportionally to simulator time units [1.6, 4.8] minutes.

4. Network Topology (IMPORTANT):
   The Backblaze Drive Stats dataset contains NO physical data center or network
   topology measurements. All network coordinates (x, y) and link weights
   are generated according to a MODELED topology (random_geometric, ring, or star)
   with seeded determinism, and are clearly tagged as `topology_modeled = True`.
"""

import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .algorithms import dist, build_sparse_graph, Point, Node, Edge


REQUIRED_COLUMNS = [
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "failure",
    "smart_5_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_197_raw",
    "smart_198_raw",
]
OPTIONAL_COLUMNS = ["smart_9_raw"]


@dataclass
class BackblazeConfig:
    """Central configuration for Backblaze Drive Stats parsing and node modeling."""
    data_dir: Path = Path("backend/data/backblaze")

    # Storage assumptions
    utilization: float = 0.70          # Assumed usable capacity utilization factor
    transfer_rate_mb_s: float = 100.0  # Assumed network recovery transfer speed in MB/s

    # Scaling bounds (matching synthetic benchmark space)
    val_min: float = 38.0
    val_max: float = 100.0
    k_min: float = 0.012
    k_max: float = 0.062
    rec_time_min: float = 1.6
    rec_time_max: float = 4.8
    tb_min: float = 0.5   # Reference minimum capacity in TB (500 GB)
    tb_max: float = 14.0  # Reference maximum capacity in TB

    # SMART Risk Weights (sum to 1.0)
    w_smart_5: float = 0.35    # Reallocated sectors
    w_smart_187: float = 0.25  # Reported uncorrectable errors
    w_smart_188: float = 0.10  # Command timeouts
    w_smart_197: float = 0.15  # Current pending sectors
    w_smart_198: float = 0.15  # Offline uncorrectable sectors
    risk_scale: float = 8.0    # tanh normalization denominator

    # Degradation Thresholds (raw values exceeding these flag the drive as degraded)
    thresh_smart_5: float = 0.0
    thresh_smart_187: float = 0.0
    thresh_smart_197: float = 0.0
    thresh_smart_198: float = 0.0


DEFAULT_CONFIG = BackblazeConfig()


# ---------------------------------------------------------------------------
# Chunked CSV Reader & Validation
# ---------------------------------------------------------------------------

def validate_csv_columns(columns: List[str]) -> None:
    """Ensure all required Backblaze Drive Stats columns exist."""
    col_set = set(columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in col_set]
    if missing:
        raise ValueError(
            f"Backblaze dataset is missing required column(s): {missing}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )


def find_csv_files(data_dir: Path) -> List[Path]:
    """Find all CSV files inside the data directory recursively."""
    if not data_dir.exists():
        return []
    files = sorted(data_dir.glob("**/*.csv"))
    return files


def read_backblaze_chunks(
    file_path: Path,
    chunksize: int = 10000,
    filter_date_range: Optional[Tuple[str, str]] = None,
):
    """Generator yielding cleaned DataFrames chunk by chunk from a CSV file.
    Only required and optional columns are loaded into memory."""
    # Peek header to validate
    header_df = pd.read_csv(file_path, nrows=0)
    validate_csv_columns(header_df.columns.tolist())

    usecols = [c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if c in header_df.columns]

    for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        # Clean null values
        chunk["date"] = chunk["date"].astype(str)
        chunk["serial_number"] = chunk["serial_number"].astype(str)
        chunk["model"] = chunk["model"].fillna("UNKNOWN").astype(str)
        chunk["capacity_bytes"] = pd.to_numeric(chunk["capacity_bytes"], errors="coerce").fillna(0.0)
        chunk["failure"] = pd.to_numeric(chunk["failure"], errors="coerce").fillna(0).astype(int)

        for sc in ["smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw"]:
            if sc in chunk.columns:
                chunk[sc] = pd.to_numeric(chunk[sc], errors="coerce").fillna(0.0)
            else:
                chunk[sc] = 0.0

        if "smart_9_raw" in chunk.columns:
            chunk["smart_9_raw"] = pd.to_numeric(chunk["smart_9_raw"], errors="coerce").fillna(0.0)

        if filter_date_range:
            start_d, end_d = filter_date_range
            chunk = chunk[(chunk["date"] >= start_d) & (chunk["date"] <= end_d)]

        if not chunk.empty:
            yield chunk


# ---------------------------------------------------------------------------
# Degradation & Risk Calculation
# ---------------------------------------------------------------------------

def is_drive_degraded(drive: Dict[str, Any], config: BackblazeConfig = DEFAULT_CONFIG) -> bool:
    """Return True if drive has non-zero failure flag or any SMART attribute exceeds threshold."""
    if drive.get("failure", 0) == 1:
        return True
    if float(drive.get("smart_5_raw", 0.0)) > config.thresh_smart_5:
        return True
    if float(drive.get("smart_187_raw", 0.0)) > config.thresh_smart_187:
        return True
    if float(drive.get("smart_197_raw", 0.0)) > config.thresh_smart_197:
        return True
    if float(drive.get("smart_198_raw", 0.0)) > config.thresh_smart_198:
        return True
    return False


def calculate_risk_score(drive: Dict[str, Any], config: BackblazeConfig = DEFAULT_CONFIG) -> float:
    """Compute normalized risk score R_norm in [0.0, 1.0) via monotonic hyperbolic tangent."""
    s5 = max(0.0, float(drive.get("smart_5_raw", 0.0)))
    s187 = max(0.0, float(drive.get("smart_187_raw", 0.0)))
    s188 = max(0.0, float(drive.get("smart_188_raw", 0.0)))
    s197 = max(0.0, float(drive.get("smart_197_raw", 0.0)))
    s198 = max(0.0, float(drive.get("smart_198_raw", 0.0)))
    fail = 1.0 if drive.get("failure", 0) == 1 else 0.0

    raw_risk = (
        config.w_smart_5 * math.log1p(s5)
        + config.w_smart_187 * math.log1p(s187)
        + config.w_smart_188 * math.log1p(s188)
        + config.w_smart_197 * math.log1p(s197)
        + config.w_smart_198 * math.log1p(s198)
        + 5.0 * fail
    )
    norm_risk = math.tanh(raw_risk / config.risk_scale)
    return max(0.0, min(norm_risk, 0.9999))


# ---------------------------------------------------------------------------
# Incident Builder (Step 2)
# ---------------------------------------------------------------------------

def build_incidents(
    data_dir: Path,
    date_range: Optional[Tuple[str, str]] = None,
    min_degraded: int = 3,
    cluster_size: int = 9,
    seed: int = 42,
    model: Optional[str] = None,
    config: BackblazeConfig = DEFAULT_CONFIG,
) -> List[Dict[str, Any]]:
    """Scan Backblaze CSVs and assemble reproducible storage cluster emergency incidents.

    An incident is a date on which a group of `cluster_size` drives contains at
    least `min_degraded` degraded drives. Grouping is deterministic given `seed`.
    """
    csv_files = find_csv_files(data_dir)
    if not csv_files:
        return []

    rng = random.Random(seed)

    # Collect daily records partitioned by date and optionally model
    daily_drives: Dict[str, List[Dict[str, Any]]] = {}

    for file_path in csv_files:
        for chunk in read_backblaze_chunks(file_path, filter_date_range=date_range):
            if model:
                chunk = chunk[chunk["model"] == model]
            if chunk.empty:
                continue
            records = chunk.to_dict(orient="records")
            for rec in records:
                d = rec["date"]
                daily_drives.setdefault(d, []).append(rec)

    incidents: List[Dict[str, Any]] = []

    for date_str in sorted(daily_drives.keys()):
        drives_on_date = daily_drives[date_str]
        if len(drives_on_date) < cluster_size:
            continue

        # Sort drives deterministically by serial number before sampling
        drives_on_date.sort(key=lambda d: d["serial_number"])

        # Identify degraded drives
        degraded = [d for d in drives_on_date if is_drive_degraded(d, config)]
        healthy = [d for d in drives_on_date if not is_drive_degraded(d, config)]

        if len(degraded) < min_degraded:
            continue

        # Form reproducible clusters containing at least min_degraded drives
        # Seed the permutation per date for repeatable yet diverse clusters
        date_seed = seed + int(hash(date_str) % 1_000_000)
        date_rng = random.Random(date_seed)

        deg_sample = date_rng.sample(degraded, min(len(degraded), min_degraded))
        remaining_needed = cluster_size - len(deg_sample)

        pool = [d for d in drives_on_date if d["serial_number"] not in {x["serial_number"] for x in deg_sample}]
        if len(pool) < remaining_needed:
            continue

        cluster_drives = deg_sample + date_rng.sample(pool, remaining_needed)
        # Deterministic ordering within the cluster
        cluster_drives.sort(key=lambda d: d["serial_number"])

        incidents.append({
            "incident_id": f"inc-{date_str}-{cluster_size}-{len(incidents)+1}",
            "date": date_str,
            "cluster_size": cluster_size,
            "degraded_count": sum(1 for d in cluster_drives if is_drive_degraded(d, config)),
            "model_summary": list({d["model"] for d in cluster_drives}),
            "drives": cluster_drives,
        })

    return incidents


# ---------------------------------------------------------------------------
# Node Adapter & Topology Generator (Step 3)
# ---------------------------------------------------------------------------

def generate_modeled_topology(
    hub: Point,
    n: int,
    topology: str = "random_geometric",
    seed: int = 42,
) -> List[Point]:
    """Generate (x, y) coordinates for n nodes under a modeled network topology."""
    rng = random.Random(seed)
    pts: List[Point] = []

    if topology == "ring":
        center_x, center_y = 540.0, 320.0
        rx, ry = 260.0, 200.0
        angle_step = (2 * math.pi) / max(n, 1)
        for i in range(n):
            angle = i * angle_step + rng.uniform(-0.05, 0.05)
            pts.append({
                "x": round(center_x + rx * math.cos(angle), 1),
                "y": round(center_y + ry * math.sin(angle), 1),
            })

    elif topology == "star":
        # Nodes radiate from a cluster switch or central backbone
        cx, cy = 540.0, 320.0
        spokes = max(3, n // 3)
        for i in range(n):
            spoke_idx = i % spokes
            tier = (i // spokes) + 1
            angle = (spoke_idx * (2 * math.pi / spokes))
            radius = tier * (240.0 / max((n // spokes) + 1, 1))
            pts.append({
                "x": round(cx + radius * math.cos(angle) + rng.uniform(-10, 10), 1),
                "y": round(cy + radius * math.sin(angle) + rng.uniform(-10, 10), 1),
            })

    else:  # "random_geometric" (default)
        x_range = (230.0, 860.0)
        y_range = (40.0, 600.0)
        min_sep = 85.0
        hub_exclusion = 140.0
        tries = 0
        while len(pts) < n and tries < 5000:
            tries += 1
            cand = {"x": round(rng.uniform(*x_range), 1), "y": round(rng.uniform(*y_range), 1)}
            if any(dist(cand, p) < min_sep for p in pts):
                continue
            if dist(cand, hub) < hub_exclusion:
                continue
            pts.append(cand)

        # Fallback if rejection sampling is exhausted
        while len(pts) < n:
            pts.append({
                "x": round(rng.uniform(250, 850), 1),
                "y": round(rng.uniform(60, 580), 1),
            })

    return pts


def drives_to_nodes(
    drives: List[Dict[str, Any]],
    hub: Optional[Point] = None,
    topology: str = "random_geometric",
    seed: int = 42,
    config: BackblazeConfig = DEFAULT_CONFIG,
) -> Tuple[List[Node], List[Edge], Dict[str, Any]]:
    """Transform Backblaze drive telemetry records into DVR-TSP nodes and edges.

    Returns:
        nodes: List[Node] adhering strictly to existing node dict format
               (id, idx, x, y, value, corruption, recoveryTime).
        edges: List[Edge] adhering strictly to existing edge dict format (a, b, w).
        metadata: Dict holding drive details, risk scores, and modeling disclosure.
    """
    if hub is None:
        hub = {"x": 78.0, "y": 320.0}

    n = len(drives)
    pts = generate_modeled_topology(hub, n, topology=topology, seed=seed)

    nodes: List[Node] = []
    drive_metadata: List[Dict[str, Any]] = []

    for i, (drive, pt) in enumerate(zip(drives, pts)):
        cap_bytes = max(0.0, float(drive.get("capacity_bytes", 0.0)))
        cap_tb = cap_bytes / 1e12
        eff_tb = cap_tb * config.utilization

        # 1. Recoverable Value V0
        tb_span = max(config.tb_max - config.tb_min, 0.1)
        tb_frac = max(0.0, min(1.0, (eff_tb - config.tb_min) / tb_span))
        value = config.val_min + (config.val_max - config.val_min) * tb_frac

        # 2. Risk Score and Corruption Rate k
        risk_norm = calculate_risk_score(drive, config)
        corruption = config.k_min + (config.k_max - config.k_min) * risk_norm

        # 3. Recovery Time
        rec_time = config.rec_time_min + (config.rec_time_max - config.rec_time_min) * tb_frac

        nid = f"N{i + 1}"
        node_dict = {
            "id": nid,
            "idx": i,
            "x": pt["x"],
            "y": pt["y"],
            "value": round(value, 1),
            "corruption": round(corruption, 4),
            "recoveryTime": round(rec_time, 1),
        }
        nodes.append(node_dict)

        drive_metadata.append({
            "node_id": nid,
            "serial_number": str(drive.get("serial_number", "")),
            "model": str(drive.get("model", "")),
            "capacity_tb": round(cap_tb, 2),
            "risk_score": round(risk_norm, 4),
            "is_degraded": is_drive_degraded(drive, config),
            "failure": int(drive.get("failure", 0)),
            "smart_5_raw": float(drive.get("smart_5_raw", 0.0)),
            "smart_187_raw": float(drive.get("smart_187_raw", 0.0)),
            "smart_188_raw": float(drive.get("smart_188_raw", 0.0)),
            "smart_197_raw": float(drive.get("smart_197_raw", 0.0)),
            "smart_198_raw": float(drive.get("smart_198_raw", 0.0)),
        })

    edges = build_sparse_graph(hub, nodes, k=3)

    scenario_metadata = {
        "source": "backblaze",
        "topology": topology,
        "topology_modeled": True,
        "topology_notice": "Network coordinates and edge weights are modeled, not physically measured from Backblaze.",
        "seed": seed,
        "drives": drive_metadata,
        "config": {
            "utilization": config.utilization,
            "transfer_rate_mb_s": config.transfer_rate_mb_s,
            "k_bounds": [config.k_min, config.k_max],
            "val_bounds": [config.val_min, config.val_max],
        },
    }

    return nodes, edges, scenario_metadata
