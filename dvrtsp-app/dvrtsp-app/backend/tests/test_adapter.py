import math
import pytest

from app.backblaze import (
    BackblazeConfig,
    calculate_risk_score,
    is_drive_degraded,
    drives_to_nodes,
    generate_modeled_topology,
)


def test_monotonicity_smart_errors():
    """Verify that increasing any SMART error count monotonically increases risk and decay rate."""
    base_drive = {
        "capacity_bytes": 4e12,
        "failure": 0,
        "smart_5_raw": 0.0,
        "smart_187_raw": 0.0,
        "smart_188_raw": 0.0,
        "smart_197_raw": 0.0,
        "smart_198_raw": 0.0,
    }

    base_risk = calculate_risk_score(base_drive)
    assert base_risk == 0.0

    for attr in ["smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw"]:
        prev_risk = base_risk
        for val in [1.0, 10.0, 100.0, 1000.0]:
            drive = dict(base_drive)
            drive[attr] = val
            cur_risk = calculate_risk_score(drive)
            assert cur_risk > prev_risk, f"Expected risk to increase for {attr}={val}"
            prev_risk = cur_risk


def test_failure_flag_impact():
    """Verify failure=1 sharply escalates the risk score."""
    drive_ok = {"capacity_bytes": 4e12, "failure": 0}
    drive_fail = {"capacity_bytes": 4e12, "failure": 1}
    assert calculate_risk_score(drive_fail) > calculate_risk_score(drive_ok)
    assert is_drive_degraded(drive_fail) is True


def test_adapter_bounds_and_no_nan_inf():
    """Verify all adapted node metrics fall strictly within configured bounds without NaN/Inf."""
    config = BackblazeConfig()
    hub = {"x": 78.0, "y": 320.0}

    test_drives = [
        # Normal small drive
        {"serial_number": "D1", "model": "M1", "capacity_bytes": 500e9, "failure": 0},
        # Extreme degraded huge drive
        {
            "serial_number": "D2",
            "model": "M2",
            "capacity_bytes": 20e12,
            "failure": 1,
            "smart_5_raw": 99999,
            "smart_187_raw": 99999,
            "smart_188_raw": 99999,
            "smart_197_raw": 99999,
            "smart_198_raw": 99999,
        },
        # Zero capacity edge case
        {"serial_number": "D3", "model": "M3", "capacity_bytes": 0, "failure": 0},
        # Negative / corrupt values
        {"serial_number": "D4", "model": "M4", "capacity_bytes": -100, "failure": 0, "smart_5_raw": -5},
    ]

    nodes, edges, meta = drives_to_nodes(test_drives, hub=hub, config=config)

    assert len(nodes) == len(test_drives)
    assert meta["topology_modeled"] is True

    for n in nodes:
        # Check no NaN / Inf
        assert not math.isnan(n["value"])
        assert not math.isinf(n["value"])
        assert not math.isnan(n["corruption"])
        assert not math.isinf(n["corruption"])
        assert not math.isnan(n["recoveryTime"])
        assert not math.isinf(n["recoveryTime"])

        # Check bounds
        assert config.val_min <= n["value"] <= config.val_max
        assert config.k_min <= n["corruption"] <= config.k_max
        assert config.rec_time_min <= n["recoveryTime"] <= config.rec_time_max


def test_modeled_topologies():
    """Verify different modeled topologies generate valid coordinates."""
    hub = {"x": 78.0, "y": 320.0}
    for topo in ["random_geometric", "ring", "star"]:
        pts = generate_modeled_topology(hub, 9, topology=topo, seed=42)
        assert len(pts) == 9
        for p in pts:
            assert 0 <= p["x"] <= 1000
            assert 0 <= p["y"] <= 1000
