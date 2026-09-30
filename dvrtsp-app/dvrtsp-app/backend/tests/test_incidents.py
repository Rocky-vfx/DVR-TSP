from pathlib import Path
import pytest
import pandas as pd

from app.backblaze import (
    build_incidents,
    validate_csv_columns,
    read_backblaze_chunks,
    REQUIRED_COLUMNS,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_incident_builder_determinism():
    """Verify that calling build_incidents with the same seed returns identical output."""
    inc1 = build_incidents(FIXTURES_DIR, cluster_size=9, seed=42)
    inc2 = build_incidents(FIXTURES_DIR, cluster_size=9, seed=42)

    assert len(inc1) == len(inc2)
    assert len(inc1) > 0

    for a, b in zip(inc1, inc2):
        assert a["incident_id"] == b["incident_id"]
        assert a["date"] == b["date"]
        assert [d["serial_number"] for d in a["drives"]] == [d["serial_number"] for d in b["drives"]]


def test_missing_column_validation(tmp_path):
    """Verify that missing required columns raises a descriptive ValueError."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("date,serial_number,model\n2024-01-01,SN1,M1\n")

    with pytest.raises(ValueError) as exc:
        validate_csv_columns(["date", "serial_number", "model"])
    assert "missing required column" in str(exc.value)

    with pytest.raises(ValueError):
        list(read_backblaze_chunks(bad_csv))


def test_empty_file_handling(tmp_path):
    """Verify that empty files or directories return empty lists without crashing."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    inc = build_incidents(empty_dir, cluster_size=9, seed=42)
    assert inc == []


def test_insufficient_drives_filtering():
    """Verify that requests for clusters larger than available drives return empty lists."""
    inc = build_incidents(FIXTURES_DIR, cluster_size=50, seed=42)
    assert inc == []
