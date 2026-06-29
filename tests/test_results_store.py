"""Tests for persisted analysis results."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from utils import results_store as rs


@pytest.fixture
def temp_results_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(rs, "RESULTS_DIR", Path(tmp))
        yield Path(tmp)


def test_save_and_list_vocal_range(temp_results_dir):
    record = rs.save_result(
        "vocal_range",
        {"low_note": "C3", "high_note": "G4", "modal_note": "E4", "low_hz": 130.0},
        input_meta={"mode": "record", "duration_sec": 10.0},
    )
    assert record["id"]
    assert (temp_results_dir / Path(record["file"]).name).exists()

    loaded = rs.list_results(feature="vocal_range")
    assert len(loaded) == 1
    assert loaded[0]["metrics"]["low_note"] == "C3"


def test_sanitize_strips_plot_arrays(temp_results_dir):
    record = rs.save_result(
        "voice_comparison",
        {
            "mean_cents_deviation": 12.5,
            "ref_f0_plot": [0.0, 220.0, 225.0],
            "user_f0_plot": [0.0, 218.0, 230.0],
        },
    )
    metrics = record["metrics"]
    assert "ref_f0_plot" not in metrics
    assert metrics["ref_f0_plot_frame_count"] == 3
    assert "ref_f0_plot_median_hz" in metrics


def test_pitch_accuracy_tier():
    assert rs.pitch_accuracy_tier(4.0) == "professional"
    assert rs.pitch_accuracy_tier(12.0) == "acceptable_live"
    assert rs.pitch_accuracy_tier(150.0) == "out_of_tune"


def test_save_numpy_bool(temp_results_dir):
    record = rs.save_result(
        "vocal_range",
        {
            "low_note": "C3",
            "high_note": "G4",
            "is_nasal": np.bool_(True),
            "low_hz": np.float64(130.5),
        },
    )
    assert record["metrics"]["is_nasal"] is True
    assert isinstance(record["metrics"]["low_hz"], float)
