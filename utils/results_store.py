"""Persist analysis results for accuracy benchmarking against competitor tools."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Reference tiers used when comparing pitch accuracy (industry norms).
PITCH_BENCHMARKS = {
    "professional_cents": 5,
    "acceptable_live_cents": 15,
    "noticeable_off_cents": 25,
    "oot_singing_threshold_cents": 100,
}

COMPETITOR_SERVICES = {
    "vocal_range": [
        {
            "name": "Vocal Range Calculator",
            "url": "https://vocalrangecalculator.com/",
            "overlap": "Low/high notes, voice type, register test",
            "compare_fields": ["low_note", "high_note", "modal_note", "register"],
        },
        {
            "name": "Vocal Range Test",
            "url": "https://vocalrangetest.com/",
            "overlap": "Real-time pitch, voice type (Bass–Soprano), cents accuracy claims",
            "compare_fields": ["low_note", "high_note", "modal_note"],
        },
        {
            "name": "Singing Range Test",
            "url": "https://singingrangetest.com/",
            "overlap": "Register map, passaggio, tessitura, pitch notation",
            "compare_fields": ["low_note", "high_note", "register"],
        },
        {
            "name": "MixButton Vocal Range Test",
            "url": "https://mixbutton.com/music-tools/frequency-and-pitch/vocal-range-test",
            "overlap": "Record low/high separately, voice type classification",
            "compare_fields": ["low_note", "high_note"],
        },
    ],
    "scale_matcher": [
        {
            "name": "Found Guitar Capo Transposer",
            "url": "https://found-tools.com/en/guitar-capo-transposer/",
            "overlap": "Capo fret + transposed open-chord shapes for target key",
            "compare_fields": ["vocal_key", "capo_fret", "chord_shape", "transposed_progression"],
        },
        {
            "name": "Guitar Tool Hub Capo Calculator",
            "url": "https://guitartoolhub.com/capo-calculator",
            "overlap": "Capo position from song key + open shapes (G, C, D, A, E)",
            "compare_fields": ["vocal_key", "capo_fret", "chord_shape"],
        },
        {
            "name": "Capo Captain (Songnotes)",
            "url": "https://songnotes.net/tools/capo-captain",
            "overlap": "Capo + chord family combinations for any fret",
            "compare_fields": ["capo_fret", "chord_shape"],
        },
        {
            "name": "TransposeChord",
            "url": "https://transposechord.com/",
            "overlap": "Chord transposition and capo helper (manual key entry)",
            "compare_fields": ["transposed_progression"],
        },
    ],
    "voice_comparison": [
        {
            "name": "Pitch Detector — Pitch Accuracy Checker",
            "url": "https://pitchdetector.com/pitch-accuracy-checker/",
            "overlap": "Real-time cents deviation vs target note (±5 pro, ±15 live)",
            "compare_fields": ["mean_cents_deviation"],
        },
        {
            "name": "Pitchy (MWM)",
            "url": "https://mwm.ai/apps/pitchy-sing-on-pitch/6749875369",
            "overlap": "Upload song, compare your pitch line to original vocals",
            "compare_fields": ["mean_cents_deviation"],
        },
        {
            "name": "PitchMonitor (MWM)",
            "url": "https://mwm.ai/apps/pitchmonitor-vocal-visual/6751228637",
            "overlap": "Dual-track reference comparison with live cents meter",
            "compare_fields": ["mean_cents_deviation", "has_vibrato"],
        },
    ],
}


def _ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def _to_json_safe(value: Any) -> Any:
    """Convert numpy/scalar types to native JSON-serializable Python values."""
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Drop large plot arrays and coerce types before persisting."""
    cleaned = dict(metrics)
    for key in ("ref_f0_plot", "user_f0_plot"):
        if key in cleaned:
            series = cleaned.pop(key)
            if isinstance(series, list) and series:
                voiced = [v for v in series if v and v > 0]
                cleaned[f"{key}_frame_count"] = len(series)
                if voiced:
                    cleaned[f"{key}_median_hz"] = float(sorted(voiced)[len(voiced) // 2])
    return _to_json_safe(cleaned)


def save_result(
    feature: str,
    metrics: dict[str, Any],
    *,
    input_meta: Optional[dict[str, Any]] = None,
    notes: str = "",
) -> dict[str, Any]:
    """
    Save one analysis run as JSON under results/.
    Returns the saved record (including id and file path).
    """
    _ensure_results_dir()
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature": feature,
        "app": "RayMozic",
        "input": input_meta or {},
        "metrics": _sanitize_metrics(metrics),
        "benchmark": {
            "pitch_tiers_cents": PITCH_BENCHMARKS,
            "competitors": COMPETITOR_SERVICES.get(feature, []),
            "notes": notes,
        },
    }
    filename = f"{record['timestamp'].replace(':', '-').replace('.', '-')}_{feature}_{record['id'][:8]}.json"
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(_to_json_safe(record), indent=2), encoding="utf-8")
    record["file"] = str(path)
    return record


def list_results(feature: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
    """Load saved results, newest first."""
    if not RESULTS_DIR.exists():
        return []

    records: list[dict[str, Any]] = []
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if feature and data.get("feature") != feature:
            continue
        data["file"] = str(path)
        records.append(data)
        if len(records) >= limit:
            break
    return records


def pitch_accuracy_tier(mean_cents: float) -> str:
    """Map mean cents deviation to a human-readable benchmark tier."""
    if mean_cents <= PITCH_BENCHMARKS["professional_cents"]:
        return "professional"
    if mean_cents <= PITCH_BENCHMARKS["acceptable_live_cents"]:
        return "acceptable_live"
    if mean_cents <= PITCH_BENCHMARKS["noticeable_off_cents"]:
        return "borderline"
    if mean_cents <= PITCH_BENCHMARKS["oot_singing_threshold_cents"]:
        return "noticeable_off"
    return "out_of_tune"
