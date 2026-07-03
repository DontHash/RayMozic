"""Tests for capo-as-transpose-up planner (fixed finger shapes)."""

import numpy as np
import pytest

from utils.play_along import (
    sounding_at_capo,
    build_capo_options,
    build_play_plan,
    infer_key_from_chords,
    vocal_match_score,
)
from features.scale_matcher import transpose_progression

SR = 22050


def test_sounding_at_capo_zero_is_identity():
    finger = ["Am", "Em", "Dm", "F"]
    assert sounding_at_capo(finger, 0) == finger


def test_sounding_at_capo_three_transposes_up():
    finger = ["Am", "Em", "Dm", "F"]
    assert sounding_at_capo(finger, 3) == ["Cm", "Gm", "Fm", "G#"]


def test_finger_shapes_stay_same_across_capo_rows():
    finger = ["Am", "Em", "Dm", "F"]
    rows = build_capo_options(finger, "C", "minor")
    for row in rows:
        assert row["finger_chords"] == finger


def test_sounding_chords_differ_by_capo():
    finger = ["Am", "Em", "Dm", "F"]
    rows = build_capo_options(finger, "C", "minor")
    soundings = [tuple(r["sounding_chords"]) for r in rows]
    assert len(set(soundings)) > 1  # not all identical


def test_capo_three_cm_progression_infers_c_minor():
    sounding = sounding_at_capo(["Am", "Em", "Dm", "F"], 3)
    key, mode, fit = infer_key_from_chords(sounding)
    assert key == "C"
    assert mode == "minor"
    assert fit >= 0.5


def test_vocal_match_exact_key():
    score = vocal_match_score("C", "minor", "C", "minor", 1.0)
    assert score >= 0.95


def test_build_play_plan_with_finger_progression():
    t = np.linspace(0, 2.0, int(SR * 2), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 261.63 * t).astype(np.float32)
    plan = build_play_plan(audio, sr=SR, finger_progression="Am Em Dm F")
    assert plan["finger_progression"] == ["Am", "Em", "Dm", "F"]
    assert plan["best_capo"] is not None
    assert len(plan["capo_by_fret"]) == 8
    assert "capo_zero_ok" in plan
    assert plan["alternatives"]


def test_transpose_roundtrip():
    prog = "Am Em Dm F"
    up = transpose_progression(prog, 3)
    down = transpose_progression(up, -3)
    assert down == prog
