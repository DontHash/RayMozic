"""Tests for progression capo / scale map (fixed sounding chords)."""

import pytest

from utils.progression_capo_map import (
    build_progression_capo_map,
    compatible_keys,
    finger_shapes_for_sounding,
)


def test_finger_shapes_capo_zero():
    assert finger_shapes_for_sounding(["G", "Em", "C", "D"], 0) == ["G", "Em", "C", "D"]


def test_finger_shapes_capo_three():
    assert finger_shapes_for_sounding(["G", "Em", "C", "D"], 3) == ["E", "C#m", "A", "B"]


def test_sounding_fixed_across_capo_rows():
    m = build_progression_capo_map("G Em C D")
    for row in m["capo_rows"]:
        assert row["sounding_chords"] == ["G", "Em", "C", "D"]


def test_finger_shapes_differ_by_capo():
    m = build_progression_capo_map("G Em C D")
    fingers = [tuple(r["finger_chords"]) for r in m["capo_rows"]]
    assert len(set(fingers)) > 1


def test_infers_major_pop_progression():
    m = build_progression_capo_map("G Em C D")
    assert m["diatonic_fit"] >= 0.75
    keys = {f"{c['key']} {c['mode']}" for c in m["compatible_keys"]}
    assert "G major" in keys


def test_compatible_keys_includes_g_major():
    keys = compatible_keys(["G", "Em", "C", "D"])
    assert any(c["key"] == "G" and c["mode"] == "major" for c in keys)


def test_shape_families_count():
    m = build_progression_capo_map("G Em C D")
    assert len(m["shape_family_rows"]) == 6
