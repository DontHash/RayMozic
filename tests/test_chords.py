"""Tests for HPS detection, pitch smoothing, chord voicings, and progressions."""
import numpy as np
import pytest

from utils.dsp_live import PitchSmoother, detect_pitch_hps, frequency_to_note
from utils.music_theory import chord_pitch_classes, diatonic_chords, name_to_pc, parse_chord
from utils.chord_voicing import recommend_voicings
from utils.progression import generate_progression, parse_degrees
from utils.tuner import TUNINGS, get_targets, note_to_frequency

SR = 22050


def _harmonic_tone(f0, seconds=0.5, sr=SR):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * f0 * t)
    sig += 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
    sig += 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
    return sig.astype(np.float32)


# --- HPS detection -----------------------------------------------------------

@pytest.mark.parametrize("f0,expected", [(110.0, "A2"), (196.0, "G3"), (220.0, "A3")])
def test_hps_detects_fundamental_not_harmonic(f0, expected):
    reading = detect_pitch_hps(_harmonic_tone(f0), SR, fmin=60, fmax=1000)
    assert reading.voiced
    assert reading.note_label == expected


def test_hps_silence_unvoiced():
    assert detect_pitch_hps(np.zeros(SR // 2, dtype=np.float32), SR).voiced is False


def test_hps_respects_a4_reference():
    # At A4=432, 432 Hz should read as A4 with ~0 cents.
    reading = detect_pitch_hps(_harmonic_tone(432.0), SR, fmin=60, fmax=1000, a4=432.0)
    assert reading.note_label == "A4"
    assert abs(reading.cents) < 10


# --- Pitch smoother ----------------------------------------------------------

def test_smoother_octave_correction():
    sm = PitchSmoother()
    for f in [220.0, 221.0, 219.0, 220.0]:
        sm.update(f, True)
    # A single octave-up spike should be pulled back near 220.
    out = sm.update(440.0, True)
    assert abs(out - 220.0) < 20


def test_smoother_dropout_hold():
    sm = PitchSmoother(max_hold=3)
    for f in [110.0, 110.0, 110.0]:
        sm.update(f, True)
    held = sm.update(0.0, False)  # brief dropout
    assert held is not None and abs(held - 110.0) < 5


def test_smoother_releases_after_long_silence():
    sm = PitchSmoother(max_hold=2)
    sm.update(110.0, True)
    for _ in range(5):
        out = sm.update(0.0, False)
    assert out is None


# --- Music theory ------------------------------------------------------------

def test_name_to_pc_flats():
    assert name_to_pc("Bb") == name_to_pc("A#")
    assert name_to_pc("C") == 0


def test_parse_chord_qualities():
    assert parse_chord("Am")[2] == [0, 3, 7]
    assert parse_chord("G7")[2] == [0, 4, 7, 10]
    assert chord_pitch_classes("C") == {0, 4, 7}


def test_diatonic_major():
    chords = diatonic_chords(name_to_pc("C"), "major")
    names = [c["name"] for c in chords]
    assert names == ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]


# --- Chord voicings ----------------------------------------------------------

@pytest.mark.parametrize("chord,shape", [
    ("E", "0 2 2 1 0 0"),
    ("Em", "0 2 2 0 0 0"),
    ("C", "x 3 2 0 1 0"),
    ("Am", "x 0 2 2 1 0"),
    ("D", "x x 0 2 3 2"),
])
def test_voicing_matches_standard_open_chord(chord, shape):
    top = recommend_voicings(chord)[0]
    assert top.diagram_str == shape


def test_voicing_only_chord_tones():
    for v in recommend_voicings("G"):
        sounding = {(v.open_midis[i] + f) % 12 for i, f in enumerate(v.frets) if f is not None}
        assert sounding <= chord_pitch_classes("G")


def test_voicing_full_coverage_and_finger_limit():
    top = recommend_voicings("C")[0]
    assert top.coverage == 1.0
    assert top.fingers <= 4


# --- Progressions ------------------------------------------------------------

def test_progression_pop():
    result = generate_progression("G", "major", "1-5-6-4")
    assert result["chord_names"] == ["G", "D", "Em", "C"]


def test_progression_roman_parsing():
    assert parse_degrees("I-V-vi-IV") == [1, 5, 6, 4]
    assert parse_degrees("1 4 5") == [1, 4, 5]


def test_progression_sevenths():
    result = generate_progression("C", "major", "2-5-1", sevenths=True)
    assert result["chord_names"] == ["Dm7", "G7", "Cmaj7"]


# --- Tuner extensions --------------------------------------------------------

def test_low_c_tuning_present():
    assert any("Low C" in name for name in TUNINGS)


def test_a4_reference_shifts_targets():
    std_440 = get_targets("Standard (E A D G B E)", a4=440.0)
    std_432 = get_targets("Standard (E A D G B E)", a4=432.0)
    assert std_432[0].frequency < std_440[0].frequency


def test_custom_tuning_targets():
    targets = get_targets(custom_notes=["D2", "A2", "D3", "G3", "A3", "D4"])
    assert len(targets) == 6
    assert targets[0].label == "D2"
