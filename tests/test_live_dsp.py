"""Tests for live pitch detection, note mapping, and the guitar tuner."""
import numpy as np
import pytest

from utils.dsp_live import (
    compute_spectrum,
    detect_pitch,
    dominant_frequency,
    frequency_to_note,
)
from utils.tuner import (
    TUNINGS,
    get_targets,
    nearest_string,
    note_to_frequency,
    tuning_direction,
)

SR = 22050


def _sine(freq, seconds=0.5, sr=SR, amp=0.5):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.parametrize("freq,expected", [(440.0, "A4"), (261.63, "C4"), (82.41, "E2")])
def test_frequency_to_note_labels(freq, expected):
    reading = frequency_to_note(freq)
    assert reading.note_label == expected
    assert abs(reading.cents) < 5


def test_frequency_to_note_handles_invalid():
    assert frequency_to_note(0).voiced is False
    assert frequency_to_note(-10).note_label == ""


@pytest.mark.parametrize("freq", [110.0, 220.0, 440.0])
def test_detect_pitch_sine(freq):
    reading = detect_pitch(_sine(freq), SR)
    assert reading.voiced
    # Within a quartertone of the true frequency.
    assert abs(reading.frequency - freq) / freq < 0.03


def test_detect_pitch_low_e_guitar():
    reading = detect_pitch(_sine(82.41), SR, fmin=60, fmax=1000)
    assert reading.voiced
    assert reading.note_label == "E2"


def test_detect_pitch_silence_is_unvoiced():
    silent = np.zeros(SR // 2, dtype=np.float32)
    assert detect_pitch(silent, SR).voiced is False


def test_compute_spectrum_peak_matches_tone():
    freqs, mags = compute_spectrum(_sine(440.0), SR, fmax=2000.0)
    assert abs(dominant_frequency(freqs, mags) - 440.0) < 15


def test_note_to_frequency_roundtrip():
    assert abs(note_to_frequency("A4") - 440.0) < 0.01
    assert abs(note_to_frequency("E2") - 82.41) < 0.5


def test_tuner_nearest_string_standard():
    match = nearest_string(110.0, "Standard (E A D G B E)")
    assert match is not None
    assert match.string.label == "A2"
    assert match.in_tune


def test_tuner_flat_string_direction():
    # Slightly flat A2 (~108 Hz).
    match = nearest_string(108.0, "Standard (E A D G B E)")
    assert match.cents_off < 0
    assert "up" in tuning_direction(match.cents_off).lower()


def test_all_tunings_have_six_strings():
    for name in TUNINGS:
        assert len(get_targets(name)) == 6
