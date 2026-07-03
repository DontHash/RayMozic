"""Tests for acoustic voice-feature extraction and register classification."""

import numpy as np
import pytest

from utils.voice_features import (
    extract_features,
    harmonic_to_noise_ratio,
    spectral_tilt,
)
from utils.register import classify_register_features

SR = 22050


def _harmonic_tone(f0, amps, secs=1.0, noise=0.0):
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    sig = np.zeros_like(t)
    for k, a in enumerate(amps, start=1):
        if f0 * k < SR / 2:
            sig += a * np.sin(2 * np.pi * f0 * k * t)
    if noise:
        sig += noise * np.random.default_rng(0).standard_normal(len(t))
    return (sig / np.max(np.abs(sig)) * 0.5).astype(np.float32)


def test_shallow_vs_steep_tilt():
    # Many strong harmonics -> shallow (less negative) tilt.
    shallow = _harmonic_tone(160, [1, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3])
    # Energy concentrated at f0 -> steep tilt.
    steep = _harmonic_tone(520, [1, 0.15, 0.05])
    assert spectral_tilt(shallow, SR) > spectral_tilt(steep, SR)


def test_hnr_clean_higher_than_noisy():
    clean = _harmonic_tone(200, [1, 0.5, 0.3])
    noisy = _harmonic_tone(200, [1, 0.5, 0.3], noise=0.8)
    assert harmonic_to_noise_ratio(clean, SR) > harmonic_to_noise_ratio(noisy, SR)


def test_chest_low_strong_harmonics_classifies_chest():
    chest = _harmonic_tone(160, [1, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.3])
    reg = classify_register_features(extract_features(chest, SR))
    assert reg.register == "Chest"
    assert reg.confidence > 0.4
    assert reg.reasons


def test_head_high_steep_tilt_classifies_head():
    head = _harmonic_tone(520, [1, 0.2, 0.08, 0.03])
    reg = classify_register_features(extract_features(head, SR))
    assert reg.register in {"Head", "Falsetto"}


def test_belt_flag_high_pitch_shallow_tilt():
    belt = _harmonic_tone(450, [1, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4])
    reg = classify_register_features(extract_features(belt, SR))
    assert reg.is_belt is True


def test_scores_sum_and_reasons_present():
    tone = _harmonic_tone(220, [1, 0.6, 0.4, 0.3])
    reg = classify_register_features(extract_features(tone, SR))
    assert set(reg.scores.keys()) == {"Chest", "Mixed", "Head", "Falsetto"}
    assert all(isinstance(v, float) for v in reg.scores.values())
    assert isinstance(reg.reasons, list) and reg.reasons
