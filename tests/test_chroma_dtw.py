"""Quick checks for chroma sanitization and voice comparison guards."""
import numpy as np
import pytest

from utils.audio_io import AudioValidationError, validate_audio_signal
from utils.chroma_utils import extract_chroma_stft, sanitize_chroma, extract_mean_chroma
from features.comparator import compare_voices, VoiceComparisonError


SR = 22050


def _tonal_audio(duration_sec: float = 2.0, freq: float = 220.0) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(SR * duration_sec), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_validate_rejects_silent_audio():
    silent = np.zeros(SR)
    with pytest.raises(AudioValidationError, match="silent"):
        validate_audio_signal(silent, SR, label="Test")


def test_validate_rejects_short_audio():
    short = _tonal_audio(0.1)
    with pytest.raises(AudioValidationError, match="too short"):
        validate_audio_signal(short, SR, label="Test")


def test_sanitize_chroma_zero_norm_columns():
    chroma = np.zeros((12, 5))
    out = sanitize_chroma(chroma)
    assert not np.isnan(out).any()
    norms = np.linalg.norm(out, axis=0)
    assert np.all(norms > 0)


def test_extract_chroma_stft_on_silent_uniform_columns():
    silent = np.zeros(SR * 2)
    chroma = extract_chroma_stft(silent, SR)
    assert chroma.shape[0] == 12
    assert chroma.shape[1] >= 1
    assert not np.isnan(chroma).any()
    norms = np.linalg.norm(chroma, axis=0)
    assert np.all(norms > 0)


def test_compare_voices_rejects_silent_user():
    ref = _tonal_audio()
    user = np.zeros(SR * 2)
    with pytest.raises(AudioValidationError):
        compare_voices(user, ref, sr=SR)


def test_compare_voices_tonal_audio():
    ref = _tonal_audio(freq=220.0)
    user = _tonal_audio(freq=225.0)
    result = compare_voices(user, ref, sr=SR)
    assert "mean_cents_deviation" in result
    assert np.isfinite(result["mean_cents_deviation"])
    assert len(result["ref_f0_plot"]) > 0


def test_extract_mean_chroma_finite():
    audio = _tonal_audio()
    mean = extract_mean_chroma(audio, SR)
    assert mean.shape == (12,)
    assert np.isfinite(mean).all()
