import librosa
import numpy as np

from utils.pitch_utils import hz_to_note
from utils.audio_io import validate_audio_signal, AudioValidationError
from utils.voice_features import extract_features
from utils.register import classify_register_features


def analyze_vocal_range(audio: np.ndarray, sr: int = 22050) -> dict:
    """
    Analyze vocal range and register.

    Range comes from the YIN f0 track (5th/95th percentile + median). The
    register is classified acoustically (spectral tilt, HF energy, HNR, pitch)
    rather than by pitch alone — see utils/register.classify_register_features.
    """
    try:
        validate_audio_signal(audio, sr, label="Vocal")
    except AudioValidationError as exc:
        return {"error": str(exc)}

    feat = extract_features(audio, sr)
    if feat.f0_median <= 0:
        return {"error": "No pitched voice detected."}

    reg = classify_register_features(feat)

    return {
        "low_note": hz_to_note(feat.f0_low),
        "high_note": hz_to_note(feat.f0_high),
        "modal_note": hz_to_note(feat.f0_median),
        "low_hz": feat.f0_low,
        "high_hz": feat.f0_high,
        "modal_hz": feat.f0_median,
        "register": reg.register,
        "register_confidence": reg.confidence,
        "is_belt": reg.is_belt,
        "is_nasal": reg.is_nasal,
        "register_reasons": reg.reasons,
        "register_scores": reg.scores,
        # Acoustic features (exposed for display / benchmarking).
        "spectral_tilt_db_per_khz": round(feat.spectral_tilt, 2),
        "hnr_db": round(feat.hnr_db, 2),
        "hf_energy_ratio": round(feat.hf_ratio, 3),
        "spectral_centroid_hz": round(feat.centroid_hz, 1),
    }
