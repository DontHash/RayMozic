"""Acoustic feature extraction for voice-register analysis.

Registers (chest / mixed / head / falsetto) and belt vs. head are not just a
function of pitch: they correlate with the *spectral* signature of the glottal
source and vocal tract. This module extracts robust, real-time-friendly
features that let us classify register acoustically rather than by pitch alone:

- f0 statistics (median / spread)                  — pitch context
- spectral tilt (dB per kHz)                        — how fast harmonics decay
- harmonic-to-noise ratio (HNR, Boersma method)     — tone clarity / breathiness
- high-frequency energy ratio + spectral centroid   — spectral balance / brightness
- nasal band energy ratio                           — nasal resonance flag

Grounding (see dsp_music_project.md "References"):
- Chest-like phonation: shallower spectral tilt, stronger high harmonics.
- Head/falsetto: steeper tilt, energy concentrated near the fundamental.
- Belt: high pitch but chest-like (shallow) spectral tilt + strong HF energy.
This is the "feature vector -> weighted rule" approach; formant/glottal
inverse-filtering (QA, closed quotient) is left as future work.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import welch


@dataclass
class VoiceFeatures:
    f0_median: float          # Hz, median of voiced frames
    f0_low: float             # Hz, 5th percentile
    f0_high: float            # Hz, 95th percentile
    spectral_tilt: float      # dB per kHz (negative = harmonics decay with freq)
    hnr_db: float             # harmonic-to-noise ratio in dB
    hf_ratio: float           # fraction of energy above 1.5 kHz (0..1)
    centroid_hz: float        # spectral centroid (brightness)
    nasal_ratio: float        # nasal-band energy vs mid-band
    voiced_fraction: float    # fraction of frames that were voiced


def estimate_f0_track(audio: np.ndarray, sr: int) -> np.ndarray:
    """Voiced f0 values (Hz) via YIN over the singing range."""
    f0 = librosa.yin(
        audio,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
    )
    return f0[np.isfinite(f0) & (f0 > 0)]


def spectral_tilt(audio: np.ndarray, sr: int, fmin: float = 100.0, fmax: float = 5000.0) -> float:
    """
    Spectral tilt as the slope of a linear fit to the log-magnitude spectrum
    (dB) vs frequency, expressed in dB per kHz. A less-negative tilt means the
    voice carries more high-harmonic energy (chest/belt-like); a steeply
    negative tilt means energy is concentrated low (head/falsetto-like).
    """
    freqs, psd = welch(audio, fs=sr, nperseg=min(2048, len(audio)))
    band = (freqs >= fmin) & (freqs <= fmax) & (psd > 0)
    if np.count_nonzero(band) < 8:
        return 0.0
    f = freqs[band]
    db = 10.0 * np.log10(psd[band])
    # Weight the fit by linear power so harmonic peaks dominate and the near-
    # silent bins between harmonics don't drag the slope down (which would make
    # every harmonic tone look artificially steep).
    weights = psd[band]
    slope_per_hz = np.polyfit(f, db, 1, w=weights)[0]
    return float(slope_per_hz * 1000.0)  # dB per kHz


def harmonic_to_noise_ratio(audio: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 1200.0) -> float:
    """
    HNR via Boersma's short-term autocorrelation method:
        HNR = 10 * log10( r_max / (1 - r_max) )
    where r_max is the normalized autocorrelation peak at the fundamental lag.
    High HNR = clean, periodic tone; low HNR = breathy/noisy.
    """
    x = np.asarray(audio, dtype=np.float64)
    x = x - np.mean(x)
    if len(x) < 512 or np.allclose(x, 0):
        return 0.0

    window = np.hanning(len(x))
    xw = x * window
    acf = np.correlate(xw, xw, mode="full")[len(xw) - 1:]
    if acf[0] <= 0:
        return 0.0
    acf = acf / acf[0]

    min_lag = max(int(sr / fmax), 1)
    max_lag = min(int(sr / fmin), len(acf) - 1)
    if max_lag <= min_lag:
        return 0.0

    r_max = float(np.max(acf[min_lag:max_lag]))
    r_max = float(np.clip(r_max, 1e-6, 0.999999))
    return 10.0 * np.log10(r_max / (1.0 - r_max))


def spectral_balance(audio: np.ndarray, sr: int, split_hz: float = 1500.0):
    """Return (hf_ratio, centroid_hz): energy fraction above split_hz and the
    spectral centroid (brightness)."""
    freqs, psd = welch(audio, fs=sr, nperseg=min(2048, len(audio)))
    total = float(np.sum(psd))
    if total <= 0:
        return 0.0, 0.0
    hf = float(np.sum(psd[freqs >= split_hz]))
    centroid = float(np.sum(freqs * psd) / total)
    return hf / total, centroid


def nasal_energy_ratio(audio: np.ndarray, sr: int) -> float:
    """
    Nasal resonance proxy: energy in the nasal murmur / anti-formant bands
    (~250 Hz and ~2-3 kHz) relative to the mid band (500-1500 Hz).
    """
    freqs, psd = welch(audio, fs=sr, nperseg=min(1024, len(audio)))
    low = float(np.mean(psd[(freqs > 200) & (freqs < 300)]) or 0.0)
    high = float(np.mean(psd[(freqs > 2000) & (freqs < 3000)]) or 0.0)
    mid = float(np.mean(psd[(freqs > 500) & (freqs < 1500)]) or 0.0)
    if mid <= 0:
        mid = 1e-10
    return (low + high) / (2.0 * mid)


def extract_features(audio: np.ndarray, sr: int) -> VoiceFeatures:
    """Extract the full acoustic feature vector for register analysis."""
    f0_track = estimate_f0_track(audio, sr)
    n_frames = max(len(librosa.yin(audio, fmin=65.0, fmax=2093.0, sr=sr)), 1)
    voiced_fraction = float(len(f0_track) / n_frames) if n_frames else 0.0

    if len(f0_track) == 0:
        f0_median = f0_low = f0_high = 0.0
    else:
        f0_median = float(np.median(f0_track))
        f0_low = float(np.percentile(f0_track, 5))
        f0_high = float(np.percentile(f0_track, 95))

    tilt = spectral_tilt(audio, sr)
    hnr = harmonic_to_noise_ratio(audio, sr)
    hf_ratio, centroid = spectral_balance(audio, sr)
    nasal = nasal_energy_ratio(audio, sr)

    return VoiceFeatures(
        f0_median=f0_median,
        f0_low=f0_low,
        f0_high=f0_high,
        spectral_tilt=tilt,
        hnr_db=hnr,
        hf_ratio=hf_ratio,
        centroid_hz=centroid,
        nasal_ratio=nasal,
        voiced_fraction=voiced_fraction,
    )
