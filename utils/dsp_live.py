"""Real-time / frame-based pitch detection and FFT spectrum for live feedback.

Designed to run on short audio frames (a rolling buffer) so it can drive a
tuner needle and a live spectrum display, GuitarTuna-style.

Primary pitch estimator: normalized autocorrelation with a "first strong peak"
rule (robust against octave errors and accurate at low frequencies like the
guitar low-E at ~82 Hz). The FFT is used for the spectrum visualization and to
mark which frequency bin drives the detected note.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Search range covering guitar low-E (~82 Hz) up through high vocal / harmonics.
DEFAULT_FMIN = 60.0
DEFAULT_FMAX = 1200.0

# Minimum RMS to consider a frame "voiced" (below this = silence/noise).
SILENCE_RMS = 2e-3

# Default concert pitch reference (adjustable 432..445, both repos support this).
DEFAULT_A4 = 440.0


@dataclass
class PitchReading:
    """Result of analyzing one audio frame."""

    frequency: float          # detected fundamental in Hz (0.0 if none)
    note: str                 # e.g. "A" (empty if none)
    octave: int               # scientific octave (e.g. 4 for A4)
    note_label: str           # e.g. "A4" (empty if none)
    midi: int                 # nearest MIDI note number
    cents: float              # signed deviation from the nearest note (-50..+50)
    target_hz: float          # ideal frequency of the nearest note
    confidence: float         # 0..1 autocorrelation peak strength
    voiced: bool              # True if a pitch was confidently detected


def frequency_to_note(freq: float, a4: float = DEFAULT_A4) -> PitchReading:
    """Map a frequency to the nearest equal-tempered note for the given A4 reference."""
    if freq is None or freq <= 0 or not np.isfinite(freq):
        return PitchReading(0.0, "", 0, "", 0, 0.0, 0.0, 0.0, False)

    midi_float = 69.0 + 12.0 * np.log2(freq / a4)
    midi = int(round(midi_float))
    target_hz = a4 * (2.0 ** ((midi - 69) / 12.0))
    cents = 1200.0 * np.log2(freq / target_hz)

    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return PitchReading(
        frequency=float(freq),
        note=name,
        octave=int(octave),
        note_label=f"{name}{octave}",
        midi=midi,
        cents=float(cents),
        target_hz=float(target_hz),
        confidence=0.0,
        voiced=True,
    )


def _parabolic_interpolate(values: np.ndarray, idx: int) -> float:
    """Sub-sample peak location via parabolic interpolation around idx."""
    if idx <= 0 or idx >= len(values) - 1:
        return float(idx)
    a, b, c = values[idx - 1], values[idx], values[idx + 1]
    denom = a - 2.0 * b + c
    if denom == 0:
        return float(idx)
    return idx + 0.5 * (a - c) / denom


def detect_pitch(
    frame: np.ndarray,
    sr: int,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    confidence_threshold: float = 0.5,
    a4: float = DEFAULT_A4,
) -> PitchReading:
    """
    Estimate the fundamental frequency of one audio frame.

    Uses windowed normalized autocorrelation and selects the first lag whose
    peak exceeds a fraction of the global maximum, which avoids picking an
    octave multiple of the true period.
    """
    if frame is None or len(frame) < 256:
        return PitchReading(0.0, "", 0, "", 0, 0.0, 0.0, 0.0, False)

    x = np.asarray(frame, dtype=np.float64)
    x = x - np.mean(x)

    rms = float(np.sqrt(np.mean(x * x)))
    if not np.isfinite(rms) or rms < SILENCE_RMS:
        return PitchReading(0.0, "", 0, "", 0, 0.0, 0.0, 0.0, False)

    n = len(x)
    windowed = x * np.hanning(n)

    # Autocorrelation via FFT (Wiener-Khinchin).
    fft_size = next_fast_len(2 * n)
    spec = rfft(windowed, fft_size)
    acf = irfft(spec * np.conjugate(spec), fft_size)[:n]

    if acf[0] <= 0:
        return PitchReading(0.0, "", 0, "", 0, 0.0, 0.0, 0.0, False)
    acf = acf / acf[0]  # normalize so zero-lag == 1.0

    min_lag = max(int(sr / fmax), 1)
    max_lag = min(int(sr / fmin), n - 1)
    if max_lag <= min_lag:
        return PitchReading(0.0, "", 0, "", 0, 0.0, 0.0, 0.0, False)

    search = acf[min_lag:max_lag]
    global_max = float(np.max(search))
    if global_max <= 0:
        return PitchReading(0.0, "", 0, "", 0, 0.0, 0.0, 0.0, False)

    # First peak that reaches 80% of the global max = true period.
    threshold = 0.8 * global_max
    peak_rel = int(np.argmax(search))
    for i in range(1, len(search) - 1):
        if (
            search[i] > threshold
            and search[i] >= search[i - 1]
            and search[i] >= search[i + 1]
        ):
            peak_rel = i
            break

    peak_lag = peak_rel + min_lag
    interp_lag = _parabolic_interpolate(acf, peak_lag)
    if interp_lag <= 0:
        return PitchReading(0.0, "", 0, "", 0, 0.0, 0.0, 0.0, False)

    freq = sr / interp_lag
    confidence = float(np.clip(global_max, 0.0, 1.0))

    reading = frequency_to_note(freq, a4)
    reading.confidence = confidence
    reading.voiced = confidence >= confidence_threshold
    if not reading.voiced:
        # Keep the frequency for display but flag as low-confidence.
        return reading
    return reading


def detect_pitch_hps(
    frame: np.ndarray,
    sr: int,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    num_hps: int = 3,
    zero_pad: int = 3,
    clarity_threshold: float = 4.0,
    a4: float = DEFAULT_A4,
) -> PitchReading:
    """
    Harmonic Product Spectrum pitch detection (mechanism from TomSchimansky's
    GuitarTuner): a zero-padded, Hann-windowed FFT is multiplied by decimated
    copies of itself so harmonics reinforce the true fundamental, then the
    loudest bin is the pitch. Zero-padding boosts frequency resolution for
    sub-cent accuracy near A4.
    """
    if frame is None or len(frame) < 256:
        return PitchReading(0.0, "", 0, "", 0, 0.0, 0.0, 0.0, False)

    x = np.asarray(frame, dtype=np.float64)
    x = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x * x)))
    if not np.isfinite(rms) or rms < SILENCE_RMS:
        return PitchReading(0.0, "", 0, "", 0, 0.0, 0.0, 0.0, False)

    n = len(x)
    windowed = x * np.hanning(n)
    padded = np.pad(windowed, (0, n * max(zero_pad, 0)), "constant")

    mag = np.abs(rfft(padded))
    freqs = rfftfreq(len(padded), d=1.0 / sr)

    # High-pass: ignore everything below fmin (kills mains hum / DC lobe).
    mag[freqs < fmin] = 0.0

    # Harmonic Product Spectrum.
    hps = mag.copy()
    for i in range(2, num_hps + 1):
        decimated = mag[::i]
        hps[: len(decimated)] *= decimated

    # Restrict the peak search to the expected fundamental range.
    hps[freqs > fmax] = 0.0
    if not np.any(hps > 0):
        return PitchReading(0.0, "", 0, "", 0, 0.0, 0.0, 0.0, False)

    peak = int(np.argmax(hps))
    interp = _parabolic_interpolate(hps, peak)
    bin_hz = freqs[1] - freqs[0] if len(freqs) > 1 else sr / len(padded)
    freq = interp * bin_hz

    # Clarity = peak prominence over the median of active bins.
    active = hps[hps > 0]
    median = float(np.median(active)) if active.size else 1.0
    clarity = float(hps[peak] / median) if median > 0 else 0.0

    reading = frequency_to_note(freq, a4)
    reading.confidence = float(np.clip(clarity / (clarity + 10.0), 0.0, 1.0))
    reading.voiced = clarity >= clarity_threshold and freq > 0
    return reading


class PitchSmoother:
    """Temporal smoothing for a live pitch stream (mechanism from MoChord).

    Combines median smoothing (jitter), a short dropout hold (prevents UI
    flicker when a frame is briefly unvoiced), and octave-jump correction
    (rejects single-frame harmonic/subharmonic flips).
    """

    def __init__(self, median_window: int = 5, max_hold: int = 3, octave_tol_cents: float = 45.0):
        self.history: deque[float] = deque(maxlen=max(median_window, 1))
        self.max_hold = max_hold
        self.octave_tol_cents = octave_tol_cents
        self.last_stable: Optional[float] = None
        self._hold = 0

    def reset(self) -> None:
        self.history.clear()
        self.last_stable = None
        self._hold = 0

    def update(self, freq: float, voiced: bool) -> Optional[float]:
        """Feed one frame; returns the smoothed frequency or None if silent."""
        if not voiced or freq is None or freq <= 0:
            self._hold += 1
            if self._hold <= self.max_hold and self.last_stable:
                return self.last_stable  # brief dropout hold
            self.history.clear()
            return None

        self._hold = 0
        corrected = self._correct_octave(freq)
        self.history.append(corrected)
        smoothed = float(np.median(self.history))
        self.last_stable = smoothed
        return smoothed

    def _correct_octave(self, freq: float) -> float:
        """Snap an octave-away reading back toward the recent stable pitch."""
        if self.last_stable is None or self.last_stable <= 0:
            return freq
        best = freq
        best_cents = abs(1200.0 * np.log2(freq / self.last_stable))
        for factor in (0.5, 2.0):
            cand = freq * factor
            cents = abs(1200.0 * np.log2(cand / self.last_stable))
            if cents < best_cents:
                best_cents, best = cents, cand
        # Only accept the octave correction if it lands very close to stable.
        if best is not freq and best_cents <= self.octave_tol_cents:
            return best
        return freq


def compute_spectrum(
    frame: np.ndarray,
    sr: int,
    fmax: float = 2000.0,
    n_fft: Optional[int] = None,
):
    """
    Compute a single-sided magnitude spectrum for display.

    Returns (freqs, magnitudes) truncated to [0, fmax]. Magnitudes are linear
    and normalized to a 0..1 peak so the plot scale stays stable frame-to-frame.
    """
    if frame is None or len(frame) < 64:
        return np.array([0.0]), np.array([0.0])

    x = np.asarray(frame, dtype=np.float64)
    x = x - np.mean(x)
    n = len(x)
    if n_fft is None:
        n_fft = next_fast_len(n)

    windowed = x * np.hanning(n)
    mags = np.abs(rfft(windowed, n_fft))
    freqs = rfftfreq(n_fft, d=1.0 / sr)

    mask = freqs <= fmax
    freqs = freqs[mask]
    mags = mags[mask]

    peak = float(np.max(mags)) if mags.size else 0.0
    if peak > 0:
        mags = mags / peak
    return freqs, mags


def dominant_frequency(freqs: np.ndarray, mags: np.ndarray, fmin: float = DEFAULT_FMIN) -> float:
    """Return the frequency of the largest spectral peak above fmin."""
    if freqs.size == 0:
        return 0.0
    valid = freqs >= fmin
    if not np.any(valid):
        return 0.0
    idx = int(np.argmax(np.where(valid, mags, -np.inf)))
    return float(freqs[idx])
