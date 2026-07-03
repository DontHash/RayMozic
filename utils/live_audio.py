"""Browser microphone capture via WebRTC, feeding the live pitch detector.

Uses streamlit-webrtc to receive audio frames from the user's browser (works
over the network, unlike server-side sounddevice recording). A rolling buffer
of recent samples is analyzed every iteration so the UI can update ~live.
"""

from __future__ import annotations

import queue
from typing import Callable, Optional

import numpy as np

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except Exception:  # pragma: no cover - import guard for environments without webrtc
    WEBRTC_AVAILABLE = False

from utils.dsp_live import (
    DEFAULT_A4,
    DEFAULT_FMAX,
    DEFAULT_FMIN,
    PitchSmoother,
    compute_spectrum,
    detect_pitch,
    detect_pitch_hps,
    frequency_to_note,
)

RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


def create_audio_streamer(key: str):
    """Create a send-only audio WebRTC streamer (mic -> server).

    Returns None if WebRTC is unavailable or the streamer can't be created in
    the current context (e.g. no live browser session), so callers can degrade
    gracefully instead of crashing the page.
    """
    if not WEBRTC_AVAILABLE:
        return None
    try:
        return webrtc_streamer(
            key=key,
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"audio": True, "video": False},
        )
    except Exception:
        return None


def frame_to_mono(frame) -> tuple[np.ndarray, int]:
    """Convert an av.AudioFrame to a mono float32 array in [-1, 1] and its SR."""
    arr = frame.to_ndarray()
    sr = int(frame.sample_rate)

    if np.issubdtype(arr.dtype, np.integer):
        max_val = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / max_val
    else:
        arr = arr.astype(np.float32)

    channels = 1
    try:
        channels = len(frame.layout.channels)
    except Exception:
        channels = 1

    planar = getattr(frame.format, "is_planar", False)

    if arr.ndim == 2:
        if planar:
            mono = arr.mean(axis=0)
        else:
            flat = arr.reshape(-1)
            mono = flat.reshape(-1, channels).mean(axis=1) if channels > 1 else flat
    else:
        mono = arr
    return mono.astype(np.float32), sr


def live_pitch_loop(
    webrtc_ctx,
    on_reading: Callable,
    *,
    buffer_seconds: float = 0.25,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    spectrum_fmax: float = 2000.0,
    method: str = "autocorr",
    a4: float = DEFAULT_A4,
    smooth: bool = True,
    should_continue: Optional[Callable[[], bool]] = None,
) -> None:
    """
    Pull audio frames while the stream is playing and invoke `on_reading` with
    the latest analysis. `on_reading(reading, freqs, mags, buffer, sr)` renders.

    `method` selects the detector: "hps" (Harmonic Product Spectrum, GuitarTuner
    mechanism) or "autocorr" (autocorrelation). When `smooth` is True, a
    MoChord-style PitchSmoother (median + dropout hold + octave correction) is
    applied before mapping to a note.

    Blocks until the stream stops (the Streamlit script run owns this loop).
    """
    if webrtc_ctx is None or webrtc_ctx.audio_receiver is None:
        return

    buffer = np.zeros(0, dtype=np.float32)
    sr = 48000  # updated from the first frame
    smoother = PitchSmoother() if smooth else None

    while True:
        if not webrtc_ctx.state.playing:
            break
        if should_continue is not None and not should_continue():
            break

        try:
            frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            continue
        except Exception:
            break

        for frame in frames:
            mono, sr = frame_to_mono(frame)
            buffer = np.concatenate([buffer, mono])

        max_len = int(buffer_seconds * sr)
        if len(buffer) > max_len:
            buffer = buffer[-max_len:]

        # Need at least a couple periods of the lowest expected note.
        min_needed = int(sr / fmin * 2)
        if len(buffer) < min_needed:
            continue

        if method == "hps":
            reading = detect_pitch_hps(buffer, sr, fmin=fmin, fmax=fmax, a4=a4)
        else:
            reading = detect_pitch(buffer, sr, fmin=fmin, fmax=fmax, a4=a4)

        if smoother is not None:
            smoothed = smoother.update(reading.frequency, reading.voiced)
            if smoothed is not None and smoothed > 0:
                conf = reading.confidence
                reading = frequency_to_note(smoothed, a4)
                reading.confidence = conf
                reading.voiced = True
            elif smoothed is None:
                reading.voiced = False

        freqs, mags = compute_spectrum(buffer, sr, fmax=spectrum_fmax)
        on_reading(reading, freqs, mags, buffer, sr)
