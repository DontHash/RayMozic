import librosa
import sounddevice as sd
import numpy as np
import tempfile
import os
from typing import Optional

# Extensions accepted by Streamlit file_uploader (no leading dot).
# WAV/FLAC/AIFF decode via soundfile; MP3/OGG/M4A/AAC/WEBM/WMA use audioread
# and require FFmpeg installed and on PATH.
SUPPORTED_AUDIO_EXTENSIONS = (
    "wav",
    "mp3",
    "flac",
    "ogg",
    "oga",
    "m4a",
    "aac",
    "webm",
    "aiff",
    "aif",
    "wma",
)

AUDIO_UPLOAD_TYPES = list(SUPPORTED_AUDIO_EXTENSIONS)

_NATIVE_SUFFIXES = {".wav", ".flac", ".aiff", ".aif"}
_FFMPEG_SUFFIXES = {
    ".mp3",
    ".ogg",
    ".oga",
    ".m4a",
    ".aac",
    ".webm",
    ".wma",
}


class AudioLoadError(Exception):
    """Raised when an audio file cannot be decoded."""


class AudioValidationError(Exception):
    """Raised when audio is too short, silent, or unsuitable for analysis."""


MIN_ANALYSIS_DURATION_SEC = 0.5
MIN_AUDIO_RMS = 1e-5


def validate_audio_signal(
    audio: np.ndarray,
    sr: int,
    *,
    label: str = "Audio",
    min_duration_sec: float = MIN_ANALYSIS_DURATION_SEC,
    min_rms: float = MIN_AUDIO_RMS,
) -> None:
    """Ensure audio has enough length and level for pitch/chroma analysis."""
    if audio is None or audio.size == 0:
        raise AudioValidationError(
            f"{label} is empty. Upload or record a vocal track with audible content."
        )

    duration = len(audio) / sr
    if duration < min_duration_sec:
        raise AudioValidationError(
            f"{label} is too short ({duration:.2f}s). "
            f"Need at least {min_duration_sec:.1f}s of audio."
        )

    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
    if not np.isfinite(rms) or rms < min_rms:
        raise AudioValidationError(
            f"{label} appears silent or nearly silent. "
            "Check microphone levels or file content."
        )


def _suffix_from_filename(filename: Optional[str]) -> str:
    """Return a file suffix suitable for temp files and decoder selection."""
    if not filename:
        return ".bin"
    ext = os.path.splitext(filename)[1].lower()
    known = {f".{e}" for e in SUPPORTED_AUDIO_EXTENSIONS}
    if ext in known:
        return ext
    return ext if ext else ".bin"


def _format_load_hint(suffix: str) -> str:
    if suffix in _NATIVE_SUFFIXES:
        return "This format is decoded natively via soundfile."
    if suffix in _FFMPEG_SUFFIXES:
        return (
            "This format is decoded via librosa/audioread and requires "
            "FFmpeg installed and available on PATH."
        )
    return (
        "Supported uploads: "
        + ", ".join(SUPPORTED_AUDIO_EXTENSIONS)
        + ". Compressed formats need FFmpeg on PATH."
    )


def _load_audio_path(path: str, sr: int = 22050) -> np.ndarray:
    suffix = _suffix_from_filename(path)
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as exc:
        hint = _format_load_hint(suffix)
        name = os.path.basename(path)
        raise AudioLoadError(
            f"Could not load audio file '{name}' ({suffix or 'unknown type'}). {hint}"
        ) from exc
    return y


def record_audio(duration: float, sr: int = 22050) -> np.ndarray:
    """
    Record audio from the default microphone.
    duration: seconds to record
    sr: sample rate
    Returns 1D numpy array of float32 audio.
    """
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def load_audio_file(file_path: str, sr: int = 22050) -> np.ndarray:
    """Load an audio file from a filesystem path."""
    return _load_audio_path(file_path, sr=sr)


def load_uploaded_audio(uploaded_file, sr: int = 22050) -> np.ndarray:
    """
    Load an audio file from a Streamlit UploadedFile object.
    Writes to a temp file with the original extension so librosa/audioread
    can select the correct decoder (soundfile or FFmpeg-backed audioread).
    """
    suffix = _suffix_from_filename(getattr(uploaded_file, "name", None))
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        return _load_audio_path(temp_path, sr=sr)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
