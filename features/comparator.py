import librosa
import numpy as np
from scipy.signal import butter, filtfilt

from utils.audio_io import validate_audio_signal
from utils.chroma_utils import extract_chroma_stft


class VoiceComparisonError(Exception):
    """Raised when voice comparison cannot be completed."""


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return b, a


def compare_voices(user_audio: np.ndarray, ref_audio: np.ndarray, sr: int = 22050) -> dict:
    """
    Compare user vocal audio against a reference track.
    Returns DTW pitch deviation, vibrato stats, and dynamic correlation.
    """
    validate_audio_signal(user_audio, sr, label="Your vocal")
    validate_audio_signal(ref_audio, sr, label="Reference track")

    # 1. Pitch Track Both
    f0_ref = librosa.yin(ref_audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    f0_user = librosa.yin(user_audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

    # 2. Extract Chroma and Align with DTW
    chroma_ref = extract_chroma_stft(ref_audio, sr)
    chroma_user = extract_chroma_stft(user_audio, sr)

    try:
        D, wp = librosa.sequence.dtw(chroma_ref, chroma_user, metric='cosine')
    except librosa.util.exceptions.ParameterError as exc:
        raise VoiceComparisonError(
            "Could not align pitch content between the two tracks. "
            "Ensure both recordings contain clear, sustained vocals."
        ) from exc

    # 3. Compute Pitch Deviation along warping path
    cents_errors = []
    ref_aligned_f0 = []
    user_aligned_f0 = []

    for ref_frame, user_frame in wp:
        f_ref = f0_ref[min(ref_frame, len(f0_ref)-1)]
        f_user = f0_user[min(user_frame, len(f0_user)-1)]

        ref_aligned_f0.append(f_ref)
        user_aligned_f0.append(f_user)

        if f_ref > 0 and f_user > 0:
            cents = 1200 * np.log2(f_user / f_ref)
            cents_errors.append(cents)

    mean_deviation = np.mean(np.abs(cents_errors)) if len(cents_errors) > 0 else 0.0

    # 4. Vibrato Detection on User Audio
    f0_voiced = f0_user[f0_user > 0]
    has_vibrato = False
    vibrato_rate = 0.0

    if len(f0_voiced) > 100:  # Need enough frames for FFT
        hop_length = 512
        f0_fps = sr / hop_length

        b, a = butter_lowpass(15, f0_fps)
        f0_smooth = filtfilt(b, a, f0_voiced)
        pitch_modulation = f0_voiced - f0_smooth

        mod_fft = np.abs(np.fft.rfft(pitch_modulation))
        mod_freqs = np.fft.rfftfreq(len(pitch_modulation), 1/f0_fps)

        vibrato_range = (mod_freqs > 4) & (mod_freqs < 9)
        if np.any(vibrato_range):
            peak_idx = np.argmax(mod_fft[vibrato_range])
            peak_val = mod_fft[vibrato_range][peak_idx]

            # Simple threshold for "has vibrato" (heuristic)
            if peak_val > np.mean(mod_fft) * 2:
                has_vibrato = True
                vibrato_rate = mod_freqs[vibrato_range][peak_idx]

    # 5. RMS Energy Correlation
    rms_ref = librosa.feature.rms(y=ref_audio)[0]
    rms_user = librosa.feature.rms(y=user_audio)[0]

    # Align RMS using DTW path
    rms_ref_aligned = np.array([rms_ref[min(r, len(rms_ref)-1)] for r, u in wp])
    rms_user_aligned = np.array([rms_user[min(u, len(rms_user)-1)] for r, u in wp])

    dynamic_corr = 0.0
    if len(rms_ref_aligned) > 1 and len(rms_user_aligned) > 1:
        corr_matrix = np.corrcoef(rms_ref_aligned, rms_user_aligned)
        if not np.isnan(corr_matrix[0, 1]):
            dynamic_corr = corr_matrix[0, 1]

    return {
        "mean_cents_deviation": mean_deviation,
        "has_vibrato": has_vibrato,
        "vibrato_rate": vibrato_rate,
        "dynamic_correlation": dynamic_corr,
        # Return aligned f0s for plotting (reverse wp to chronological order)
        "ref_f0_plot": ref_aligned_f0[::-1],
        "user_f0_plot": user_aligned_f0[::-1]
    }
