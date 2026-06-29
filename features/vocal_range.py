import librosa
import numpy as np
from scipy.signal import welch
from utils.pitch_utils import hz_to_note, classify_register
from utils.audio_io import validate_audio_signal, AudioValidationError

def analyze_vocal_range(audio: np.ndarray, sr: int = 22050) -> dict:
    """
    Analyzes the vocal range, modal pitch, register, and checks for nasal formants.
    Returns a dictionary of the results.
    """
    try:
        validate_audio_signal(audio, sr, label="Vocal")
    except AudioValidationError as exc:
        return {"error": str(exc)}

    # 1. Pitch Tracking (using pYIN for better vocal accuracy, fallback to YIN if needed)
    # pYIN is slower but avoids octave errors. We'll use YIN here for speed in Streamlit, 
    # but in a production app, pYIN is better. Let's stick to YIN as per spec for fast response,
    # or use pYIN if the audio is short enough. We'll use librosa.yin for responsiveness.
    f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    f0_voiced = f0[f0 > 0]
    
    if len(f0_voiced) == 0:
        return {"error": "No pitched voice detected."}
        
    # 2. Compute Range
    low_hz = np.percentile(f0_voiced, 5)
    high_hz = np.percentile(f0_voiced, 95)
    modal_hz = np.median(f0_voiced)
    
    # 3. Classify Register
    register = classify_register(modal_hz)
    
    # 4. Nasal Detection via Formant Energy (PSD)
    freqs, psd = welch(audio, fs=sr, nperseg=1024)
    
    nasal_low = np.mean(psd[(freqs > 200) & (freqs < 300)])
    nasal_high = np.mean(psd[(freqs > 2000) & (freqs < 3000)])
    mid_band = np.mean(psd[(freqs > 500) & (freqs < 1500)])
    
    # Avoid division by zero
    if mid_band == 0: mid_band = 1e-10
    
    is_nasal = (nasal_low > mid_band * 1.5) and (nasal_high > mid_band * 1.2)
    
    return {
        "low_note": hz_to_note(low_hz),
        "high_note": hz_to_note(high_hz),
        "modal_note": hz_to_note(modal_hz),
        "low_hz": float(low_hz),
        "high_hz": float(high_hz),
        "modal_hz": float(modal_hz),
        "register": register,
        "is_nasal": is_nasal
    }
