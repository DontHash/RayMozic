import numpy as np
import librosa

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

_CHROMA_EPS = 1e-8


def sanitize_chroma(chroma: np.ndarray, eps: float = _CHROMA_EPS) -> np.ndarray:
    """
    Make a chromagram safe for cosine distance and DTW.
    Silent STFT frames produce all-zero columns; cosine distance is undefined (NaN).
    """
    chroma = np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    zero_cols = (norms < eps).flatten()
    chroma = chroma / np.maximum(norms, eps)
    if np.any(zero_cols):
        uniform = 1.0 / np.sqrt(chroma.shape[0])
        chroma[:, zero_cols] = uniform
    return chroma


def extract_chroma_stft(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract a sanitized chromagram suitable for DTW alignment."""
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma = sanitize_chroma(chroma)
    if chroma.shape[1] < 1:
        raise ValueError("Chroma extraction produced no time frames.")
    return chroma

# Krumhansl-Schmuckler Key Profiles
# Weights for all 12 pitch classes relative to the tonic
KS_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def extract_mean_chroma(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract the mean chromagram across the entire audio signal."""
    chroma = extract_chroma_stft(audio, sr)
    mean_chroma = chroma.mean(axis=1)
    mean_chroma = np.nan_to_num(mean_chroma, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(mean_chroma)
    if norm < _CHROMA_EPS:
        return np.full(chroma.shape[0], 1.0 / np.sqrt(chroma.shape[0]))
    return mean_chroma / norm

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def get_key_profile(root_idx: int, mode: str = 'major') -> np.ndarray:
    """
    Generate the KS key profile shifted to the target root index.
    root_idx: 0=C, 1=C#, etc.
    mode: 'major' or 'minor'
    """
    base_profile = KS_MAJOR_PROFILE if mode == 'major' else KS_MINOR_PROFILE
    # Shift the profile so the root aligns with root_idx
    return np.roll(base_profile, root_idx)

def detect_key(chroma_mean: np.ndarray) -> tuple[str, str, float]:
    """
    Detect the most likely musical key given a mean chromagram.
    Returns (root_note, mode, confidence_score).
    """
    best_score = -1.0
    best_key = "C"
    best_mode = "major"
    
    for i, note in enumerate(NOTE_NAMES):
        # Check major
        maj_profile = get_key_profile(i, 'major')
        maj_score = cosine_sim(chroma_mean, maj_profile)
        if maj_score > best_score:
            best_score = maj_score
            best_key = note
            best_mode = "major"
            
        # Check minor
        min_profile = get_key_profile(i, 'minor')
        min_score = cosine_sim(chroma_mean, min_profile)
        if min_score > best_score:
            best_score = min_score
            best_key = note
            best_mode = "minor"
            
    return best_key, best_mode, best_score
