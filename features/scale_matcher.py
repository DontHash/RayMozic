from utils.chroma_utils import extract_mean_chroma, detect_key, NOTE_NAMES
from utils.audio_io import validate_audio_signal
import numpy as np

# Standard guitar keys (open chords friendly)
GUITAR_KEYS = {'G': 7, 'A': 9, 'C': 0, 'D': 2, 'E': 4, 'F': 5}

CHORD_TO_SEMITONE = {'C':0, 'C#':1, 'Db':1, 'D':2, 'D#':3, 'Eb':3, 'E':4,
                     'F':5, 'F#':6, 'Gb':6, 'G':7, 'G#':8, 'Ab':8, 'A':9,
                     'A#':10, 'Bb':10, 'B':11}

def match_guitar_scale(audio: np.ndarray, sr: int = 22050) -> dict:
    """
    Finds the vocal key and matches it to the best guitar capo position.
    """
    validate_audio_signal(audio, sr, label="Vocal")

    # 1. Get vocal key
    chroma_mean = extract_mean_chroma(audio, sr)
    vocal_key, vocal_mode, confidence = detect_key(chroma_mean)
    vocal_key_idx = NOTE_NAMES.index(vocal_key)
    
    # 2. Capo Recommendations
    recommendations = []
    for shape_key, shape_idx in GUITAR_KEYS.items():
        capo = (vocal_key_idx - shape_idx) % 12
        if capo > 6:
            # If capo is too high on the neck, invert direction (play higher chord shape with no capo or lower capo)
            # Actually, standard capo logic: if capo > 6, it might be better to just use a different shape.
            # We'll just penalize it or keep it as an option.
            pass
            
        recommendations.append({
            'chord_shape': shape_key,
            'capo_fret': capo,
            # We can use a simple score: lower capo is better, but this is a heuristic
            'score': 12 - capo 
        })
        
    # Sort by lowest capo fret
    recommendations.sort(key=lambda x: x['capo_fret'])
    
    return {
        "vocal_key": vocal_key,
        "vocal_mode": vocal_mode,
        "confidence": confidence,
        "recommendations": recommendations[:3] # Top 3
    }

def transpose_progression(progression: str, semitones: int) -> str:
    """
    Transpose a space-separated chord progression by N semitones.
    e.g., 'G Em C D' by +2 -> 'A F#m D E'
    """
    if not progression.strip():
        return ""
        
    chords = progression.split()
    transposed = []
    
    for chord in chords:
        # Extract root and quality
        # Simple extraction: root is first 1 or 2 chars (e.g. C, C#)
        if len(chord) > 1 and chord[1] in ['#', 'b']:
            root = chord[:2]
            quality = chord[2:]
        else:
            root = chord[:1]
            quality = chord[1:]
            
        if root not in CHORD_TO_SEMITONE:
            transposed.append(chord) # Leave as is if unparseable
            continue
            
        new_idx = (CHORD_TO_SEMITONE[root] + semitones) % 12
        new_root = NOTE_NAMES[new_idx]
        transposed.append(new_root + quality)
        
    return " ".join(transposed)
