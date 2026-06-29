import numpy as np
import librosa

def hz_to_note(hz: float) -> str:
    """Convert a frequency in Hz to a MIDI note name (e.g., 'C4')."""
    if hz <= 0:
        return "Unknown"
    midi = 69 + 12 * np.log2(hz / 440.0)
    return librosa.midi_to_note(int(round(midi)))

def note_to_hz(note_name: str) -> float:
    """Convert a MIDI note name to its frequency in Hz."""
    return librosa.note_to_hz(note_name)

def hz_to_midi(hz: float) -> float:
    """Convert a frequency in Hz to its exact fractional MIDI number."""
    if hz <= 0:
        return 0.0
    return 69 + 12 * np.log2(hz / 440.0)

def cents_deviation(f_measured: float, f_target: float) -> float:
    """Calculate the deviation in cents between a measured and target frequency."""
    if f_measured <= 0 or f_target <= 0:
        return 0.0
    return 1200 * np.log2(f_measured / f_target)

def classify_register(hz: float) -> str:
    """
    Classify the voice register based on approximate physiological boundaries.
    Chest voice: < 350 Hz
    Mixed / Neck: 350 - 600 Hz
    Head / Falsetto: > 600 Hz
    """
    if hz < 350:
        return "Chest Voice"
    elif hz < 600:
        return "Mixed / Neck Voice"
    else:
        return "Head Voice / Falsetto"
