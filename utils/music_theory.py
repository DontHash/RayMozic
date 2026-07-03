"""Shared music-theory primitives: notes, chord parsing, and diatonic scales.

Used by the chord-voicing recommender and the progression generator (features
inspired by MoChord). Everything is expressed in pitch classes (0 = C).
"""

from __future__ import annotations

from typing import Optional

from utils.dsp_live import NOTE_NAMES

FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "Cb": "B", "Fb": "E",
}

# Chord quality -> semitone intervals from the root.
CHORD_INTERVALS: dict[str, list[int]] = {
    "": [0, 4, 7],
    "maj": [0, 4, 7],
    "M": [0, 4, 7],
    "m": [0, 3, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "5": [0, 7],
    "6": [0, 4, 7, 9],
    "m6": [0, 3, 7, 9],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "M7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "min7": [0, 3, 7, 10],
    "m7b5": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "add9": [0, 4, 7, 2],
    "9": [0, 4, 7, 10, 2],
    "m9": [0, 3, 7, 10, 2],
}

# Diatonic triad qualities for major and natural-minor scales.
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
NATURAL_MINOR = [0, 2, 3, 5, 7, 8, 10]

MAJOR_TRIAD_QUALITIES = ["", "m", "m", "", "", "m", "dim"]
MINOR_TRIAD_QUALITIES = ["m", "dim", "", "m", "m", "", ""]

# Degree-specific seventh-chord qualities (dominant 7 lands on V / VII).
MAJOR_SEVENTH_QUALITIES = ["maj7", "m7", "m7", "maj7", "7", "m7", "m7b5"]
MINOR_SEVENTH_QUALITIES = ["m7", "m7b5", "maj7", "m7", "m7", "maj7", "7"]

MAJOR_ROMAN = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
MINOR_ROMAN = ["i", "ii°", "III", "iv", "v", "VI", "VII"]

# Harmonic function per scale degree (0-indexed).
MAJOR_FUNCTION = ["Tonic", "Subdominant", "Tonic", "Subdominant", "Dominant", "Tonic", "Dominant"]
MINOR_FUNCTION = ["Tonic", "Subdominant", "Tonic", "Subdominant", "Dominant", "Subdominant", "Dominant"]


def name_to_pc(name: str) -> int:
    """Pitch class (0..11) for a note name like 'C', 'F#', 'Bb'."""
    name = name.strip().capitalize() if len(name) == 1 else name.strip()[:1].upper() + name.strip()[1:]
    name = FLAT_TO_SHARP.get(name, name)
    if name not in NOTE_NAMES:
        raise ValueError(f"Unknown note name: {name!r}")
    return NOTE_NAMES.index(name)


def pc_to_name(pc: int) -> str:
    return NOTE_NAMES[pc % 12]


def parse_chord(symbol: str) -> tuple[int, str, list[int]]:
    """
    Parse a chord symbol into (root_pc, quality, interval_list).
    e.g. 'C' -> (0, '', [0,4,7]); 'F#m7' -> (6, 'm7', [0,3,7,10]).
    """
    s = symbol.strip()
    if not s:
        raise ValueError("Empty chord symbol")

    # Root: letter + optional accidental.
    root = s[0].upper()
    idx = 1
    if idx < len(s) and s[idx] in ("#", "b"):
        root += s[idx]
        idx += 1
    root_pc = name_to_pc(root)

    quality = s[idx:]
    if quality not in CHORD_INTERVALS:
        # Tolerate a few aliases; default to major triad.
        quality = {"maj9": "9", "sus": "sus4"}.get(quality, quality)
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS[""])
    return root_pc, quality, intervals


def chord_pitch_classes(symbol: str) -> set[int]:
    root_pc, _, intervals = parse_chord(symbol)
    return {(root_pc + iv) % 12 for iv in intervals}


def note_label_to_midi(label: str) -> int:
    """MIDI number for a scientific label like 'E2' (E2 = 40)."""
    i = len(label)
    while i > 0 and (label[i - 1].isdigit() or label[i - 1] == "-"):
        i -= 1
    name, octave = label[:i], int(label[i:])
    name = FLAT_TO_SHARP.get(name, name)
    return (octave + 1) * 12 + NOTE_NAMES.index(name)


def diatonic_chords(root_pc: int, mode: str = "major", sevenths: bool = False) -> list[dict]:
    """
    Build the seven diatonic chords of a key.
    Returns a list of dicts with degree, roman, name, quality, function.
    """
    if mode.lower().startswith("min"):
        scale, romans, funcs = NATURAL_MINOR, MINOR_ROMAN, MINOR_FUNCTION
        triads, sevenths_q = MINOR_TRIAD_QUALITIES, MINOR_SEVENTH_QUALITIES
    else:
        scale, romans, funcs = MAJOR_SCALE, MAJOR_ROMAN, MAJOR_FUNCTION
        triads, sevenths_q = MAJOR_TRIAD_QUALITIES, MAJOR_SEVENTH_QUALITIES

    chords = []
    for degree in range(7):
        chord_root = (root_pc + scale[degree]) % 12
        quality = sevenths_q[degree] if sevenths else triads[degree]
        name = pc_to_name(chord_root) + quality
        chords.append(
            {
                "degree": degree + 1,
                "roman": romans[degree],
                "name": name,
                "quality": quality,
                "function": funcs[degree],
                "root_pc": chord_root,
            }
        )
    return chords
