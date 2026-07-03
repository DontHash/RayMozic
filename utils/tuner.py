"""Guitar tuner: tuning presets and nearest-string matching.

Each preset maps a string label (thickest -> thinnest) to a target note name.
Target frequencies are derived from equal temperament (A4 = 440 Hz), so the
tuner and the pitch detector share one reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.dsp_live import DEFAULT_A4, NOTE_NAMES

A4_HZ = DEFAULT_A4

# Adjustable concert-pitch range (both GuitarTuner and MoChord expose this).
A4_MIN = 432.0
A4_MAX = 445.0


def note_to_frequency(note_label: str, a4: float = A4_HZ) -> float:
    """Convert a scientific note label (e.g. 'E2', 'A#3') to Hz for the given A4."""
    label = note_label.strip()
    # Split trailing octave digits (allow negative just in case).
    i = len(label)
    while i > 0 and (label[i - 1].isdigit() or label[i - 1] == "-"):
        i -= 1
    name, octave = label[:i], int(label[i:])
    if name not in NOTE_NAMES:
        raise ValueError(f"Unknown note name: {name!r}")
    midi = (octave + 1) * 12 + NOTE_NAMES.index(name)
    return a4 * (2.0 ** ((midi - 69) / 12.0))


# Ordered thickest (6th) string -> thinnest (1st) string.
TUNINGS: dict[str, list[str]] = {
    "Standard (E A D G B E)": ["E2", "A2", "D3", "G3", "B3", "E4"],
    "Drop D (D A D G B E)": ["D2", "A2", "D3", "G3", "B3", "E4"],
    "Low C (C G D G A D)": ["C2", "G2", "D3", "G3", "A3", "D4"],
    "Half Step Down (Eb Ab Db Gb Bb Eb)": ["D#2", "G#2", "C#3", "F#3", "A#3", "D#4"],
    "Full Step Down (D G C F A D)": ["D2", "G2", "C3", "F3", "A3", "D4"],
    "Open G (D G D G B D)": ["D2", "G2", "D3", "G3", "B3", "D4"],
    "Open D (D A D F# A D)": ["D2", "A2", "D3", "F#3", "A3", "D4"],
    "DADGAD (D A D G A D)": ["D2", "A2", "D3", "G3", "A3", "D4"],
}

DEFAULT_TUNING = "Standard (E A D G B E)"


@dataclass
class StringTarget:
    label: str          # note label, e.g. "E2"
    frequency: float    # target Hz
    string_number: int  # 6 (thickest) .. 1 (thinnest)


@dataclass
class TunerMatch:
    string: StringTarget
    detected_hz: float
    cents_off: float     # signed: negative = flat, positive = sharp
    in_tune: bool


def get_targets(
    tuning_name: str = DEFAULT_TUNING,
    a4: float = A4_HZ,
    custom_notes: Optional[list[str]] = None,
) -> list[StringTarget]:
    """Return the string targets for a tuning, thickest string first.

    Pass `custom_notes` (6 labels, thickest first) to define a custom tuning;
    otherwise the named preset is used. All frequencies use the given A4.
    """
    notes = custom_notes if custom_notes else TUNINGS.get(tuning_name, TUNINGS[DEFAULT_TUNING])
    targets = []
    n = len(notes)
    for idx, label in enumerate(notes):
        targets.append(
            StringTarget(
                label=label,
                frequency=note_to_frequency(label, a4),
                string_number=n - idx,  # first entry is the 6th string
            )
        )
    return targets


def nearest_string(
    detected_hz: float,
    tuning_name: str = DEFAULT_TUNING,
    in_tune_cents: float = 5.0,
    a4: float = A4_HZ,
    custom_notes: Optional[list[str]] = None,
) -> Optional[TunerMatch]:
    """
    Find which string of the tuning the detected pitch is closest to, and how
    far off it is in cents. Returns None if no pitch was detected.
    """
    if detected_hz is None or detected_hz <= 0:
        return None

    targets = get_targets(tuning_name, a4=a4, custom_notes=custom_notes)
    best: Optional[StringTarget] = None
    best_cents = float("inf")
    for target in targets:
        cents = 1200.0 * np.log2(detected_hz / target.frequency)
        if abs(cents) < abs(best_cents):
            best_cents = cents
            best = target

    if best is None:
        return None

    return TunerMatch(
        string=best,
        detected_hz=float(detected_hz),
        cents_off=float(best_cents),
        in_tune=abs(best_cents) <= in_tune_cents,
    )


def tuning_direction(cents_off: float, in_tune_cents: float = 5.0) -> str:
    """Human-readable instruction for the player."""
    if abs(cents_off) <= in_tune_cents:
        return "In tune"
    return "Tune down (too sharp)" if cents_off > 0 else "Tune up (too flat)"
