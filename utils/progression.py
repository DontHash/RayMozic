"""Local chord-progression generator (MoChord's offline/fallback approach).

Given a key, mode, and a degree pattern (numbers or Roman numerals), build a
diatonic progression with chord names, Roman numerals, and harmonic function.
No external AI/API is required — this is the deterministic local path.
"""

from __future__ import annotations

import re
from typing import Optional

from utils.music_theory import diatonic_chords, name_to_pc

ROMAN_TO_DEGREE = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
}


def parse_degrees(pattern: str) -> list[int]:
    """
    Parse a degree pattern into 1-based scale degrees.
    Accepts '1-5-6-4', '1 5 6 4', or Roman numerals 'I-V-vi-IV'.
    """
    tokens = re.split(r"[\s,\-]+", pattern.strip())
    degrees: list[int] = []
    for tok in tokens:
        if not tok:
            continue
        if tok.isdigit():
            d = int(tok)
            if 1 <= d <= 7:
                degrees.append(d)
        else:
            roman = re.sub(r"[^IViv]", "", tok).upper()
            if roman in ROMAN_TO_DEGREE:
                degrees.append(ROMAN_TO_DEGREE[roman])
    return degrees


def generate_progression(
    key: str,
    mode: str = "major",
    pattern: str = "1-5-6-4",
    sevenths: bool = False,
) -> dict:
    """
    Build a diatonic progression.

    Returns a dict with the key/mode, the resolved chord list (each item has
    name, roman, function, degree), and the full diatonic set for reference.
    """
    root_pc = name_to_pc(key)
    table = diatonic_chords(root_pc, mode=mode, sevenths=sevenths)
    degrees = parse_degrees(pattern) or [1, 5, 6, 4]

    sequence = []
    for d in degrees:
        chord = table[d - 1]
        sequence.append(
            {
                "degree": d,
                "roman": chord["roman"],
                "name": chord["name"],
                "function": chord["function"],
            }
        )

    return {
        "key": key,
        "mode": mode,
        "pattern": "-".join(str(d) for d in degrees),
        "progression": sequence,
        "diatonic": table,
        "chord_names": [c["name"] for c in sequence],
    }


# A few common, musically useful patterns to offer as presets.
COMMON_PATTERNS = {
    "Pop I–V–vi–IV": "1-5-6-4",
    "50s I–vi–IV–V": "1-6-4-5",
    "Pachelbel I–V–vi–iii–IV–I–IV–V": "1-5-6-3-4-1-4-5",
    "Blues I–IV–V": "1-4-5",
    "ii–V–I (jazz)": "2-5-1",
    "Andalusian i–VII–VI–V": "1-7-6-5",
}
