"""Smart guitar chord-voicing recommender (mechanism inspired by MoChord).

Given a chord symbol and a tuning, search the fretboard for playable voicings
and score them by chord-tone coverage, root/bass placement, open-string use,
fret span, muting structure, and hand ergonomics (finger count, barre).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utils.music_theory import note_label_to_midi, parse_chord
from utils.tuner import DEFAULT_TUNING, TUNINGS

MAX_FRET = 12
WINDOW = 3  # frets reachable above the base fret (base..base+WINDOW)


@dataclass
class Voicing:
    frets: list[Optional[int]]      # per string, thickest->thinnest; None = muted
    open_midis: list[int]           # open-string MIDI, thickest->thinnest
    score: float = 0.0
    fingers: int = 0
    is_barre: bool = False
    base_fret: int = 0
    coverage: float = 0.0
    bass_is_root: bool = False
    played_notes: list[str] = field(default_factory=list)

    @property
    def diagram_str(self) -> str:
        """e.g. 'x 3 2 0 1 0' (thickest string first)."""
        return " ".join("x" if f is None else str(f) for f in self.frets)


def _open_midis(tuning_name: str, custom_notes: Optional[list[str]]) -> list[int]:
    notes = custom_notes if custom_notes else TUNINGS.get(tuning_name, TUNINGS[DEFAULT_TUNING])
    return [note_label_to_midi(n) for n in notes]


def _fretted_span(frets: list[Optional[int]]) -> int:
    fretted = [f for f in frets if f is not None and f > 0]
    return (max(fretted) - min(fretted)) if fretted else 0


def _has_internal_mute(frets: list[Optional[int]]) -> bool:
    """True if a muted string sits between two sounded strings (broken pattern)."""
    played_idx = [i for i, f in enumerate(frets) if f is not None]
    if len(played_idx) < 2:
        return False
    for i in range(played_idx[0], played_idx[-1] + 1):
        if frets[i] is None:
            return True
    return False


def _estimate_fingers(frets: list[Optional[int]]) -> tuple[int, bool]:
    """Estimate finger count and whether a barre is likely."""
    fretted = [(i, f) for i, f in enumerate(frets) if f is not None and f > 0]
    if not fretted:
        return 0, False
    min_fret = min(f for _, f in fretted)
    max_fret = max(f for _, f in fretted)
    at_min = [i for i, f in fretted if f == min_fret]
    # A real barre needs 2+ strings at the lowest fret AND notes fretted above
    # it (index finger bars while other fingers fret higher). Two same-fret
    # notes with nothing above are just two separate fingers, not a barre.
    is_barre = len(at_min) >= 2 and max_fret > min_fret
    if is_barre:
        fingers = 1 + sum(1 for _, f in fretted if f > min_fret)
    else:
        fingers = len(fretted)
    return fingers, is_barre


def _score(voicing_frets: list[Optional[int]], open_midis: list[int], chord_pcs: set[int], root_pc: int) -> Optional[Voicing]:
    played = [(i, f) for i, f in enumerate(voicing_frets) if f is not None]
    if len(played) < 3:
        return None  # need at least a triad's worth of strings

    span = _fretted_span(voicing_frets)
    if span > WINDOW + 1:  # unplayable stretch
        return None

    fingers, is_barre = _estimate_fingers(voicing_frets)
    if fingers > 4:
        return None

    sounding_pcs = {(open_midis[i] + f) % 12 for i, f in played}
    covered = sounding_pcs & chord_pcs
    if not covered:
        return None
    # Reject non-chord tones (keep voicings strictly diatonic to the chord).
    if sounding_pcs - chord_pcs:
        return None

    coverage = len(covered) / len(chord_pcs)

    # Bass = lowest (thickest) sounding string.
    bass_idx = min(i for i, _ in played)
    bass_pc = (open_midis[bass_idx] + voicing_frets[bass_idx]) % 12
    bass_is_root = bass_pc == root_pc

    open_strings = sum(1 for _, f in played if f == 0)

    fretted_positions = [f for _, f in played if f > 0]
    min_fret = min(fretted_positions) if fretted_positions else 0

    score = 0.0
    score += 40.0 * coverage
    score += 10.0 if root_pc in sounding_pcs else -20.0
    score += 15.0 if bass_is_root else 0.0
    score += 2.0 * open_strings
    score += 1.5 * len(played)            # fuller chords preferred
    score -= 3.0 * max(0, span - 2)        # mild stretch penalty
    score -= 4.0 if is_barre else 0.0      # barres slightly harder
    score -= 1.2 * min_fret                # prefer positions low on the neck
    if _has_internal_mute(voicing_frets):
        score -= 8.0

    note_names = []
    from utils.music_theory import pc_to_name
    for i, f in played:
        note_names.append(pc_to_name((open_midis[i] + f) % 12))

    return Voicing(
        frets=list(voicing_frets),
        open_midis=open_midis,
        score=round(score, 2),
        fingers=fingers,
        is_barre=is_barre,
        base_fret=min([f for _, f in played if f > 0], default=0),
        coverage=round(coverage, 2),
        bass_is_root=bass_is_root,
        played_notes=note_names,
    )


def _candidate_for_base(base: int, open_midis: list[int], chord_pcs: set[int], root_pc: int) -> list[Optional[int]]:
    """
    Build one voicing for a base fret: make the bass the root when possible,
    then fill each remaining string with the lowest in-window chord tone.
    """
    n = len(open_midis)
    allowed = [0] + list(range(base, base + WINDOW + 1)) if base > 0 else list(range(0, WINDOW + 1))
    allowed = sorted(set(f for f in allowed if 0 <= f <= MAX_FRET))

    def chord_frets(string_idx: int) -> list[int]:
        return [f for f in allowed if (open_midis[string_idx] + f) % 12 in chord_pcs]

    # Find the thickest string that can play the root -> that becomes the bass.
    bass_string = None
    for i in range(n):
        if any((open_midis[i] + f) % 12 == root_pc for f in allowed):
            bass_string = i
            break

    frets: list[Optional[int]] = [None] * n
    for i in range(n):
        options = chord_frets(i)
        if not options:
            frets[i] = None
            continue
        if bass_string is not None and i < bass_string:
            frets[i] = None  # mute below the root so the bass is the root
        elif i == bass_string:
            root_options = [f for f in options if (open_midis[i] + f) % 12 == root_pc]
            frets[i] = min(root_options) if root_options else min(options)
        else:
            frets[i] = min(options)
    return frets


def recommend_voicings(
    symbol: str,
    tuning_name: str = DEFAULT_TUNING,
    custom_notes: Optional[list[str]] = None,
    top_n: int = 4,
) -> list[Voicing]:
    """Return the top scored voicings for a chord symbol on the given tuning."""
    root_pc, _, intervals = parse_chord(symbol)
    chord_pcs = {(root_pc + iv) % 12 for iv in intervals}
    open_midis = _open_midis(tuning_name, custom_notes)

    seen: set[tuple] = set()
    voicings: list[Voicing] = []
    for base in range(0, MAX_FRET - WINDOW + 1):
        frets = _candidate_for_base(base, open_midis, chord_pcs, root_pc)
        key = tuple(frets)
        if key in seen:
            continue
        seen.add(key)
        scored = _score(frets, open_midis, chord_pcs, root_pc)
        if scored is not None:
            voicings.append(scored)

    voicings.sort(key=lambda v: v.score, reverse=True)
    return voicings[:top_n]
