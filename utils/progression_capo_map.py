"""Map a sounding-key chord progression across capo positions and open shapes.

Opposite of the voice-matching tab: here the **sounding** progression is fixed
(what the listener hears). Each capo fret or open-chord family shows which
**finger shapes** you play to produce that same sound — and which keys/scales
the progression fits.
"""

from __future__ import annotations

from utils.music_theory import chord_pitch_classes, diatonic_chords, name_to_pc, pc_to_name
from features.scale_matcher import GUITAR_KEYS, transpose_progression
from utils.play_along import infer_key_from_chords

def finger_shapes_for_sounding(sounding_chords: list[str], capo_fret: int) -> list[str]:
    """Finger shapes to play so the audience hears `sounding_chords` with capo on `capo_fret`."""
    if capo_fret == 0:
        return list(sounding_chords)
    joined = transpose_progression(" ".join(sounding_chords), -capo_fret)
    return joined.split() if joined else []


def compatible_keys(chord_names: list[str], min_fit: float = 0.75) -> list[dict]:
    """Keys/modes where the progression is mostly diatonic."""
    if not chord_names:
        return []

    results = []
    for root_pc in range(12):
        for mode in ("major", "minor"):
            table = diatonic_chords(root_pc, mode)
            hits = 0
            roman_line = []
            for ch in chord_names:
                try:
                    pcs = chord_pitch_classes(ch)
                except ValueError:
                    roman_line.append("?")
                    continue
                matched = False
                for d in table:
                    if pcs == chord_pitch_classes(d["name"]):
                        hits += 1
                        roman_line.append(d["roman"])
                        matched = True
                        break
                if not matched:
                    roman_line.append("·")

            fit = hits / len(chord_names)
            if fit >= min_fit:
                results.append({
                    "key": pc_to_name(root_pc),
                    "mode": mode,
                    "fit": round(fit, 3),
                    "roman": " – ".join(roman_line),
                })

    results.sort(key=lambda r: (-r["fit"], r["key"]))
    return results


def _capo_row_explanation(capo: int, finger: list[str], sounding: list[str], key: str, mode: str) -> str:
    f = " · ".join(finger)
    s = " · ".join(sounding)
    if capo == 0:
        return f"Capo **0** — finger **{f}** → sounds **{s}** (in **{key} {mode}**)."
    return (
        f"Capo **{capo}** — finger **{f}** → still sounds **{s}** "
        f"(same progression, **{key} {mode}**)."
    )


def _shape_family_explanation(shape_key: str, capo: int, finger: list[str], sounding: list[str]) -> str:
    f = " · ".join(finger)
    s = " · ".join(sounding)
    if capo == 0:
        return f"**{shape_key}** shapes, no capo: finger **{f}** → sounds **{s}**."
    return (
        f"**{shape_key}** family + capo **{capo}**: finger **{f}** → sounds **{s}**."
    )


def build_progression_capo_map(
    progression: str,
    target_key: str | None = None,
    target_mode: str | None = None,
    max_fret: int = 7,
) -> dict:
    """
    Explore a sounding-key progression across capo frets and open-chord families.

    `progression` is what the listener hears (e.g. "G Em C D").
    Optional `target_key` / `target_mode` override the inferred key for shape-family math.
    """
    sounding = progression.strip().split()
    if not sounding:
        raise ValueError("Enter at least one chord.")

    inf_key, inf_mode, diatonic_fit = infer_key_from_chords(sounding)
    key = target_key or inf_key
    mode = target_mode or inf_mode
    key_pc = name_to_pc(key)

    # Capo fret table — sounding fixed, finger shapes change.
    capo_rows = []
    for capo in range(max_fret + 1):
        finger = finger_shapes_for_sounding(sounding, capo)
        capo_rows.append({
            "capo_fret": capo,
            "finger_chords": finger,
            "sounding_chords": list(sounding),
            "inferred_key": key,
            "inferred_mode": mode,
            "explanation": _capo_row_explanation(capo, finger, sounding, key, mode),
        })

    # Open-chord family table — capo chosen so each family reaches `key`.
    shape_rows = []
    for shape_key, shape_pc in GUITAR_KEYS.items():
        capo = (key_pc - shape_pc) % 12
        practical = capo <= 7
        finger = finger_shapes_for_sounding(sounding, capo)
        shape_rows.append({
            "chord_shape": shape_key,
            "capo_fret": capo,
            "finger_chords": finger,
            "sounding_chords": list(sounding),
            "sounding_key": key,
            "sounding_mode": mode,
            "practical": practical,
            "recommended": practical and capo <= 5,
            "explanation": _shape_family_explanation(shape_key, capo, finger, sounding),
        })
    shape_rows.sort(key=lambda r: (not r["practical"], r["capo_fret"]))

    compat = compatible_keys(sounding)

    return {
        "sounding_chords": sounding,
        "inferred_key": inf_key,
        "inferred_mode": inf_mode,
        "diatonic_fit": round(diatonic_fit, 3),
        "target_key": key,
        "target_mode": mode,
        "compatible_keys": compat,
        "capo_rows": capo_rows,
        "shape_family_rows": shape_rows,
        "summary": (
            f"**{' · '.join(sounding)}** fits **{inf_key} {inf_mode}** "
            f"({diatonic_fit:.0%} diatonic). The tables below show how to play the "
            f"**same sounding progression** at different capo frets or with different "
            f"open-chord families."
        ),
    }
