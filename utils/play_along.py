"""Play-along planner: fixed finger shapes + capo → different sounding keys.

The user's chord shapes stay the same (what they physically play). A capo
*raises* pitch, so each fret produces a *different* sounding progression.
We score every capo 0–7 against the detected vocal key and recommend the best
fit — plus optional alternative progressions in the singer's key.
"""

from __future__ import annotations

from utils.music_theory import (
    chord_pitch_classes,
    diatonic_chords,
    name_to_pc,
    pc_to_name,
)
from utils.progression import COMMON_PATTERNS, generate_progression
from features.scale_matcher import match_guitar_scale, transpose_progression

DEFAULT_PATTERN = {
    "major": "1-5-6-4",
    "minor": "1-7-6-5",
}

PATTERN_LABELS = {
    "1-5-6-4": "Pop (I–V–vi–IV)",
    "1-6-4-5": "50s ballad (I–vi–IV–V)",
    "1-4-5": "Blues (I–IV–V)",
    "1-7-6-5": "Andalusian (i–VII–VI–V)",
    "2-5-1": "Jazz turnaround (ii–V–I)",
}

def _mode_key(mode: str) -> str:
    return "minor" if mode.lower().startswith("min") else "major"


def sounding_at_capo(finger_chords: list[str], capo_fret: int) -> list[str]:
    """What the audience hears when you play `finger_chords` with capo on `capo_fret`."""
    if capo_fret == 0:
        return list(finger_chords)
    joined = transpose_progression(" ".join(finger_chords), capo_fret)
    return joined.split() if joined else []


def infer_key_from_chords(chord_names: list[str]) -> tuple[str, str, float]:
    """Best-fit key/mode for a chord list + diatonic fit score 0..1."""
    if not chord_names:
        return "C", "major", 0.0

    best_key, best_mode, best_fit = "C", "major", 0.0
    for root_pc in range(12):
        for mode in ("major", "minor"):
            table = diatonic_chords(root_pc, mode)
            hits = 0
            for ch in chord_names:
                try:
                    pcs = chord_pitch_classes(ch)
                except ValueError:
                    continue
                if any(pcs == chord_pitch_classes(d["name"]) for d in table):
                    hits += 1
            fit = hits / len(chord_names)
            if fit > best_fit:
                best_fit = fit
                best_key = pc_to_name(root_pc)
                best_mode = mode
    return best_key, best_mode, best_fit


def vocal_match_score(
    inferred_key: str,
    inferred_mode: str,
    vocal_key: str,
    vocal_mode: str,
    diatonic_fit: float,
) -> float:
    """0..1 how well a capo row's sounding key aligns with the singer's key."""
    inf_pc = name_to_pc(inferred_key)
    voc_pc = name_to_pc(vocal_key)
    inf_minor = _mode_key(inferred_mode) == "minor"
    voc_minor = _mode_key(vocal_mode) == "minor"

    if inf_pc == voc_pc and inf_minor == voc_minor:
        root_score = 1.0
    elif inf_minor != voc_minor:
        # Relative major/minor (e.g. C major ↔ A minor).
        if not inf_minor and (inf_pc + 9) % 12 == voc_pc and voc_minor:
            root_score = 0.88
        elif inf_minor and (inf_pc + 3) % 12 == voc_pc and not voc_minor:
            root_score = 0.88
        else:
            dist = min((inf_pc - voc_pc) % 12, (voc_pc - inf_pc) % 12)
            root_score = max(0.0, 1.0 - dist / 6.0) * 0.5
    else:
        dist = min((inf_pc - voc_pc) % 12, (voc_pc - inf_pc) % 12)
        root_score = max(0.0, 1.0 - dist / 6.0)

    return round(root_score * 0.7 + diatonic_fit * 0.3, 3)


def _capo_explanation(
    capo: int,
    finger: list[str],
    sounding: list[str],
    inferred_key: str,
    inferred_mode: str,
    vocal_key: str,
    vocal_mode: str,
    match_score: float,
) -> str:
    finger_str = " · ".join(finger)
    sound_str = " · ".join(sounding)
    mode_word = inferred_mode
    if capo == 0:
        lead = f"**No capo.** Finger **{finger_str}** — they sound like **{sound_str}** (~{inferred_key} {mode_word})."
    else:
        lead = (
            f"**Capo fret {capo}.** Keep fingering **{finger_str}** — "
            f"they sound like **{sound_str}** (~{inferred_key} {mode_word})."
        )
    if match_score >= 0.75:
        tail = f" Strong match for your voice ({vocal_key} {vocal_mode})."
    elif match_score >= 0.5:
        tail = f" Partial match for your voice ({vocal_key} {vocal_mode}) — usable but not ideal."
    else:
        tail = f" Weak match for your voice ({vocal_key} {vocal_mode}) — try another capo fret."
    return lead + tail


def build_capo_options(
    finger_chords: list[str],
    vocal_key: str,
    vocal_mode: str,
    max_fret: int = 7,
) -> list[dict]:
    """Scan capo frets; finger shapes are identical, sounding chords change each row."""
    rows = []
    for capo in range(max_fret + 1):
        sounding = sounding_at_capo(finger_chords, capo)
        inf_key, inf_mode, diatonic_fit = infer_key_from_chords(sounding)
        score = vocal_match_score(inf_key, inf_mode, vocal_key, vocal_mode, diatonic_fit)
        rows.append({
            "capo_fret": capo,
            "finger_chords": list(finger_chords),
            "sounding_chords": sounding,
            "inferred_key": inf_key,
            "inferred_mode": inf_mode,
            "diatonic_fit": round(diatonic_fit, 3),
            "match_score": score,
            "matches_voice": score >= 0.75,
            "practical": capo <= 7,
            "explanation": _capo_explanation(
                capo, finger_chords, sounding, inf_key, inf_mode,
                vocal_key, vocal_mode, score,
            ),
        })
    rows.sort(key=lambda r: (-r["match_score"], r["capo_fret"]))
    return rows


def suggest_alternative_progressions(
    vocal_key: str,
    vocal_mode: str,
    sevenths: bool = False,
) -> list[dict]:
    """Other diatonic progressions in the singer's key (different sounding chords)."""
    mode = _mode_key(vocal_mode)
    alts = []
    patterns = [
        ("1-5-6-4", PATTERN_LABELS["1-5-6-4"]),
        ("1-6-4-5", PATTERN_LABELS["1-6-4-5"]),
        ("1-4-5", PATTERN_LABELS["1-4-5"]),
    ]
    if mode == "minor":
        patterns.append(("1-7-6-5", PATTERN_LABELS["1-7-6-5"]))

    for pat, label in patterns:
        meta = generate_progression(vocal_key, mode, pat, sevenths=sevenths)
        alts.append({
            "pattern": pat,
            "pattern_label": label,
            "sounding_chords": meta["chord_names"],
            "progression_meta": meta,
        })
    return alts


def _auto_finger_progression(
    vocal_mode: str, pattern: str, sevenths: bool,
) -> tuple[list[str], dict]:
    """Comfortable open shapes (G major or A minor); capo shifts them to the vocal key."""
    mode = _mode_key(vocal_mode)
    if mode == "minor":
        open_key, open_mode = "A", "minor"
    else:
        open_key, open_mode = "G", "major"
    meta = generate_progression(open_key, open_mode, pattern, sevenths=sevenths)
    return meta["chord_names"], meta


def _beginner_summary(
    vocal_key: str,
    vocal_mode: str,
    finger: list[str],
    best: dict | None,
    capo0: dict | None,
) -> str:
    finger_str = " · ".join(finger)
    base = (
        f"Your voice sits in **{vocal_key} {vocal_mode}**. "
        f"Keep fingering **{finger_str}** and move the capo — each fret changes "
        f"what the listener hears."
    )
    if best:
        base += (
            f" Best fit: **capo {best['capo_fret']}** → sounds like "
            f"**{' · '.join(best['sounding_chords'])}** ({best['inferred_key']} {best['inferred_mode']})."
        )
    if capo0 and capo0["match_score"] < 0.5 and best and best["capo_fret"] != 0:
        base += (
            f" Open (capo 0) only scores {capo0['match_score']:.0%} against your voice — "
            f"capo {best['capo_fret']} works better."
        )
    return base


def build_play_plan(
    audio,
    sr: int = 22050,
    pattern: str | None = None,
    sevenths: bool = False,
    finger_progression: str | None = None,
) -> dict:
    """
    Build a play-along plan from vocal audio.

    `finger_progression` = chord shapes the user fingers at the nut (e.g. "Am Em Dm F").
    If omitted, we suggest comfortable open shapes in G / Am.
    """
    match = match_guitar_scale(audio, sr)
    mode = _mode_key(match["vocal_mode"])
    vocal_key = match["vocal_key"]
    pattern_used = pattern or DEFAULT_PATTERN[mode]
    pattern_label = PATTERN_LABELS.get(pattern_used, pattern_used)

    if finger_progression and finger_progression.strip():
        finger_chords = finger_progression.strip().split()
        prog_meta = None
        pattern_label = "Your chord shapes"
    else:
        finger_chords, prog_meta = _auto_finger_progression(
            match["vocal_mode"], pattern_used, sevenths,
        )

    capo_options = build_capo_options(finger_chords, vocal_key, match["vocal_mode"])
    best_capo = capo_options[0] if capo_options else None
    capo0 = next((r for r in capo_options if r["capo_fret"] == 0), None)

    # Sorted by capo fret for display table.
    capo_by_fret = sorted(capo_options, key=lambda r: r["capo_fret"])

    alternatives = suggest_alternative_progressions(vocal_key, match["vocal_mode"], sevenths)

    return {
        **match,
        "mode_normalized": mode,
        "pattern": pattern_used,
        "pattern_label": pattern_label,
        "finger_progression": finger_chords,
        "progression_meta": prog_meta,
        "capo_options": capo_options,
        "capo_by_fret": capo_by_fret,
        "best_capo": best_capo,
        "capo_zero": capo0,
        "capo_zero_ok": capo0["matches_voice"] if capo0 else False,
        "alternatives": alternatives,
        "beginner_summary": _beginner_summary(
            vocal_key, match["vocal_mode"], finger_chords, best_capo, capo0,
        ),
    }
