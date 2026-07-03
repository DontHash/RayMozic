"""Acoustic voice-register classification from a feature vector.

Combines pitch context with spectral evidence (tilt, high-frequency energy,
HNR) using transparent weighted rules, so the decision is explainable rather
than a black box. Returns the register, a 0..1 confidence, per-register scores,
and the human-readable reasons that drove the call.

Register model:
- Chest    : lower/mid pitch, shallow spectral tilt, strong high harmonics
- Mixed    : transitional pitch, intermediate spectral balance
- Head     : higher pitch, steep spectral tilt, energy near the fundamental
- Falsetto : high pitch + steep tilt + low HNR (breathy, weak harmonics)
Flags (not mutually exclusive with the base register):
- Belt     : high pitch BUT chest-like shallow tilt + strong HF energy
- Nasal    : elevated nasal-band energy ratio
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.voice_features import VoiceFeatures

REGISTERS = ["Chest", "Mixed", "Head", "Falsetto"]


@dataclass
class RegisterResult:
    register: str
    confidence: float
    is_belt: bool
    is_nasal: bool
    scores: dict[str, float]
    reasons: list[str]


def _pitch_scores(f0: float) -> dict[str, float]:
    """Soft pitch membership per register (overlapping ranges, not hard cutoffs)."""
    scores = {r: 0.0 for r in REGISTERS}
    if f0 <= 0:
        return scores
    # Approximate physiological centres (Hz); membership falls off away from them.
    centres = {"Chest": 180.0, "Mixed": 330.0, "Head": 520.0, "Falsetto": 700.0}
    for reg, centre in centres.items():
        # Gaussian-ish membership in log-frequency (musical) space.
        import math
        octaves = math.log2(f0 / centre)
        scores[reg] = math.exp(-(octaves ** 2) / (2 * 0.35 ** 2))
    return scores


def classify_register_features(feat: VoiceFeatures) -> RegisterResult:
    """Classify register from acoustic features with an explainable rule set."""
    reasons: list[str] = []
    scores = _pitch_scores(feat.f0_median)

    tilt = feat.spectral_tilt      # dB/kHz, typically negative
    hf = feat.hf_ratio             # 0..1 energy above 1.5 kHz
    hnr = feat.hnr_db

    # --- Spectral tilt evidence -------------------------------------------------
    # Shallow tilt (closer to 0) => strong upper harmonics => chest/belt-like.
    # Steep tilt (very negative) => energy near f0 => head/falsetto-like.
    if tilt > -6.0:
        scores["Chest"] += 0.8
        scores["Mixed"] += 0.2
        reasons.append(f"Shallow spectral tilt ({tilt:.1f} dB/kHz) → strong upper harmonics (chest-like)")
    elif tilt > -12.0:
        scores["Mixed"] += 0.6
        scores["Chest"] += 0.2
        scores["Head"] += 0.2
        reasons.append(f"Moderate spectral tilt ({tilt:.1f} dB/kHz) → transitional balance (mixed)")
    else:
        scores["Head"] += 0.7
        scores["Falsetto"] += 0.4
        reasons.append(f"Steep spectral tilt ({tilt:.1f} dB/kHz) → energy near the fundamental (head-like)")

    # --- High-frequency energy --------------------------------------------------
    if hf > 0.25:
        scores["Chest"] += 0.4
        reasons.append(f"High HF energy ({hf*100:.0f}% above 1.5 kHz) → bright, harmonically rich tone")
    elif hf < 0.08:
        scores["Head"] += 0.3
        scores["Falsetto"] += 0.3
        reasons.append(f"Low HF energy ({hf*100:.0f}% above 1.5 kHz) → dark, few upper harmonics")

    # --- HNR / breathiness ------------------------------------------------------
    if hnr < 7.0 and feat.f0_median > 350.0:
        scores["Falsetto"] += 0.5
        reasons.append(f"Low HNR ({hnr:.1f} dB) at high pitch → breathy (falsetto-like)")
    elif hnr > 15.0:
        scores["Chest"] += 0.2
        reasons.append(f"High HNR ({hnr:.1f} dB) → clean, well-adducted tone")

    # --- Decide -----------------------------------------------------------------
    best = max(scores, key=scores.get)
    total = sum(scores.values()) or 1.0
    confidence = scores[best] / total

    # --- Belt flag: high pitch with chest-like (shallow) tilt + strong HF -------
    is_belt = feat.f0_median > 380.0 and tilt > -8.0 and hf > 0.2
    if is_belt:
        reasons.append("High pitch with chest-like tilt & strong HF → belt characteristics")

    is_nasal = feat.nasal_ratio > 1.4
    if is_nasal:
        reasons.append(f"Elevated nasal-band energy (ratio {feat.nasal_ratio:.2f}) → nasal resonance")

    return RegisterResult(
        register=best,
        confidence=float(round(confidence, 3)),
        is_belt=bool(is_belt),
        is_nasal=bool(is_nasal),
        scores={k: float(round(v, 3)) for k, v in scores.items()},
        reasons=reasons,
    )
