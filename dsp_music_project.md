# DSP Music Analysis & Guitar Resonance App — Project Spec

> **Status: Live App (v0.2)** — The app now does GuitarTuna-style **live microphone** analysis with a calibrated tuner needle and a live FFT spectrum, plus a dedicated guitar tuner. Server-side recording (sounddevice) is removed from the UI; input is either **live browser mic** or **file upload**. This document describes project objectives, the DSP foundations, and what the codebase delivers. Future enhancements are listed separately.

---

## Project Overview

A Python-based audio analysis app with four core capabilities:

1. **Guitar Tuner (Live)** — real-time pitch tracking from the browser mic, a calibrated needle (cents off target), selectable tunings, and a live FFT spectrum with the detected fundamental marked
2. **Vocal Range & Register Detector** — live or upload; identify range, register (chest / neck / head), and nasal tone
3. **Key, Capo & Chords** — detect a vocal clip's key, auto-generate a diatonic chord progression, and show beginner-friendly capo guides (finger shapes vs. sounding chords for every open-chord family)
4. **Voice Modulation Comparator** — compare an uploaded vocal against a reference track (pitch, vibrato, dynamics)

**Guitar only.** No other instruments in scope.

---

## Live Feedback Design (v0.2 / v0.3)

- **Browser mic capture:** `streamlit-webrtc` streams audio frames from the browser to Python (works over the network, unlike server-side `sounddevice`). A rolling buffer of recent samples is analyzed every loop iteration.
- **Pitch detection (live):** two selectable detectors in `utils/dsp_live.py`:
  - **Autocorrelation** with a "first strong peak" rule — robust against octave errors, accurate at low frequencies (guitar low-E ≈ 82 Hz), parabolic sub-sample interpolation.
  - **HPS (Harmonic Product Spectrum)** on a zero-padded, Hann-windowed FFT — mechanism adopted from **TomSchimansky/GuitarTuner**: the spectrum is multiplied by decimated copies of itself so harmonics reinforce the fundamental; zero-padding gives sub-cent resolution near A4.
- **Temporal smoothing:** `PitchSmoother` (mechanism from **Mocha-Yuan/MoChord**) applies median smoothing, a short dropout hold (prevents needle flicker), and octave-jump correction (rejects single-frame harmonic/subharmonic flips). RMS gating + a clarity threshold reject silence and unstable frames.
- **FFT spectrum:** `numpy` real FFT (Hann-windowed) produces the single-sided magnitude spectrum shown live; the detected fundamental is marked with a vertical line so you can see which frequency drives the note.
- **Needle UI:** a Plotly gauge (`utils/visuals.py`) shows cents deviation (−50..+50) with a green in-tune band; constantly recalibrated each frame.
- **Tuner stability & confirmation:** the tuner tab uses a note-stability counter (only locks a string after a few consistent frames) and confirms "in tune" after several consecutive in-tune frames — GuitarTuner's stability mechanism.
- **Tuner:** `utils/tuner.py` maps the detected pitch to the nearest string of the selected tuning and reports tune-up/down direction. Tunings: Standard, Drop D, **Low C**, Half/Full step down, Open G, Open D, DADGAD, plus **Custom** (define six strings). **Adjustable A4 reference (432–445 Hz)** recalculates all targets.

## Chord Intelligence (v0.3, MoChord-inspired)

- **Smart voicing recommender** (`utils/chord_voicing.py`): parses a chord symbol, searches the fretboard (frets 0–12) per position window, and scores candidate voicings by chord-tone coverage, root presence, bass-is-root, open-string use, fret span, internal-mute penalty, neck position, and hand ergonomics (finger count with barre detection). Rejects >4-finger and over-stretched shapes. Reproduces the standard open chords (E `022100`, C `x32010`, G `320003`, Am `x02210`, D `xx0232`, …).
- **Progression generator** (`utils/progression.py`): a deterministic, offline diatonic generator (MoChord's local/fallback path — no external AI/API). Given a key, mode, and degree pattern (numbers `1-5-6-4` or Roman `I-V-vi-IV`), it returns chord names, Roman numerals, and harmonic function, with a beginner (triads) / pro (7th chords, dominant V) toggle and common-pattern presets.
- **Shared music theory** (`utils/music_theory.py`): note/pitch-class conversion, chord parsing (`CHORD_INTERVALS`), and diatonic scale/seventh tables.

---

## Tech Stack

```
Python 3.10+
librosa          # audio loading, STFT, chroma, YIN, DTW (batch/upload analysis)
numpy            # DSP math, FFT, autocorrelation
scipy            # FFT helpers, filtering, Welch PSD, Butterworth filters
streamlit        # UI
streamlit-webrtc # live browser microphone capture (WebRTC)
plotly           # tuner needle gauge + live FFT spectrum
av               # audio frame decoding for WebRTC
matplotlib       # legacy/optional plots
sounddevice      # legacy (no longer used by the UI)
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app (Windows):

```powershell
.\run.ps1
```

Or directly:

```bash
streamlit run app.py
```

**Note:** MP3 and other compressed formats require FFmpeg on PATH. WAV, FLAC, and AIFF decode natively via soundfile.

---

## Mathematical Foundations

These concepts underpin all three features. The implementation uses librosa and scipy rather than reimplementing algorithms from scratch.

### 1. Fourier Transform (STFT)

Every audio frame is decomposed into frequency components:

```
X(k, n) = Σ x(n + m) · w(m) · e^(−j2πkm/N)
```

- `N` = FFT size (librosa default)
- Frequency of bin k: `f_k = k · (sample_rate / N)`

**Resource:** https://ccrma.stanford.edu/~jos/mdft/

### 2. Pitch (f0) Detection — YIN

```
d(τ) = Σ (x(t) − x(t + τ))²
d'(τ) = d(τ) / [(1/τ) Σ d(j)]
f0 = sample_rate / argmin(d'(τ))
```

Used via `librosa.yin(y, fmin, fmax, sr=22050)` across all features.

**Resource:** https://librosa.org/doc/latest/generated/librosa.yin.html

### 3. Musical Note — Frequency Mapping

```
MIDI note number: n = 69 + 12 · log2(f / 440)
Frequency from MIDI: f = 440 · 2^((n − 69) / 12)
Cents deviation: cents = 1200 · log2(f_measured / f_target)
```

Implemented in `utils/pitch_utils.py`.

**Resource:** https://pages.mtu.edu/~suits/notefreqs.html

### 4. Voice Register Frequency Bands

| Register      | Approx. Frequency Range |
|---------------|-------------------------|
| Chest voice   | 80 Hz – 350 Hz          |
| Mixed/Neck    | 300 Hz – 600 Hz         |
| Head/Falsetto | 500 Hz – 1200 Hz        |
| Nasal         | Formant resonance ~250 Hz and ~2500 Hz |

Register classification uses modal (median) f0. Nasal detection uses Welch PSD band-energy ratios.

### 5. Harmonic-to-Noise Ratio (HNR)

```
HNR = 10 · log10 (harmonic power / noise power)
```

Relevant for future voice-quality metrics; not yet surfaced in the UI.

### 6. Chroma Features

```
chroma[p] = Σ |X(k)|²  for all k where note(k) mod 12 == p
```

Used in scale matching and voice comparison. Key detection uses **Krumhansl–Schmuckler profiles** (major and minor) with cosine similarity, not a simple argmax on raw chroma energy.

### 7. Capo & Transposition Math

A capo on fret `n` raises all strings by `n` semitones:

```
capo_fret = (K_singer − K_standard) mod 12
```

Chord transposition is implemented in `features/scale_matcher.py`.

**Resource:** https://www.musictheory.net/lessons

### 8. Dynamic Time Warping (DTW)

Aligns two chroma sequences of different lengths/tempos:

```
DTW(i, j) = dist(i, j) + min(DTW(i−1, j), DTW(i, j−1), DTW(i−1, j−1))
```

Used via `librosa.sequence.dtw()` in the voice comparator. Chroma columns are sanitized before DTW to avoid NaN cosine distances on silent frames.

**Resource:** https://librosa.org/doc/latest/generated/librosa.sequence.dtw.html

---

## Feature 1 — Vocal Range & Register Detector

### Goal

Given a recorded vocal sample (5–15 seconds of sustained singing or scale runs):

- Output: lowest note, highest note, natural/modal pitch, register (with confidence + evidence), belt & nasal flags, and the raw acoustic features that drove the decision.

### Current Implementation (`features/vocal_range.py`, `utils/voice_features.py`, `utils/register.py`)

Registers are treated as **acoustic + laryngeal patterns**, not pitch labels. The
classifier combines pitch with the spectral signature of the voice so the same
note can read as chest, head, or belt depending on how the harmonics are shaped.

- **Input:** Upload (WAV, MP3, FLAC, etc.) or live microphone via WebRTC.
- **Pitch tracking:** YIN over C2–C7 at 22,050 Hz; range = 5th/95th percentile, modal = median.
- **Acoustic feature vector (`utils/voice_features.py`):**
  - **Spectral tilt** (dB/kHz) — power-weighted log-magnitude slope; shallow tilt = strong upper harmonics (chest/belt), steep tilt = energy near f0 (head/falsetto).
  - **HNR** — Boersma short-term autocorrelation method (`10·log10(r/(1−r))`); clean vs. breathy tone.
  - **HF energy ratio** and **spectral centroid** — spectral balance / brightness.
  - **Nasal-band ratio** — energy near 250 Hz + 2–3 kHz vs. the 500–1500 Hz mid-band.
- **Register decision (`utils/register.py`):** transparent weighted rules over pitch + tilt + HF energy + HNR → `{Chest, Mixed, Head, Falsetto}` with a 0–1 confidence, per-register scores, and a human-readable list of the evidence. A **belt** flag fires on high pitch with chest-like (shallow) tilt and strong HF energy; a **nasal** flag fires on elevated nasal-band energy.
- **Validation:** Rejects empty, silent, or too-short audio (`utils/audio_io.validate_audio_signal`).

Scope note: formant estimation (F1–F4 via LP/WLP) and glottal inverse filtering
(closed quotient, amplitude quotient QA) are documented in **References** as the
next tier of features; the current build uses the spectral-envelope proxies
above, which are robust in real time and require no per-period glottal modeling.

---

## Feature 2 — Key, Capo & Chords (Play-Along Planner)

### Goal

Given a vocal recording, give beginners **actionable guitar guidance** — not just a key label:

- Singer's pitch center (key + mode)
- An **auto-generated chord progression** that fits the detected key (Pop I–V–vi–IV for major, Andalusian for minor)
- **Every capo option** (G / A / C / D / E / F shapes): finger chords vs. sounding chords, with plain-English explanations
- Optional customization for advanced users (pattern presets, custom progression, 7th chords)
- Fretboard diagrams for the recommended capo row

### Current Implementation (`utils/play_along.py`, `features/scale_matcher.py`, `utils/progression.py`)

- **Input:** Uploaded clip + your **finger shapes** (e.g. `Am Em Dm F` — what you play with no capo)
- **Key detection:** Mean chroma + Krumhansl–Schmuckler major/minor profiles
- **Capo scan (0–7):** Finger shapes stay **fixed**; capo *raises* pitch so each fret produces **different sounding chords**. We score each row against your detected key and recommend the best capo.
- **Capo 0 warning:** If open strings don't fit your voice, we say so explicitly and point to a better fret.
- **Alternatives:** Other diatonic progressions in your vocal key if you want different material (not capo-shifted copies).
- **Auto mode:** If you leave shapes blank, we suggest comfortable G / Am open patterns and still run the capo scan.

### Guitar Standard Tuning (open strings)

```
E2 = 82.41 Hz    A2 = 110.00 Hz    D3 = 146.83 Hz
G3 = 196.00 Hz   B3 = 246.94 Hz    E4 = 329.63 Hz
```

---

## Feature 3 — Voice Modulation Comparator

### Goal

Load original song audio + user recording. Compare:

- Pitch accuracy (note-by-note deviation in cents)
- Vibrato presence and rate
- Dynamic range (loudness envelope shape)
- Chroma similarity (overall key/scale matching via DTW alignment)

### Current Implementation (`features/comparator.py`)

- **Input:** Reference track (sidebar upload) + user upload (no local recording)
- **Pitch tracking:** YIN on both tracks
- **Alignment:** DTW on sanitized chroma STFT features
- **Pitch deviation:** Mean absolute cents error along the warping path
- **Vibrato:** Butterworth low-pass on voiced f0, FFT of pitch modulation in 4–9 Hz band
- **Dynamics:** RMS envelope correlation aligned via the same DTW path
- **Visualization:** Plotly plot of aligned reference vs. user pitch in the UI

---

## Feature 4 — Guitar Tuner (Live)

### Goal

Tune a guitar in real time from the browser microphone, GuitarTuna-style.

### Current Implementation (`utils/tuner.py`, `utils/dsp_live.py`, `utils/live_audio.py`)

- **Input:** Live browser mic via WebRTC
- **Tunings:** Standard, Drop D, Half/Full step down, Open G, Open D, DADGAD
- **Pitch detection:** autocorrelation + parabolic interpolation on a rolling buffer
- **Needle:** Plotly gauge shows cents off the nearest string; green band = in tune (±5 ¢)
- **Spectrum:** live FFT with the detected fundamental marked
- **Guidance:** reports the nearest string, target Hz, and tune-up / tune-down direction

---

## Project File Structure

```
RayMozic/
├── app.py                  # Streamlit UI entry point (5 tabs)
├── run.ps1                 # Windows launcher
├── requirements.txt
├── dsp_music_project.md    # This document
│
├── features/
│   ├── vocal_range.py      # Vocal range/register (upload)
│   ├── scale_matcher.py    # Key detection + capo recommendations
│   └── comparator.py       # Reference-track comparison
│
├── utils/
│   ├── audio_io.py         # load + validate uploaded audio
│   ├── pitch_utils.py      # hz↔note, cents
│   ├── chroma_utils.py     # chroma extraction, KS key detection
│   ├── dsp_live.py         # autocorr + HPS pitch detection, PitchSmoother, FFT
│   ├── voice_features.py   # spectral tilt, HNR, HF ratio, centroid, nasal ratio
│   ├── register.py         # acoustic register classifier (explainable rules)
│   ├── tuner.py            # tuning presets (+ Low C, custom), A4 reference
│   ├── music_theory.py     # notes, chord parsing, diatonic scales/sevenths
│   ├── chord_voicing.py    # fretboard voicing search + scoring (MoChord)
│   ├── progression.py      # diatonic progression generator (MoChord local path)
│   ├── play_along.py       # voice + fixed finger shapes → best capo
│   ├── progression_capo_map.py  # fixed sounding prog → capo / scale map
│   ├── visuals.py          # Plotly tuner meter, string pills, spectrum, fretboard
│   ├── live_audio.py       # WebRTC mic capture + live processing loop
│   └── results_store.py    # persist results + competitor benchmarks
│
└── tests/
    ├── test_chroma_dtw.py       # chroma/DTW guards
    ├── test_results_store.py    # result persistence
    ├── test_live_dsp.py         # live pitch detection + tuner
    ├── test_voice_features.py   # acoustic features + register classifier
    └── test_chords.py           # HPS, smoother, voicings, progressions
```

---

## UI Flow (Streamlit)

```
Sidebar: Upload reference audio (for Voice Comparator)

Tab 1: Guitar Tuner (Live)
  - Select tuning; START mic
  - Output: strobe-style horizontal meter (flat ◀ in-tune ▶ sharp) with big note
    name + FLAT/SHARP/IN-TUNE hint, a string-pill row highlighting the active
    string, and the live FFT spectrum with the detected fundamental marked

Tab 2: Vocal Range
  - Live mic (accumulating range) OR upload
  - Output: Low / Modal / High note; register with confidence bar + acoustic
    feature metrics (tilt, HNR, HF energy, brightness) + "why this register"
    evidence; belt & nasal flags; FFT spectrum

Tab 3: Key, Capo & Chords (voice match)
  - Upload vocal + your finger shapes → best capo to match your voice
  - Same shapes at each fret sound *different*; we score voice fit

Tab 4: Progression & Capo Map (theory / exploration)
  - Enter sounding progression (e.g. G Em C D) → keys/scales it fits
  - Capo table: same sound at every fret, finger shapes change
  - Open-family table (G/A/C/D/E/F): capo + shapes to reach that sound in the key

Tab 5: Voice Comparator
  - Requires sidebar reference upload; upload your vocal
  - Output: mean cents deviation, vibrato, dynamic similarity %, pitch alignment plot

Tab 5: Results & Benchmarks
  - Saved-run history, JSON export, competitor comparison links
```

---

## Delivered in v0.2

- Live browser-mic pitch tracking (WebRTC) with a calibrated tuner needle
- Live FFT spectrum with the detected fundamental highlighted
- Dedicated Guitar Tuner tab with tuning presets and tune-up/down guidance
- Removed server-side `sounddevice` recording from the UI (upload or live only)
- Capo ranking now flags impractical high-fret positions instead of dead-branching

## Delivered in v0.3 (from GuitarTuner + MoChord)

- **HPS pitch detector** (GuitarTuner) selectable alongside autocorrelation
- **PitchSmoother**: median smoothing, dropout hold, octave-jump correction (MoChord)
- **Note-stability counter + in-tune confirmation** in the tuner (GuitarTuner)
- **Adjustable A4 reference (432–445 Hz)**, **Low C** tuning, **custom tuning** (both)
- **Smart chord-voicing recommender** with fretboard diagrams (MoChord)
- **Offline diatonic progression generator** (MoChord's local/fallback path)

## Delivered in v0.5 (unified play-along)

- **Merged Guitar Scale Matcher + Chord Progression** into one **Key, Capo & Chords** tab
- **Auto chord progression** from detected key (Pop pattern for major, Andalusian for minor)
- **Full capo playbook** — all six open-chord families with finger shapes, sounding chords, and beginner explanations
- **Capo transpose fix** — shapes are transposed *down* by capo fret (correct physical fingering)
- **Beginner-first layout** with optional customization expander and integrated fretboard diagrams

## Delivered in v0.4 (acoustic register + tuner UI)

- **Feature-based register classifier** — spectral tilt (power-weighted), HNR (Boersma autocorrelation), HF-energy ratio, spectral centroid, and nasal-band ratio combined via explainable weighted rules into `{Chest, Mixed, Head, Falsetto}` with confidence, per-register scores, and human-readable evidence.
- **Belt detection** — high pitch with chest-like shallow tilt + strong HF energy.
- **Redesigned tuner UI** — strobe-style horizontal meter with directional FLAT/SHARP/IN-TUNE hints, an active-string pill row, and Hz→target readout (replaces the single radial needle).
- **Removed** the redundant pitch-only `classify_register` and the meaningless "modal pitch vs nearest note" needle on the vocal upload view.

## Credits / Upstream Mechanisms

- **[TomSchimansky/GuitarTuner](https://github.com/TomSchimansky/GuitarTuner)** — HPS on a zero-padded FFT buffer, cents/note math, needle smoothing, note-stability and in-tune confirmation, adjustable A4.
- **[Mocha-Yuan/MoChord](https://github.com/Mocha-Yuan/MoChord)** — tuner signal chain (RMS gating, clarity threshold, median smoothing, dropout hold, octave-jump correction), tuning presets + custom + A4 range, smart voicing scoring, and the local (non-AI) progression generation path.

## Future Enhancements (Not Yet Implemented)

These objectives remain valid but are not yet built:

- **Formant estimation (F1–F4)** via LP / weighted LP + root finding (for vowel/register nuance).
- **Glottal source modeling** via LP inverse filtering to derive closed quotient and normalized amplitude quotient (QA).
- Optional small trained classifier (e.g. logistic regression) over the existing feature vector once labeled data is collected.
- Live per-frame register voting (currently the register decision runs on uploaded/segmented audio, which needs a spectral window).
- Reference-tone playback per string in the tuner.
- AI-backed progression generation (MoChord uses DeepSeek; ours is local-only).
- Metronome / practice-mode loop and Tone.js-style chord audition.
- Sample audio fixtures for end-to-end tests.

---

## Key References

| Topic | URL |
|-------|-----|
| STFT / Fourier math | https://ccrma.stanford.edu/~jos/mdft/ |
| YIN pitch algorithm | http://audition.ens.fr/adc/pdf/2002_JASA_YIN.pdf |
| librosa YIN docs | https://librosa.org/doc/latest/generated/librosa.yin.html |
| Note frequency table | https://pages.mtu.edu/~suits/notefreqs.html |
| Music theory (scales, keys) | https://www.musictheory.net/lessons |
| Krumhansl key profiles | https://www.jstor.org/stable/40285499 |
| Guitar acoustics / PASP | https://ccrma.stanford.edu/~jos/pasp/ |
| Vocal formants / registers | https://home.cc.umanitoba.ca/~robh/acoustics.html |
| Voice science | https://www.voicescienceworks.org/ |
| Praat (voice analysis tool) | https://www.fon.hum.uva.nl/praat/ |
| Chest vs. head register acoustics/laryngeal markers | https://pubmed.ncbi.nlm.nih.gov/ (search "Laryngeal and Acoustic Analysis of Chest and Head Register") |
| HNR (Boersma autocorrelation method) | https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf |
| Formant estimation from LP data | https://en.wikipedia.org/wiki/Linear_predictive_coding |
| Weighted LP for high-pitched singing formants | https://acris.aalto.fi/ (search "weighted linear prediction formant singing") |
| Source–filter / glottal modeling (Rabiner & Schafer) | https://www.pearson.com/en-us/subject-catalog/p/digital-processing-of-speech-signals/ |
| DTW in librosa | https://librosa.org/doc/latest/generated/librosa.sequence.dtw.html |
| MIR evaluation metrics | https://craffel.github.io/mir_eval/ |
