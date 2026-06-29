# DSP Music Analysis & Guitar Resonance App — Project Spec

> **Status: Foundation App (v0.1)** — Core features are implemented and wired through a Streamlit UI. This document describes project objectives, the DSP foundations they rely on, and what the current codebase delivers. Future enhancements are listed separately; removed are build steps and pseudocode for work not yet performed.

---

## Project Overview

A Python-based audio analysis app with three core features:

1. **Vocal Range & Register Detector** — identify singer's range and voice register (chest, neck/head, nasal)
2. **Guitar Scale Resonance Matcher** — match singer's natural pitch center to optimal guitar key/capo position
3. **Voice Modulation Comparator** — compare user's singing against a reference track (original artist)

**Guitar only.** No other instruments in scope.

---

## Tech Stack

```
Python 3.10+
librosa          # audio loading, STFT, pitch tracking, chroma, DTW
numpy            # DSP math
scipy            # signal filtering, Welch PSD, Butterworth filters
sounddevice      # real-time mic recording
matplotlib       # pitch alignment plots in the comparator tab
streamlit        # UI
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

- Output: lowest note, highest note, natural/modal pitch, dominant register, nasal tone flag

### Current Implementation (`features/vocal_range.py`)

- **Input:** Upload (WAV, MP3, FLAC, etc.) or microphone recording via Streamlit
- **Pitch tracking:** YIN over C2–C7 at 22,050 Hz sample rate
- **Range:** 5th/95th percentile for low/high; median for modal pitch
- **Register:** Classified from modal f0 (`utils/pitch_utils.classify_register`)
- **Nasal detection:** Welch PSD comparing 200–300 Hz and 2000–3000 Hz bands against a 500–1500 Hz mid-band
- **Validation:** Rejects empty, silent, or too-short audio (`utils/audio_io.validate_audio_signal`)

---

## Feature 2 — Guitar Scale Resonance Matcher

### Goal

Given a vocal recording, determine:

- Singer's pitch center (key)
- Which standard guitar key/chord shapes resonate best
- Recommended capo fret
- Alternative chord progressions in matched key

### Current Implementation (`features/scale_matcher.py`)

- **Input:** Vocal audio from Tab 1 session state (run Vocal Range Analysis first)
- **Key detection:** Mean chroma + Krumhansl–Schmuckler major/minor profiles (`utils/chroma_utils.detect_key`)
- **Capo recommendations:** Top 3 options across guitar-friendly open shapes (G, A, C, D, E, F), ranked by lowest capo fret
- **Chord transposition:** Space-separated progression transposed by semitones (e.g. `G Em C D`)

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

- **Input:** Reference track (sidebar upload) + user upload or microphone recording
- **Pitch tracking:** YIN on both tracks
- **Alignment:** DTW on sanitized chroma STFT features
- **Pitch deviation:** Mean absolute cents error along the warping path
- **Vibrato:** Butterworth low-pass on voiced f0, FFT of pitch modulation in 4–9 Hz band
- **Dynamics:** RMS envelope correlation aligned via the same DTW path
- **Visualization:** Matplotlib plot of aligned reference vs. user pitch in the UI

---

## Project File Structure

```
RayMozic/
├── app.py                  # Streamlit UI entry point
├── run.ps1                 # Windows launcher (uses .venv Streamlit)
├── requirements.txt
├── dsp_music_project.md    # This document
│
├── features/
│   ├── vocal_range.py      # Feature 1
│   ├── scale_matcher.py    # Feature 2
│   └── comparator.py       # Feature 3
│
├── utils/
│   ├── audio_io.py         # record, load, validate audio
│   ├── pitch_utils.py      # hz↔note, cents, register classification
│   └── chroma_utils.py     # chroma extraction, KS key detection, sanitization
│
└── tests/
    └── test_chroma_dtw.py  # chroma sanitization, validation, comparator guards
```

---

## UI Flow (Streamlit)

```
Sidebar: Upload reference audio (Feature 3)

Tab 1: Vocal Range Analysis
  - Upload or record (5–15 sec)
  - Output: Low / Modal / High note, register, nasal flag
  - Saves vocal audio to session state for Tab 2

Tab 2: Guitar Scale Matcher
  - Requires Tab 1 analysis first
  - Input: chord progression text (default "G Em C D")
  - Output: Detected key/mode, confidence, top 3 capo recommendations, transposed chords

Tab 3: Voice Comparator
  - Requires sidebar reference upload
  - Upload or record user vocal
  - Output: Mean cents deviation, vibrato, dynamic similarity %, pitch alignment plot
```

---

## Future Enhancements (Not Yet Implemented)

These objectives remain valid but are not part of the foundation app:

- Capo inversion heuristic when `capo_fret > 6` (full alternate-shape logic)
- Per-frame dominant register voting (currently uses modal pitch only)
- Cosine similarity scoring of guitar keys against singer chroma (current ranking uses capo fret heuristic)
- `data/guitar_keys.json` external chord/capo reference data
- HNR and broader voice-quality metrics in the UI
- pYIN pitch tracking option for improved octave accuracy
- Expanded test coverage and sample audio fixtures

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
| DTW in librosa | https://librosa.org/doc/latest/generated/librosa.sequence.dtw.html |
| MIR evaluation metrics | https://craffel.github.io/mir_eval/ |
