# DSP Report Source Document — RayMozic

> **How to use this file:** Feed this document to Claude, ChatGPT, or another LLM and ask it to write a formal DSP lab/project report. The LLM should keep simple English for most sections. Use proper DSP terms only where needed. Expand each section into full report paragraphs. Do not copy long code blocks into the report. Use the placeholders below for your personal details, screenshots, and demo video.

---

## Instructions for the LLM (do not print in final report)

- Target audience: DSP subject report for college level
- Tone: clear and simple. Avoid long sentences and decorative punctuation
- Length: concise report, not a full manual of every app feature
- Focus on **Digital Signal Processing** concepts, not UI details
- Include scientific terms where appropriate: FFT, STFT, fundamental frequency, autocorrelation, chroma, HNR, spectral tilt, DTW, etc.
- Leave figure placeholders as captions in the final report
- Implementation section should mention GitHub link only, no pasted source code
- Simulation/demo video link goes in Results section as placeholder

---

## 1. Cover Page

| Field | Value |
|-------|-------|
| **Project Title** | RayMozic: Live Digital Signal Processing for Guitar Tuning and Vocal Analysis |
| **Subject** | Digital Signal Processing |
| **Student Name** | `[YOUR FULL NAME]` |
| **Roll Number** | `[YOUR ROLL NUMBER]` |
| **Department / Class** | `[YOUR DEPARTMENT]` |
| **College** | `[YOUR COLLEGE NAME]` |
| **Academic Year** | `[YEAR]` |
| **Submission Date** | `[DATE]` |

---

## 2. Abstract

RayMozic is a web based music analysis application built in Python. It uses live microphone input and uploaded audio files to study sound in the frequency domain. The main DSP tasks are pitch detection, spectrum analysis, vocal range measurement, musical key detection, and voice comparison against a reference track.

The system uses the Fast Fourier Transform to show how energy is spread across frequencies. Pitch is found using autocorrelation and Harmonic Product Spectrum methods for live guitar tuning, and the YIN algorithm for vocal analysis. For key detection, the app builds a chroma vector from the short time Fourier transform and matches it to known major and minor key profiles. Vocal register is estimated using acoustic features such as spectral tilt and harmonic to noise ratio, not pitch alone.

The project shows that basic DSP blocks can be combined into a practical tool for musicians. Results appear as live tuner meters, spectrum plots, range charts, and comparison metrics. Full source code is available on GitHub. A demo video will be attached separately.

**Word count target for LLM:** about 150 to 200 words

---

## 3. Introduction

### 3.1 Background

Sound is a time domain signal. Digital Signal Processing lets us move this signal into the frequency domain using the Fourier Transform. Once we see frequency components, we can measure pitch, study harmonics, detect musical keys, and compare two voices mathematically.

Many music apps show notes and tunings, but the signal processing steps are often hidden. This project makes those steps visible. The user can see the FFT spectrum, the detected fundamental frequency, and how analysis results change in real time.

### 3.2 Related Theory (short)

**Sampling:** Audio is captured at 22050 Hz sample rate for analysis.

**STFT and FFT:** Audio is split into short frames. Each frame is windowed (Hann window) and passed through FFT to get magnitude spectrum.

**Fundamental frequency (f0):** The lowest strong periodic component of a voiced sound. Pitch in music is derived from f0.

**Autocorrelation:** A time domain method. The signal is correlated with delayed copies of itself. The delay at the strongest peak gives the period, and period inverse gives frequency.

**Harmonic Product Spectrum (HPS):** A frequency domain method. The magnitude spectrum is multiplied by downsampled copies of itself so that harmonic peaks reinforce the true fundamental.

**YIN algorithm:** A improved pitch tracker for voice. It uses a normalized difference function to find the best lag for f0.

**Chroma features:** The 12 pitch classes (C, C#, D, ... B) summed across octaves. Used for key detection.

**Cents:** A log unit for pitch error. 1200 cents = one octave. Used in the guitar tuner.

**Harmonic to Noise Ratio (HNR):** Measures how periodic vs noisy a voice signal is.

**Spectral tilt:** How fast harmonic energy drops as frequency increases. Helps distinguish chest like vs head like voice quality.

**Dynamic Time Warping (DTW):** Aligns two time series that may run at different speeds. Used to compare pitch tracks of two singers.

### 3.3 Purpose

The purpose of this project is to:

1. Apply core DSP methods to real audio from microphone and file upload
2. Build a live feedback system so users can see frequency analysis while they play or sing
3. Support practical musician tasks: tune guitar, measure vocal range, find song key, compare voice to a reference
4. Show the link between theory (FFT, pitch, chroma) and working software

---

## 4. Problem Statement

Musicians often need to know if they are in tune, what key they are singing in, and how their voice compares to a reference performance. Doing this by ear alone is hard for beginners.

The problem this project solves:

- How to detect pitch accurately from live microphone audio, including low guitar strings
- How to display frequency information so the user understands what the software measured
- How to estimate vocal range and register using signal features, not guesswork
- How to detect the musical key of a vocal melody using chroma analysis
- How to compare two vocal recordings using pitch alignment and simple similarity metrics

The challenge is to run these DSP steps fast enough for live use in a web browser, while still giving stable and readable output.

---

## 5. Methodology

### 5.1 Overall Signal Flow

```
Audio input (live mic or file)
    → Preprocess (mono, validate level, buffer recent frames)
    → Pitch track OR spectrum OR chroma (depends on feature)
    → Feature extraction (f0, cents, HNR, tilt, chroma vector, etc.)
    → Decision / display (note name, tuner meter, key, register, comparison score)
```

### 5.2 Live Guitar Tuner Pipeline

1. Browser sends audio frames through WebRTC
2. Rolling buffer keeps the last few hundred milliseconds
3. Hann window applied before FFT
4. Pitch detected by **autocorrelation** or **HPS** (user selectable)
5. **PitchSmoother** reduces flicker: median filter, dropout hold, octave jump correction
6. Detected f0 converted to note name and cents offset from target string frequency
7. FFT magnitude spectrum plotted with fundamental marked

### 5.3 Vocal Range and Register Pipeline

1. Uploaded or live audio analyzed with **YIN** over singing range (about C2 to C7)
2. Low, high, and modal pitch taken from percentile and median of voiced frames
3. Acoustic features extracted:
   - Spectral tilt (power weighted log spectrum slope)
   - HNR (autocorrelation based, Boersma style)
   - High frequency energy ratio and spectral centroid
4. Register classified with simple weighted rules using pitch plus spectral features

### 5.4 Key and Capo Analysis

1. Mean chroma vector computed from STFT
2. Key matched using **Krumhansl Schmuckler** major and minor profiles
3. User chord shapes kept fixed; capo position shifts sounding key
4. Each capo fret scored against detected vocal key

### 5.5 Voice Comparison

1. YIN pitch tracks extracted for user vocal and reference track
2. Chroma features aligned with **DTW**
3. Mean cents deviation measured along alignment path
4. Vibrato checked by low pass filtering pitch track and FFT of modulation (4 to 9 Hz band)
5. Dynamic similarity from RMS envelope correlation

### 5.6 Tools and Libraries

| Tool | Role in DSP |
|------|-------------|
| Python | Main language |
| NumPy | FFT, arrays, math |
| SciPy | Welch PSD, Butterworth filter |
| librosa | YIN, STFT, chroma, DTW |
| Streamlit | User interface |
| streamlit-webrtc | Live microphone capture |
| Plotly | Tuner meter and spectrum plots |

---

## 6. Implementation (Code / Simulation)

The full project is implemented in Python. Source code, folder structure, and setup steps are available at:

**GitHub Repository:** `[INSERT YOUR GITHUB REPO URL HERE]`

Main code areas (for report mention only, not for pasting code):

| Module | DSP role |
|--------|----------|
| `utils/dsp_live.py` | Live autocorrelation, HPS, FFT spectrum, pitch smoothing |
| `utils/visuals.py` | Spectrum, tuner meter, range bar plots |
| `utils/live_audio.py` | WebRTC audio loop |
| `utils/voice_features.py` | Spectral tilt, HNR, nasal band ratio |
| `utils/register.py` | Register classification from features |
| `features/vocal_range.py` | Vocal range analysis |
| `utils/chroma_utils.py` | Chroma and key detection |
| `features/comparator.py` | Voice comparison with DTW |
| `app.py` | Streamlit application entry point |

**How to run:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

No separate hardware simulation was used. The system runs on real microphone input and uploaded WAV/MP3 files. Processing is software based simulation of DSP blocks on live audio.

---

## 7. Results

> Replace each placeholder below with your own screenshot or graph when preparing the final report.

### 7.1 Live Guitar Tuner

**Figure 1:** `[INSERT SCREENSHOT: tuner meter showing note name and cents offset]`

The tuner shows how many cents the detected string is flat or sharp. The horizontal meter gives quick visual feedback. HPS and autocorrelation both track the fundamental frequency from the live spectrum.

**Figure 2:** `[INSERT SCREENSHOT: FFT spectrum with fundamental marked]`

The spectrum plot shows magnitude vs frequency in Hz. The marked line shows which frequency was chosen as f0. This connects the math to what the user hears.

### 7.2 Vocal Range and Register

**Figure 3:** `[INSERT SCREENSHOT: vocal range bar and detected low/modal/high notes]`

YIN pitch tracking over time gives the singer's usable range. Register result uses spectral tilt and HNR together with pitch.

**Figure 4:** `[INSERT SCREENSHOT: register metrics table or evidence panel]`

### 7.3 Key Detection and Capo Match

**Figure 5:** `[INSERT SCREENSHOT: detected key and capo match bar chart]`

Chroma based key detection finds the best fit key. Capo scan shows which fret best matches the singer's key while keeping the same finger chord shapes.

### 7.4 Voice Comparison

**Figure 6:** `[INSERT SCREENSHOT: pitch alignment or comparison metrics]`

DTW alignment allows fair comparison even when timing differs. Output includes mean cents error, vibrato detection, and dynamic similarity percentage.

### 7.5 Demo Video

**Video link:** `[INSERT DEMO VIDEO URL HERE]`

Short screen recording showing live tuner, vocal analysis, and key detection in use.

### 7.6 Sample Result Summary (fill with your own numbers after testing)

| Test | Input | DSP method | Sample output |
|------|-------|------------|---------------|
| Guitar low E string | Live mic | HPS | ~82 Hz, within ±5 cents when tuned |
| Sustained vocal note | Upload | YIN | Stable f0 track, range in Hz and note names |
| Melody clip | Upload | Chroma + KS profiles | Key detected with confidence score |
| User vs reference vocal | Two uploads | DTW + YIN | Mean cents deviation reported |

---

## 8. Conclusion

This project shows that standard DSP techniques can be combined into a useful music analysis tool. FFT and spectrum plots make frequency content visible. Autocorrelation and HPS give workable live pitch tracking for guitar. YIN and chroma features support vocal and key analysis. Simple acoustic features like spectral tilt and HNR add more detail than pitch alone for register study.

The live web interface helps connect theory to practice. A musician can see the spectrum and cents meter while playing, which makes the processing easier to understand than a black box app.

**Limitations:**

- Register detection uses rule based logic, not a trained machine learning model
- Key detection can confuse relative major and minor keys on short clips
- Live performance depends on microphone quality and background noise
- Formant and glottal source analysis (closed quotient, F1 to F4) are not yet implemented

**Future work:** add formant estimation using linear prediction, improve noise rejection, and collect labeled singing data for better register classification.

---

## 9. References

Use these in IEEE or APA format as required by your college. The LLM should format them properly.

1. de Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator for speech and music. *JASA*.
2. Krumhansl, C. L., & Kessler, E. J. (1982). Tracing the dynamic changes in perceived tonal organization in a spatial representation of musical keys. *Psychological Review*.
3. Boersma, P. (1993). Accurate short term analysis of the vocal signal. *Proceedings of the Institute of Phonetic Sciences*.
4. Rabiner, L. R., & Schafer, R. W. *Digital Processing of Speech Signals*. Prentice Hall.
5. librosa documentation — YIN, chroma, DTW: https://librosa.org/doc/latest/
6. Julius O. Smith, STFT and DFT resources: https://ccrma.stanford.edu/~jos/mdft/
7. Tom Schimansky, GuitarTuner (HPS reference): https://github.com/TomSchimansky/GuitarTuner
8. NumPy FFT documentation: https://numpy.org/doc/stable/reference/routines.fft.html
9. SciPy signal processing: https://docs.scipy.org/doc/scipy/reference/signal.html
10. Streamlit and streamlit-webrtc documentation for live audio capture

---

## Appendix: One Paragraph Project Summary (for LLM opening)

RayMozic is a Python and Streamlit web app for guitar and voice analysis. It uses FFT based spectrum display, autocorrelation and Harmonic Product Spectrum for live tuning, YIN for vocal pitch, chroma vectors for key detection, and DTW for voice comparison. The app is aimed at making DSP visible and useful for musicians rather than hiding the math behind a simple note display.

---

## Appendix: Keywords for Report

Digital Signal Processing, FFT, STFT, fundamental frequency, pitch detection, autocorrelation, Harmonic Product Spectrum, YIN, chroma, key detection, vocal range, spectral tilt, HNR, DTW, real time audio, WebRTC, music information retrieval
