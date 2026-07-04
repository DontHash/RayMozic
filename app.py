import json
from typing import Optional

import numpy as np
import streamlit as st

from utils.audio_io import (
    AUDIO_UPLOAD_TYPES,
    AudioLoadError,
    AudioValidationError,
    load_uploaded_audio,
)
from utils.results_store import (
    COMPETITOR_SERVICES,
    PITCH_BENCHMARKS,
    list_results,
    pitch_accuracy_tier,
    save_result,
)
from utils.dsp_live import (
    DEFAULT_A4,
    DEFAULT_FMIN,
    compute_spectrum,
    dominant_frequency,
    frequency_to_note,
)
from utils.visuals import (
    capo_match_figure,
    fretboard_figure,
    needle_figure,
    range_progress_figure,
    spectrum_figure,
    string_status_figure,
    tuner_meter_figure,
)
from utils.tuner import (
    A4_MAX,
    A4_MIN,
    DEFAULT_TUNING,
    TUNINGS,
    get_targets,
    nearest_string,
    note_to_frequency,
    tuning_direction,
)
from utils.live_audio import WEBRTC_AVAILABLE, create_audio_streamer, live_pitch_loop
from utils.chord_voicing import recommend_voicings
from utils.play_along import build_play_plan
from utils.progression import COMMON_PATTERNS
from utils.progression_capo_map import build_progression_capo_map

from features.vocal_range import analyze_vocal_range
from features.comparator import compare_voices, VoiceComparisonError

SR = 22050

st.set_page_config(page_title="RayMozic", layout="wide", initial_sidebar_state="collapsed")
st.title("RayMozic")
st.caption("Live tuner · vocal analysis · capo & chord tools")

with st.sidebar:
    st.header("Reference Audio")
    st.markdown("Upload the original artist's track for the Voice Comparator.")
    ref_file = st.file_uploader("Upload Reference Track", type=AUDIO_UPLOAD_TYPES)
    if not WEBRTC_AVAILABLE:
        st.warning("Live microphone unavailable: `streamlit-webrtc` failed to import.")


def _input_meta(mode: str, audio: np.ndarray, sr: int, filename: Optional[str] = None) -> dict:
    return {
        "mode": mode,
        "filename": filename,
        "duration_sec": round(len(audio) / sr, 2),
        "sample_rate": sr,
    }


def analysis_spectrum(audio: np.ndarray, sr: int, seconds: float = 1.0, fmax: float = 2000.0):
    """FFT spectrum of the highest-energy ~1s window of an uploaded clip."""
    n = int(seconds * sr)
    if len(audio) > n:
        hop = max(n // 2, 1)
        best_start, best_energy = 0, -1.0
        for start in range(0, len(audio) - n + 1, hop):
            energy = float(np.sum(audio[start:start + n] ** 2))
            if energy > best_energy:
                best_energy, best_start = energy, start
        seg = audio[best_start:best_start + n]
    else:
        seg = audio
    return compute_spectrum(seg, sr, fmax=fmax)


tab_tuner, tab_vocal, tab_play, tab_capo_map, tab_compare, tab_results = st.tabs(
    [
        "Guitar Tuner (Live)",
        "Vocal Range",
        "Key, Capo & Chords",
        "Progression & Capo Map",
        "Voice Comparator",
        "Results & Benchmarks",
    ]
)


# ----------------------------------------------------------------------------
# TAB 1 — Guitar Tuner (live mic, needle + spectrum)
# ----------------------------------------------------------------------------
with tab_tuner:
    st.header("Guitar tuner")

    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    tuning_choices = list(TUNINGS.keys()) + ["Custom"]
    tuning_name = ctrl1.selectbox("Tuning", tuning_choices, index=tuning_choices.index(DEFAULT_TUNING))
    a4_ref = ctrl2.slider("A4 reference (Hz)", int(A4_MIN), int(A4_MAX), int(DEFAULT_A4), 1)
    method_label = ctrl3.selectbox("Detection", ["HPS (harmonic)", "Autocorrelation"])
    method = "hps" if method_label.startswith("HPS") else "autocorr"

    custom_notes = None
    if tuning_name == "Custom":
        default_custom = ["E2", "A2", "D3", "G3", "B3", "E4"]
        cc = st.columns(6)
        custom_notes = [
            cc[i].text_input(f"String {6 - i}", default_custom[i], key=f"custom_str_{i}")
            for i in range(6)
        ]

    try:
        targets = get_targets(tuning_name, a4=float(a4_ref), custom_notes=custom_notes)
    except ValueError as exc:
        st.error(f"Invalid custom tuning: {exc}")
        targets = get_targets(DEFAULT_TUNING, a4=float(a4_ref))

    if not WEBRTC_AVAILABLE:
        st.error("Live tuner requires `streamlit-webrtc`. Install it and restart.")
    else:
        st.caption("Click **START**, allow microphone access, then play one string at a time.")
        ctx = create_audio_streamer("tuner-stream")

        strings_ph = st.empty()
        meter_ph = st.empty()
        status_ph = st.empty()
        spec_ph = st.empty()
        info_ph = st.empty()

        strings_ph.plotly_chart(
            string_status_figure(targets, None, {}),
            width="stretch",
            key="tuner_strings_idle",
        )
        meter_ph.plotly_chart(
            tuner_meter_figure(None, "--"),
            width="stretch",
            key="tuner_meter_idle",
        )

        if ctx and ctx.state.playing:
            # Stability state (GuitarTuner mechanism): only accept a note after a
            # few consistent frames, and confirm "in tune" after several in a row.
            tstate = {"i": 0, "note": None, "note_hits": 0, "in_tune_hits": 0}
            HITS_TO_LOCK = 3
            HITS_TO_CONFIRM = 6

            def render_tuner(reading, freqs, mags, buffer, sr):
                tstate["i"] += 1
                i = tstate["i"]
                match = (
                    nearest_string(reading.frequency, tuning_name, a4=float(a4_ref), custom_notes=custom_notes)
                    if reading.voiced else None
                )
                if match:
                    if match.string.label == tstate["note"]:
                        tstate["note_hits"] += 1
                    else:
                        tstate["note"] = match.string.label
                        tstate["note_hits"] = 1

                    cents = match.cents_off
                    if match.in_tune:
                        tstate["in_tune_hits"] += 1
                    else:
                        tstate["in_tune_hits"] = 0

                    note_label = match.string.label
                    active = match.string.string_number
                    if tstate["in_tune_hits"] >= HITS_TO_CONFIRM:
                        status_ph.caption(f"{match.string.label} — in tune")
                    elif tstate["note_hits"] >= HITS_TO_LOCK:
                        status_ph.caption(f"{match.string.label} — {tuning_direction(cents)}")
                    else:
                        status_ph.empty()

                    strings_ph.plotly_chart(
                        string_status_figure(targets, active, {active: cents}),
                        width="stretch",
                        key=f"tuner_strings_{i}",
                    )
                    meter_ph.plotly_chart(
                        tuner_meter_figure(
                            cents, note_label,
                            target_hz=match.string.frequency,
                            detected_hz=reading.frequency,
                        ),
                        width="stretch",
                        key=f"tuner_meter_{i}",
                    )
                else:
                    tstate["in_tune_hits"] = 0
                    status_ph.empty()
                    strings_ph.plotly_chart(
                        string_status_figure(targets, None, {}),
                        width="stretch",
                        key=f"tuner_strings_{i}",
                    )
                    meter_ph.plotly_chart(
                        tuner_meter_figure(None, "--"),
                        width="stretch",
                        key=f"tuner_meter_{i}",
                    )

                spec_ph.plotly_chart(
                    spectrum_figure(
                        freqs, mags,
                        highlight_hz=reading.frequency if reading.voiced else None,
                        fmax=1500.0,
                    ),
                    width="stretch",
                    key=f"tuner_spec_{i}",
                )

            live_pitch_loop(
                ctx,
                render_tuner,
                buffer_seconds=0.35,
                fmin=60.0,
                fmax=1000.0,
                spectrum_fmax=1500.0,
                method=method,
                a4=float(a4_ref),
                smooth=True,
            )
        else:
            info_ph.caption("Press START to begin.")


# ----------------------------------------------------------------------------
# TAB 2 — Vocal Range (live mic OR upload)
# ----------------------------------------------------------------------------
with tab_vocal:
    st.header("Vocal range")

    mode = st.radio("Input mode", ["Live Microphone", "Upload File"], key="vocal_mode", horizontal=True)

    if mode == "Live Microphone":
        if not WEBRTC_AVAILABLE:
            st.error("Live mode requires `streamlit-webrtc`.")
        else:
            st.caption("Sing sustained notes and glides. Your range accumulates while the mic is on.")
            c1, c2 = st.columns([1, 1])
            if c2.button("Reset live range", key="reset_live_range"):
                st.session_state["live_vocal_freqs"] = []

            st.session_state.setdefault("live_vocal_freqs", [])
            ctx = create_audio_streamer("vocal-stream")

            needle_ph = st.empty()
            range_ph = st.empty()
            spec_ph = st.empty()
            stats_ph = st.empty()

            needle_ph.plotly_chart(
                needle_figure(None, "--", "Waiting for audio..."),
                width="stretch",
                key="vocal_needle_idle",
            )

            if ctx and ctx.state.playing:
                frame_counter = {"i": 0}

                def render_vocal(reading, freqs, mags, buffer, sr):
                    frame_counter["i"] += 1
                    i = frame_counter["i"]
                    if reading.voiced and reading.frequency > 0:
                        st.session_state["live_vocal_freqs"].append(reading.frequency)
                        # Keep memory bounded (~last 20k frames).
                        if len(st.session_state["live_vocal_freqs"]) > 20000:
                            st.session_state["live_vocal_freqs"] = st.session_state["live_vocal_freqs"][-20000:]

                    detail = f"{reading.frequency:.1f} Hz" if reading.voiced else "Listening..."
                    needle_ph.plotly_chart(
                        needle_figure(
                            reading.cents if reading.voiced else None,
                            reading.note_label or "--",
                            detail,
                        ),
                        width="stretch",
                        key=f"vocal_needle_{i}",
                    )

                    freqs_hist = np.array(st.session_state["live_vocal_freqs"])
                    if freqs_hist.size > 5:
                        low = float(np.percentile(freqs_hist, 5))
                        high = float(np.percentile(freqs_hist, 95))
                        range_ph.plotly_chart(
                            range_progress_figure(low, high, reading.frequency if reading.voiced else 0),
                            width="stretch",
                            key=f"vocal_range_{i}",
                        )

                    spec_ph.plotly_chart(
                        spectrum_figure(
                            freqs, mags,
                            highlight_hz=reading.frequency if reading.voiced else None,
                            fmax=2000.0,
                        ),
                        width="stretch",
                        key=f"vocal_spec_{i}",
                    )

                live_pitch_loop(
                    ctx,
                    render_vocal,
                    buffer_seconds=0.3,
                    fmin=70.0,
                    fmax=1200.0,
                    spectrum_fmax=2000.0,
                )

            # Summary after stopping (session_state persists across reruns).
            freqs_hist = np.array(st.session_state.get("live_vocal_freqs", []))
            if freqs_hist.size > 10:
                low = float(np.percentile(freqs_hist, 5))
                high = float(np.percentile(freqs_hist, 95))
                modal = float(np.median(freqs_hist))
                low_n = frequency_to_note(low)
                high_n = frequency_to_note(high)
                modal_n = frequency_to_note(modal)
                stats_ph.container()
                with stats_ph.container():
                    st.subheader("Live Range Summary")
                    a, b, c = st.columns(3)
                    a.metric("Lowest", low_n.note_label, f"{low:.1f} Hz")
                    b.metric("Modal", modal_n.note_label, f"{modal:.1f} Hz")
                    c.metric("Highest", high_n.note_label, f"{high:.1f} Hz")
                    if st.button("Save live range result", key="save_live_range"):
                        payload = {
                            "low_note": low_n.note_label,
                            "high_note": high_n.note_label,
                            "modal_note": modal_n.note_label,
                            "low_hz": low,
                            "high_hz": high,
                            "modal_hz": modal,
                            "frames": int(freqs_hist.size),
                        }
                        saved = save_result(
                            "vocal_range",
                            payload,
                            input_meta={"mode": "live_microphone", "sample_rate": "browser"},
                        )
                        st.success(f"Saved — `{saved['id'][:8]}`")

    else:  # Upload File
        vocal_file = st.file_uploader("Upload Vocal Track", type=AUDIO_UPLOAD_TYPES, key="vocal_upload_1")
        vocal_audio = None
        if vocal_file is not None:
            st.audio(vocal_file)
            if st.button("Run Vocal Analysis", type="primary", key="run_1_upload"):
                with st.spinner("Analyzing vocal range..."):
                    try:
                        vocal_audio = load_uploaded_audio(vocal_file, sr=SR)
                    except AudioLoadError as exc:
                        st.error(str(exc))

        if vocal_audio is not None:
            results = analyze_vocal_range(vocal_audio, sr=SR)
            if "error" in results:
                st.error(results["error"])
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Lowest Note", results["low_note"], f"{results['low_hz']:.1f} Hz")
                col2.metric("Modal (Natural) Note", results["modal_note"], f"{results['modal_hz']:.1f} Hz")
                col3.metric("Highest Note", results["high_note"], f"{results['high_hz']:.1f} Hz")

                reg_label = results["register"]
                if results.get("is_belt"):
                    reg_label += " (belt characteristics)"
                if results.get("is_nasal"):
                    reg_label += " · nasal resonance"
                st.subheader("Register")
                st.write(reg_label)
                st.progress(
                    min(max(results.get("register_confidence", 0.0), 0.0), 1.0),
                    text=f"{results.get('register_confidence', 0.0)*100:.0f}% confidence",
                )

                fcol1, fcol2, fcol3, fcol4 = st.columns(4)
                fcol1.metric("Spectral tilt", f"{results['spectral_tilt_db_per_khz']:.1f} dB/kHz")
                fcol2.metric("HNR", f"{results['hnr_db']:.1f} dB")
                fcol3.metric("HF energy", f"{results['hf_energy_ratio']*100:.0f}%")
                fcol4.metric("Brightness", f"{results['spectral_centroid_hz']:.0f} Hz")

                reasons = results.get("register_reasons", [])
                if reasons:
                    with st.expander("Register evidence"):
                        for r in reasons:
                            st.write(f"· {r}")

                freqs, mags = analysis_spectrum(vocal_audio, SR, fmax=2000.0)
                st.plotly_chart(
                    spectrum_figure(freqs, mags, highlight_hz=results["modal_hz"], fmax=2000.0),
                    width="stretch",
                    key="vocal_upload_spec",
                )

                saved = save_result(
                    "vocal_range",
                    results,
                    input_meta=_input_meta("upload_file", vocal_audio, SR, vocal_file.name),
                )
                st.caption(f"Result saved for benchmarking — `{saved['id'][:8]}`")
                st.session_state["last_vocal_audio"] = vocal_audio


# ----------------------------------------------------------------------------
# TAB 3 — Key, Capo & Chords (scale match + auto progression + capo guide)
# ----------------------------------------------------------------------------
with tab_play:
    st.header("Key, capo & chords")

    src = st.radio(
        "Audio source",
        ["Upload File", "Use last vocal analysis"],
        key="play_src",
        horizontal=True,
    )
    play_audio = None

    if src == "Upload File":
        play_file = st.file_uploader(
            "Upload vocal or melody (sing along with your guitar)",
            type=AUDIO_UPLOAD_TYPES,
            key="play_upload",
        )
        if play_file is not None:
            st.audio(play_file)
            if st.button("Analyze & build chord plan", type="primary", key="play_analyze_upload"):
                try:
                    play_audio = load_uploaded_audio(play_file, sr=SR)
                except AudioLoadError as exc:
                    st.error(str(exc))
    else:
        if "last_vocal_audio" in st.session_state:
            if st.button("Analyze last vocal recording", type="primary", key="play_analyze_session"):
                play_audio = st.session_state["last_vocal_audio"]
        else:
            st.caption("Run vocal analysis first, or upload here.")

    finger_prog = st.text_input(
        "Your chord shapes (what you finger with no capo)",
        "Am Em Dm F",
        key="play_finger_chords",
        help="These shapes stay the same. We try capo 0, 1, 2… and tell you which "
             "fret makes them fit your voice. Leave default or enter your own.",
    )

    with st.expander("More options", expanded=False):
        preset = st.selectbox(
            "Auto pattern (only if chord box is empty)",
            ["Auto (based on key)"] + list(COMMON_PATTERNS.keys()),
            key="play_preset",
        )
        use_sevenths = st.checkbox("Use 7th chords (auto mode)", key="play_sevenths")
        pattern_input = None if preset == "Auto (based on key)" else COMMON_PATTERNS[preset]

    if play_audio is not None:
        try:
            plan = build_play_plan(
                play_audio,
                sr=SR,
                pattern=pattern_input or None,
                sevenths=use_sevenths,
                finger_progression=finger_prog or None,
            )
        except AudioValidationError as exc:
            st.error(str(exc))
        else:
            st.session_state["last_play_plan"] = plan
            finger = plan["finger_progression"]
            best = plan["best_capo"]

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Key", f"{plan['vocal_key']} {plan['vocal_mode']}")
            c2.metric("Confidence", f"{plan['confidence']:.0%}")
            c3.metric("Your shapes", " ".join(finger))
            if best:
                c4.metric("Best capo", f"fret {best['capo_fret']}", f"{best['match_score']:.0%} match")

            if best:
                st.write(
                    f"Play **{' '.join(best['finger_chords'])}** at capo **{best['capo_fret']}** "
                    f"→ sounds **{' '.join(best['sounding_chords'])}** ({best['inferred_key']} {best['inferred_mode']})"
                )
            elif plan.get("capo_zero") and not plan.get("capo_zero_ok"):
                z = plan["capo_zero"]
                st.write(f"Open strings score {z['match_score']:.0%} against your voice.")

            root_markers = [
                note_to_frequency(f"{plan['vocal_key']}{o}")
                for o in range(2, 6)
                if note_to_frequency(f"{plan['vocal_key']}{o}") <= 2000.0
            ]
            freqs, mags = analysis_spectrum(play_audio, SR, fmax=2000.0)
            dom = dominant_frequency(freqs, mags)
            st.plotly_chart(
                spectrum_figure(
                    freqs, mags, highlight_hz=dom, fmax=2000.0,
                    extra_markers_hz=root_markers,
                ),
                width="stretch",
                key="play_spec",
            )

            if best:
                st.plotly_chart(
                    capo_match_figure(plan["capo_by_fret"], highlight_fret=best["capo_fret"]),
                    width="stretch",
                    key="play_capo_chart",
                )

            st.dataframe(
                [
                    {
                        "Capo": row["capo_fret"],
                        "Finger": " ".join(row["finger_chords"]),
                        "Sounds": " ".join(row["sounding_chords"]),
                        "Key": f"{row['inferred_key']} {row['inferred_mode']}",
                        "Match": f"{row['match_score']:.0%}",
                    }
                    for row in plan["capo_by_fret"]
                ],
                width="stretch",
                hide_index=True,
            )

            with st.expander("Other progressions in your key"):
                for alt in plan.get("alternatives", [])[:3]:
                    meta = alt["progression_meta"]
                    chords = " → ".join(c["name"] for c in meta["progression"])
                    st.write(f"{alt['pattern_label']}: {chords}")

            with st.expander("Fretboard diagrams"):
                fing_cols = st.columns(min(len(finger), 4))
                for col, chord in zip(fing_cols, finger[:4]):
                    with col:
                        try:
                            voicings = recommend_voicings(chord, top_n=1)
                            if voicings:
                                v = voicings[0]
                                st.plotly_chart(
                                    fretboard_figure(v.frets, v.open_midis, title=chord),
                                    width="stretch",
                                    key=f"play_v_{chord}",
                                )
                                st.caption(f"{v.diagram_str} · {v.fingers}f")
                        except ValueError:
                            st.write(chord)

            saved = save_result(
                "scale_matcher",
                {
                    "vocal_key": plan["vocal_key"],
                    "vocal_mode": plan["vocal_mode"],
                    "confidence": plan["confidence"],
                    "finger_progression": plan["finger_progression"],
                    "best_capo_fret": best["capo_fret"] if best else None,
                    "best_match_score": best["match_score"] if best else None,
                    "best_sounding_chords": best["sounding_chords"] if best else [],
                    "capo_options": [
                        {k: v for k, v in r.items() if k != "explanation"}
                        for r in plan["capo_by_fret"]
                    ],
                    "dominant_hz": dom,
                },
                input_meta=_input_meta("play_along", play_audio, SR),
            )
            st.caption(f"Saved for benchmarking — `{saved['id'][:8]}`")


# ----------------------------------------------------------------------------
# TAB 4 — Progression & Capo Map (sounding fixed → finger shapes per capo/scale)
# ----------------------------------------------------------------------------
with tab_capo_map:
    st.header("Progression & capo map")

    map_prog = st.text_input(
        "Sounding chord progression",
        "G Em C D",
        key="capo_map_prog",
        help="Space-separated chords as heard (e.g. the key you are playing in).",
    )

    mc1, mc2 = st.columns(2)
    override_key = mc1.text_input("Override key (optional)", "", key="capo_map_key")
    override_mode = mc2.selectbox(
        "Override mode (optional)",
        ["Auto", "major", "minor"],
        key="capo_map_mode",
    )

    if st.button("Map progression", type="primary", key="capo_map_run"):
        try:
            capo_map = build_progression_capo_map(
                map_prog,
                target_key=override_key.strip() or None,
                target_mode=None if override_mode == "Auto" else override_mode,
            )
            st.session_state["capo_map_result"] = capo_map
        except ValueError as exc:
            st.error(str(exc))

    if "capo_map_result" in st.session_state:
        m = st.session_state["capo_map_result"]

        st.divider()
        k1, k2, k3 = st.columns(3)
        k1.metric("Key", f"{m['inferred_key']} {m['inferred_mode']}")
        k2.metric("Diatonic fit", f"{m['diatonic_fit']:.0%}")
        k3.metric("Chords", " ".join(m["sounding_chords"]))

        if m["compatible_keys"]:
            st.dataframe(
                [
                    {
                        "Key": f"{c['key']} {c['mode']}",
                        "Fit": f"{c['fit']:.0%}",
                        "Roman": c["roman"],
                    }
                    for c in m["compatible_keys"][:12]
                ],
                width="stretch",
                hide_index=True,
            )

        st.caption("Capo fret → finger shapes (sound stays the same)")
        st.dataframe(
            [
                {
                    "Capo": row["capo_fret"],
                    "Finger": " ".join(row["finger_chords"]),
                    "Sounds": " ".join(row["sounding_chords"]),
                }
                for row in m["capo_rows"]
            ],
            width="stretch",
            hide_index=True,
        )

        st.caption(f"Open families in {m['target_key']} {m['target_mode']}")
        st.dataframe(
            [
                {
                    "Family": row["chord_shape"],
                    "Capo": row["capo_fret"],
                    "Finger": " ".join(row["finger_chords"]),
                }
                for row in m["shape_family_rows"]
            ],
            width="stretch",
            hide_index=True,
        )


# ----------------------------------------------------------------------------
# TAB 5 — Voice Comparator (upload only)
# ----------------------------------------------------------------------------
with tab_compare:
    st.header("Voice comparison")

    if ref_file is None:
        st.warning("Upload a Reference Track in the sidebar first.")
    else:
        user_file_3 = st.file_uploader("Upload Your Vocal Track", type=AUDIO_UPLOAD_TYPES, key="vocal_upload_3")
        user_audio_3 = None
        ref_audio = None

        if st.button("Run Comparison", type="primary", key="run_3_upload") and user_file_3 is not None:
            with st.spinner("Aligning and comparing..."):
                try:
                    user_audio_3 = load_uploaded_audio(user_file_3, sr=SR)
                    ref_audio = load_uploaded_audio(ref_file, sr=SR)
                except AudioLoadError as exc:
                    st.error(str(exc))
                    user_audio_3 = None
                    ref_audio = None

        if user_audio_3 is not None and ref_audio is not None:
            try:
                comp_results = compare_voices(user_audio_3, ref_audio, sr=SR)
            except (AudioValidationError, VoiceComparisonError) as exc:
                st.error(str(exc))
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Pitch Error", f"{comp_results['mean_cents_deviation']:.0f} cents")
                col2.metric(
                    "Vibrato Detected",
                    "Yes" if comp_results["has_vibrato"] else "No",
                    f"{comp_results['vibrato_rate']:.1f} Hz" if comp_results["has_vibrato"] else None,
                )
                col3.metric("Dynamic Similarity", f"{comp_results['dynamic_correlation'] * 100:.1f}%")

                st.subheader("Pitch Alignment (DTW)")
                import plotly.graph_objects as go

                align_fig = go.Figure()
                align_fig.add_trace(go.Scatter(y=comp_results["ref_f0_plot"], name="Reference", line={"color": "#4aa3ff"}))
                align_fig.add_trace(go.Scatter(y=comp_results["user_f0_plot"], name="You", line={"color": "#e67e22"}))
                align_fig.update_layout(
                    height=320, xaxis_title="Aligned frames", yaxis_title="Frequency (Hz)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e0e0e0"},
                )
                st.plotly_chart(align_fig, width="stretch", key="compare_align")

                tier = pitch_accuracy_tier(comp_results["mean_cents_deviation"])
                saved = save_result(
                    "voice_comparison",
                    {**comp_results, "accuracy_tier": tier},
                    input_meta={
                        **_input_meta("upload_file", user_audio_3, SR, user_file_3.name),
                        "reference_filename": getattr(ref_file, "name", None),
                    },
                    notes=f"Pitch tier: {tier}",
                )
                st.caption(
                    f"Result saved — `{saved['id'][:8]}`. Tier: **{tier.replace('_', ' ')}** "
                    f"(pro ≤{PITCH_BENCHMARKS['professional_cents']}¢, live ≤{PITCH_BENCHMARKS['acceptable_live_cents']}¢)"
                )


# ----------------------------------------------------------------------------
# TAB 5 — Results & Benchmarks
# ----------------------------------------------------------------------------
with tab_results:
    st.header("Results")

    feature_filter = st.selectbox("Filter by feature", ["All", "vocal_range", "scale_matcher", "voice_comparison"])
    selected_feature = None if feature_filter == "All" else feature_filter
    records = list_results(feature=selected_feature, limit=30)

    if not records:
        st.info("No saved results yet. Run an analysis first.")
    else:
        summary_rows = []
        for rec in records:
            m = rec.get("metrics", {})
            row = {
                "time": rec.get("timestamp", "")[:19].replace("T", " "),
                "feature": rec.get("feature"),
                "id": rec.get("id", "")[:8],
            }
            if rec["feature"] == "vocal_range":
                row.update(low=m.get("low_note"), high=m.get("high_note"), modal=m.get("modal_note"), register=m.get("register"))
            elif rec["feature"] == "scale_matcher":
                row.update(key=f"{m.get('vocal_key')} {m.get('vocal_mode')}", capo=m.get("top_capo_fret"), shape=m.get("top_chord_shape"))
            elif rec["feature"] == "voice_comparison":
                row.update(cents=round(m.get("mean_cents_deviation", 0), 1), tier=m.get("accuracy_tier"), vibrato=m.get("has_vibrato"))
            summary_rows.append(row)
        st.dataframe(summary_rows, width="stretch", hide_index=True)

        pick = st.selectbox(
            "Inspect saved result",
            options=range(len(records)),
            format_func=lambda i: f"{records[i]['timestamp'][:19]} — {records[i]['feature']} ({records[i]['id'][:8]})",
        )
        chosen = records[pick]
        st.download_button(
            "Download JSON",
            data=json.dumps(chosen, indent=2),
            file_name=f"raymozic_{chosen['feature']}_{chosen['id'][:8]}.json",
            mime="application/json",
        )
        st.json(chosen)

    st.subheader("Competitor services to compare against")
    for feature, services in COMPETITOR_SERVICES.items():
        with st.expander(feature.replace("_", " ").title()):
            for svc in services:
                st.markdown(f"**[{svc['name']}]({svc['url']})**")
                st.write(svc["overlap"])
                st.caption(f"Compare fields: {', '.join(svc['compare_fields'])}")
