import json
from typing import Optional

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Import utils
from utils.audio_io import (
    AUDIO_UPLOAD_TYPES,
    AudioLoadError,
    AudioValidationError,
    load_uploaded_audio,
    record_audio,
)
from utils.results_store import (
    COMPETITOR_SERVICES,
    PITCH_BENCHMARKS,
    list_results,
    pitch_accuracy_tier,
    save_result,
)

# Import features
from features.vocal_range import analyze_vocal_range
from features.scale_matcher import match_guitar_scale, transpose_progression
from features.comparator import compare_voices, VoiceComparisonError

# Set up page config
st.set_page_config(page_title="DSP Music Analysis", layout="wide")
st.title("DSP Music Analysis & Guitar Resonance App")
st.markdown("Analyze vocal ranges, match guitar keys, and compare vocal modulation.")

# Sidebar for Reference Audio Upload
with st.sidebar:
    st.header("Reference Audio (Feature 3)")
    st.markdown("Upload the original artist's track for comparison.")
    ref_file = st.file_uploader("Upload Reference Track", type=AUDIO_UPLOAD_TYPES)

def _input_meta(mode: str, audio: np.ndarray, sr: int, filename: Optional[str] = None) -> dict:
    return {
        "mode": mode,
        "filename": filename,
        "duration_sec": round(len(audio) / sr, 2),
        "sample_rate": sr,
    }


# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Vocal Range Analysis", "Guitar Scale Matcher", "Voice Comparator", "Results & Benchmarks"]
)

with tab1:
    st.header("Feature 1: Vocal Range & Register")
    st.markdown("Upload or record your isolated vocals to find your range and register.")
    
    input_mode = st.radio("Input Mode", ["Upload File", "Record Audio"], key="tab1_mode")
    vocal_audio = None
    sr = 22050
    
    tab1_input_mode = input_mode
    tab1_filename = None

    if input_mode == "Upload File":
        vocal_file = st.file_uploader(
            "Upload Vocal Track", type=AUDIO_UPLOAD_TYPES, key="vocal_upload_1"
        )
        if vocal_file is not None:
            st.audio(vocal_file)
            tab1_filename = vocal_file.name
            if st.button("Run Vocal Analysis", type="primary", key="run_1_upload"):
                with st.spinner("Analyzing vocal range..."):
                    try:
                        vocal_audio = load_uploaded_audio(vocal_file, sr=sr)
                    except AudioLoadError as exc:
                        st.error(str(exc))
    else:
        duration = st.slider("Record Duration (seconds)", 5, 15, 10, key="rec_dur_1")
        if st.button("Record & Analyze", type="primary", key="rec_btn_1"):
            with st.spinner(f"Recording for {duration} seconds..."):
                vocal_audio = record_audio(duration, sr=sr)
            st.success("Recording complete!")

    if vocal_audio is not None:
        results = analyze_vocal_range(vocal_audio, sr=sr)
        if "error" in results:
            st.error(results["error"])
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Lowest Note", results['low_note'], f"{results['low_hz']:.1f} Hz")
            col2.metric("Modal (Natural) Note", results['modal_note'], f"{results['modal_hz']:.1f} Hz")
            col3.metric("Highest Note", results['high_note'], f"{results['high_hz']:.1f} Hz")
            
            st.subheader("Vocal Characteristics")
            st.write(f"**Dominant Register:** {results['register']}")
            st.write(f"**Nasal Tone Detected:** {'Yes' if results['is_nasal'] else 'No'}")

            saved = save_result(
                "vocal_range",
                results,
                input_meta=_input_meta(
                    tab1_input_mode.lower().replace(" ", "_"),
                    vocal_audio,
                    sr,
                    tab1_filename,
                ),
            )
            st.caption(f"Result saved for benchmarking — `{saved['id'][:8]}`")

            # Save for Tab 2
            st.session_state['last_vocal_audio'] = vocal_audio


with tab2:
    st.header("Feature 2: Guitar Scale Matcher")
    st.markdown("Find the best capo position and transpose chords for your vocal key.")
    
    if 'last_vocal_audio' not in st.session_state:
        st.info("Please run the Vocal Range Analysis in Tab 1 first to load your voice profile.")
    else:
        chords_input = st.text_input("Original Chord Progression (e.g. G Em C D)", "G Em C D")
        
        if st.button("Match Guitar Scale", type="primary"):
            with st.spinner("Finding best guitar key..."):
                v_audio = st.session_state['last_vocal_audio']
                try:
                    match_results = match_guitar_scale(v_audio, sr=sr)
                except AudioValidationError as exc:
                    st.error(str(exc))
                else:
                    st.subheader(f"Vocal Key Detected: {match_results['vocal_key']} {match_results['vocal_mode']}")
                    st.write(f"Confidence Score: {match_results['confidence']:.2f}")

                    st.subheader("Top Capo Recommendations")
                    for i, rec in enumerate(match_results['recommendations']):
                        st.markdown(f"**{i+1}. Capo on Fret {rec['capo_fret']}** — Play in **{rec['chord_shape']}** shape")

                        if i == 0 and chords_input:
                            st.write("---")
                            st.write("**Transposed Progression for Top Recommendation:**")
                            transposed = transpose_progression(chords_input, rec['capo_fret'])
                            st.info(f"If original was {chords_input}, you might play shapes: {transposed}")

                    top = match_results["recommendations"][0]
                    transposed_top = (
                        transpose_progression(chords_input, top["capo_fret"]) if chords_input else ""
                    )
                    metrics_to_save = {
                        **match_results,
                        "original_progression": chords_input,
                        "top_capo_fret": top["capo_fret"],
                        "top_chord_shape": top["chord_shape"],
                        "transposed_progression": transposed_top,
                    }
                    saved = save_result(
                        "scale_matcher",
                        metrics_to_save,
                        input_meta=_input_meta(
                            "session_vocal",
                            st.session_state["last_vocal_audio"],
                            sr,
                        ),
                    )
                    st.caption(f"Result saved for benchmarking — `{saved['id'][:8]}`")


with tab3:
    st.header("Feature 3: Voice Modulation Comparator")
    st.markdown("Compare your vocals to the reference track.")
    
    if ref_file is None:
        st.warning("Please upload a Reference Track in the sidebar first.")
    else:
        input_mode_3 = st.radio("Input Mode", ["Upload File", "Record Audio"], key="tab3_mode")
        user_audio_3 = None
        ref_audio = None
        user_filename_3 = None
        
        if input_mode_3 == "Upload File":
            user_file_3 = st.file_uploader(
                "Upload Your Vocal Track", type=AUDIO_UPLOAD_TYPES, key="vocal_upload_3"
            )
            if st.button("Run Comparison", type="primary", key="run_3_upload"):
                if user_file_3 is not None:
                    user_filename_3 = user_file_3.name
                    with st.spinner("Aligning and comparing..."):
                        try:
                            user_audio_3 = load_uploaded_audio(user_file_3, sr=sr)
                            ref_audio = load_uploaded_audio(ref_file, sr=sr)
                        except AudioLoadError as exc:
                            st.error(str(exc))
                            user_audio_3 = None
                            ref_audio = None
        else:
            duration_3 = st.slider("Record Duration (seconds)", 5, 15, 10, key="rec_dur_3")
            if st.button("Record & Compare", type="primary", key="rec_btn_3"):
                with st.spinner(f"Recording for {duration_3} seconds..."):
                    user_audio_3 = record_audio(duration_3, sr=sr)
                with st.spinner("Aligning and comparing..."):
                    try:
                        ref_audio = load_uploaded_audio(ref_file, sr=sr)
                    except AudioLoadError as exc:
                        st.error(str(exc))
                        ref_audio = None
        
        if user_audio_3 is not None and ref_audio is not None:
            try:
                comp_results = compare_voices(user_audio_3, ref_audio, sr=sr)
            except (AudioValidationError, VoiceComparisonError) as exc:
                st.error(str(exc))
            else:
                st.subheader("Comparison Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Pitch Error", f"{comp_results['mean_cents_deviation']:.0f} cents")
                col2.metric("Vibrato Detected", "Yes" if comp_results['has_vibrato'] else "No", f"{comp_results['vibrato_rate']:.1f} Hz" if comp_results['has_vibrato'] else None)
                col3.metric("Dynamic Similarity", f"{comp_results['dynamic_correlation']*100:.1f}%")

                st.subheader("Pitch Alignment (DTW)")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(comp_results['ref_f0_plot'], label='Reference Pitch', color='blue', alpha=0.7)
                ax.plot(comp_results['user_f0_plot'], label='User Pitch', color='orange', alpha=0.7)
                ax.set_ylabel("Frequency (Hz)")
                ax.set_xlabel("Aligned Frames")
                ax.legend()
                st.pyplot(fig)

                tier = pitch_accuracy_tier(comp_results["mean_cents_deviation"])
                saved = save_result(
                    "voice_comparison",
                    {**comp_results, "accuracy_tier": tier},
                    input_meta={
                        **_input_meta(
                            input_mode_3.lower().replace(" ", "_"),
                            user_audio_3,
                            sr,
                            user_filename_3,
                        ),
                        "reference_filename": getattr(ref_file, "name", None),
                    },
                    notes=f"Pitch tier: {tier}",
                )
                st.caption(
                    f"Result saved — `{saved['id'][:8]}`. "
                    f"Tier: **{tier.replace('_', ' ')}** "
                    f"(pro ≤{PITCH_BENCHMARKS['professional_cents']}¢, "
                    f"live ≤{PITCH_BENCHMARKS['acceptable_live_cents']}¢)"
                )


with tab4:
    st.header("Saved Results & Competitor Benchmarks")
    st.markdown(
        "Every successful analysis is saved locally under `results/` as JSON. "
        "Run the same audio through a competitor site, then compare the fields listed below."
    )

    feature_filter = st.selectbox(
        "Filter by feature",
        ["All", "vocal_range", "scale_matcher", "voice_comparison"],
    )
    selected_feature = None if feature_filter == "All" else feature_filter
    records = list_results(feature=selected_feature, limit=30)

    if not records:
        st.info("No saved results yet. Run an analysis in Tabs 1–3 first.")
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
                row.update(
                    low=m.get("low_note"),
                    high=m.get("high_note"),
                    modal=m.get("modal_note"),
                    register=m.get("register"),
                )
            elif rec["feature"] == "scale_matcher":
                row.update(
                    key=f"{m.get('vocal_key')} {m.get('vocal_mode')}",
                    capo=m.get("top_capo_fret"),
                    shape=m.get("top_chord_shape"),
                )
            elif rec["feature"] == "voice_comparison":
                row.update(
                    cents=round(m.get("mean_cents_deviation", 0), 1),
                    tier=m.get("accuracy_tier"),
                    vibrato=m.get("has_vibrato"),
                )
            summary_rows.append(row)
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)

        pick = st.selectbox(
            "Inspect saved result",
            options=range(len(records)),
            format_func=lambda i: (
                f"{records[i]['timestamp'][:19]} — {records[i]['feature']} ({records[i]['id'][:8]})"
            ),
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

    st.subheader("RayMozic advantages vs. most competitors")
    st.markdown(
        """
        - **All-in-one:** range + register + nasal tone + guitar capo + reference-track comparison in one app
        - **Nasal detection:** few online range testers report formant-based nasal tone
        - **DTW alignment:** voice comparison aligns by chroma before measuring cents (tempo-robust)
        - **Vibrato + dynamics:** comparator reports vibrato rate and RMS envelope correlation, not just pitch
        - **Audio-derived key:** scale matcher detects key from your vocal recording (competitors often need manual key entry)
        """
    )
