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

# Tabs
tab1, tab2, tab3 = st.tabs(["Vocal Range Analysis", "Guitar Scale Matcher", "Voice Comparator"])

with tab1:
    st.header("Feature 1: Vocal Range & Register")
    st.markdown("Upload or record your isolated vocals to find your range and register.")
    
    input_mode = st.radio("Input Mode", ["Upload File", "Record Audio"], key="tab1_mode")
    vocal_audio = None
    sr = 22050
    
    if input_mode == "Upload File":
        vocal_file = st.file_uploader(
            "Upload Vocal Track", type=AUDIO_UPLOAD_TYPES, key="vocal_upload_1"
        )
        if vocal_file is not None:
            st.audio(vocal_file)
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


with tab3:
    st.header("Feature 3: Voice Modulation Comparator")
    st.markdown("Compare your vocals to the reference track.")
    
    if ref_file is None:
        st.warning("Please upload a Reference Track in the sidebar first.")
    else:
        input_mode_3 = st.radio("Input Mode", ["Upload File", "Record Audio"], key="tab3_mode")
        user_audio_3 = None
        ref_audio = None
        
        if input_mode_3 == "Upload File":
            user_file_3 = st.file_uploader(
                "Upload Your Vocal Track", type=AUDIO_UPLOAD_TYPES, key="vocal_upload_3"
            )
            if st.button("Run Comparison", type="primary", key="run_3_upload"):
                if user_file_3 is not None:
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
