import streamlit as st
import numpy as np
import time
from datetime import datetime
from feature_extraction import extract_features
from model import StressMoodModel
from utils import record_audio

CALIBRATION_SAMPLES = 5
RECORD_DURATION = 3
INTERVAL = 10

def calibrate():
    calibration_samples = []
    st.info(f"Calibration phase: Please provide {CALIBRATION_SAMPLES} normal voice samples.")
    for i in range(CALIBRATION_SAMPLES):
        st.write(f"Recording sample {i+1}...")
        audio, fs = record_audio(duration=RECORD_DURATION)
        feat = extract_features(audio, fs)
        calibration_samples.append(feat)
        st.success(f"Sample {i+1} recorded.")
    return np.array(calibration_samples)

def main():
    st.title("Voice-based Stress & Mood Detector")

    if "calibrated" not in st.session_state:
        st.session_state.calibrated = False
        st.session_state.detections = []
        st.session_state.running = False

    if not st.session_state.calibrated:
        if st.button("Start Calibration"):
            calibration_array = calibrate()
            st.session_state.model = StressMoodModel(clustering_method='isolation_forest')
            st.session_state.model.fit(calibration_array)
            st.session_state.calibrated = True
            st.success("Calibration completed! You can now start detection.")
    else:
        if not st.session_state.running:
            if st.button("Start Detection"):
                st.session_state.running = True
        else:
            if st.button("Stop Detection"):
                st.session_state.running = False

        # Detection display area
        detection_display = st.empty()

        while st.session_state.running:
            audio, fs = record_audio(duration=RECORD_DURATION)
            feat = extract_features(audio, fs)
            prediction = st.session_state.model.predict(feat)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.detections.append((timestamp, feat, prediction))

            msg = f"[{timestamp}] "
            if prediction == -1:
                msg += "Anomalous/stressed voice detected!"
                detection_display.error(msg)
            else:
                msg += "Normal voice/mood detected."
                detection_display.success(msg)

            # Show performance summary inside the loop
            total = len(st.session_state.detections)
            stressed = sum(1 for _, _, p in st.session_state.detections if p == -1)
            normal = total - stressed

            pitches = [f[0] for _, f, _ in st.session_state.detections]
            volumes = [f[1] for _, f, _ in st.session_state.detections]
            speeds = [f[2] for _, f, _ in st.session_state.detections]

            perf_summary = (
                f"**Performance Summary:**\n\n"
                f"- Total samples: {total}\n"
                f"- Stressed samples: {stressed}\n"
                f"- Normal samples: {normal}\n"
                f"- Average pitch: {np.mean(pitches):.2f}\n"
                f"- Average volume: {np.mean(volumes):.4f}\n"
                f"- Average speed: {np.mean(speeds):.4f}"
            )
            st.markdown(perf_summary)

            time.sleep(INTERVAL - RECORD_DURATION)

if __name__ == "__main__":
    main()
