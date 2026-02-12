import streamlit as st
import os
import sys
import time
from datetime import datetime

# --------------------------------------------------
# PATH FIX
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from inference import predict
from gradcam_inference import run_gradcam
from visual_reasoning_engine import analyze_visual_defects

# --------------------------------------------------
# OUTPUT PATH
# --------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "gradcam_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="VisionSpec Industrial QC",
    layout="wide",
    page_icon="🧠"
)

# --------------------------------------------------
# CUSTOM CSS (THIS MAKES IT PROFESSIONAL)
# --------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* Title */
.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    background: -webkit-linear-gradient(#38bdf8, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Glass Cards */
.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0px 0px 20px rgba(0,0,0,0.5);
}

/* Result badge */
.good {
    background: rgba(34,197,94,0.2);
    color: #22c55e;
    padding: 10px 18px;
    border-radius: 12px;
    font-weight: bold;
    font-size: 20px;
}

.bad {
    background: rgba(239,68,68,0.2);
    color: #ef4444;
    padding: 10px 18px;
    border-radius: 12px;
    font-weight: bold;
    font-size: 20px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: bold;
    background: linear-gradient(90deg,#0ea5e9,#22c55e);
    color: white;
    border: none;
}

.stButton > button:hover {
    transform: scale(1.03);
    transition: 0.2s;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<div class='main-title'>VisionSpec Industrial PCB Inspection</div>", unsafe_allow_html=True)
st.markdown("<center>AI-Powered Automated Quality Control System</center>", unsafe_allow_html=True)
st.write("")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("System Monitor")
st.sidebar.info("Model: VisionSpec-Net v1.0")
st.sidebar.success("Status: Online")
st.sidebar.write("GPU: Auto Detection")
st.sidebar.write("Inspection Mode: Production")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
st.markdown("### Upload PCB Board Image")
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    temp_path = os.path.join(PROJECT_ROOT, "temp_upload.jpg")

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Input Image")
        st.image(temp_path, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # RUN INSPECTION
    # --------------------------------------------------
    if st.button("Start Automated Inspection"):

        progress = st.progress(0)

        # Fake realistic scanning stages
        status = st.empty()

        status.info("Initializing vision pipeline...")
        time.sleep(1)
        progress.progress(20)

        status.info("Extracting features...")
        time.sleep(1)
        progress.progress(45)

        # Actual Model
        cnn_result = predict(temp_path)

        status.info("Running defect localization...")
        run_gradcam(temp_path)
        time.sleep(1)
        progress.progress(70)

        status.info("Performing visual reasoning analysis...")
        visual_report = analyze_visual_defects(temp_path)
        time.sleep(1)
        progress.progress(100)

        status.success("Inspection Completed")

        # --------------------------------------------------
        # RESULT PANEL
        # --------------------------------------------------
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Inspection Decision")

            if cnn_result["result"] == "GOOD":
                st.markdown("<div class='good'>GOOD BOARD</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='bad'>DEFECTIVE BOARD</div>", unsafe_allow_html=True)

            st.write("")
            st.write(f"**Model Confidence:** {cnn_result['confidence']}")
            st.write(f"**Risk Severity:** {cnn_result['severity']}")
            st.write(f"**Drift Monitor:** {cnn_result['data_drift_warning']}")

            st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------------------
        # VISUAL REPORT
        # --------------------------------------------------
        st.write("")
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Automated Visual Analysis Report")
        st.write(visual_report)
        st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------------------
        # HEATMAP
        # --------------------------------------------------
        gradcam_image = os.path.join(OUTPUT_DIR, os.path.basename(temp_path))

        if os.path.exists(gradcam_image):
            st.write("")
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Defect Localization Heatmap")
            st.image(gradcam_image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------------------
        # HISTORY LOG
        # --------------------------------------------------
        st.write("")
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Inspection Log")

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "result": cnn_result["result"],
            "confidence": cnn_result["confidence"]
        })

        for item in reversed(st.session_state.history[-5:]):
            st.write(f"{item['time']}  —  {item['result']}  (Confidence {item['confidence']})")

        st.markdown("</div>", unsafe_allow_html=True)
