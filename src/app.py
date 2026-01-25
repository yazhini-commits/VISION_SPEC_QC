import streamlit as st
import os
import sys

# --------------------------------------------------
# PATH FIX (IMPORTANT)
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# add src to python path
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from inference import predict
from gradcam_inference import run_gradcam

# --------------------------------------------------
# OUTPUT PATHS
# --------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "gradcam_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# UI CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="VisionSpec QC Dashboard",
    layout="centered"
)

st.title("VisionSpec – Bottle Quality Control")
st.write("AI-powered visual inspection system")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload bottle image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    temp_path = os.path.join(PROJECT_ROOT, "temp_upload.jpg")

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(
        temp_path,
        caption="Uploaded Image",
        use_column_width=True
    )

    if st.button("Run Inspection"):
        with st.spinner("Analyzing image..."):
            result = predict(temp_path)
            run_gradcam(temp_path)

        st.subheader("Inspection Result")

        st.markdown(f"**Result:** {result['result']}")
        st.markdown(f"**Confidence:** {result['confidence']}")
        st.markdown(f"**Severity:** {result['severity']}")
        st.markdown(f"**Data Drift Warning:** {result['data_drift_warning']}")

        gradcam_image = os.path.join(
            OUTPUT_DIR,
            os.path.basename(temp_path)
        )

        if os.path.exists(gradcam_image):
            st.subheader("Grad-CAM Explanation")
            st.image(
                gradcam_image,
                use_column_width=True
            )
