"""
Public Cloud Streamlit Application for Hybrid Drone Detection
Compatible with Streamlit Community Cloud (Mobile & Web)
"""

import os
import glob
import cv2
import numpy as np
import streamlit as st
from hybrid_detector import HybridDroneDetector

# Page Configuration
st.set_page_config(
    page_title="Hybrid Drone Detection AI",
    page_icon="🛸",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
    }
    .header-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        border-left: 5px solid #10b981;
    }
    .badge {
        background: #10b981;
        color: black;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class='header-box'>
    <h2 style='color: white; margin: 0;'>🛸 Hybrid Deep Learning Architecture for Robust Drone Detection</h2>
    <p style='color: #94a3b8; margin: 6px 0 0 0;'>
        Dual-YOLO (v8 + 11) with Weighted Boxes Fusion (WBF) | <span class='badge'>Cloud Deployment Live</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Detection Controls")
conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
iou_thresh = st.sidebar.slider("WBF IoU Fusion Threshold", min_value=0.30, max_value=0.85, value=0.55, step=0.05)

# Load Models
base_dir = os.path.dirname(os.path.abspath(__file__))
m1 = os.path.join(base_dir, "models", "yolov8_drone.pt")
m2 = os.path.join(base_dir, "models", "yolo11_drone.pt")

if not os.path.exists(m1):
    m1 = "yolov8n.pt"
if not os.path.exists(m2):
    m2 = "yolo11n.pt"

@st.cache_resource
def load_detector(p1, p2):
    return HybridDroneDetector(p1, p2, conf_thresh=0.25, iou_thresh=0.55)

detector = load_detector(m1, m2)
detector.conf_thresh = conf_thresh
detector.iou_thresh = iou_thresh

# Tabs
tab_sample, tab_cam, tab_upload = st.tabs(["🎲 Test Drone Samples (Instant)", "📷 Device Camera", "📁 Upload Image"])

input_image = None

# TAB 1: SAMPLE DRONE IMAGES
with tab_sample:
    st.markdown("### Select a Drone Test Image")
    sample_files = glob.glob(os.path.join(base_dir, "sample_images", "*.jpg")) + glob.glob(os.path.join(base_dir, "sample_images", "*.png"))
    
    if sample_files:
        cols = st.columns(min(5, len(sample_files)))
        selected_file = None
        for i, s_file in enumerate(sample_files[:5]):
            with cols[i]:
                st.image(s_file, width="stretch")
                if st.button(f"Test #{i+1}", key=f"btn_{i}"):
                    st.session_state.current_sample = s_file

        if "current_sample" in st.session_state and st.session_state.current_sample:
            input_image = cv2.imread(st.session_state.current_sample)
    else:
        st.info("No sample images found.")

# TAB 2: DEVICE CAMERA
with tab_cam:
    st.markdown("### Scan with Your Device Camera (Phone / Laptop)")
    cam_file = st.camera_input("Take a photo to test detection")
    if cam_file is not None:
        file_bytes = np.asarray(bytearray(cam_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, 1)

# TAB 3: CUSTOM UPLOAD
with tab_upload:
    st.markdown("### Upload Custom Image")
    uploaded = st.file_uploader("Upload drone photo", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, 1)

# RUN DETECTION & DISPLAY RESULTS
if input_image is not None:
    st.markdown("---")
    st.markdown("### 🔍 Hybrid Detection & Model Comparison")

    results = detector.predict(input_image)
    vis_m1 = detector.draw_detections(input_image, results, mode="model1")
    vis_m2 = detector.draw_detections(input_image, results, mode="model2")
    vis_hyb = detector.draw_detections(input_image, results, mode="hybrid")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"#### 🟠 Model 1 (YOLOv8)")
        st.caption(f"Detections: **{results['model1']['count']}**")
        st.image(cv2.cvtColor(vis_m1, cv2.COLOR_BGR2RGB), width="stretch")

    with c2:
        st.markdown(f"#### 🟡 Model 2 (YOLO11)")
        st.caption(f"Detections: **{results['model2']['count']}**")
        st.image(cv2.cvtColor(vis_m2, cv2.COLOR_BGR2RGB), width="stretch")

    with c3:
        st.markdown(f"#### 🟢 Hybrid Architecture (WBF)")
        st.caption(f"Robust Detections: **{results['hybrid']['count']}** (Fused)")
        st.image(cv2.cvtColor(vis_hyb, cv2.COLOR_BGR2RGB), width="stretch")

    count = results['hybrid']['count']
    if count > 0:
        scores = [f"{s:.2f}" for s in results['hybrid']['scores']]
        st.success(f"✅ Drone Successfully Detected! Fused Confidence: {', '.join(scores)}")
    else:
        st.warning("No drone detected at the current confidence threshold.")
