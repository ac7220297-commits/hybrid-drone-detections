"""
Hybrid Drone Detection - Public Cloud App with Real-Time Live Camera
Uses streamlit-webrtc for true real-time browser camera streaming
"""

import os
import glob
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from hybrid_detector import HybridDroneDetector

# Page Config
st.set_page_config(
    page_title="Hybrid Drone Detection",
    page_icon="🛸",
    layout="wide"
)

# Styling
st.markdown("""
<style>
    body { background-color: #0b0f19; }
    .header-box {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-left: 5px solid #10b981;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 16px;
    }
    .badge {
        background: #10b981;
        color: black;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='header-box'>
    <h2 style='color:white; margin:0;'>🛸 Hybrid Deep Learning Architecture for Robust Drone Detection</h2>
    <p style='color:#94a3b8; margin:5px 0 0 0;'>
        YOLOv8 + YOLO11 + Weighted Boxes Fusion (WBF) &nbsp;
        <span class='badge'>Live</span>
    </p>
</div>
""", unsafe_allow_html=True)

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

# Sidebar
st.sidebar.header("⚙️ Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
view_mode = st.sidebar.radio("Detection View", ["Hybrid Only (Neon Green)", "All 3 Models Side-by-Side"])
detector.conf_thresh = conf_thresh

# Real-time Video Processor Class
class DroneVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = detector
        self.view_mode = "Hybrid Only (Neon Green)"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        results = self.detector.predict(img)

        if self.view_mode == "Hybrid Only (Neon Green)":
            out = self.detector.draw_detections(img, results, mode="hybrid")
            count = results["hybrid"]["count"]
            # Status bar
            color = (0, 255, 128) if count > 0 else (80, 80, 80)
            text = f"DRONE DETECTED: {count}" if count > 0 else "Scanning..."
            cv2.rectangle(out, (8, 8), (360, 45), (0, 0, 0), -1)
            cv2.putText(out, text, (14, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        else:
            v1 = self.detector.draw_detections(img, results, mode="model1")
            v2 = self.detector.draw_detections(img, results, mode="model2")
            vh = self.detector.draw_detections(img, results, mode="hybrid")
            h, w = img.shape[:2]
            tw = w // 3
            th = int(h * tw / w)
            v1r = cv2.resize(v1, (tw, th))
            v2r = cv2.resize(v2, (tw, th))
            vhr = cv2.resize(vh, (tw, th))
            out = np.hstack([v1r, v2r, vhr])

        return av.VideoFrame.from_ndarray(out, format="bgr24")


# TABS
tab1, tab2 = st.tabs(["📹 Live Camera Detection", "🖼️ Test with Image"])

# TAB 1: LIVE REAL-TIME CAMERA
with tab1:
    st.subheader("Real-Time Live Camera Drone Detection")
    st.caption("Allow camera access when browser asks. Detection runs on every frame automatically.")

    RTC_CONFIG = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    ctx = webrtc_streamer(
        key="drone-detection",
        video_processor_factory=DroneVideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.view_mode = view_mode
        ctx.video_processor.detector.conf_thresh = conf_thresh

    st.info("▶️ Click **START** above to begin real-time drone detection from your camera.")

# TAB 2: IMAGE TEST
with tab2:
    st.subheader("Test with an Image")

    sample_files = sorted(glob.glob(os.path.join(base_dir, "sample_images", "*.jpg")))
    input_image = None

    if sample_files:
        st.markdown("**Quick Test — Click any sample drone image:**")
        cols = st.columns(min(5, len(sample_files)))
        for i, f in enumerate(sample_files[:5]):
            with cols[i]:
                st.image(f, use_container_width=True)
                if st.button(f"Test #{i+1}", key=f"s{i}"):
                    st.session_state.sample = f

        if "sample" in st.session_state:
            input_image = cv2.imread(st.session_state.sample)

    st.markdown("---")
    uploaded = st.file_uploader("Or upload your own drone image", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, 1)

    if input_image is not None:
        res = detector.predict(input_image)
        v1 = detector.draw_detections(input_image, res, mode="model1")
        v2 = detector.draw_detections(input_image, res, mode="model2")
        vh = detector.draw_detections(input_image, res, mode="hybrid")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**🟠 YOLOv8 — {res['model1']['count']} detections**")
            st.image(cv2.cvtColor(v1, cv2.COLOR_BGR2RGB), use_container_width=True)
        with c2:
            st.markdown(f"**🟡 YOLO11 — {res['model2']['count']} detections**")
            st.image(cv2.cvtColor(v2, cv2.COLOR_BGR2RGB), use_container_width=True)
        with c3:
            st.markdown(f"**🟢 Hybrid WBF — {res['hybrid']['count']} detections**")
            st.image(cv2.cvtColor(vh, cv2.COLOR_BGR2RGB), use_container_width=True)

        if res['hybrid']['count'] > 0:
            scores = [f"{s:.2f}" for s in res['hybrid']['scores']]
            st.success(f"✅ Drone Detected! Confidence: {', '.join(scores)}")
        else:
            st.warning("No drone detected. Try lowering the confidence threshold.")
