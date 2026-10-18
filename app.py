import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "best.pt"
CLASS_COLORS = {
    "car": "🟦", "motorcycle": "🟪", "helmet": "🟩",
    "no_helmet": "🟥", "number_plate": "🟨",
}

st.set_page_config(page_title="Traffic Violation Detection", page_icon="🚦", layout="wide")


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def run_detection(model, img_rgb, conf, iou):
    res = model.predict(img_rgb, conf=conf, iou=iou, imgsz=640, verbose=False)[0]
    annotated = res.plot()[:, :, ::-1]  # BGR -> RGB
    names = [model.names[int(c)] for c in res.boxes.cls]
    return res, annotated, names


def plate_crops(res, img_rgb, model):
    crops = []
    h, w = img_rgb.shape[:2]
    for box, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls):
        if model.names[int(c)] == "number_plate":
            x1, y1, x2, y2 = [int(v) for v in box]
            crop = img_rgb[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
            if crop.size:
                crops.append(crop)
    return crops


def show_summary(names):
    counts = {n: names.count(n) for n in sorted(set(names))}
    if not counts:
        st.info("Kuch detect nahi hua.")
        return counts
    cols = st.columns(len(counts))
    for col, (k, v) in zip(cols, counts.items()):
        col.metric(f"{CLASS_COLORS.get(k, '⬜')} {k}", v)
    if counts.get("no_helmet", 0) > 0:
        st.error(f"🚨 VIOLATION: {counts['no_helmet']} rider(s) bina helmet ke!")
    elif counts.get("helmet", 0) > 0:
        st.success("✅ Koi helmet violation nahi mili.")
    return counts


# ---------------- UI ----------------
st.title("🚦 AI-Powered Traffic Violation Detection System")
st.caption("YOLO11 • car, motorcycle, helmet, no_helmet, number_plate")

if not os.path.exists(MODEL_PATH):
    st.error(f"`{MODEL_PATH}` nahi mili. Apni trained `best.pt` ko app.py ke saath same folder mein rakho.")
    st.stop()

model = load_model()

with st.sidebar:
    st.header("⚙️ Settings")
    conf = st.slider("Confidence", 0.05, 0.90, 0.25, 0.05)
    iou = st.slider("IoU (NMS)", 0.10, 0.90, 0.50, 0.05)
    st.markdown("---")
    st.write("**Classes:**")
    for k, v in CLASS_COLORS.items():
        st.write(f"{v} {k}")

tab_img, tab_cam, tab_vid = st.tabs(["🖼️ Image", "📷 Camera", "🎬 Video"])


def handle_image(file):
    img = np.array(Image.open(file).convert("RGB"))
    res, annotated, names = run_detection(model, img, conf, iou)
    c1, c2 = st.columns(2)
    c1.image(img, caption="Original", use_container_width=True)
    c2.image(annotated, caption="Detection", use_container_width=True)
    counts = show_summary(names)
    if counts.get("no_helmet", 0) > 0:
        crops = plate_crops(res, img, model)
        if crops:
            st.subheader("🔎 Detected number plates")
            cols = st.columns(min(len(crops), 4))
            for i, crop in enumerate(crops):
                cols[i % len(cols)].image(crop, use_container_width=True)


with tab_img:
    f = st.file_uploader("Image upload karo", type=["jpg", "jpeg", "png", "webp"], key="img")
    if f:
        handle_image(f)

with tab_cam:
    shot = st.camera_input("Photo lo")
    if shot:
        handle_image(shot)

with tab_vid:
    v = st.file_uploader("Video upload karo (chhoti video behtar hai)", type=["mp4", "avi", "mov", "mkv"], key="vid")
    skip = st.slider("Har kitne frame baad detect karein", 1, 10, 3)
    if v and st.button("▶️ Process video"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(v.read())
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_box = st.empty()
        info_box = st.empty()
        bar = st.progress(0)
        violations, i = 0, 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if i % skip == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, annotated, names = run_detection(model, rgb, conf, iou)
                frame_box.image(annotated, use_container_width=True)
                if "no_helmet" in names:
                    violations += 1
                info_box.write(f"Frame {i}/{total} • violation frames: **{violations}**")
            i += 1
            bar.progress(min(i / total, 1.0))
        cap.release()
        os.unlink(tmp.name)
        if violations:
            st.error(f"🚨 {violations} processed frame(s) mein bina helmet rider mila.")
        else:
            st.success("✅ Video mein koi helmet violation nahi mili.")
