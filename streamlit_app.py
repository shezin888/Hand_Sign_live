import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
st.set_page_config(page_title="Hand Sign Detection", page_icon="✋", layout="wide")

# optional CSS (ok after page_config)
st.markdown("""
<style>
[data-testid="stCamera"] video,
[data-testid="stCamera"] canvas,
[data-testid="stFileUploaderDropzone"] {
    max-width: 360px !important;
}
</style>
""", unsafe_allow_html=True)

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

# ---- Config ----
MODEL_DIR = "models/hand_ssd/saved_model"
CLASS_NAMES = ["Hello", "Yes", "No", "Thank You", "I Love You"]  # ids 1..5

# ---- Load model once ----
@st.cache_resource
def load_model():
    return tf.saved_model.load(MODEL_DIR)

detect_fn = load_model()

# ---- Drawing helpers ----
def _draw_box(img, x1, y1, x2, y2, label, score):
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
    tag = f"{label} {score:.2f}"
    tw, th = 8 * len(tag), 16
    y0 = max(0, y1 - th - 2)
    draw.rectangle([x1, y0, x1 + tw, y0 + th], fill=(0, 255, 0))
    draw.text((x1 + 3, y0 + 2), tag, fill=(0, 0, 0))

# ---- Simple: show only the single best detection ----
def run_inference_top1(pil_img, score_thresh=0.70):
    img = pil_img.convert("RGB")
    W, H = img.size
    x = tf.convert_to_tensor(np.asarray(img))[tf.newaxis, ...]
    out = detect_fn(x)
    boxes  = out["detection_boxes"][0].numpy()
    scores = out["detection_scores"][0].numpy()
    classes= out["detection_classes"][0].numpy().astype(int)

    i = int(np.argmax(scores))
    s = float(scores[i])
    if s < score_thresh:
        return img

    ymin, xmin, ymax, xmax = boxes[i]
    x1 = int(np.clip(xmin * W, 0, W - 1)); x2 = int(np.clip(xmax * W, 0, W - 1))
    y1 = int(np.clip(ymin * H, 0, H - 1)); y2 = int(np.clip(ymax * H, 0, H - 1))
    w, h = x2 - x1, y2 - y1
    area_ratio = (w * h) / float(W * H)
    if w < 16 or h < 16 or area_ratio > 0.85:
        return img

    label = CLASS_NAMES[classes[i]-1] if 1 <= classes[i] <= len(CLASS_NAMES) else str(classes[i])
    _draw_box(img, x1, y1, x2, y2, label, s)
    return img

# ---- Advanced: class-agnostic NMS + filters ----
def run_inference_nms(pil_img, score_thresh=0.60, iou_thresh=0.45, max_dets=3):
    img = pil_img.convert("RGB")
    W, H = img.size
    x = tf.convert_to_tensor(np.asarray(img))[tf.newaxis, ...]
    out = detect_fn(x)

    boxes  = out["detection_boxes"][0].numpy()
    scores = out["detection_scores"][0].numpy()
    classes= out["detection_classes"][0].numpy().astype(int)

    keep = scores >= score_thresh
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    if boxes.size == 0:
        return img

    b_abs = np.stack([boxes[:,0]*H, boxes[:,1]*W, boxes[:,2]*H, boxes[:,3]*W], axis=1).astype(np.float32)
    idx = tf.image.non_max_suppression(b_abs, scores.astype(np.float32),
                                       max_output_size=max_dets, iou_threshold=iou_thresh).numpy()
    for (y1,x1,y2,x2), s, c in zip(b_abs[idx], scores[idx], classes[idx]):
        x1 = int(np.clip(x1, 0, W-1)); x2 = int(np.clip(x2, 0, W-1))
        y1 = int(np.clip(y1, 0, H-1)); y2 = int(np.clip(y2, 0, H-1))
        w, h = x2-x1, y2-y1
        if w < 16 or h < 16 or (w*h)/(W*H) > 0.85 or (w/max(h,1) < 0.25) or (w/max(h,1) > 4.0):
            continue
        label = CLASS_NAMES[c-1] if 1 <= c <= len(CLASS_NAMES) else str(c)
        _draw_box(img, x1, y1, x2, y2, label, float(s))
    return img

# ---- UI ----
st.title("✋ Hand Sign Detection")
st.caption("Live SSD MobileNet hand-sign detector. Use webcam or upload an image.")

mode = st.sidebar.radio("Mode", ["Simple (Top-1)", "Advanced (NMS)"], index=0)
if mode == "Simple (Top-1)":
    score = st.sidebar.slider("Score threshold", 0.0, 1.0, 0.70, 0.01)
else:
    score = st.sidebar.slider("Score threshold", 0.0, 1.0, 0.60, 0.01)
    iou   = st.sidebar.slider("NMS IoU", 0.10, 0.90, 0.45, 0.01)
    k     = st.sidebar.slider("Max boxes", 1, 10, 3)

tab1, tab2 = st.tabs(["📷 Webcam", "📁 Upload"])

with tab1:
    col_cam, col_out = st.columns([1, 2])
    with col_cam:
        snap = st.camera_input("Webcam", label_visibility="collapsed")
    with col_out:
        if snap:
            img = Image.open(snap)
            out = run_inference_top1(img, score_thresh=score) if mode == "Simple (Top-1)" \
                  else run_inference_nms(img, score_thresh=score, iou_thresh=iou, max_dets=k)
            st.image(out, use_column_width=True)

with tab2:
    file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if file:
        img = Image.open(file)
        out = run_inference_top1(img, score_thresh=score) if mode == "Simple (Top-1)" \
              else run_inference_nms(img, score_thresh=score, iou_thresh=iou, max_dets=k)
        st.image(out, use_column_width=True)
        
st.markdown(
    "<small>Tip: plain background + good lighting improves results. "
    "If boxes look too big/noisy, raise the score threshold or use Simple mode.</small>",
    unsafe_allow_html=True
)
