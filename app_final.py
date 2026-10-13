"""
Automatic Number Plate Detection - Streamlit web app

Run:
    python -m streamlit run app.py
"""

import time

import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Number Plate Detection", page_icon="🚘", layout="wide")

MAX_CHARS = 14          # more character-like blobs than this is a grille / texture, not a plate
PAD = 4                 # pixels added around every candidate box
MIN_BOX_W, MIN_BOX_H = 30, 10

# ---------------------------------------------------------------- styling
STYLE = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg, #1e3a8a 0%, #6d28d9 100%); border-radius: 18px;
       padding: 28px 32px; color: #ffffff; margin-bottom: 1.2rem;}
.hero h1 {color: #ffffff; margin: 0 0 6px 0; font-size: 2.1rem; padding: 0;}
.hero p {margin: 0 0 14px 0; opacity: 0.92; font-size: 1.02rem;}
.pill {display: inline-block; background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.3);
       padding: 3px 12px; border-radius: 999px; font-size: 0.82rem; margin: 0 6px 6px 0;}
.step-card {border: 1px solid rgba(128,128,128,0.25); background: rgba(128,128,128,0.07);
            border-radius: 14px; padding: 16px 18px; height: 100%;}
.step-card .num {font-weight: 700; font-size: 0.78rem; letter-spacing: 0.08em; opacity: 0.65;}
.step-card h4 {margin: 0.2rem 0 0.3rem 0;}
.step-card p {margin: 0; opacity: 0.8; font-size: 0.92rem;}
[data-testid="stMetric"] {background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.2);
                          border-radius: 14px; padding: 12px 16px;}
[data-testid="stImage"] img {border-radius: 10px;}
</style>
"""



def render_header():
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>🚘 Automatic Number Plate Detection</h1>'
        '<p>Upload a photo of a vehicle and detect its number plate.</p>'
        '</div>',
        unsafe_allow_html=True,
    )



def render_empty_state():
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="step-card"><div class="num">STEP 1</div><h4>Upload</h4>'
                "<p>Choose a photo of one or more vehicles. Clear front or rear views work best.</p></div>",
                unsafe_allow_html=True)
    c2.markdown('<div class="step-card"><div class="num">STEP 2</div><h4>Adjust settings</h4>'
                "<p>Use the sidebar sliders to tune the detection for your photo.</p></div>",
                unsafe_allow_html=True)
    c3.markdown('<div class="step-card"><div class="num">STEP 3</div><h4>Get the plate</h4>'
                "<p>See the detected plates boxed in red and download the cropped plates.</p></div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
def read_settings():
    st.sidebar.title("⚙️ Settings")
    settings = dict(
        width=st.sidebar.select_slider("Working image width (px)", options=[600, 800, 1000, 1200, 1600, 2000, 2400, 3000], value=1000),
        canny_low=st.sidebar.slider("Canny low threshold", 10, 150, 30),
        canny_high=st.sidebar.slider("Canny high threshold", 100, 400, 200),
        ar=st.sidebar.slider("Plate aspect ratio (width / height)", 1.0, 8.0, (2.0, 6.0), 0.1),
        max_pct=st.sidebar.slider("Max plate size (% of image)", 1, 20, 5),
        min_chars=st.sidebar.slider("Min character-like blobs inside a plate", 2, 8, 4),
        max_plates=st.sidebar.slider("Max plates to detect", 1, 10, 5),
    )
    st.sidebar.info(
        "**Tips**\n\n"
        "- Plate missed: raise the working width, or lower the min character-like blobs.\n"
        "- Wrong shapes boxed (windows, signs): raise the min character-like blobs, "
        "or lower the max plate size."
    )
    return settings


# ---------------------------------------------------------------- detection
def preprocess(img, work_width):
    """Resize (keeping proportions), convert to grayscale and denoise with a bilateral filter."""
    h, w = img.shape[:2]
    img = cv2.resize(img, (work_width, int(h * work_width / w)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 15, 15)
    return img, gray


def contour_boxes(gray, low, high):
    """Candidate boxes from Canny edges: contours with 4-8 corners that fill their bounding box."""
    edged = cv2.Canny(gray, low, high)
    found = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:400]

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            break                      # sorted by area, everything after is smaller
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if not (4 <= len(approx) <= 8):
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        if bw * bh > 0 and area / float(bw * bh) >= 0.5:
            boxes.append((x, y, bw, bh))
    return boxes


def sobel_boxes(gray, work_width):
    """Candidate boxes from vertical (Sobel x) edges joined by morphological closing."""
    sobel = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3))
    _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_w = max(9, int(work_width * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)    # merge characters into a block
    thresh = cv2.erode(thresh, None, iterations=1)                # remove small noise
    thresh = cv2.dilate(thresh, None, iterations=2)

    found = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    return [cv2.boundingRect(c) for c in contours]


def filter_and_pad(boxes, img_shape, ar_range, max_pct):
    """Keep boxes with a plate-like ratio and size, then pad them slightly."""
    H, W = img_shape[:2]
    min_ratio, max_ratio = ar_range
    kept = []
    for (x, y, bw, bh) in boxes:
        if bh == 0 or bw < MIN_BOX_W or bh < MIN_BOX_H:
            continue
        if not (min_ratio <= bw / float(bh) <= max_ratio):
            continue
        if bw * bh > (max_pct / 100.0) * H * W:
            continue
        x0, y0 = max(x - PAD, 0), max(y - PAD, 0)
        x1, y1 = min(x + bw + PAD, W), min(y + bh + PAD, H)
        kept.append((x0, y0, x1 - x0, y1 - y0))
    return kept[:600]


def count_characters(crop):
    """Count character-like blobs inside a candidate (Otsu threshold + connected components)."""
    h, w = crop.shape[:2]
    if h < 8 or w < 24:
        return 0
    crop = cv2.resize(crop, (240, max(int(240 * h / w), 16)))
    H, W = crop.shape

    best = 0
    for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):       # dark-on-light and light-on-dark
        _, binary = cv2.threshold(crop, 0, 255, mode + cv2.THRESH_OTSU)
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        count = 0
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if x <= 1 or x + cw >= W - 1:                         # touches left/right edge: frame or background
                continue
            if 0.3 * H <= ch <= H and 2 <= cw <= ch and area >= 0.25 * cw * ch:
                count += 1
        best = max(best, count)
    return best


def overlap_ratio(a, b):
    """Intersection area divided by the smaller box's area."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return (ix * iy) / float(min(aw * ah, bw * bh))


def score_candidates(boxes, gray, min_chars):
    """Validate candidates by their characters and score them (best first)."""
    scored = []
    for (x, y, bw, bh) in boxes:
        crop = gray[y:y + bh, x:x + bw]
        count = count_characters(crop)
        if count < min_chars or count > MAX_CHARS:
            continue
        ratio = bw / float(bh)
        score = min(count, 10) + 2 * min(float(crop.std()) / 50.0, 1.0) + (1.0 if 3.0 <= ratio <= 5.5 else 0.0)
        scored.append((score, (x, y, bw, bh)))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def remove_duplicates(scored, limit):
    """Drop overlapping boxes (best score wins), keep up to `limit`, order left to right."""
    kept = []
    for _, box in scored:
        if any(overlap_ratio(box, k) > 0.4 for k in kept):
            continue
        kept.append(box)
        if len(kept) == limit:
            break
    return sorted(kept, key=lambda b: b[0])


def detect_plates(img, s):
    """Full pipeline. Returns (resized color image, annotated image, list of cropped plates)."""
    img, gray = preprocess(img, s["width"])

    boxes = contour_boxes(gray, s["canny_low"], s["canny_high"]) + sobel_boxes(gray, s["width"])
    boxes = filter_and_pad(boxes, img.shape, s["ar"], s["max_pct"])
    plates = remove_duplicates(score_candidates(boxes, gray, s["min_chars"]), s["max_plates"])

    annotated = img.copy()
    crops = []
    for i, (x, y, bw, bh) in enumerate(plates, 1):
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(annotated, str(i), (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        crops.append(gray[y:y + bh, x:x + bw])
    return img, annotated, crops


# ---------------------------------------------------------------- output
def png_bytes(image):
    ok, buf = cv2.imencode(".png", image)
    return buf.tobytes() if ok else b""


def render_results(resized, annotated, crops, elapsed_ms):
    h, w = resized.shape[:2]
    m1, m2, m3 = st.columns(3)
    m1.metric("Plates found", len(crops))
    m2.metric("Working size", f"{w} × {h}")
    m3.metric("Processing time", f"{elapsed_ms:.0f} ms")

    if crops:
        st.success(f"Detected {len(crops)} number plate(s).")
    else:
        st.warning("No number plate detected. Try a larger working width, fewer minimum "
                   "character-like blobs, or a wider aspect ratio range.")

    st.subheader("Detected vehicle")
    st.image(annotated, channels="BGR", caption="Number plates are boxed in red and numbered left to right")
    st.download_button("⬇️ Download annotated image", png_bytes(annotated),
                       file_name="detected.png", mime="image/png")

    if crops:
        st.subheader("Cropped number plates")
        cols = st.columns(min(len(crops), 3))
        for i, crop in enumerate(crops):
            shown = crop
            if crop.shape[1] < 300:                               # enlarge small crops for display only
                scale = 300.0 / crop.shape[1]
                shown = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            with cols[i % len(cols)]:
                st.image(shown, caption=f"Plate {i + 1}")
                st.download_button(f"⬇️ Download plate {i + 1}", png_bytes(crop),
                                   file_name=f"plate_{i + 1}.png", mime="image/png", key=f"dl_{i}")


# ---------------------------------------------------------------- app
def main():
    render_header()
    settings = read_settings()

    uploaded = st.file_uploader("Drop a vehicle photo here (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        render_empty_state()
        return

    data = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not read this image. Please try a different file.")
        return

    with st.spinner("Detecting number plates..."):
        start = time.perf_counter()
        resized, annotated, crops = detect_plates(img, settings)
        elapsed_ms = (time.perf_counter() - start) * 1000

    render_results(resized, annotated, crops, elapsed_ms)


main()