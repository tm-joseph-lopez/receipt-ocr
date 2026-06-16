"""
Receipt OCR Pipeline — Streamlit App
Stack: PaddleOCR (text extraction) → Qwen2-0.5B-Instruct (structured field mapping)
"""

import os
import io
import json
import re
import warnings
from pathlib import Path
from datetime import datetime

# Suppress deprecation warnings from libraries
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Receipt Lens",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background: #f6f8fb;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: #0d1117;
  border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] .stMarkdown h3 {
  color: #58a6ff !important;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  font-weight: 600;
}
section[data-testid="stSidebar"] hr { border-color: #21262d !important; }
section[data-testid="stSidebar"] label { color: #8b949e !important; font-size: 0.82rem !important; }
section[data-testid="stSidebar"] .stToggle span { color: #c9d1d9 !important; font-size: 0.88rem !important; }

/* ── Hero header ── */
.hero {
  background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
  border-radius: 16px;
  padding: 36px 40px;
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
  border: 1px solid #21262d;
}
.hero::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 220px; height: 220px;
  background: radial-gradient(circle, rgba(88,166,255,0.12) 0%, transparent 70%);
  border-radius: 50%;
}
.hero-title {
  font-size: 2rem;
  font-weight: 700;
  color: #f0f6fc;
  letter-spacing: -0.5px;
  margin: 0 0 6px 0;
}
.hero-sub {
  font-size: 0.88rem;
  color: #8b949e;
  font-family: 'JetBrains Mono', monospace;
  margin: 0;
}
.hero-pill {
  display: inline-block;
  background: rgba(88,166,255,0.12);
  border: 1px solid rgba(88,166,255,0.3);
  color: #58a6ff;
  font-size: 0.7rem;
  font-weight: 600;
  padding: 2px 10px;
  border-radius: 20px;
  margin-right: 6px;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.3px;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
  background: #ffffff;
  border-radius: 12px;
  border: 2px dashed #d0d7de;
  padding: 8px;
  transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #58a6ff; }

/* ── Stat cards ── */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin: 24px 0;
}
.stat-card {
  background: #ffffff;
  border: 1px solid #e8ecf0;
  border-radius: 12px;
  padding: 18px 20px;
  text-align: center;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  transition: box-shadow 0.2s;
}
.stat-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
.stat-number {
  font-size: 2rem;
  font-weight: 700;
  color: #0d1117;
  line-height: 1;
  margin-bottom: 4px;
}
.stat-number.blue  { color: #0969da; }
.stat-number.green { color: #1a7f37; }
.stat-number.red   { color: #cf222e; }
.stat-number.gold  { color: #9a6700; }
.stat-label {
  font-size: 0.72rem;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  font-weight: 500;
}

/* ── Receipt cards ── */
.r-card {
  background: #ffffff;
  border: 1px solid #e8ecf0;
  border-radius: 14px;
  padding: 24px;
  margin-bottom: 16px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  transition: box-shadow 0.2s, border-color 0.2s;
}
.r-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.08); border-color: #d0d7de; }
.r-card-error { border-left: 4px solid #cf222e; }
.r-card-ok    { border-left: 4px solid #1a7f37; }

.r-filename {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  color: #0969da;
  font-weight: 500;
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.field-grid {
  display: grid;
  grid-template-columns: 120px 1fr;
  gap: 8px 12px;
  align-items: center;
}
.f-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  font-weight: 500;
}
.f-value {
  font-size: 0.92rem;
  color: #0d1117;
  font-weight: 500;
}
.f-value.total {
  font-size: 1.1rem;
  font-weight: 700;
  color: #1a7f37;
  font-family: 'JetBrains Mono', monospace;
}
.f-null {
  color: #d0d7de;
  font-style: italic;
  font-size: 0.85rem;
}

/* ── Confidence bar ── */
.conf-bar-bg {
  background: #f0f2f5;
  border-radius: 4px;
  height: 6px;
  width: 100%;
  margin-top: 4px;
}
.conf-bar-fill {
  height: 6px;
  border-radius: 4px;
  background: linear-gradient(90deg, #1a7f37, #2da44e);
}

/* ── Badge ── */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.3px;
  font-family: 'JetBrains Mono', monospace;
}
.badge-ok  { background: #dafbe1; color: #116329; border: 1px solid #aae0b4; }
.badge-err { background: #ffebe9; color: #a40e26; border: 1px solid #ffc1ba; }
.badge-warn{ background: #fff8c5; color: #7d4e00; border: 1px solid #f0cc4a; }

/* ── OCR block ── */
.ocr-block {
  background: #0d1117;
  color: #7ee787;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  padding: 16px 18px;
  border-radius: 10px;
  white-space: pre-wrap;
  max-height: 280px;
  overflow-y: auto;
  line-height: 1.7;
  border: 1px solid #21262d;
}

/* ── Progress ── */
div[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, #0969da, #58a6ff) !important;
  border-radius: 4px;
}
div[data-testid="stProgressBar"] > div {
  background: #e8ecf0 !important;
  border-radius: 4px;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #0969da, #0550ae);
  color: white;
  border: none;
  border-radius: 8px;
  font-family: 'Inter', sans-serif;
  font-weight: 600;
  font-size: 0.9rem;
  padding: 0.55rem 1.6rem;
  transition: all 0.15s;
  box-shadow: 0 2px 6px rgba(9,105,218,0.3);
}
.stButton > button:hover {
  background: linear-gradient(135deg, #0550ae, #033d8b);
  box-shadow: 0 4px 12px rgba(9,105,218,0.4);
  transform: translateY(-1px);
  color: white;
}
.stButton > button:active { transform: translateY(0); }

.stDownloadButton > button {
  background: #f6f8fa;
  color: #24292f;
  border: 1px solid #d0d7de;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.88rem;
  transition: all 0.15s;
}
.stDownloadButton > button:hover {
  background: #eaeef2;
  border-color: #b0b7be;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: #f6f8fa;
  border-radius: 10px;
  padding: 4px;
  border: 1px solid #e8ecf0;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 7px;
  font-weight: 500;
  font-size: 0.85rem;
  padding: 6px 16px;
  color: #656d76;
}
.stTabs [aria-selected="true"] {
  background: #ffffff !important;
  color: #0d1117 !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Summary table ── */
.sum-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.sum-table th {
  background: #f6f8fa;
  padding: 10px 14px;
  text-align: left;
  font-size: 0.72rem;
  font-weight: 600;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  border-bottom: 1px solid #e8ecf0;
}
.sum-table td {
  padding: 10px 14px;
  border-bottom: 1px solid #f0f2f5;
  color: #24292f;
  vertical-align: middle;
}
.sum-table tr:last-child td { border-bottom: none; }
.sum-table tr:hover td { background: #f6f8fa; }
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }

/* ── Divider ── */
hr { border-color: #e8ecf0 !important; margin: 24px 0 !important; }

/* ── Section title ── */
.section-title {
  font-size: 1rem;
  font-weight: 600;
  color: #0d1117;
  margin: 24px 0 16px 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-title::after {
  content: '';
  flex: 1;
  height: 1px;
  background: #e8ecf0;
  margin-left: 8px;
}

/* ── JSON block ── */
.json-block {
  background: #f6f8fa;
  border: 1px solid #e8ecf0;
  border-radius: 8px;
  padding: 14px 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  color: #24292f;
  white-space: pre-wrap;
  max-height: 200px;
  overflow-y: auto;
}

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
RECEIPT_FIELDS = ["store_name", "date", "total"]

SYSTEM_PROMPT = """You are a receipt parsing assistant.
Extract fields from receipt OCR text and return ONLY a valid JSON object.
Use null for missing fields. No markdown, no explanation, no text outside the JSON.

RULES:
- date must be in ISO format: YYYY-MM-DD
- total must be a plain number only, no symbols or letters

Fields: store_name, date, total
"""

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_ocr_engine():
    import torch
    from paddleocr import PaddleOCR
    return PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=torch.cuda.is_available(),
        show_log=False,
    )


@st.cache_resource(show_spinner=False)
def load_llm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    QWEN_MODEL = "Qwen/Qwen2-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=None,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    if int(torch.__version__.split(".")[0]) >= 2:
        try:
            model = torch.compile(model)
        except Exception:
            pass
    return tokenizer, model, device


# ─────────────────────────────────────────────────────────────────────────────
# OCR & LLM helpers
# ─────────────────────────────────────────────────────────────────────────────
def run_ocr(image: Image.Image) -> str:
    import numpy as np
    ocr_engine = load_ocr_engine()
    img_array = np.array(image.convert("RGB"))
    result = ocr_engine.ocr(img_array, cls=True)
    lines = []
    if result and result[0]:
        for item in result[0]:
            text, _conf = item[1]
            lines.append(text)
    return "\n".join(lines)


def _sanitize_llm_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    def _resolve_expr(m):
        parts = m.group(0).split("=")
        return parts[-1].strip()

    text = re.sub(r'[\d\s\.\+\-\*\/]+==[^,\}\]\\\"\'\n]+', _resolve_expr, text)
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return text[start:]


def _repair_truncated_json(text: str) -> str:
    stack = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack and stack[-1] == ch:
                    stack.pop()
    repaired = text.rstrip().rstrip(",")
    for closer in reversed(stack):
        repaired += closer
    return repaired


def extract_fields_with_llm(ocr_text: str) -> dict:
    import torch
    if not ocr_text.strip():
        return {field: None for field in RECEIPT_FIELDS}
    tokenizer, model, device = load_llm()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Receipt OCR text:\n\n{ocr_text}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    clean = _sanitize_llm_json(response_text)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    repaired = _repair_truncated_json(clean)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return {"raw_llm_response": response_text}


def completeness_score(fields: dict) -> int:
    """Return % of core fields that are non-null."""
    filled = sum(1 for f in RECEIPT_FIELDS if fields.get(f) is not None)
    return int((filled / len(RECEIPT_FIELDS)) * 100)


def resize_image_for_display(image: Image.Image, max_width: int = 320) -> Image.Image:
    """Resize image for display without using deprecated parameters."""
    w, h = image.size
    if w > max_width:
        ratio = max_width / w
        image = image.resize((max_width, int(h * ratio)), Image.LANCZOS)
    return image


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Settings")
    st.divider()

    st.markdown("**Display**")
    show_image   = st.toggle("Show receipt thumbnails", value=True)
    show_ocr     = st.toggle("Show raw OCR text", value=False)
    show_json    = st.toggle("Show raw JSON output", value=False)

    st.divider()
    st.markdown("**Export**")
    include_ocr_in_csv = st.toggle("Include OCR text in CSV", value=False)

    st.divider()
    st.markdown("**Pipeline**")
    conf_threshold = st.slider("Min. completeness to flag ⚠️", 0, 100, 60, step=10,
                               help="Receipts below this % completeness get a warning badge")

    st.divider()
    st.markdown("**Model cache paths**")
    st.code(
        "HuggingFace:\n~/.cache/huggingface\n\nPaddleOCR:\n~/.paddleocr",
        language=None,
    )

    st.divider()
    st.markdown("**Stack**")
    st.markdown("""
    <div style='font-size:0.78rem;color:#8b949e;line-height:1.9'>
    🔍 PaddleOCR 2.8.1<br>
    🤖 Qwen2-0.5B-Instruct<br>
    📦 Transformers ≥ 4.44<br>
    🔥 PyTorch 2.4 (CPU)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div style="margin-bottom:14px">
    <span class="hero-pill">PaddleOCR</span>
    <span class="hero-pill">Qwen2-0.5B</span>
    <span class="hero-pill">Local · No cloud</span>
  </div>
  <div class="hero-title">🧾 Receipt Lens</div>
  <p class="hero-sub">Drop receipt images → get structured data → export CSV</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# File uploader
# ─────────────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload receipt images — JPG, PNG, BMP, TIFF",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.markdown("""
    <div style='text-align:center;padding:48px 0;color:#8b949e'>
      <div style='font-size:3rem;margin-bottom:12px'>📂</div>
      <div style='font-size:1rem;font-weight:500;color:#656d76'>No images uploaded yet</div>
      <div style='font-size:0.85rem;margin-top:6px'>Drag and drop receipt images above to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.markdown(
    f"<div style='font-size:0.85rem;color:#656d76;margin:8px 0 4px'>"
    f"<b style='color:#0d1117'>{len(uploaded_files)}</b> image(s) queued</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Run button
# ─────────────────────────────────────────────────────────────────────────────
col_btn, col_note = st.columns([1, 4])
with col_btn:
    run = st.button("▶ Run Pipeline", use_container_width=True)
with col_note:
    st.markdown(
        "<div style='padding:10px 0;font-size:0.82rem;color:#8b949e'>"
        "Models load once and stay cached for the session.</div>",
        unsafe_allow_html=True,
    )

if not run:
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────────────────────
results = []
errors  = []

with st.container():
    st.markdown("<div class='section-title'>⏳ Processing</div>", unsafe_allow_html=True)

    overall_label = st.empty()
    overall_bar   = st.progress(0)
    step_label    = st.empty()
    step_bar      = st.progress(0)
    log_area      = st.empty()
    log_lines     = []

    def add_log(msg):
        if msg:
            log_lines.append(msg)
        log_area.markdown(
            "<div style='background:#0d1117;border-radius:10px;padding:14px 18px;"
            "font-family:JetBrains Mono,monospace;font-size:0.74rem;color:#8b949e;"
            "max-height:160px;overflow-y:auto;line-height:1.9;border:1px solid #21262d'>"
            + "<br>".join(log_lines[-10:])
            + "</div>",
            unsafe_allow_html=True,
        )

    def set_step(frac, text, color="#656d76"):
        step_label.markdown(
            f"<span style='font-size:0.82rem;color:{color}'>↳ {text}</span>",
            unsafe_allow_html=True,
        )
        step_bar.progress(frac)

    # ── Load models ──────────────────────────────────────────────────────────
    overall_label.markdown(
        "<span style='font-size:0.88rem;font-weight:600'>Step 1 / 3 &nbsp;·&nbsp; Loading models</span>",
        unsafe_allow_html=True,
    )
    overall_bar.progress(0)
    set_step(0.0, "Initialising PaddleOCR…")
    add_log("⏳ Loading PaddleOCR engine…")

    try:
        load_ocr_engine()
        set_step(0.5, "PaddleOCR ready ✓", "#1a7f37")
        add_log("✅ PaddleOCR ready")

        set_step(0.5, "Loading Qwen2-0.5B-Instruct (first run: ~1 GB download)…")
        add_log("⏳ Loading Qwen2-0.5B-Instruct…")
        load_llm()
        set_step(1.0, "All models loaded ✓", "#1a7f37")
        add_log("✅ Qwen LLM ready")

    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

    # ── Per-image loop ────────────────────────────────────────────────────────
    total = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        fname = uploaded_file.name
        pct   = (idx + 1) / total

        overall_label.markdown(
            f"<span style='font-size:0.88rem;font-weight:600'>"
            f"Step 2 / 3 &nbsp;·&nbsp; Receipt {idx + 1} of {total}"
            f"&nbsp; <span style='color:#8b949e;font-weight:400'>{fname}</span></span>",
            unsafe_allow_html=True,
        )
        overall_bar.progress(pct)
        add_log(f"")
        add_log(f"📄 [{idx+1}/{total}] {fname}")

        try:
            image = Image.open(uploaded_file).convert("RGB")

            # OCR
            set_step(0.1, f"Running OCR…")
            add_log("  🔍 OCR scanning…")
            raw_text = run_ocr(image)
            n_lines  = len(raw_text.splitlines())
            set_step(0.5, f"OCR complete — {n_lines} lines", "#1a7f37")
            add_log(f"  ✅ OCR done · {n_lines} lines detected")

            # LLM
            set_step(0.6, "Extracting fields with LLM…")
            add_log("  🤖 LLM parsing fields…")
            fields = extract_fields_with_llm(raw_text)
            set_step(1.0, "Fields extracted ✓", "#1a7f37")

            score   = completeness_score(fields)
            summary = {k: fields.get(k) for k in RECEIPT_FIELDS}
            add_log(f"  ✅ {summary} · completeness {score}%")

            if isinstance(fields.get("items"), list):
                fields["items"] = json.dumps(fields["items"])

            fields["source_file"]  = fname
            fields["raw_ocr_text"] = raw_text
            fields["_completeness"] = score
            fields["_processed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            results.append({
                "source_file": fname,
                "image":       image,
                "raw_ocr":     raw_text,
                "fields":      fields,
                "error":       None,
            })

        except Exception as e:
            errors.append(fname)
            add_log(f"  ❌ Error: {e}")
            results.append({
                "source_file": fname,
                "image":       None,
                "raw_ocr":     "",
                "fields":      {"_completeness": 0},
                "error":       str(e),
            })

    # ── Done ─────────────────────────────────────────────────────────────────
    ok_count  = sum(1 for r in results if not r["error"])
    overall_label.markdown(
        f"<span style='font-size:0.88rem;font-weight:600;color:#1a7f37'>"
        f"Step 3 / 3 &nbsp;·&nbsp; Complete ✓ &nbsp;"
        f"<span style='color:#8b949e;font-weight:400'>{ok_count} succeeded"
        f"{f', {len(errors)} failed' if errors else ''}</span></span>",
        unsafe_allow_html=True,
    )
    overall_bar.progress(1.0)
    set_step(1.0, "Pipeline complete ✓", "#1a7f37")
    add_log("")
    add_log(f"🎉 Done — {ok_count}/{total} receipts processed successfully")


# ─────────────────────────────────────────────────────────────────────────────
# Stats bar
# ─────────────────────────────────────────────────────────────────────────────
good   = [r for r in results if not r["error"]]
totals = []
for r in good:
    t = r["fields"].get("total")
    try:
        totals.append(float(t))
    except (TypeError, ValueError):
        pass

avg_completeness = (
    sum(r["fields"].get("_completeness", 0) for r in good) // max(len(good), 1)
)
total_spend = sum(totals)

st.markdown(f"""
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-number blue">{len(results)}</div>
    <div class="stat-label">Total Uploaded</div>
  </div>
  <div class="stat-card">
    <div class="stat-number green">{ok_count}</div>
    <div class="stat-label">Parsed OK</div>
  </div>
  <div class="stat-card">
    <div class="stat-number {'red' if errors else 'green'}">{len(errors)}</div>
    <div class="stat-label">Errors</div>
  </div>
  <div class="stat-card">
    <div class="stat-number gold">{'₱{:,.2f}'.format(total_spend) if total_spend else '—'}</div>
    <div class="stat-label">Total Spend</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs: Results | Summary Table | Export
# ─────────────────────────────────────────────────────────────────────────────
tab_results, tab_table, tab_export = st.tabs(["📋 Results", "📊 Summary Table", "📥 Export"])


# ── Tab 1: Results ────────────────────────────────────────────────────────────
with tab_results:
    if not results:
        st.info("No results yet.")
    else:
        for r in results:
            fields = r["fields"]
            score  = fields.get("_completeness", 0)

            if r["error"]:
                st.markdown(
                    f'<div class="r-card r-card-error">'
                    f'<div class="r-filename">📄 {r["source_file"]}'
                    f' &nbsp;<span class="badge badge-err">✕ Error</span></div>'
                    f'<div style="font-size:0.85rem;color:#cf222e">{r["error"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                continue

            # Badge
            if score == 100:
                badge = '<span class="badge badge-ok">✓ Complete</span>'
            elif score >= conf_threshold:
                badge = f'<span class="badge badge-warn">⚠ {score}% complete</span>'
            else:
                badge = f'<span class="badge badge-err">✕ {score}% complete</span>'

            # Field rows
            field_html = '<div class="field-grid">'
            for key in RECEIPT_FIELDS:
                val = fields.get(key)
                label = key.replace("_", " ").title()
                if val is None:
                    val_html = '<span class="f-null">—</span>'
                elif key == "total":
                    try:
                        val_html = f'<span class="f-value total">₱ {float(val):,.2f}</span>'
                    except (TypeError, ValueError):
                        val_html = f'<span class="f-value total">{val}</span>'
                else:
                    val_html = f'<span class="f-value">{val}</span>'
                field_html += f'<span class="f-label">{label}</span>{val_html}'

            # Extra fields from LLM
            skip = set(RECEIPT_FIELDS) | {"source_file", "raw_ocr_text", "_completeness", "_processed_at"}
            for key, val in fields.items():
                if key in skip or val is None:
                    continue
                label = key.replace("_", " ").title()
                field_html += f'<span class="f-label">{label}</span><span class="f-value">{val}</span>'

            field_html += "</div>"

            # Completeness bar
            bar_color = "#1a7f37" if score == 100 else ("#d4a017" if score >= conf_threshold else "#cf222e")
            conf_bar = (
                f'<div style="margin-top:14px">'
                f'<div style="font-size:0.68rem;color:#8b949e;margin-bottom:4px;font-family:JetBrains Mono,monospace">'
                f'COMPLETENESS &nbsp; {score}%</div>'
                f'<div class="conf-bar-bg"><div class="conf-bar-fill" '
                f'style="width:{score}%;background:{bar_color}"></div></div>'
                f'</div>'
            )

            processed_at = fields.get("_processed_at", "")

            if show_image and r["image"] is not None:
                img_col, info_col = st.columns([1, 2])
                with img_col:
                    display_img = resize_image_for_display(r["image"], max_width=300)
                    st.image(display_img)
                with info_col:
                    st.markdown(
                        f'<div class="r-card r-card-ok" style="margin:0;height:100%">'
                        f'<div class="r-filename">📄 {r["source_file"]} &nbsp;{badge}'
                        f'<span style="margin-left:auto;font-size:0.68rem;color:#8b949e;font-family:JetBrains Mono,monospace">{processed_at}</span></div>'
                        f'{field_html}{conf_bar}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f'<div class="r-card r-card-ok">'
                    f'<div class="r-filename">📄 {r["source_file"]} &nbsp;{badge}'
                    f'<span style="margin-left:auto;font-size:0.68rem;color:#8b949e;font-family:JetBrains Mono,monospace">{processed_at}</span></div>'
                    f'{field_html}{conf_bar}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Optional expandable sections
            if show_ocr and r["raw_ocr"]:
                with st.expander(f"Raw OCR — {r['source_file']}"):
                    st.markdown(
                        f'<div class="ocr-block">{r["raw_ocr"]}</div>',
                        unsafe_allow_html=True,
                    )

            if show_json:
                with st.expander(f"JSON output — {r['source_file']}"):
                    display_fields = {k: v for k, v in fields.items()
                                      if k not in ("raw_ocr_text", "_processed_at", "_completeness")}
                    st.markdown(
                        f'<div class="json-block">{json.dumps(display_fields, indent=2, ensure_ascii=False)}</div>',
                        unsafe_allow_html=True,
                    )


# ── Tab 2: Summary Table ──────────────────────────────────────────────────────
with tab_table:
    if not good:
        st.info("No successfully parsed receipts to display.")
    else:
        rows_html = ""
        for r in good:
            f     = r["fields"]
            score = f.get("_completeness", 0)
            store = f.get("store_name") or "<span style='color:#d0d7de'>—</span>"
            date  = f.get("date")       or "<span style='color:#d0d7de'>—</span>"
            total_val = f.get("total")
            try:
                total_disp = f"₱ {float(total_val):,.2f}"
            except (TypeError, ValueError):
                total_disp = total_val or "<span style='color:#d0d7de'>—</span>"

            badge_color = "#1a7f37" if score == 100 else ("#9a6700" if score >= conf_threshold else "#cf222e")
            badge_bg    = "#dafbe1" if score == 100 else ("#fff8c5" if score >= conf_threshold else "#ffebe9")

            rows_html += (
                f"<tr>"
                f"<td class='mono'>{r['source_file']}</td>"
                f"<td>{store}</td>"
                f"<td class='mono'>{date}</td>"
                f"<td class='mono' style='font-weight:600;color:#1a7f37'>{total_disp}</td>"
                f"<td><span style='background:{badge_bg};color:{badge_color};padding:2px 8px;"
                f"border-radius:10px;font-size:0.72rem;font-weight:600;"
                f"font-family:JetBrains Mono,monospace'>{score}%</span></td>"
                f"</tr>"
            )

        st.markdown(
            f"<div style='background:#fff;border:1px solid #e8ecf0;border-radius:12px;overflow:hidden'>"
            f"<table class='sum-table'>"
            f"<thead><tr>"
            f"<th>File</th><th>Store</th><th>Date</th><th>Total</th><th>Complete</th>"
            f"</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            f"</table></div>",
            unsafe_allow_html=True,
        )

        if totals:
            st.markdown(
                f"<div style='text-align:right;margin-top:12px;font-size:0.88rem;color:#8b949e'>"
                f"<b style='color:#0d1117'>Grand Total: </b>"
                f"<span style='font-family:JetBrains Mono,monospace;font-weight:700;"
                f"color:#1a7f37;font-size:1rem'>₱ {total_spend:,.2f}</span>"
                f"&nbsp; across {len(totals)} receipt(s)"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Tab 3: Export ─────────────────────────────────────────────────────────────
with tab_export:
    if not good:
        st.info("No results to export yet.")
    else:
        rows = []
        for r in good:
            row = {}
            for k, v in r["fields"].items():
                if k == "raw_ocr_text" and not include_ocr_in_csv:
                    continue
                if k.startswith("_"):
                    continue
                row[k] = v
            row["completeness_pct"] = r["fields"].get("_completeness", 0)
            row["processed_at"]     = r["fields"].get("_processed_at", "")
            rows.append(row)

        df = pd.DataFrame(rows)
        priority = ["source_file", "store_name", "date", "total", "completeness_pct", "processed_at"]
        other    = [c for c in df.columns if c not in priority]
        df = df[[c for c in priority if c in df.columns] + other]

        csv_bytes  = df.to_csv(index=False).encode("utf-8")
        json_bytes = json.dumps(
            [r["fields"] for r in good], indent=2, ensure_ascii=False
        ).encode("utf-8")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        st.markdown("<div class='section-title'>Download</div>", unsafe_allow_html=True)
        col_csv, col_json, col_spacer = st.columns([1, 1, 3])
        with col_csv:
            st.download_button(
                label="⬇ Download CSV",
                data=csv_bytes,
                file_name=f"receipts_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_json:
            st.download_button(
                label="⬇ Download JSON",
                data=json_bytes,
                file_name=f"receipts_{ts}.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("<div class='section-title'>Preview</div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
