"""
Receipt OCR Pipeline — Streamlit App
Stack: PaddleOCR (text extraction) → Qwen2-0.5B-Instruct (structured field mapping)
"""

import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Receipt OCR Pipeline",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — clean, functional, data-tool aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #2a2d3a;
  }
  section[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
  }

  /* Top bar header */
  .header-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 0 6px 0;
    border-bottom: 2px solid #2563eb;
    margin-bottom: 24px;
  }
  .header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #1e293b;
    letter-spacing: -0.5px;
  }
  .header-sub {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 300;
    margin-top: 2px;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* Status badges */
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  .badge-ok   { background: #dcfce7; color: #166534; }
  .badge-warn { background: #fef9c3; color: #854d0e; }
  .badge-err  { background: #fee2e2; color: #991b1b; }

  /* Receipt card */
  .receipt-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .receipt-filename {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #2563eb;
    margin-bottom: 12px;
    font-weight: 600;
  }
  .field-row {
    display: flex;
    gap: 8px;
    margin-bottom: 6px;
    align-items: baseline;
  }
  .field-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #94a3b8;
    width: 110px;
    flex-shrink: 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .field-value {
    font-size: 0.9rem;
    color: #0f172a;
    font-weight: 400;
  }
  .field-value-null {
    font-size: 0.85rem;
    color: #cbd5e1;
    font-style: italic;
  }

  /* Monospace OCR block */
  .ocr-block {
    background: #0f1117;
    color: #a3e635;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    padding: 14px 16px;
    border-radius: 8px;
    white-space: pre-wrap;
    max-height: 260px;
    overflow-y: auto;
    line-height: 1.6;
  }

  /* Progress bar strip */
  div[data-testid="stProgressBar"] > div > div {
    background: #2563eb !important;
  }

  /* Metric tweaks */
  [data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 18px;
  }

  /* Buttons */
  .stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: background 0.15s;
  }
  .stButton > button:hover {
    background: #1d4ed8;
    color: white;
  }

  /* Download button */
  .stDownloadButton > button {
    background: #f1f5f9;
    color: #1e293b;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
  }
  .stDownloadButton > button:hover {
    background: #e2e8f0;
  }
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
# Lazy-load heavy models (cached so they only load once)
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
# OCR & LLM helpers (identical logic to notebook, Streamlit-friendly)
# ─────────────────────────────────────────────────────────────────────────────
def run_ocr(image: Image.Image) -> str:
    """Run PaddleOCR on a PIL Image and return joined text."""
    ocr_engine = load_ocr_engine()

    # PaddleOCR needs a file path or numpy array
    import numpy as np
    img_array = np.array(image.convert("RGB"))
    result = ocr_engine.ocr(img_array, cls=True)

    lines = []
    if result and result[0]:
        for item in result[0]:
            text, _confidence = item[1]
            lines.append(text)
    return "\n".join(lines)


def _sanitize_llm_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    def _resolve_expr(m):
        parts = m.group(0).split("=")
        return parts[-1].strip()

    text = re.sub(r"[\d\s\.\+\-\*\/]+==[^,\}\]\\"'\n]+", _resolve_expr, text)

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
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()

    show_ocr_text = st.toggle("Show raw OCR text", value=False)
    show_image_preview = st.toggle("Show image preview", value=True)

    st.divider()
    st.markdown("**Model info**")
    st.markdown(
        """
        - **OCR:** PaddleOCR (English, angle-aware)
        - **LLM:** Qwen2-0.5B-Instruct
        - **Fields extracted:** store name, date, total
        """
    )
    st.divider()
    st.markdown("**How to use**")
    st.markdown(
        """
        1. Upload one or more receipt images
        2. Click **Run pipeline**
        3. Review results and download CSV
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <span style="font-size:2rem;">🧾</span>
  <div>
    <div class="header-title">Receipt OCR Pipeline</div>
    <div class="header-sub">PaddleOCR → Qwen2-0.5B-Instruct → structured CSV</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# File uploader
# ─────────────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload receipt images",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True,
    help="Supports JPG, PNG, BMP, TIFF. Upload as many as you need.",
)

if not uploaded_files:
    st.info("Upload one or more receipt images above to get started.", icon="📂")
    st.stop()

st.markdown(f"**{len(uploaded_files)} image(s) ready**")

# ─────────────────────────────────────────────────────────────────────────────
# Run button
# ─────────────────────────────────────────────────────────────────────────────
col_run, col_spacer = st.columns([1, 5])
with col_run:
    run = st.button("▶ Run pipeline", use_container_width=True)

if not run:
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────────────────────
results = []
errors  = []

progress_bar  = st.progress(0, text="Initialising models…")
status_text   = st.empty()

# Pre-load models before the loop (shows one spinner)
with st.spinner("Loading OCR engine and LLM (first run may take a minute)…"):
    try:
        load_ocr_engine()
        load_llm()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

total = len(uploaded_files)

for idx, uploaded_file in enumerate(uploaded_files):
    fname = uploaded_file.name
    status_text.markdown(f"Processing **{fname}** ({idx + 1} / {total})…")
    progress_bar.progress((idx) / total, text=f"{fname}")

    try:
        image = Image.open(uploaded_file).convert("RGB")

        # ── Step 1: OCR ────────────────────────────────────────────────────
        raw_text = run_ocr(image)

        # ── Step 2: LLM field extraction ───────────────────────────────────
        fields = extract_fields_with_llm(raw_text)

        if isinstance(fields.get("items"), list):
            fields["items"] = json.dumps(fields["items"])

        fields["source_file"]  = fname
        fields["raw_ocr_text"] = raw_text

        results.append({
            "source_file": fname,
            "image":       image,
            "raw_ocr":     raw_text,
            "fields":      fields,
            "error":       None,
        })

    except Exception as e:
        errors.append(fname)
        results.append({
            "source_file": fname,
            "image":       None,
            "raw_ocr":     "",
            "fields":      {},
            "error":       str(e),
        })

progress_bar.progress(1.0, text="Done!")
status_text.empty()


# ─────────────────────────────────────────────────────────────────────────────
# Summary metrics
# ─────────────────────────────────────────────────────────────────────────────
ok_count  = sum(1 for r in results if r["error"] is None)
err_count = len(errors)

m1, m2, m3 = st.columns(3)
m1.metric("Receipts processed", ok_count)
m2.metric("Fields per receipt", len(RECEIPT_FIELDS))
m3.metric("Errors", err_count)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Per-receipt results
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Results")

for r in results:
    with st.container():
        if r["error"]:
            st.markdown(
                f'<div class="receipt-card">'
                f'<div class="receipt-filename">{r["source_file"]}</div>'
                f'<span class="badge badge-err">ERROR</span>&nbsp; {r["error"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
            continue

        fields = r["fields"]

        # Build field rows HTML
        field_rows_html = ""
        for key in RECEIPT_FIELDS:
            val = fields.get(key)
            if val is None:
                val_html = '<span class="field-value-null">—</span>'
            else:
                val_html = f'<span class="field-value">{val}</span>'
            field_rows_html += (
                f'<div class="field-row">'
                f'<span class="field-label">{key}</span>'
                f'{val_html}'
                f'</div>'
            )

        # Extra unknown fields (LLM sometimes adds more)
        for key, val in fields.items():
            if key in RECEIPT_FIELDS or key in ("source_file", "raw_ocr_text"):
                continue
            val_html = f'<span class="field-value">{val}</span>'
            field_rows_html += (
                f'<div class="field-row">'
                f'<span class="field-label">{key}</span>'
                f'{val_html}'
                f'</div>'
            )

        # Two-column layout: image + fields
        if show_image_preview and r["image"] is not None:
            img_col, fields_col = st.columns([1, 2])
            with img_col:
                thumb = r["image"].copy()
                thumb.thumbnail((300, 450))
                st.image(thumb, use_column_width=True)
            with fields_col:
                st.markdown(
                    f'<div class="receipt-card" style="margin-top:0">'
                    f'<div class="receipt-filename">{r["source_file"]}</div>'
                    f'<span class="badge badge-ok">✓ Parsed</span>'
                    f'<div style="margin-top:14px">{field_rows_html}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<div class="receipt-card">'
                f'<div class="receipt-filename">{r["source_file"]}</div>'
                f'<span class="badge badge-ok">✓ Parsed</span>'
                f'<div style="margin-top:14px">{field_rows_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if show_ocr_text and r["raw_ocr"]:
            with st.expander(f"Raw OCR text — {r['source_file']}"):
                st.markdown(
                    f'<div class="ocr-block">{r["raw_ocr"]}</div>',
                    unsafe_allow_html=True,
                )

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# CSV download
# ─────────────────────────────────────────────────────────────────────────────
good_results = [r for r in results if r["error"] is None]

if good_results:
    rows = []
    for r in good_results:
        row = {k: v for k, v in r["fields"].items() if k != "image"}
        rows.append(row)

    df = pd.DataFrame(rows)

    # Prioritise useful columns first
    priority = ["source_file", "store_name", "date", "total", "raw_ocr_text"]
    other    = [c for c in df.columns if c not in priority]
    df = df[[c for c in priority if c in df.columns] + other]

    csv_bytes = df.to_csv(index=False).encode("utf-8")

    dl_col, preview_col = st.columns([1, 3])
    with dl_col:
        st.download_button(
            label="⬇ Download CSV",
            data=csv_bytes,
            file_name="receipt_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with preview_col:
        st.markdown(f"*{len(df)} row(s) — preview below*")

    preview_cols = [c for c in priority if c in df.columns and c != "raw_ocr_text"]
    st.dataframe(df[preview_cols], use_container_width=True, hide_index=True)
