"""
Streamlit web interface for the U-Net Image Denoiser.

Run:
    streamlit run app/streamlit_app.py
"""

import os
import io
import sys
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import UNetDenoiser


# ---------------- CONFIG ----------------
APP_TITLE = "Image Denoiser"
APP_SUBTITLE = "U-Net Convolutional Autoencoder"
MODEL_PATH = "models/autoencoder.pth"
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


# ---------------- STYLING ----------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1d2e 100%);
    }
    h1 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #e6e6e6;
    }
    .subtitle {
        color: #9aa0b4;
        font-size: 1.1rem;
        margin-top: -15px;
        margin-bottom: 25px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #2a2e45 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #2e3450;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .metric-label {
        color: #8b92b0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #4ade80;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-delta {
        color: #facc15;
        font-size: 0.9rem;
        margin-top: 4px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7c7ff5 0%, #a78bfa 100%);
        transform: translateY(-1px);
    }
    div[data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid #2e3450;
    }
    .upload-note {
        color: #9aa0b4;
        font-size: 0.9rem;
        padding: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetDenoiser().to(device)
    if not os.path.exists(MODEL_PATH):
        return None, device
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


# ---------------- HELPERS ----------------
def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma / 255.0, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def denoise(model, device, noisy_img):
    t = torch.from_numpy(noisy_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = model(t).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(out, 0.0, 1.0)


def compute_psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim(pred, target):
    return structural_similarity(target, pred, channel_axis=2, data_range=1.0)


def load_uploaded(file_bytes, size=256):
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    pil_img = pil_img.resize((size, size), Image.BICUBIC)
    return np.asarray(pil_img).astype(np.float32) / 255.0


def load_sample(path, size=256):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img.astype(np.float32) / 255.0


# ---------------- UI ----------------
st.markdown(f"<h1>✨ {APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">{APP_SUBTITLE} · Semester 3 Deep Learning Project</div>',
    unsafe_allow_html=True,
)

# Load model
model, device = load_model()
if model is None:
    st.error(
        "Model file not found at `models/autoencoder.pth`. "
        "Train the model first: `python src/train.py`"
    )
    st.stop()

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    sigma = st.radio(
        "Noise level (sigma)",
        options=[15, 25, 50],
        index=1,
        format_func=lambda x: {
            15: "Mild (σ=15)",
            25: "Moderate (σ=25)",
            50: "Severe (σ=50)",
        }[x],
    )
    st.markdown("---")
    st.caption("U-Net · 1.93M parameters")
    st.caption("Trained on Div2K_Random100 · 40 epochs")
    st.caption(f"Device: `{device}`")


# Input mode
st.markdown("### 1. Choose an image")
mode = st.radio(
    "Input mode",
    ["📤 Upload your own", "🖼️ Pick a sample"],
    horizontal=True,
    label_visibility="collapsed",
)

clean_img = None
source_name = None

if mode == "📤 Upload your own":
    uploaded = st.file_uploader(
        "Drop a JPG or PNG",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        clean_img = load_uploaded(uploaded.getvalue())
        source_name = uploaded.name
    else:
        st.markdown(
            '<div class="upload-note">No file uploaded yet. '
            "Or switch to <b>Pick a sample</b> to try an image from the dataset.</div>",
            unsafe_allow_html=True,
        )
else:
    sample_files = sorted([
        f for f in os.listdir(SAMPLES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if not sample_files:
        st.warning("No sample images found in app/samples/")
        st.stop()

    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = sample_files[0]

    cols = st.columns(len(sample_files))
    for i, fname in enumerate(sample_files):
        with cols[i]:
            img = Image.open(os.path.join(SAMPLES_DIR, fname))
            st.image(img, use_container_width=True)
            if st.button(f"Use {fname}", key=f"btn_{fname}"):
                st.session_state.selected_sample = fname
                st.rerun()

    clean_img = load_sample(
        os.path.join(SAMPLES_DIR, st.session_state.selected_sample)
    )
    source_name = st.session_state.selected_sample


# Denoise button
if clean_img is not None:
    st.markdown(f"*Selected: `{source_name}`*")

    if st.button("🚀 Denoise Image"):
        with st.spinner("Running U-Net inference..."):
            noisy_img = add_gaussian_noise(clean_img, sigma)
            denoised_img = denoise(model, device, noisy_img)

            psnr_noisy = compute_psnr(noisy_img, clean_img)
            psnr_denoised = compute_psnr(denoised_img, clean_img)
            ssim_noisy = compute_ssim(noisy_img, clean_img)
            ssim_denoised = compute_ssim(denoised_img, clean_img)

        st.markdown("### 2. Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Noisy Input**")
            st.image(noisy_img, use_container_width=True)
        with col2:
            st.markdown("**Denoised (U-Net)**")
            st.image(denoised_img, use_container_width=True)
        with col3:
            st.markdown("**Original (Clean)**")
            st.image(clean_img, use_container_width=True)

        st.markdown("### 3. Metrics")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PSNR · Noisy</div>
                <div class="metric-value">{psnr_noisy:.2f}</div>
                <div class="metric-delta">dB</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            delta = psnr_denoised - psnr_noisy
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PSNR · Denoised</div>
                <div class="metric-value">{psnr_denoised:.2f}</div>
                <div class="metric-delta">+{delta:.2f} dB</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">SSIM · Noisy</div>
                <div class="metric-value">{ssim_noisy:.4f}</div>
                <div class="metric-delta">&nbsp;</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            delta_s = ssim_denoised - ssim_noisy
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">SSIM · Denoised</div>
                <div class="metric-value">{ssim_denoised:.4f}</div>
                <div class="metric-delta">+{delta_s:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption(
            "U-Net trained on Div2K_Random100 · 40 epochs · sigma=25"
        )