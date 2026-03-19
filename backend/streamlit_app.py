import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import matplotlib.pyplot as plt

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PulmoScan · Respiratory AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #060a0f !important;
    font-family: 'DM Mono', monospace;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,200,150,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0,120,255,0.05) 0%, transparent 60%),
        #060a0f !important;
}
[data-testid="stHeader"], footer { display: none !important; }
[data-testid="block-container"] {
    padding: 0 2rem 2rem !important;
    max-width: 1200px !important;
}

/* ── Streamlit bordered containers → panels ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 18px !important;
    padding: 4px !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stVerticalBlockBorderWrapper"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,229,160,0.45), transparent);
    border-radius: 18px 18px 0 0;
    z-index: 1;
}

/* ── Nav ── */
.nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 0;
}
.nav-brand { display: flex; align-items: center; gap: 12px; }
.nav-logo {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #00e5a0, #00a8ff);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    animation: pulse-ring 2.5s infinite;
}
@keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0    rgba(0,229,160,0.35); }
    70%  { box-shadow: 0 0 0 10px rgba(0,229,160,0);    }
    100% { box-shadow: 0 0 0 0    rgba(0,229,160,0);    }
}
.nav-name {
    font-family: 'Syne', sans-serif;
    font-weight: 800; font-size: 20px;
    color: #fff; letter-spacing: -0.3px;
}
.nav-tag {
    font-size: 11px; color: rgba(255,255,255,0.3);
    letter-spacing: 2px; text-transform: uppercase;
}
.nav-badge {
    background: rgba(0,229,160,0.1);
    border: 1px solid rgba(0,229,160,0.25);
    color: #00e5a0; font-size: 11px;
    padding: 4px 14px; border-radius: 100px; letter-spacing: 1px;
}

/* ── Hero ── */
.hero { padding: 48px 0 32px; }
.hero-eyebrow {
    font-size: 11px; letter-spacing: 3px;
    text-transform: uppercase; color: #00e5a0;
    margin-bottom: 14px;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(34px, 4.5vw, 56px);
    color: #fff; line-height: 1.1;
    letter-spacing: -1px; margin-bottom: 14px;
}
.hero-title em { font-style: italic; color: #00e5a0; }
.hero-sub {
    font-size: 13px; color: rgba(255,255,255,0.38);
    max-width: 460px; line-height: 1.8; font-weight: 300;
}

/* ── Labels ── */
.panel-label {
    font-size: 10px; letter-spacing: 3px;
    text-transform: uppercase; color: rgba(255,255,255,0.28);
    margin-bottom: 4px;
}
.section-title {
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 12px; letter-spacing: 2px;
    text-transform: uppercase; color: rgba(255,255,255,0.4);
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 16px;
}
.section-title::after {
    content: ''; flex: 1; height: 1px;
    background: rgba(255,255,255,0.06);
}

/* ── Text input ── */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 14px !important;
    padding: 14px 18px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    caret-color: #00e5a0 !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(0,229,160,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,229,160,0.08) !important;
    outline: none !important;
}
[data-testid="stTextInput"] label {
    font-size: 11px !important; letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.38) !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 400 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(0,229,160,0.3) !important;
    background: rgba(0,229,160,0.03) !important;
}
[data-testid="stFileUploader"] label {
    color: rgba(255,255,255,0.45) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important; letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #00e5a0, #00c87a) !important;
    color: #060a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 13px !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
    padding: 14px 40px !important;
    border-radius: 100px !important; border: none !important;
    box-shadow: 0 8px 28px rgba(0,229,160,0.22) !important;
    transition: all 0.15s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 36px rgba(0,229,160,0.35) !important;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
    margin: 32px 0;
}

/* ── Result card (pure HTML, no widgets inside) ── */
.result-card {
    border-radius: 16px; padding: 28px;
    position: relative; overflow: hidden;
}
.result-card.healthy {
    background: linear-gradient(135deg, rgba(0,229,160,0.09), rgba(0,200,130,0.03));
    border: 1px solid rgba(0,229,160,0.2);
}
.result-card.disease {
    background: linear-gradient(135deg, rgba(255,80,80,0.09), rgba(200,40,40,0.03));
    border: 1px solid rgba(255,80,80,0.2);
}
.result-eyebrow {
    font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
    color: rgba(255,255,255,0.3); margin-bottom: 10px;
}
.result-diagnosis {
    font-family: 'DM Serif Display', serif;
    font-size: 52px; line-height: 1;
    letter-spacing: -1.5px; margin-bottom: 10px;
}
.result-diagnosis.healthy { color: #00e5a0; }
.result-diagnosis.disease { color: #ff6060; }
.result-confidence {
    font-size: 13px; color: rgba(255,255,255,0.4); margin-bottom: 16px;
}
.result-confidence span { color: rgba(255,255,255,0.85); font-weight: 500; }
.stat-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px; padding: 6px 14px;
    font-size: 12px; color: rgba(255,255,255,0.45);
}
.stat-dot { width: 6px; height: 6px; border-radius: 50%; background: #00e5a0; flex-shrink: 0; }
.stat-dot.red { background: #ff6060; }

/* ── Probability bars ── */
.prob-row { display: flex; align-items: center; gap: 12px; margin-bottom: 11px; }
.prob-label {
    font-size: 11px; font-family: 'DM Mono', monospace;
    color: rgba(255,255,255,0.45); width: 110px; flex-shrink: 0;
}
.prob-bar-track {
    flex: 1; height: 4px;
    background: rgba(255,255,255,0.06); border-radius: 100px; overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 100px; }
.prob-value {
    font-size: 11px; font-family: 'DM Mono', monospace;
    color: rgba(255,255,255,0.3); width: 36px; text-align: right; flex-shrink: 0;
}

/* ── Misc ── */
audio { width: 100%; border-radius: 100px; opacity: 0.75; }
[data-testid="stAlert"] {
    background: rgba(255,200,0,0.06) !important;
    border: 1px solid rgba(255,200,0,0.15) !important;
    border-radius: 12px !important;
    color: rgba(255,200,0,0.8) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: rgba(0,229,160,0.2); border-radius: 100px; }
</style>
""", unsafe_allow_html=True)


# ─── Model ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

LABELS = ['Asthma','Bronchiectasis','Bronchiolitis','COPD','Healthy','LRTI','Pneumonia','URTI']

LABEL_INFO = {
    'Asthma':         ('Chronic airway inflammation causing wheezing.',      '#f0a500'),
    'Bronchiectasis': ('Permanent widening of bronchial tubes.',             '#e06c75'),
    'Bronchiolitis':  ('Inflammation of small bronchioles.',                 '#e06c75'),
    'COPD':           ('Progressive obstruction of airflow in the lungs.',   '#e06c75'),
    'Healthy':        ('No respiratory abnormalities detected.',             '#00e5a0'),
    'LRTI':           ('Lower respiratory tract infection.',                 '#e06c75'),
    'Pneumonia':      ('Lung infection — alveoli filled with fluid.',        '#e06c75'),
    'URTI':           ('Upper respiratory tract infection.',                 '#f0a500'),
}


# ─── Feature Extraction ──────────────────────────────────────────────────────
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    target_length = 5 * sr
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)

    def fix(x):
        return np.pad(x, ((0,0),(0, max(0, 259 - x.shape[1]))))[:, :259]

    mfcc   = np.expand_dims(fix(mfcc),   axis=-1)
    chroma = np.expand_dims(fix(chroma), axis=-1)
    mel    = np.expand_dims(fix(mel),    axis=-1)
    return mfcc, chroma, mel, audio, sr


# ─── Plot helper ─────────────────────────────────────────────────────────────
def style_ax(ax, title):
    ax.set_facecolor('#0c1117')
    ax.set_title(title, fontsize=9, color='#667788',
                 fontfamily='monospace', pad=8, fontweight='normal')
    ax.tick_params(colors='#445566', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2535')


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Nav ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav">
  <div class="nav-brand">
    <div class="nav-logo">🫁</div>
    <div>
      <div class="nav-name">Respiratory Classifier</div>
      <div class="nav-tag">Respiratory AI</div>
    </div>
  </div>
  <div class="nav-badge">● SYSTEM READY</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">// AI-powered auscultation analysis</div>
  <h1 class="hero-title">Detect respiratory<br><em>disorders</em> from sound.</h1>
  <p class="hero-sub">
    Upload a lung auscultation recording and receive an instant AI-powered
    differential across 8 respiratory conditions.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Input Panel ──────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="panel-label">// Patient intake</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        name = st.text_input("Patient Name", placeholder="e.g. John Doe")
    with col_b:
        uploaded_file = st.file_uploader("Auscultation Recording (.wav)", type=["wav"])
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("⬡  Run Analysis")

# ── Results ──────────────────────────────────────────────────────────────────
if run:
    if not uploaded_file or not name:
        st.warning("Please enter the patient name and upload a .wav recording.")
    else:
        with st.spinner("Analysing auscultation patterns…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            mfcc, chroma, mel, audio, sr = extract_features(tmp_path)
            pred = model.predict([
                np.expand_dims(mfcc,   0),
                np.expand_dims(chroma, 0),
                np.expand_dims(mel,    0),
            ])
            probs      = pred[0]
            idx        = int(np.argmax(probs))
            label      = LABELS[idx]
            conf       = float(np.max(probs))
            info, accent = LABEL_INFO[label]
            is_healthy = label == "Healthy"

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Row 1: Diagnosis card + Playback panel ───────────────────────────
        c1, c2 = st.columns([3, 2], gap="large")

        with c1:
            card_cls = "healthy" if is_healthy else "disease"
            diag_cls = "healthy" if is_healthy else "disease"
            dot_cls  = ""        if is_healthy else "red"
            st.markdown(f"""
            <div class="result-card {card_cls}">
              <div class="result-eyebrow">Primary Diagnosis · {name}</div>
              <div class="result-diagnosis {diag_cls}">{label}</div>
              <div class="result-confidence">Confidence: <span>{conf*100:.1f}%</span></div>
              <div class="stat-pill">
                <div class="stat-dot {dot_cls}"></div>
                {info}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            with st.container(border=True):
                st.markdown('<div class="panel-label">// Playback</div>', unsafe_allow_html=True)
                uploaded_file.seek(0)
                st.audio(uploaded_file)

                st.markdown("<br>", unsafe_allow_html=True)

                # Confidence meter
                fig_c, ax_c = plt.subplots(figsize=(4, 1.2))
                fig_c.patch.set_facecolor('#0c1117')
                ax_c.barh([' '], [1],    color='#1a2535', height=0.4)
                ax_c.barh([' '], [conf], color=accent,    height=0.4)
                ax_c.set_xlim(0, 1)
                ax_c.set_xticks([0, 0.5, 1])
                ax_c.set_xticklabels(['0%', '50%', '100%'], fontsize=7, color='#445566')
                ax_c.set_yticks([])
                style_ax(ax_c, f"Confidence — {conf*100:.1f}%")
                fig_c.tight_layout(pad=0.5)
                st.pyplot(fig_c, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 2: Probabilities + Signal Analysis ───────────────────────────
        c3, c4 = st.columns([1, 1], gap="large")

        with c3:
            with st.container(border=True):
                st.markdown('<div class="section-title">Differential Probabilities</div>',
                            unsafe_allow_html=True)
                sorted_idx = np.argsort(probs)[::-1]
                bars_html = ""
                for i in sorted_idx:
                    p         = float(probs[i])
                    is_top    = (i == idx)
                    fill      = accent if is_top else "#1e3048"
                    lbl_color = "#ffffff" if is_top else "rgba(255,255,255,0.45)"
                    val_color = "#ffffff" if is_top else "rgba(255,255,255,0.3)"
                    bars_html += f"""
                    <div class="prob-row">
                      <div class="prob-label" style="color:{lbl_color};">{LABELS[i]}</div>
                      <div class="prob-bar-track">
                        <div class="prob-bar-fill"
                             style="width:{p*100:.1f}%;background:{fill};"></div>
                      </div>
                      <div class="prob-value" style="color:{val_color};">{p*100:.0f}%</div>
                    </div>"""
                st.markdown(bars_html, unsafe_allow_html=True)

        with c4:
            with st.container(border=True):
                st.markdown('<div class="section-title">Signal Analysis</div>',
                            unsafe_allow_html=True)

                fig2, axes = plt.subplots(2, 1, figsize=(5, 4.0))
                fig2.patch.set_facecolor('#0c1117')

                # Waveform
                t = np.linspace(0, len(audio) / sr, num=len(audio))
                axes[0].set_facecolor('#0c1117')
                axes[0].plot(t, audio, color='#00e5a0', linewidth=0.4, alpha=0.85)
                axes[0].fill_between(t, audio, alpha=0.08, color='#00e5a0')
                style_ax(axes[0], 'Waveform')

                # MFCC
                mfcc_disp = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
                axes[1].set_facecolor('#0c1117')
                axes[1].imshow(mfcc_disp, aspect='auto', origin='lower',
                               cmap='inferno', interpolation='bilinear')
                style_ax(axes[1], 'MFCC Coefficients')

                fig2.tight_layout(pad=1.2)
                st.pyplot(fig2, use_container_width=True)

        # ── Footer ───────────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; margin-top:48px; padding-bottom:32px;
                    font-size:11px; color:rgba(255,255,255,0.14);
                    font-family:'DM Mono',monospace; letter-spacing:1px;">
            For research and clinical decision support only ·
            Not a substitute for professional medical advice
        </div>
        """, unsafe_allow_html=True)