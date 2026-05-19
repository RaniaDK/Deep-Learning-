"""
╔══════════════════════════════════════════════════════════════╗
║   🛡️  Theft Detection — Streamlit Inference App              ║
║   Projet Deep Learning · M. Abdallah Khemais                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import time
import os
import gdown
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Theft Detection — DL",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Fond principal */
.stApp {
    background: #060A14;
    color: #F9FAFB;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1F2937;
}

/* Boutons */
.stButton > button {
    background: linear-gradient(135deg, #EF4444, #B91C1C);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 13px;
    padding: 10px 24px;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 20px #EF444466;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0F172A;
    border: 2px dashed #1F2937;
    border-radius: 12px;
    padding: 20px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #EF4444;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #0F172A;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 14px;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #6B7280;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #EF4444;
    border-bottom-color: #EF4444;
}

/* Headers */
h1, h2, h3 { font-family: 'Sora', sans-serif; font-weight: 800; }

/* Alert / info boxes */
.theft-alert {
    background: #450A0A;
    border: 1.5px solid #EF4444;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    animation: pulseRed 2s infinite;
}
.normal-alert {
    background: #052E16;
    border: 1.5px solid #10B981;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 0 0 #EF444433; }
    50%       { box-shadow: 0 0 0 12px #EF444400; }
}

/* Code-style text */
.mono { font-family: 'Space Mono', monospace; font-size: 11px; color: #6B7280; }
.tag  { 
    font-family: 'Space Mono', monospace; font-size: 10px;
    background: #1F2937; color: #9CA3AF;
    padding: 2px 8px; border-radius: 4px;
}

/* Progress bar personnalisée */
.stProgress > div > div > div { background: #EF4444; }

/* Selectbox */
[data-testid="stSelectbox"] { font-family: 'Space Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONSTANTES & CONFIG
# ══════════════════════════════════════════════════════════════
CLASS_NAMES   = ["normal", "theft"]
IMG_SIZE      = 224
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ── IDs Google Drive pour télécharger les modèles ─────────────
# Remplace ces IDs par tes vrais IDs après upload sur Drive
GDRIVE_IDS = {
    "Baseline CNN":     "TON_ID_GDRIVE_baseline",
    "ResNet-50":        "TON_ID_GDRIVE_resnet50",
    "EfficientNet-B0":  "TON_ID_GDRIVE_efficientnet",
    "MobileNetV3":      "TON_ID_GDRIVE_mobilenet",
    "VGG-16":           "TON_ID_GDRIVE_vgg16",
}

MODEL_FILES = {
    "Baseline CNN":     "baseline_cnn.pth",
    "ResNet-50":        "resnet50.pth",
    "EfficientNet-B0":  "efficientnet_b0.pth",
    "MobileNetV3":      "mobilenet_v3.pth",
    "VGG-16":           "vgg16.pth",
}

MODEL_COLORS = {
    "Baseline CNN":    "#6B7280",
    "ResNet-50":       "#3B82F6",
    "EfficientNet-B0": "#8B5CF6",
    "MobileNetV3":     "#F59E0B",
    "VGG-16":          "#EF4444",
}

# ══════════════════════════════════════════════════════════════
# ARCHITECTURES (doivent être identiques au training)
# ══════════════════════════════════════════════════════════════
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 128),     nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))


def build_model(arch: str) -> nn.Module:
    """Reconstruit l'architecture identique à l'entraînement."""
    num_classes = 2
    if arch == "Baseline CNN":
        return SimpleCNN(num_classes)

    elif arch == "ResNet-50":
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(
            nn.Linear(m.fc.in_features, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        return m

    elif arch == "EfficientNet-B0":
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 128),
            nn.ReLU(),       nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
        return m

    elif arch == "MobileNetV3":
        m = models.mobilenet_v3_small(weights=None)
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Sequential(
            nn.Linear(in_f, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
        return m

    elif arch == "VGG-16":
        m = models.vgg16(weights=None)
        in_f = m.classifier[6].in_features
        m.classifier[6] = nn.Sequential(
            nn.Linear(in_f, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes)
        )
        return m

    raise ValueError(f"Modèle inconnu : {arch}")


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DU MODÈLE (avec cache)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model(arch: str) -> nn.Module:
    """Charge le modèle depuis le fichier .pth local."""
    path = MODELS_DIR / MODEL_FILES[arch]

    # ── Téléchargement depuis Google Drive si absent ──────────
    if not path.exists():
        gdrive_id = GDRIVE_IDS.get(arch)
        if gdrive_id and not gdrive_id.startswith("TON_"):
            with st.spinner(f"⬇️ Téléchargement {arch} depuis Google Drive..."):
                url = f"https://drive.google.com/uc?id={gdrive_id}"
                gdown.download(url, str(path), quiet=False)
        else:
            return None  # Pas encore uploadé

    model = build_model(arch)
    ckpt  = torch.load(path, map_location=DEVICE)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    return model


# ══════════════════════════════════════════════════════════════
# INFÉRENCE
# ══════════════════════════════════════════════════════════════
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(model: nn.Module, img: Image.Image):
    """Retourne (classe, confiance, probas, temps_ms)."""
    tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        logits = model(tensor)
        probas = torch.softmax(logits, dim=1)[0].cpu().numpy()
    elapsed_ms = (time.time() - t0) * 1000
    pred_idx   = int(np.argmax(probas))
    return CLASS_NAMES[pred_idx], float(probas[pred_idx]), probas, elapsed_ms


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
        <div style='font-size:36px; margin-bottom:6px;'>🛡️</div>
        <div style='font-family:Sora; font-weight:800; font-size:15px; color:#F9FAFB;'>
            Theft Detection
        </div>
        <div style='font-family:Space Mono,monospace; font-size:10px; color:#6B7280; margin-top:4px;'>
            Deep Learning · Inférence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Sélection du modèle ───────────────────────────────────
    st.markdown('<p style="font-family:Space Mono,monospace;font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">MODÈLE</p>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        label="",
        options=list(MODEL_FILES.keys()),
        label_visibility="collapsed"
    )

    color = MODEL_COLORS[selected_model]
    st.markdown(f"""
    <div style='background:#111827;border:1px solid {color}44;border-radius:8px;padding:12px;margin:10px 0;'>
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>
            <div style='width:8px;height:8px;border-radius:50%;background:{color};box-shadow:0 0 6px {color};'></div>
            <span style='font-family:Space Mono,monospace;font-size:11px;color:{color};font-weight:700;'>{selected_model}</span>
        </div>
        <div style='font-family:Space Mono,monospace;font-size:10px;color:#4B5563;'>
            Device : {"🟢 GPU" if DEVICE.type == "cuda" else "🔵 CPU"} · {str(DEVICE).upper()}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Classes ───────────────────────────────────────────────
    st.markdown('<p style="font-family:Space Mono,monospace;font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">CLASSES</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style='display:flex;flex-direction:column;gap:6px;'>
        <div style='background:#052E16;border:1px solid #10B98144;border-radius:6px;padding:8px 12px;display:flex;align-items:center;gap:8px;'>
            <div style='width:8px;height:8px;border-radius:50%;background:#10B981;'></div>
            <span style='font-family:Space Mono,monospace;font-size:11px;color:#10B981;'>normal</span>
            <span style='font-family:Space Mono,monospace;font-size:10px;color:#374151;margin-left:auto;'>class 0</span>
        </div>
        <div style='background:#450A0A;border:1px solid #EF444444;border-radius:6px;padding:8px 12px;display:flex;align-items:center;gap:8px;'>
            <div style='width:8px;height:8px;border-radius:50%;background:#EF4444;'></div>
            <span style='font-family:Space Mono,monospace;font-size:11px;color:#EF4444;'>theft</span>
            <span style='font-family:Space Mono,monospace;font-size:10px;color:#374151;margin-left:auto;'>class 1</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Infos projet ──────────────────────────────────────────
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:10px;color:#4B5563;line-height:1.8;'>
        <div>📦 Dataset : 9 747 images</div>
        <div>🏋️ Train  : 8 771 images</div>
        <div>✅ Valid  : 600 images</div>
        <div>🧪 Test   : 376 images</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style='display:flex;align-items:center;gap:14px;margin-bottom:24px;'>
    <div style='font-size:32px;'>🛡️</div>
    <div>
        <h1 style='color:#F9FAFB;margin:0;font-size:24px;'>Theft & Shoplifting Detection</h1>
        <p style='color:#6B7280;font-family:Space Mono,monospace;font-size:11px;margin:0;'>
            Inférence en ligne · Module Deep Learning
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab_infer, tab_results, tab_about = st.tabs([
    "🔍  Inférence", "📊  Résultats & Comparaison", "📋  À propos du projet"
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — INFÉRENCE
# ══════════════════════════════════════════════════════════════
with tab_infer:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<p style="font-family:Space Mono,monospace;font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">CHARGER UNE IMAGE</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            label="",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
            help="Image de caméra de surveillance (JPG, PNG, WEBP)"
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption=f"📷 {uploaded.name}", use_container_width=True)

            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            run_btn = st.button("⚡  Analyser l'image", type="primary")
        else:
            st.markdown("""
            <div style='text-align:center;padding:40px;color:#374151;font-family:Space Mono,monospace;font-size:11px;'>
                <div style='font-size:40px;margin-bottom:12px;'>📷</div>
                Glissez une image de vidéosurveillance<br>ou cliquez pour parcourir
            </div>
            """, unsafe_allow_html=True)
            run_btn = False

    with col_result:
        st.markdown('<p style="font-family:Space Mono,monospace;font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">RÉSULTAT</p>', unsafe_allow_html=True)

        if uploaded and run_btn:
            # ── Chargement modèle ─────────────────────────────
            with st.spinner(f"Chargement {selected_model}..."):
                model = load_model(selected_model)

            if model is None:
                st.error(f"⚠️ Modèle **{selected_model}** non disponible.\n\nUploadez le fichier `.pth` dans le dossier `models/` ou configurez Google Drive.")
            else:
                # ── Inférence ─────────────────────────────────
                with st.spinner("Analyse en cours..."):
                    pred_class, confidence, probas, elapsed_ms = predict(model, img)

                # ── Résultat principal ────────────────────────
                if pred_class == "theft":
                    st.markdown(f"""
                    <div class="theft-alert">
                        <div style='font-size:48px;margin-bottom:8px;'>🚨</div>
                        <div style='font-family:Sora,sans-serif;font-weight:800;font-size:28px;color:#EF4444;'>VOL DÉTECTÉ</div>
                        <div style='font-family:Space Mono,monospace;font-size:22px;color:#FCA5A5;margin-top:6px;'>
                            {confidence*100:.1f}% de confiance
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="normal-alert">
                        <div style='font-size:48px;margin-bottom:8px;'>✅</div>
                        <div style='font-family:Sora,sans-serif;font-weight:800;font-size:28px;color:#10B981;'>COMPORTEMENT NORMAL</div>
                        <div style='font-family:Space Mono,monospace;font-size:22px;color:#6EE7B7;margin-top:6px;'>
                            {confidence*100:.1f}% de confiance
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

                # ── Métriques rapides ─────────────────────────
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Modèle", selected_model.split("-")[0])
                with m2:
                    st.metric("Confiance", f"{confidence*100:.1f}%")
                with m3:
                    st.metric("Temps", f"{elapsed_ms:.0f} ms")

                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

                # ── Graphique probabilités ────────────────────
                fig = go.Figure()
                bar_colors = ["#10B981" if c == "normal" else "#EF4444" for c in CLASS_NAMES]
                fig.add_trace(go.Bar(
                    x=CLASS_NAMES,
                    y=[p * 100 for p in probas],
                    marker_color=bar_colors,
                    marker_line_width=0,
                    text=[f"{p*100:.1f}%" for p in probas],
                    textposition="auto",
                    textfont=dict(family="Space Mono", color="white", size=13),
                ))
                fig.update_layout(
                    title=dict(text="Probabilités par classe", font=dict(family="Sora", color="#F9FAFB", size=13)),
                    paper_bgcolor="#0F172A",
                    plot_bgcolor="#0F172A",
                    font=dict(family="Space Mono", color="#9CA3AF"),
                    xaxis=dict(showgrid=False, color="#6B7280"),
                    yaxis=dict(showgrid=True, gridcolor="#1F2937", range=[0, 105], ticksuffix="%", color="#6B7280"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=220,
                )
                st.plotly_chart(fig, use_container_width=True)

        elif not uploaded:
            st.markdown("""
            <div style='background:#0F172A;border:1px solid #1F2937;border-radius:12px;
                        padding:60px 20px;text-align:center;color:#374151;'>
                <div style='font-size:36px;margin-bottom:12px;'>⚡</div>
                <div style='font-family:Space Mono,monospace;font-size:12px;'>
                    Chargez une image pour démarrer l'analyse
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — RÉSULTATS & COMPARAISON
# ══════════════════════════════════════════════════════════════
with tab_results:
    st.markdown('<p style="font-family:Space Mono,monospace;font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;">COMPARAISON DES MODÈLES — Test Set</p>', unsafe_allow_html=True)

    # ── Données de résultats (à remplacer par tes vrais chiffres après entraînement) ──
    results_data = {
        "Modèle":       ["Baseline CNN", "ResNet-50", "EfficientNet-B0", "MobileNetV3", "VGG-16", "ResNet-50+FT"],
        "Test Acc":     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # ← remplir après entraînement
        "Test Loss":    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # ← remplir après entraînement
        "Best Val Acc": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # ← remplir après entraînement
        "Params":       ["~2.1M", "~25M", "~5.3M", "~2.5M", "~138M", "~25M"],
        "LR":           ["1e-3", "3e-4", "3e-4", "3e-4", "1e-4", "1e-5"],
    }

    colors_list = ["#6B7280","#3B82F6","#8B5CF6","#F59E0B","#EF4444","#10B981"]

    # ── Affichage si résultats disponibles ───────────────────
    has_results = any(v != 0.0 for v in results_data["Test Acc"])

    if not has_results:
        st.info("ℹ️ Les résultats s'afficheront ici après l'entraînement. Mettez à jour `results_data` dans `app.py` avec vos métriques réelles.")

        # ── Placeholder graphique de démonstration ────────────
        st.markdown("**Exemple de visualisation (données simulées) :**")
        demo_accs = [0.72, 0.88, 0.86, 0.84, 0.87, 0.91]
        results_data["Test Acc"] = demo_accs

    # ── Bar chart Test Accuracy ───────────────────────────────
    best_acc = max(results_data["Test Acc"])
    bar_colors = [colors_list[i] for i in range(len(results_data["Modèle"]))]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=results_data["Modèle"],
        y=[a * 100 for a in results_data["Test Acc"]],
        marker_color=bar_colors,
        marker_line_width=0,
        text=[f"{a*100:.1f}%" for a in results_data["Test Acc"]],
        textposition="outside",
        textfont=dict(family="Space Mono", color="white", size=11),
    ))
    fig_bar.add_hline(
        y=best_acc * 100, line_dash="dash", line_color="#F59E0B",
        annotation_text=f"Best: {best_acc*100:.1f}%",
        annotation_font=dict(color="#F59E0B", family="Space Mono"),
    )
    fig_bar.update_layout(
        title=dict(text="Test Accuracy par Modèle", font=dict(family="Sora", color="#F9FAFB", size=14)),
        paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
        font=dict(family="Space Mono", color="#9CA3AF"),
        xaxis=dict(showgrid=False, color="#6B7280"),
        yaxis=dict(showgrid=True, gridcolor="#1F2937", range=[0, 110], ticksuffix="%", color="#6B7280"),
        margin=dict(l=10, r=10, t=50, b=10), height=320,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Tableau comparatif ────────────────────────────────────
    st.markdown('<p style="font-family:Space Mono,monospace;font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px;">TABLEAU DÉTAILLÉ</p>', unsafe_allow_html=True)

    best_model_name = results_data["Modèle"][results_data["Test Acc"].index(best_acc)]
    for i, (name, acc, params, lr) in enumerate(zip(
        results_data["Modèle"], results_data["Test Acc"],
        results_data["Params"], results_data["LR"]
    )):
        is_best = (name == best_model_name)
        c = colors_list[i]
        badge = "🏆 MEILLEUR" if is_best else ""
        st.markdown(f"""
        <div style='background:{"#0F172A" if not is_best else "#111827"};
                    border:{"1.5px solid "+c if is_best else "1px solid #1F2937"};
                    border-radius:10px;padding:12px 16px;margin-bottom:8px;
                    display:flex;align-items:center;gap:16px;
                    box-shadow:{"0 0 16px "+c+"33" if is_best else "none"};'>
            <div style='width:10px;height:10px;border-radius:50%;background:{c};flex-shrink:0;
                        box-shadow:{"0 0 8px "+c if is_best else "none"};'></div>
            <div style='flex:1;'>
                <span style='font-family:Sora,sans-serif;font-weight:700;font-size:13px;color:#F9FAFB;'>{name}</span>
                <span style='font-family:Space Mono,monospace;font-size:10px;color:#4B5563;margin-left:10px;'>LR={lr} · {params}</span>
            </div>
            <div style='font-family:Space Mono,monospace;font-size:16px;font-weight:700;color:{c};'>
                {acc*100:.1f}%
            </div>
            <div style='font-family:Space Mono,monospace;font-size:10px;color:#F59E0B;min-width:80px;text-align:right;'>
                {badge}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Pipeline P4 + P5 résumé ───────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    col_p4, col_p5 = st.columns(2, gap="medium")

    with col_p4:
        st.markdown("""
        <div style='background:#0F172A;border:1px solid #10B98144;border-radius:12px;padding:18px;'>
            <div style='font-family:Space Mono,monospace;font-size:10px;color:#10B981;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;'>P4 — BALANCE</div>
            <div style='font-size:13px;color:#E5E7EB;margin-bottom:8px;font-weight:600;'>Class Weights</div>
            <div style='font-family:Space Mono,monospace;font-size:10px;color:#6B7280;line-height:1.8;'>
                ✓ WeightedRandomSampler<br>
                ✓ CrossEntropy + class weights<br>
                ✓ Label smoothing = 0.1<br>
                ✓ Subset équilibré 1 500/classe
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_p5:
        st.markdown("""
        <div style='background:#0F172A;border:1px solid #8B5CF644;border-radius:12px;padding:18px;'>
            <div style='font-family:Space Mono,monospace;font-size:10px;color:#8B5CF6;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;'>P5 — FINE-TUNING</div>
            <div style='font-size:13px;color:#E5E7EB;margin-bottom:8px;font-weight:600;'>Progressive Unfreeze</div>
            <div style='font-family:Space Mono,monospace;font-size:10px;color:#6B7280;line-height:1.8;'>
                ✓ Phase 1 : Head · lr=3e-4 · 4ep<br>
                ✓ Phase 2 : Blocs partiels · lr=1e-4 · 4ep<br>
                ✓ Phase 3 : Full unfreeze · lr=1e-5 · 6ep<br>
                ✓ CosineAnnealingLR phase finale
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — À PROPOS
# ══════════════════════════════════════════════════════════════
with tab_about:
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("""
        <div style='background:#0F172A;border:1px solid #1F2937;border-radius:14px;padding:24px;margin-bottom:16px;'>
            <div style='font-family:Space Mono,monospace;font-size:10px;color:#6B7280;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;'>PROJET</div>
            <div style='color:#F9FAFB;font-weight:700;font-size:16px;margin-bottom:6px;'>
                Détection de Vol en Vidéosurveillance
            </div>
            <div style='color:#9CA3AF;font-size:13px;line-height:1.7;'>
                Classification binaire d'images de caméras de surveillance 
                pour identifier automatiquement les comportements de vol ou shoplifting.
            </div>
        </div>

        <div style='background:#0F172A;border:1px solid #1F2937;border-radius:14px;padding:24px;margin-bottom:16px;'>
            <div style='font-family:Space Mono,monospace;font-size:10px;color:#6B7280;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;'>DATASET</div>
            <div style='font-family:Space Mono,monospace;font-size:11px;color:#9CA3AF;line-height:2;'>
                <div>📦 Source 1 : <span style='color:#3B82F6;'>pranaytlt/theft-new</span></div>
                <div>📦 Source 2 : <span style='color:#3B82F6;'>rex-jy68d/theft-detection-ksxxh</span></div>
                <div>🗂️ Format  : YOLO (images + labels .txt)</div>
                <div>🔢 Total   : 9 747 images</div>
                <div>🏷️ Classes : normal (ID=0) · theft (ID=1,2,3)</div>
            </div>
        </div>

        <div style='background:#0F172A;border:1px solid #1F2937;border-radius:14px;padding:24px;'>
            <div style='font-family:Space Mono,monospace;font-size:10px;color:#6B7280;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;'>TECHNOLOGIE</div>
            <div style='display:flex;flex-wrap:wrap;gap:8px;'>
                {"".join([f'<span style="background:#1F2937;color:#9CA3AF;font-family:Space Mono,monospace;font-size:10px;padding:4px 10px;border-radius:4px;">{t}</span>' for t in ["PyTorch", "torchvision", "Streamlit", "Plotly", "Kaggle GPU T4", "Python 3.12"]])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style='background:#0F172A;border:1px solid #1F2937;border-radius:14px;padding:24px;margin-bottom:16px;'>
            <div style='font-family:Space Mono,monospace;font-size:10px;color:#6B7280;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;'>MODÈLES</div>
        """, unsafe_allow_html=True)

        for name, color in MODEL_COLORS.items():
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #1F293755;'>
                <div style='width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0;'></div>
                <span style='font-family:Space Mono,monospace;font-size:11px;color:#E5E7EB;flex:1;'>{name}</span>
                <span style='font-family:Space Mono,monospace;font-size:10px;color:#4B5563;'>
                    {"From scratch" if name == "Baseline CNN" else "Transfer Learning"}
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#0F172A;border:1px solid #EF444433;border-radius:14px;padding:24px;'>
            <div style='font-family:Space Mono,monospace;font-size:10px;color:#EF4444;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;'>DÉPLOIEMENT</div>
            <div style='font-family:Space Mono,monospace;font-size:11px;color:#9CA3AF;line-height:2;'>
                <div>🚀 Plateforme : <span style='color:#EF4444;'>Streamlit Cloud</span></div>
                <div>📁 Modèles   : Google Drive (auto-download)</div>
                <div>🔧 Config    : requirements.txt</div>
                <div>📌 Python    : 3.10+</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Guide déploiement ─────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    with st.expander("📖 Guide de déploiement Streamlit Cloud", expanded=False):
        st.markdown("""
        #### 1. Préparer les fichiers
        ```
        ton-repo/
        ├── app.py
        ├── requirements.txt
        └── models/          ← optionnel (ou Google Drive)
        ```

        #### 2. Uploader les modèles sur Google Drive
        - Upload chaque `.pth` sur Drive
        - Partager → "Tout le monde avec le lien"
        - Copier l'ID (dans l'URL après `/d/`)
        - Remplir `GDRIVE_IDS` dans `app.py`

        #### 3. Déployer sur Streamlit Cloud
        ```bash
        # 1. Push ton repo sur GitHub
        git add app.py requirements.txt
        git commit -m "deploy theft detection app"
        git push

        # 2. Aller sur https://share.streamlit.io
        # 3. New app → sélectionner ton repo
        # 4. Main file: app.py → Deploy!
        ```

        #### 4. Variables d'environnement (optionnel)
        Dans les Settings Streamlit Cloud, tu peux ajouter des secrets pour sécuriser les IDs Drive.
        """)