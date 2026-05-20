# 🛡️ Theft & Shoplifting Detection — Deep Learning

> Classification binaire d'images de vidéosurveillance par Deep Learning  
> Module Deep Learning · Encadrant : **M. Abdallah Khemais**

---

## Problème

Détecter automatiquement les comportements de vol dans des images de caméras de surveillance.

| Classe | Description | Label |
|--------|-------------|-------|
| 🟢 **normal** | Comportement ordinaire | 0 |
| 🔴 **theft** | Vol / shoplifting détecté | 1 |

---

##  Dataset

| Propriété | Détail |
|-----------|--------|
| **Source 1** | pranaytlt/theft-new — Roboflow |
| **Source 2** | rex-jy68d/theft-detection-ksxxh — Roboflow |
| **Format** | YOLO (images + labels .txt) |
| **Train** | 8 771 images |
| **Valid** | 600 images |
| **Test** | 376 images |
| **Total** | **9 747 images** |

### Mapping YOLO → binaire

```
ID 0 (person)      → normal
ID 1 (theft)       → theft
ID 2 (shoplifting) → theft
ID 3 (suspect)     → theft
```

R�gle : une image est `theft` si les annotations theft sont majoritaires.

---

##  Pipeline

```
Dataset YOLO → ImageFolder (normal/ theft/)
    ↓ P4 — WeightedSampler + Class Weights + Label Smoothing 0.1
    ↓ Augmentation : Flip, Rotation, ColorJitter
    ↓ Subset équilibré 1 500/classe (3 000 train)
    ↓ 5 modèles × 10 epochs · Kaggle GPU T4
    ↓ P5 — Fine-Tuning Progressif (3 phases)
    ↓ Évaluation + Déploiement Streamlit
```

---

##  Résultats

| Modèle | Type | Params | Test Acc | Val Acc |
|--------|------|--------|----------|---------|
| Baseline CNN | From scratch | 2.6M | 57.18% | 80.33% |
| ResNet-50 | Transfer Learning | 23.8M | 59.04% | 88.33% |
| EfficientNet-B0 | Transfer Learning | 4.2M | 58.78% | 68.50% |
| MobileNetV3 | Transfer Learning | 1.6M | 58.24% | 82.17% |
| VGG-16 | Transfer Learning | 134M | 56.65% | 90.83% |
| **ResNet-50 + FT 🏆** | **Fine-Tuning** | **23.8M** | **59.84%** | **90.50%** |

---

##  P5 — Fine-Tuning Progressif (ResNet-50)

| Phase | Couches | LR | Epochs |
|-------|---------|-----|--------|
| Phase 1 | Head uniquement | 3e-4 | 4 |
| Phase 2 | Derniers blocs | 1e-4 | 4 |
| Phase 3 | Réseau complet | 1e-5 | 6 |

R�sultat : 59.04% → **59.84%** (↑ +0.80%)

---

## Structure

```
Deep-Learning-/
├── deep-learning-final.ipynb
├── app.py
├── requirements.txt
├── models/
│   ├── baseline_cnn.pth         (10.2 MB)
│   ├── resnet50.pth             (95.4 MB)
│   ├── efficientnet_b0.pth      (17.0 MB)
│   ├── mobilenet_v3.pth         (6.7  MB)
│   └── resnet50_finetuned.pth   (95.4 MB)
└── README.md
```

---

##  Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Application déployée : https://raniadk-deep-learning--app-syqctp.streamlit.app

---

##  Technologies

| Outil | Usage |
|-------|-------|
| PyTorch 2.2 | Framework Deep Learning |
| torchvision | Architectures pré-entraînées |
| Streamlit | Interface web d'inférence |
| Plotly | Graphiques interactifs |
| Kaggle GPU T4 | Entraînement |

---

##  Concepts clés

- **Gradient vanishing** → résolu par ResNet (skip connections)
- **Transfer Learning** → features ImageNet réutilisées pour la surveillance
- **P4 Balance** → WeightedSampler + class weights contre le déséquilibre
- **P5 Fine-Tuning** → dégel progressif pour adapter le backbone

---

Module Deep Learning 
