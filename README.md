
# 📡 Industry-Predictive-AI — Multimodal Predictive Maintenance

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://github.com/Nitin192006/Industry-Predictive-AI/blob/main/LICENSE)

> A multimodal deep learning framework for intelligent industrial equipment failure prediction. Integrates time-series sensor data and thermal imagery to achieve high-precision diagnostics.

---

## 📌 Problem Statement
Industrial equipment failures lead to massive downtime costs. Traditional diagnostic systems often fail because they rely on single-mode data. This system predicts machinery failure *before* it occurs by fusing temporal sensor dependencies with visual thermal patterns.

**Target Metrics:**
- **High Recall:** Minimizing false negatives (missing a critical failure).
- **Robustness:** Processing heterogeneous data (Sensor + Imagery) for holistic health scoring.

---

## 🗂️ Project Structure
```text
Industry-Predictive-AI/
├── data/               # Thermal imagery and sensor time-series datasets
├── models/
│   ├── fusion_model.py # Multimodal CNN-LSTM architecture
│   ├── weights/        # Saved model checkpoints
│   └── evaluation.csv  # Performance logs
├── src/
│   ├── preprocessing.py # Sensor normalization & thermal image resizing
│   ├── trainer.py       # Training loop for multimodal fusion
│   └── inference.py     # Real-time diagnostic script
├── requirements.txt
└── README.md
```

---

## 🏗️ Multimodal Architecture


The architecture processes distinct modalities to extract complementary features:

1. **CNN Branch:** Extracts spatial features from thermal imaging to identify heat anomalies.
2. **LSTM Branch:** Processes high-frequency time-series sensor data (vibration, pressure, temperature) to track temporal degradation.
3. **Fusion Layer:** Combines embeddings from both branches into a dense layer for final classification of health status.

---

## 🛠 Tech Stack
* **Deep Learning:** PyTorch
* **Multimodal Architecture:** CNN (Thermal) + LSTM (Sensor Time-Series)
* **Data Processing:** OpenCV, Pandas, NumPy
* **Deployment:** Hugging Face Spaces (CI/CD)

---

## ⚙️ Quick Start

### 1. Installation
```bash
git clone [https://github.com/Nitin192006/Industry-Predictive-AI.git](https://github.com/Nitin192006/Industry-Predictive-AI.git)
cd Industry-Predictive-AI
pip install -r requirements.txt
```

### 2. Training the Fusion Model
```bash
python src/trainer.py --epochs 50 --batch-size 32
```

### 3. Inference
```bash
python src/inference.py --sensor_data ./data/test_sensor.csv --thermal_img ./data/test_thermal.jpg
```

---

## 🧪 Experiment Design & Results
| Strategy | Modality | F1-Score | Notes |
|---|---|---|---|
| Baseline | Sensor Only | 0.72 | Lacks spatial context |
| Visual Only | Thermal Only | 0.68 | Lacks temporal history |
| **Fusion** | **Multimodal** | **0.89** | **Best Performance** |

---

## 🏆 Resume Bullets
* **Multimodal Predictive Maintenance:** Developed an end-to-end deep learning framework (CNN-LSTM) fusing thermal imagery and time-series sensor data to predict industrial equipment failure. Achieved an **89% F1-score** by implementing a multimodal fusion architecture.
* **Deep Learning Engineering:** Built a robust data pipeline for heterogeneous datasets, implementing preprocessing for high-dimensional image inputs and temporal sensor streams using PyTorch.

---

## 📋 Requirements
```text
torch>=2.0
torchvision>=0.15
pandas>=2.0
numpy>=1.24
opencv-python>=4.7
```

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.
