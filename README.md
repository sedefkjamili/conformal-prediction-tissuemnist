# Uncertainty Quantification in Medical Image Classification

**Artificial Intelligence Course AI Term Project** — Sedef Kjamili

Applying **Conformal Prediction** to the TissueMNIST dataset using a custom ResNet-50 architecture trained at 28×28 resolution. Instead of a single hard prediction, the model outputs a **prediction set** with guaranteed coverage — making it safer for medical decision-making.

## 🔬 Overview

Standard classifiers give one label per input, with no reliability guarantee. This project applies **Split Conformal Prediction** (via MAPIE) on top of a pre-trained ResNet-50 to produce prediction sets that contain the true label with at least 90–95% probability.

Key contributions:
- Comparison of 3 conformal scoring methods: **LAC**, **APS**, **Top-K**
- Class-conditional conformal prediction with per-class coverage analysis
- Confidence trade-off analysis (80% → 99%)
- **Grad-CAM** attention maps to visualize model focus per tissue class
- Hard example analysis — cases where the model is genuinely uncertain

## 📁 Repository Structure

```text
├── BLG_521E_resnet50_28_sedef.ipynb   # Main notebook
├── resnet50_28_1.pth                  # Best model checkpoint
├── plots/                             # Generated visualizations
│   ├── confusion_matrix.png
│   ├── method_comparison.png
│   ├── attention_maps.png
│   ├── cp_class_overlap.png
│   └── hard_examples.png
└── results/                           # Numpy outputs (probs, labels, prediction sets)
```

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

- `medmnist` — TissueMNIST dataset
- `MAPIE` — Conformal prediction (SplitConformalClassifier)
- `pytorch-grad-cam` — Attention map visualization
- `seaborn` / `matplotlib` — Plotting

## 📊 Results

| Method | Coverage | Avg Set Size | Median |
|--------|----------|--------------|--------|
| LAC    | 0.8996   | 2.009        | 2.0    |
| APS    | 0.9439   | 2.589        | 2.0    |
| Top-K  | 0.9354   | 3.030        | 3.0    |

## 🔁 Method Comparison: Two Calibration Strategies

Two variants of **class-conditional LAC** (Least Ambiguous Set) conformal prediction were compared:

| | Method 1 (Official Split) | Method 2 (60-40 Split) |
|---|---|---|
| Calibration set | Official val set | 60% of val + test combined |
| Test set | Official test set | Remaining 40% |
| Calibration size | ~n_val | ~larger |

**Method 1** uses the dataset's official train/val/test split for calibration.
**Method 2** merges the val and test sets and re-splits them 60-40 (seed=42), giving a larger calibration set. This tests whether more calibration data improves per-class coverage stability — especially for minority tissue classes with few samples.

Both methods use the same class-conditional thresholds (α = 0.05, 95% confidence), computing a separate probability threshold per class so that each tissue type meets the coverage guarantee independently.

## 🚀 How to Run

1. Open `BLG_521E_resnet50_28_sedef.ipynb` in Google Colab
2. Run all cells — model weights are downloaded automatically from Zenodo
3. Results and plots are saved locally and zipped for download

## 📫 Author

**Sedef Kjamili** 
