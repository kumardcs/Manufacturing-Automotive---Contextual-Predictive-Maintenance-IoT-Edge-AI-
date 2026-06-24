# Contextual Predictive Maintenance (IoT Edge AI)
### Infotact Technical Internship Program — Project 1

![Python](https://img.shields.io/badge/Python-3.10-blue) ![LightGBM](https://img.shields.io/badge/Model-LightGBM-green) ![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## 📌 Problem Statement

Most existing predictive maintenance systems rely exclusively on internal diagnostic signals and fail in real-world deployment. A machine's failure is rarely isolated — it is heavily influenced by external factors such as weather conditions, factory load, and environmental stress.

This project builds an advanced **Contextual Data Fusion Framework** that integrates internal IoT telemetry (vibration, temperature, current) with external environmental signals to accurately predict mechanical failures before they happen using robust ensemble modeling.

---

## 🎯 Objective

- Fuse internal IoT sensor data with external contextual signals
- Build a LightGBM-based classification pipeline to predict machine failures
- Handle severe class imbalance (<2% failure rate) using SMOTE within cross-validation
- Achieve **Macro F1 ≥ 0.85** on the test set
- Validate model robustness against synthetic noise injection

---

## 📁 Project Structure

```
predictive-maintenance/
│
├── notebooks/
│   ├── week1_signal_processing.ipynb
│   ├── week2_data_fusion.ipynb
│   ├── week3_lightgbm_modeling.ipynb
│   └── week4_noise_analysis.ipynb
│
├── src/
│   ├── feature_engineering.py
│   ├── data_fusion.py
│   ├── model.py
│   └── evaluate.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

> **Note:** `data/` and `models/` directories are excluded via `.gitignore`. See dataset section below for download instructions.

---

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset**

The AI4I 2020 Predictive Maintenance Dataset is a synthetic industrial dataset modeled after real-world manufacturing equipment behavior. It was created to benchmark machine learning algorithms for predictive maintenance tasks in industrial IoT environments.

The dataset contains **10,000 data points** with **14 features**, collected from a milling machine operating under varying conditions. Each record represents a single operational cycle of the machine.

### Sensor Features

| Feature | Description |
|---|---|
| `Air temperature [K]` | Ambient air temperature around the machine |
| `Process temperature [K]` | Internal operating temperature of the process |
| `Rotational speed [rpm]` | Speed of the rotating component |
| `Torque [Nm]` | Torque applied during the process |
| `Tool wear [min]` | Cumulative tool usage time in minutes |

### Target Variable

`Machine failure` — binary label (0 = no failure, 1 = failure). Failures account for less than **2% of the dataset**, making it a heavily imbalanced classification problem.

### Failure Modes

The dataset also includes 5 independent failure type columns representing specific failure causes:

| Column | Failure Type |
|---|---|
| `TWF` | Tool Wear Failure |
| `HDF` | Heat Dissipation Failure |
| `PWF` | Power Failure |
| `OSF` | Overstrain Failure |
| `RNF` | Random Failure |

### Download

Download from UCI Machine Learning Repository:
🔗 https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Place the file in the root directory as `ai4i2020.csv` before running any notebooks.

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/predictive-maintenance.git
cd predictive-maintenance
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `ai4i2020.csv` from the link above and place it in the root folder.

### 4. Run notebooks in order
```
week1 → week2 → week3 → week4
```

---

## 🧪 Tech Stack

| Category | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| ML Model | LightGBM |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Version Control | Git, GitHub |

---

## 🗓️ 4-Week Engineering Roadmap

### Week 1 — IoT Telemetry Ingestion & Signal Processing
- Load AI4I dataset
- Compute rolling mean, standard deviation, and variance over time windows for each sensor column
- Create baseline feature set for modeling

### Week 2 — Contextual Data Fusion & Feature Engineering
- Simulate external contextual data (ambient temperature, load density)
- Merge external signals with internal sensor telemetry on timestamps
- Conduct ablation study to prove external features improve predictive power

### Week 3 — Imbalanced Classification & LightGBM Modeling
- Set up 5-fold stratified cross-validation pipeline
- Apply SMOTE strictly inside training folds to prevent data leakage
- Train LightGBM classifier, optimize for Macro F1

### Week 4 — Noise Sensitivity Analysis & Threshold Tuning
- Inject synthetic Gaussian noise into test data
- Plot Precision-Recall curves
- Tune decision threshold to balance false alarms vs. missed failures

---

## 📈 Results

| Metric | Score |
|---|---|
| Macro F1 | TBD |
| Precision | TBD |
| Recall | TBD |
| ROC-AUC | TBD |

> Results will be updated upon project completion.

---

## 📌 Key Implementation Notes

- **SMOTE is applied only inside CV folds** — not before splitting, to strictly prevent data leakage
- **No CSV or model weight files are pushed to GitHub** — use `.gitignore` from Day 1
- **Jupyter notebooks are committed with cleared outputs** — use `Kernel → Restart & Clear Output` before every commit
- Commit messages follow semantic format: `feat: implement rolling window features (fixes #2)`

---

## 🔗 References

- [AI4I 2020 Dataset — UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [SHAP Explainability](https://shap.readthedocs.io/)

---

## 👤 Author

**Pranav**
B.Tech — Electronics and Communication Engineering
Central University of Jammu
Infotact Solutions & Co. — DS/ML Intern, 2026
