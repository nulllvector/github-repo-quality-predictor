# 🔍 GitHub Repository Quality Predictor

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Benchmark Accuracy](https://img.shields.io/badge/OOD_Accuracy-97.71%25-brightgreen)](#-out-of-distribution-ood-benchmark-results)

An end-to-end Machine Learning pipeline that predicts GitHub repository quality based on engagement metrics, maintenance recency, and development hygiene. 

---

## 📌 Project Roadmap

| Phase | Focus | Status |
| :--- | :--- | :--- |
| **Version 1** | **Binary Baseline & OOD Generalization**: 100-pt rubric, stratified training, explainability, unseen-language benchmark | **Completed** ✅ |
| **Version 2** | **Multi-Class & Velocity Engineering**: 3-tier quality grading, star/fork velocity, CI/CD detection, Gradient Boosting | *Next* 🚀 |
| **Version 3** | **Deep Content & NLP**: README embeddings, AST cyclomatic complexity analysis, interactive web demo | *Planned* 📋 |

---

## 📊 Version 1: Key Achievements

* **19,166 Clean Repositories**: Scraped via GitHub REST API across 8 popularity tiers and 4 recency windows to eliminate the API's native recency bias.
* **Leak-Free ML Pipeline**: Stratified 80/20 train-test split with scaling strictly fitted on training data.
* **Rigorous Out-of-Distribution (OOD) Testing**: Validated on an external benchmark of **Go** and **Rust** repositories across active, recent, stale, and dormant date windows.
* **Generalization**: Random Forest achieved **97.71% accuracy** on unseen languages (virtually identical to in-distribution test accuracy of 97.91%).

---

## 🏆 Benchmark & Evaluation Results

### 1. In-Distribution (Python, JavaScript, TypeScript) vs. Out-of-Distribution (Go, Rust)

| Model | In-Distribution Test Acc | OOD Benchmark Acc | OOD Precision | OOD Recall | OOD F1-Score | OOD ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **97.91%** | **97.71%** | **99.07%** | **97.55%** | **0.9830** | **0.9987** |
| **Logistic Regression** | 89.31% | 92.50% | 96.77% | 92.02% | 0.9434 | 0.9769 |

### 2. Sub-Group Slice Analysis (Language Invariance)
To verify that the model did not learn ecosystem-specific bias, performance was measured across unseen languages:

| Language | Repositories Tested | Random Forest Accuracy | Logistic Regression Accuracy |
| :--- | :---: | :---: | :---: |
| **Go** | 239 | **98.33%** | 92.05% |
| **Rust** | 240 | **97.08%** | 92.92% |

### 3. Confusion Matrix (External Benchmark)

```
Random Forest (479 repos)          Logistic Regression (479 repos)
Predicted:   Low   High            Predicted:   Low   High
Actual Low:  151      3            Actual Low:  144     10
Actual High:   8    318            Actual High:  26    300
```

---

## 📐 Data Engineering & 100-Point Quality Rubric

To train our baseline classifiers without human labeling bottlenecks, we engineered a deterministic 100-point quality rubric:

$$\text{Quality Score} = S_{\text{stars}} + S_{\text{forks}} + S_{\text{pushed}} + S_{\text{updated}} + S_{\text{license}} + S_{\text{owner}}$$

| Component | Weight | Logic |
| :--- | :---: | :--- |
| **Forks Count** | 25 pts | Logarithmic scale ($\log_{10}(1 + \text{forks}) / \log_{10}(1 + 2000)$) |
| **Push Recency** | 25 pts | Stepped: $\le 30\text{d}$ (25), $\le 90\text{d}$ (18), $\le 180\text{d}$ (10), $\le 365\text{d}$ (4), $>365\text{d}$ (0) |
| **Stars Count** | 20 pts | Logarithmic scale ($\log_{10}(1 + \text{stars}) / \log_{10}(1 + 10000)$) |
| **License Presence**| 15 pts | Binary indicator (Open Source hygiene) |
| **Owner Type** | 10 pts | Organization = 10 pts, Individual User = 0 pts |
| **Update Recency** | 5 pts | Stepped: $\le 7\text{d}$ (5), $\le 30\text{d}$ (3), $\le 90\text{d}$ (1) |

* **Target Cutoff**: Set at median score (**42.5**), producing an exact **50.02% / 49.98%** class balance across 19,166 samples.

---

## 💡 Key Engineering Insights

1. **GitHub API Recency Trap**: Unfiltered GitHub searches sort by `updated desc` by default, clustering 100% of samples within the last 18 days. We solved this by explicitly partitioning queries across 4 distinct date windows (`<90d`, `90d–1yr`, `1–3yr`, `>3yr`).
2. **Language Invariance**: Feature importance analysis revealed language one-hot features (`lang_Python`, `lang_JavaScript`, `lang_TypeScript`) had near-zero importance ($< 0.002$). The models make decisions purely on structural signals.
3. **The Cold-Start Limitation (The Motivation for V2)**: In Version 1, quality is derived from metadata (social proof & maturity). A 2-day-old project with pristine code and 4 stars will be labeled low quality simply due to lack of time to gain traction. Version 2 solves this via **rate/velocity features** (stars/day) and **engineering rigor indicators** (CI/CD workflows, automated test suites).

---

## 📁 Repository Structure

```text
github-repository-quality-predictor/
├── data/
│   ├── raw/                       # 19,179 raw repos from API (git-ignored)
│   └── processed/                 # 19,166 preprocessed repos (git-ignored)
├── models/                        # Serialized model artifacts (git-ignored)
│   ├── logistic_regression_v1.pkl
│   ├── random_forest_v1.pkl
│   └── scaler_v1.pkl
├── notebooks/
│   ├── eda.ipynb                  # Feature distributions & correlation heatmaps
│   └── model_training.ipynb       # Split, scaling, training, feature importances
├── src/
│   ├── data_collection.py         # Rate-limit resilient API scraper
│   ├── preprocessing.py           # Preprocessing & reusable 100-pt rubric
│   └── benchmark_evaluation.py    # Out-of-Distribution evaluation suite
├── .env                           # GitHub API token (git-ignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Quickstart & Reproduction

### 1. Clone & Setup Environment

```bash
git clone https://github.com/<your-username>/github-repository-quality-predictor.git
cd github-repository-quality-predictor

# Create & activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure GitHub Token

Create a `.env` file in the root directory:

```env
GITHUB_TOKEN=ghp_yourPersonalAccessTokenHere
```

### 3. Run Pipeline Scripts

```bash
# Optional: Collect 19k training repositories
python src/data_collection.py

# Preprocess & apply rubric
python src/preprocessing.py

# Run the live Out-of-Distribution benchmark (Go & Rust across recency windows)
python src/benchmark_evaluation.py
```

---

## 🛠️ Tech Stack

* **Language**: Python 3.13
* **Machine Learning**: `scikit-learn`, `joblib`
* **Data Processing & Math**: `pandas`, `numpy`
* **Visualization**: `matplotlib`, `seaborn`
* **Data Ingestion**: `requests`, GitHub Search REST API v3
