# 🔍 GitHub Repository Quality Predictor

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-2E7D32?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-5.0.0-1E88E5?logo=optuna&logoColor=white)](https://optuna.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Macro F1-Score](https://img.shields.io/badge/V2_Macro_F1-96.15%25-brightgreen)](#-version-2-modeling--performance)
[![Multi-Class ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9966-success)](#-version-2-modeling--performance)

An end-to-end Machine Learning system that predicts GitHub repository quality through **multi-class grading, velocity feature engineering, Bayesian hyperparameter optimization, and adversarial stress-testing**.

---

## 📌 Project Architecture & Evolution

```
┌─────────────────────────────────┐      ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│           VERSION 1             │      │           VERSION 2             │      │           VERSION 3             │
│   Binary Baseline (Complete)    │ ───> │  Multi-Class SOTA (Complete)    │ ───> │  Neural & AST NLP (Planned)     │
│ • 100-pt Binary Median Cutoff   │      │ • 3-Tier Quality Classification │      │ • README Semantic Embeddings    │
│ • Logistic Regression vs. RF    │      │ • Velocity & S-Curve Prior      │      │ • AST Cyclomatic Complexity     │
│ • Unseen Go/Rust OOD Testing    │      │ • Tuned LightGBM vs. RF         │      │ • CI/CD & Test Suite Extraction │
│ • 97.71% Baseline Accuracy      │      │ • Optuna TPE Bayesian Tuning    │      │ • Interactive Live Web Demo     │
│                                 │      │ • 3-Phase Benchmark Suite       │      │                                 │
└─────────────────────────────────┘      └─────────────────────────────────┘      └─────────────────────────────────┘
```

---

## 🚀 Version 2: Major Innovations & Highlights

* **3-Tier Multi-Class Quality Grading**: Transitions from simplistic binary labels to a nuanced 3-tier hierarchy:
  - **Tier 0 (Low / Dormant)**: Score $< 40$ (~33.4% of dataset)
  - **Tier 1 (Emerging / Moderate)**: Score $40 - 65$ (~37.9% of dataset)
  - **Tier 2 (Production-Grade)**: Score $\ge 65$ (~28.7% of dataset)
* **Decoupled Velocity & Bayesian Priors**: Solves the "Young Repository Penalty." Young repos with explosive adoption (`astral-sh/uv`, `ollama/ollama`) earn Tier 2 status via smoothed logarithmic velocity ($\frac{\log_{10}(1+\text{stars})}{\log_{10}(1+\text{age}+30)}$).
* **Champion Gradient Boosting (LightGBM)**: Employs leaf-wise tree growth with 256 histogram bins (`uint8`), reducing memory consumption by $80\%$ and outperforming Random Forest by **$+4.0\%$ accuracy on the hardest sandwiched class (Tier 1)**.
* **Bayesian Hyperparameter Optimization (Optuna)**: 30-trial Tree-structured Parzen Estimator (TPE) search across 5-fold Stratified Cross-Validation, proving model stability within a wide, regularized minimum.
* **Centralized Single-Source-of-Truth Feature Pipeline**: Refactored `src/features.py` module ensuring zero code duplication across data collection, training, and testing suites.
* **Rigorous 3-Phase Benchmark Suite**: Stress-tested against 15 live landmark repositories, 6-year temporal lifecycle simulations, and 5 adversarial metadata attacks.

---

## 📐 The Upgraded 100-Point Quality Rubric (V2)

$$\text{Quality Score} = S_{\text{traction}} + S_{\text{maintenance}} + S_{\text{hygiene}} + S_{\text{docs}} + S_{\text{governance}}$$

| Category | Max Pts | Mathematical Formulation & Engineering Rationale |
| :--- | :---: | :--- |
| **1. Traction & Velocity** | **35 pts** | **Dual-Path Decoupled Logic**: $\max(S_{\text{cumulative}}, S_{\text{velocity}})$<br>• *Cumulative*: Forks $22\text{ pts}$ ($\frac{\log_{10}(1+\text{forks})}{\log_{10}(1+2000)}$) + Stars $13\text{ pts}$ ($\frac{\log_{10}(1+\text{stars})}{\log_{10}(1+10000)}$) *(heavily weights forks over passive stars)*.<br>• *Velocity*: $\frac{\log_{10}(1+\text{stars})}{\log_{10}(1+\text{age}+30)} \times 25\text{ pts}$ *(Bayesian prior prevents 1-day noise explosions)*. |
| **2. Maintenance & Liveness** | **25 pts** | **Push Recency (20 pts)**: $\le 30\text{d}$ (20), $\le 90\text{d}$ (14), $\le 180\text{d}$ (8), $\le 365\text{d}$ (3), $>365\text{d}$ (0).<br>**Update Recency (5 pts)**: $\le 14\text{d}$ (5), $\le 60\text{d}$ (2). |
| **3. Open Source Hygiene** | **15 pts** | **Licensing Integrity**: Standard OSI-approved license = $15\text{ pts}$, custom/proprietary license = $8\text{ pts}$, unlicensed = $0\text{ pts}$. |
| **4. Documentation & Packaging** | **15 pts** | **Description Length (10 pts)**: $\ge 30\text{ chars}$ (10), $>0$ (5).<br>**Curated Topics (5 pts)**: $\ge 3\text{ topics}$ (5), $>0$ (2). |
| **5. Governance & Community** | **10 pts** | **Organization Backed (6 pts)**: Organization ownership = 6, individual user = 0.<br>**Community Tooling (4 pts)**: Active Discussions, Wiki, or GitHub Pages = 4. |

---

## 📊 Version 2 Modeling & Performance

### 1. Multi-Class Holdout Test Scoreboard (3,834 Repositories)

Evaluated on an untouched 20% holdout test set with stratified class balance:

| Model | Paradigm | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Multi-Class ROC-AUC (OvR) | Model Size | Inference Speed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (v2)** | Bagging (100 Trees) | 93.74% | 0.9387 | 0.9389 | 0.9381 | 0.9882 | 8.99 MB | ~4.2 ms / repo |
| **Default LightGBM** | Gradient Boosting | 96.04% | 0.9602 | 0.9613 | 0.9607 | 0.9966 | 1.28 MB | **~0.2 ms / repo** |
| **Tuned LightGBM (Champion)** | **Optuna TPE Boosted** | **96.17%** | **0.9612** | **0.9619** | **0.9615** | **0.9966** | **1.28 MB** | **~0.2 ms / repo** |
| **Soft-Voting Ensemble** | RF + Tuned LGB Blend | 96.11% | 0.9608 | 0.9614 | 0.9610 | 0.9965 | 10.27 MB | ~4.4 ms / repo |

> **Production Champion**: **Tuned LightGBM** was selected as the sole production model because it achieves top accuracy, executes $20\times$ faster than Random Forest, and has a compact $1.28\text{ MB}$ footprint ideal for serverless deployment.

---

### 2. Normalized Confusion Matrix (LightGBM v2)

```
                       Predicted Tier 0    Predicted Tier 1    Predicted Tier 2
Actual Tier 0 (Low)        98.44%               1.56%               0.00%
Actual Tier 1 (Emerging)    2.20%              94.77%               3.03%
Actual Tier 2 (Production)  0.00%               3.83%              96.17%
```

* **0.00% Catastrophic Error Rate**: Zero Tier 0 repositories are ever misclassified as Tier 2, and zero Tier 2 repositories are ever misclassified as Tier 0.
* **Adjacent Error Bounding**: 100% of errors occur strictly across adjacent tier borders (Tier 0 $\leftrightarrow$ Tier 1, or Tier 1 $\leftrightarrow$ Tier 2).

---

### 3. Feature Importance Analysis (Breaking the Vanity Star Bias)

In naive GitHub models, raw bookmark counts dominate decisions. Our feature engineering successfully dethroned raw stars:

```
Top Features Ranked by LightGBM Split Frequency:
 1. days_since_last_push    ██████████████████████████████ (1,248 splits)
 2. log_star_velocity       ████████████████████████████   (1,202 splits)
 3. description_length      ████████████████████           (915 splits)
 4. maintenance_ratio       ██████████████                 (612 splits)
 5. forks_count             ████████████                   (542 splits)
 6. license_type_osi        ████████████                   (524 splits)
 ...
15. stargazers_count        ████████                       (398 splits)
```

Raw `stargazers_count` dropped to **#15**—the model makes decisions based on active maintenance recency and velocity, not historical bookmark accumulation.

---

## 🧪 The 3-Phase Benchmark Suite

To validate the model beyond standard test splits, we implemented three dedicated stress-testing suites:

### **Suite 1: Landmark Archetypes Benchmark (`src/benchmark_01_archetypes.py`)**
Evaluates 15 live iconic real-world repositories via the GitHub REST API:

| Repository | Category | Stars | Inactive | Predicted Tier | Confidence | Automated Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `facebook/react` | Production Titan | 252k | 0d | **Tier 2** | 100.0% | High community scale + active push recency + verified license |
| `torvalds/linux` | Production Titan | 254k | 0d | **Tier 2** | 99.2% | High community scale + active push recency + verified license |
| `psf/black` | Production Titan | 42k | 2d | **Tier 2** | 100.0% | Official Python standard tool, active maintenance |
| `kubernetes/kubernetes` | Production Titan | 128k | 0d | **Tier 2** | 100.0% | Cloud infrastructure standard, enterprise backing |
| `astral-sh/uv` | Viral Rocket (Young) | 91k | 0d | **Tier 2** | 100.0% | Extreme adoption velocity decoupled young repo age |
| `ollama/ollama` | Viral Rocket (Young) | 186k | 0d | **Tier 2** | 100.0% | Extreme adoption velocity decoupled young repo age |
| `angular/angular.js` | Dormant Legend | 58k | 938d | **Tier 2** | 100.0% | Massive historical scale (58k★, 27k⑂) outscores 938d inactivity |
| `request/request` | Dormant Legend | 25k | 814d | **Tier 2** | 99.2% | Massive historical scale (25k★, 3k⑂) outscores 814d inactivity |
| `octocat/Hello-World` | Tutorial / Scrapbook | 3.9k | 807d | **Tier 1** | 99.8% | Moderate historical stars (3.9k★) elevated above Tier 0 cutoff |

---

### **Suite 2: Temporal Lifecycle Simulation ("Time Travel") (`src/benchmark_02_lifecycle.py`)**
Simulates continuous repository evolution over 6 years using empirical historical snapshots:

```text
1. FastAPI Rising Titan Path (Genesis to Global Standard):
   Day 1 (Genesis, no license)     --> Tier 0 (Low / Dormant)       [99.5% conf]
   Day 2 (MIT license + README)    --> Tier 1 (Emerging / Moderate) [95.6% conf]
   Day 14 (85 stars, 6/day vel)    --> Tier 2 (Production-Grade)   [86.5% conf]
   Day 180 (Viral Post-HN, 3.4k★)  --> Tier 2 (Production-Grade)   [99.6% conf]
   Year 6 (Global Titan, 104k★)    --> Tier 2 (Production-Grade)   [96.1% conf]
   >> Graduation Path: T0 ──> T1 ──> T2 ──> T2 ──> T2

2. The Inactivity Decay of a Mid-Sized Tool (600 Stars):
   Push: Today                     --> Tier 2 (Production-Grade)   [79.2% conf]
   Push: 30 days ago               --> Tier 1 (Emerging / Moderate) [75.9% conf]
   Push: 180 days ago              --> Tier 1 (Emerging / Moderate) [99.2% conf]
   Push: 365+ days ago             --> Tier 1 (Emerging / Moderate) [96.1% conf]
   >> Demotion Path:   T2 ──> T1 ──> T1 ──> T1
```

---

### **Suite 3: Adversarial & Gaming Robustness (`src/benchmark_03_adversarial.py`)**
Stress-tests the decision boundaries against intentional metadata gaming:

| Attack ID | Scenario Name | Attack Vector | Baseline | Attacked | Defense Verdict |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **ATTACK-01** | The $50 Fake Star-Bomb | Injected 10,000 bot stars into an abandoned 0-fork repo | Tier 0 (100%) | **Tier 0 (96.2%)** | **DEFENDED**: Lack of forks and dead push prevented Tier 2 jump. |
| **ATTACK-02** | The Zombie Commit Bot | Daily automated cron commits pushing empty commits | Tier 0 (100%) | **Tier 0 (99.9%)** | **DEFENDED**: Green squares without community adoption cannot trick the model. |
| **ATTACK-03** | SEO / Topic Stuffing | Stuffed 20 trending AI buzzwords into empty repo | Tier 0 (100%) | **Tier 0 (100%)** | **DEFENDED**: Keywords max out at 15 pts, far below Tier 1/2 cutoffs. |
| **ATTACK-04** | Synthetic Fork Farm | Botnet generated 5,000 forks on a 2-star repo | Tier 1 (68.5%) | **Tier 1 (99.3%)** | **DEFENDED**: Star deficit kept it strictly sealed inside Tier 1. |
| **ATTACK-05** | License Stripping Sabotage | Deleted open-source license from Facebook React | Tier 2 (94.2%) | **Tier 2 (57.5%)** | **SENSITIVE**: Tier 2 probability plummeted by **-36.7%**, proving active license compliance. |

---

## 💡 The "Metadata Glass Ceiling" & What Comes Next (Version 3)

The most important scientific discovery of Version 2 is the **Metadata Glass Ceiling**:
> *A tabular model trained purely on GitHub REST metadata (`stars`, `forks`, `issues`, `dates`, `size`, `license`) achieves near-flawless accuracy (~96%) in quantifying popularity, adoption velocity, and maintenance recency.*
>
> *However, metadata ALONE cannot inspect the code. It cannot know that `github/gitignore` is just a list of text files rather than executable software, or verify whether a repository has rigorous unit test suites and clean architecture.*

### **The Version 3 Roadmap:**
1. **Multimodal NLP Analysis**: Embedding repository `README.md` and descriptions using transformer language models (e.g., CodeBERT / MiniLM) to categorize project purpose (e.g. distinguishing software libraries from asset lists and tutorial scrapbooks).
2. **Abstract Syntax Tree (AST) Inspection**: Cloning repository shallow trees to measure cyclomatic complexity, docstring coverage, and static code smells.
3. **CI/CD & Engineering Hygiene Signals**: Detecting automated test suite execution (GitHub Actions workflows, pytest/jest configs, coverage badges).
4. **Interactive Web Application**: A full-stack Streamlit / FastAPI web application enabling real-time URL scoring and repository diagnostic breakdowns.

---

## 📁 Repository Structure

```text
github-repository-quality-predictor/
├── data/
│   ├── raw/github_repos.csv             # 19,179 raw repos scraped from GitHub API
│   └── processed/github_repos_v2.csv    # 19,166 preprocessed repos (26 features + quality_tier)
├── models/
│   ├── lightgbm_v2.pkl                  # 1.28 MB Tuned Champion Model
│   └── random_forest_v2.pkl             # 8.99 MB Multi-Class Bagging Benchmark
├── notebooks/
│   ├── eda.ipynb                        # 22-cell deep statistical EDA with visual themes
│   └── modeling_v2.ipynb                # Complete V2 training, Optuna tuning & evaluation
├── src/
│   ├── features.py                      # Single Source of Truth feature extraction module
│   ├── preprocessing.py                 # V2 dataset preprocessing & 100-pt rubric
│   ├── data_collection.py               # GitHub REST API collection engine
│   ├── benchmark_01_archetypes.py       # Suite 1: Landmark Archetypes Benchmark
│   ├── benchmark_02_lifecycle.py        # Suite 2: Temporal Lifecycle Simulation
│   └── benchmark_03_adversarial.py      # Suite 3: Adversarial & Gaming Robustness
├── requirements.txt                     # Pinned project dependencies
├── .gitignore                           # Git exclusion rules
├── LICENSE                              # MIT Open Source License
└── README.md                            # Comprehensive project documentation
```

---

## ⚡ Quickstart & Installation

### 1. Clone & Set Up Virtual Environment
```powershell
git clone https://github.com/nulllvector/github-repository-quality-predictor.git
cd github-repository-quality-predictor

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure GitHub API Token (Optional but Recommended)
Create a `.env` file in the root directory:
```env
GITHUB_TOKEN=your_personal_access_token_here
```

### 3. Run the Benchmark Suites
```powershell
# Suite 1: Landmark Archetypes (Live API)
python src/benchmark_01_archetypes.py

# Suite 2: Temporal Lifecycle Simulation ("Time Travel")
python src/benchmark_02_lifecycle.py

# Suite 3: Adversarial & Gaming Robustness
python src/benchmark_03_adversarial.py
```

### 4. Run the Jupyter Notebooks
```powershell
jupyter notebook notebooks/modeling_v2.ipynb
```

---

## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
