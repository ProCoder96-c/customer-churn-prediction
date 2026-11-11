# 📉 Customer Churn Prediction Model

> **Academic Project** | Python • Scikit-learn | Aug 2024 – Nov 2024

A binary classification pipeline to predict customer churn using **Logistic Regression** and **Random Forest**, achieving **~90% accuracy** (Random Forest, ROC-AUC 0.944) with robust handling of class imbalance via **SMOTE**.

---

## 🎯 Project Highlights

| | |
|---|---|
| **Task** | Binary Classification (Churn / No Churn) |
| **Best Accuracy** | 89.7% (Random Forest) |
| **Best ROC-AUC** | 0.944 (Random Forest) |
| **Imbalance Handling** | SMOTE (Synthetic Minority Oversampling) |
| **Models** | Logistic Regression, Random Forest |
| **Dataset** | Synthetic telecom churn dataset (~3,000 customers) |

---

## 📁 Project Structure

```
customer-churn-prediction/
├── src/
│   └── churn_model.py        # Full ML pipeline
├── outputs/
│   └── plots/
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       ├── feature_importance.png
│       └── smote_distribution.png
├── requirements.txt
└── README.md
```

---

## ⚙️ Pipeline Overview

```
Raw Data → Feature Engineering → Train/Test Split
    → SMOTE (balance classes) → StandardScaler
    → Model Training (LR + RF) → Evaluation + Plots
```

### 1. Feature Engineering
- `avg_monthly_spend` — total charges / tenure
- `calls_per_month` — support calls normalized by tenure
- `high_support_user` — binary flag for heavy support users
- `long_tenure` — binary flag for customers > 36 months
- `monthly_contract` — binary flag for month-to-month contracts

### 2. Class Imbalance → SMOTE
Original dataset has ~22% churn rate. SMOTE synthetically oversamples the minority class in the training set, balancing it to 50/50 without data leakage.

### 3. Models
| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 83.3% | Baseline, interpretable coefficients. ROC-AUC 0.885 |
| Random Forest | 89.7% | Best performer, used for feature importance. ROC-AUC 0.944, churn-class recall 0.79 |

### 4. Visualisations
- **Confusion Matrix** — per-class prediction breakdown
- **ROC Curve** — AUC comparison across models
- **Feature Importance** — top drivers of churn (Random Forest)
- **SMOTE Distribution** — before vs after class balancing

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/ProCoder96-c/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/churn_model.py
```

Outputs (plots) will be saved to `outputs/plots/`.

---

## 📊 Sample Outputs

### ROC Curves
Both models clearly outperform random classification: Random Forest AUC = 0.944, Logistic Regression AUC = 0.885.

### Feature Importance (Random Forest)
Top churn drivers: `tenure_months`, `monthly_charges`, `support_calls`, `contract_type`, `avg_monthly_spend`

---

## 🛠 Tech Stack

- **Python 3.10+**
- `scikit-learn` — model training, evaluation, preprocessing
- `imbalanced-learn` — SMOTE
- `matplotlib` / `seaborn` — visualisation
- `pandas` / `numpy` — data manipulation

---

## 📝 Key Learnings

- SMOTE must be applied **only on training data** to prevent data leakage
- Random Forest consistently outperforms Logistic Regression on imbalanced tabular data
- Feature engineering (derived ratios, binary flags) meaningfully improved recall on the minority class
- ROC-AUC is a more reliable metric than accuracy when classes are imbalanced

---

*Academic project — all data is synthetically generated.*
