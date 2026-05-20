"""
Customer Churn Prediction Model
Academic Project | Aug 2024 – Nov 2024

Binary classification using Logistic Regression + Random Forest
with SMOTE for class imbalance handling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
os.makedirs("outputs/plots", exist_ok=True)

# ─────────────────────────────────────────────
# 1. Simulate realistic customer churn dataset
# ─────────────────────────────────────────────
def generate_churn_data(n_samples=3000, random_state=42):
    """Generate a synthetic but realistic customer churn dataset."""
    np.random.seed(random_state)

    X_raw, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        weights=[0.78, 0.22],   # ~22% churn rate (class imbalance)
        flip_y=0.02,
        random_state=random_state
    )

    feature_names = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "num_products",
        "support_calls",
        "contract_type",     # 0=month-to-month, 1=1yr, 2=2yr
        "payment_method",    # encoded
        "internet_service",  # encoded
        "online_security",
        "tech_support"
    ]

    df = pd.DataFrame(X_raw, columns=feature_names)

    # Make features more interpretable
    df["tenure_months"]   = (df["tenure_months"] * 12 + 36).clip(1, 72).astype(int)
    df["monthly_charges"] = (df["monthly_charges"] * 20 + 65).clip(20, 120).round(2)
    df["total_charges"]   = (df["tenure_months"] * df["monthly_charges"]).round(2)
    df["num_products"]    = (df["num_products"] * 1.5 + 3).clip(1, 6).astype(int)
    df["support_calls"]   = (df["support_calls"] * 2 + 2).clip(0, 10).astype(int)
    df["contract_type"]   = (df["contract_type"] * 1 + 1).clip(0, 2).astype(int)
    df["churn"]           = y

    return df


# ─────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────
def feature_engineering(df):
    """Create derived features that improve model signal."""
    df = df.copy()
    df["avg_monthly_spend"]    = df["total_charges"] / df["tenure_months"].replace(0, 1)
    df["calls_per_month"]      = df["support_calls"] / df["tenure_months"].replace(0, 1)
    df["high_support_user"]    = (df["support_calls"] > 4).astype(int)
    df["long_tenure"]          = (df["tenure_months"] > 36).astype(int)
    df["monthly_contract"]     = (df["contract_type"] == 0).astype(int)
    return df


# ─────────────────────────────────────────────
# 3. Preprocessing & SMOTE
# ─────────────────────────────────────────────
def preprocess(df):
    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Before SMOTE → Train class distribution: {dict(y_train.value_counts())}")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"After  SMOTE → Train class distribution: {dict(pd.Series(y_train_res).value_counts())}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train_res, y_test, X.columns.tolist(), scaler


# ─────────────────────────────────────────────
# 4. Train Models
# ─────────────────────────────────────────────
def train_models(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, random_state=42, C=0.5)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return lr, rf


# ─────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring="accuracy")
    print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return y_pred, acc


# ─────────────────────────────────────────────
# 6. Visualisations
# ─────────────────────────────────────────────
def plot_confusion_matrices(lr, rf, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)

    for ax, model, name in zip(axes, [lr, rf], ["Logistic Regression", "Random Forest"]):
        cm = confusion_matrix(y_test, model.predict(X_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig("outputs/plots/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: outputs/plots/confusion_matrices.png")


def plot_roc_curves(lr, rf, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))

    for model, name, color in zip(
        [lr, rf],
        ["Logistic Regression", "Random Forest"],
        ["#4C72B0", "#DD8452"]
    ):
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Churn Prediction", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/plots/roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: outputs/plots/roc_curves.png")


def plot_feature_importance(rf, feature_names):
    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_feat = [feature_names[i] for i in indices]
    sorted_imp  = importances[indices]

    palette = ["#2ecc71" if i < 5 else "#95a5a6" for i in range(len(sorted_feat))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(sorted_feat[::-1], sorted_imp[::-1], color=palette[::-1], edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Feature Importance — Random Forest", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, sorted_imp[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/plots/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: outputs/plots/feature_importance.png")


def plot_class_distribution(df_before, y_after_smote):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Class Distribution: Before vs After SMOTE", fontsize=13, fontweight="bold")

    colors = ["#3498db", "#e74c3c"]
    labels = ["No Churn", "Churn"]

    for ax, counts, title in zip(
        axes,
        [df_before["churn"].value_counts().sort_index(),
         pd.Series(y_after_smote).value_counts().sort_index()],
        ["Before SMOTE", "After SMOTE"]
    ):
        ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 20, str(v), ha="center", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/plots/smote_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: outputs/plots/smote_distribution.png")


# ─────────────────────────────────────────────
# 7. Main Pipeline
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("   Customer Churn Prediction — ML Pipeline")
    print("="*55)

    # Data
    print("\n[1/5] Generating dataset...")
    df = generate_churn_data()
    print(f"      Shape: {df.shape} | Churn rate: {df['churn'].mean():.1%}")

    # Feature Engineering
    print("\n[2/5] Feature engineering...")
    df = feature_engineering(df)
    print(f"      Features after engineering: {df.shape[1] - 1}")

    # Preprocessing
    print("\n[3/5] Preprocessing + SMOTE...")
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(df)

    # Train
    print("\n[4/5] Training models...")
    lr, rf = train_models(X_train, y_train)
    print("      ✓ Logistic Regression trained")
    print("      ✓ Random Forest trained")

    # Evaluate
    print("\n[5/5] Evaluation + Visualisations...")
    evaluate(lr, X_test, y_test, "Logistic Regression")
    evaluate(rf, X_test, y_test, "Random Forest")

    # Plots
    print("\nGenerating plots...")
    plot_confusion_matrices(lr, rf, X_test, y_test)
    plot_roc_curves(lr, rf, X_test, y_test)
    plot_feature_importance(rf, feature_names)
    plot_class_distribution(df, y_train)

    print("\n" + "="*55)
    print("   Pipeline complete. All outputs saved to outputs/plots/")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
