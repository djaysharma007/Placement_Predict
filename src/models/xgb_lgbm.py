import os
import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = os.path.join(
    "src",
    "data",
    "raw_placement_data.csv"
)

FIGURES_DIR = os.path.join(
    "reports",
    "figures"
)

os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading placement dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. SELECT FEATURES
# ============================================================

FEATURES = [
    "branch",
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count"
]

TARGET = "placement_status"

df = df[FEATURES + [TARGET]].dropna()

print(
    f"Samples after removing missing values: "
    f"{len(df)}"
)


# ============================================================
# 3. CONVERT COLLEGE TIER
# ============================================================

df["college_tier"] = (
    df["college_tier"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
)

df["college_tier"] = pd.to_numeric(
    df["college_tier"],
    errors="coerce"
)

df = df.dropna(
    subset=["college_tier"]
)


# ============================================================
# 4. ONE-HOT ENCODE BRANCH
# ============================================================

df = pd.get_dummies(
    df,
    columns=["branch"],
    drop_first=False
)

for column in df.columns:

    if df[column].dtype == "bool":

        df[column] = df[column].astype(int)


# ============================================================
# 5. CONVERT TARGET TO 0/1
# ============================================================

target_labels = sorted(
    df[TARGET].astype(str).unique()
)

target_mapping = {
    label: index
    for index, label in enumerate(target_labels)
}

df[TARGET] = (
    df[TARGET]
    .astype(str)
    .map(target_mapping)
)

print(
    f"Target mapping: {target_mapping}"
)

if df[TARGET].nunique() != 2:

    raise ValueError(
        "placement_status must contain exactly two classes."
    )


# ============================================================
# 6. CREATE X AND y
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]

print(
    f"Number of features: {X.shape[1]}"
)

print(
    f"Target classes: {target_labels}"
)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


# ============================================================
# 8. XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST CLASSIFICATION")
print("=" * 60)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

start_time = time.time()

xgb_model.fit(
    X_train,
    y_train
)

xgb_training_time = time.time() - start_time

xgb_predictions = xgb_model.predict(
    X_test
)

xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)

print(
    f"Training Time: "
    f"{xgb_training_time:.4f} seconds"
)

print(
    f"Test Accuracy: "
    f"{xgb_accuracy:.4f}"
)

print("\nXGBoost Classification Report:")

print(
    classification_report(
        y_test,
        xgb_predictions,
        target_names=target_labels,
        zero_division=0
    )
)


# ============================================================
# 9. LIGHTGBM
# ============================================================

print("\n" + "=" * 60)
print("LIGHTGBM CLASSIFICATION")
print("=" * 60)

lgbm_model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1,
    n_jobs=-1
)

start_time = time.time()

lgbm_model.fit(
    X_train,
    y_train
)

lgbm_training_time = time.time() - start_time

lgbm_predictions = lgbm_model.predict(
    X_test
)

lgbm_accuracy = accuracy_score(
    y_test,
    lgbm_predictions
)

print(
    f"Training Time: "
    f"{lgbm_training_time:.4f} seconds"
)

print(
    f"Test Accuracy: "
    f"{lgbm_accuracy:.4f}"
)

print("\nLightGBM Classification Report:")

print(
    classification_report(
        y_test,
        lgbm_predictions,
        target_names=target_labels,
        zero_division=0
    )
)


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST VS LIGHTGBM")
print("=" * 60)

print(
    f"XGBoost  Accuracy: "
    f"{xgb_accuracy:.4f}"
)

print(
    f"LightGBM Accuracy: "
    f"{lgbm_accuracy:.4f}"
)

print(
    f"XGBoost  Time    : "
    f"{xgb_training_time:.4f} seconds"
)

print(
    f"LightGBM Time    : "
    f"{lgbm_training_time:.4f} seconds"
)


comparison_df = pd.DataFrame({
    "Model": [
        "XGBoost",
        "LightGBM"
    ],
    "Accuracy": [
        xgb_accuracy,
        lgbm_accuracy
    ],
    "Training_Time_Seconds": [
        xgb_training_time,
        lgbm_training_time
    ]
})

comparison_path = os.path.join(
    FIGURES_DIR,
    "xgb_lightgbm_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

print(
    f"Saved: {comparison_path}"
)


# ============================================================
# 11. SHAP ANALYSIS - XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("SHAP ANALYSIS - XGBOOST")
print("=" * 60)

print("Creating SHAP explainer...")

xgb_explainer = shap.TreeExplainer(
    xgb_model
)

# Use a sample for faster SHAP processing

shap_sample = X_test.sample(
    n=min(2000, len(X_test)),
    random_state=42
)

print(
    f"Calculating SHAP values for "
    f"{len(shap_sample)} samples..."
)

xgb_shap_values = xgb_explainer(
    shap_sample
)


# ============================================================
# 12. SHAP FEATURE IMPORTANCE CSV
# ============================================================

shap_importance = pd.DataFrame({
    "Feature": X.columns,
    "Mean_Absolute_SHAP": (
        abs(xgb_shap_values.values).mean(axis=0)
    )
})

shap_importance = shap_importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

shap_csv_path = os.path.join(
    FIGURES_DIR,
    "xgb_shap_feature_importance.csv"
)

shap_importance.to_csv(
    shap_csv_path,
    index=False
)

print(
    f"Saved: {shap_csv_path}"
)

print("\nTop influential features:")

print(
    shap_importance.head(10).to_string(
        index=False
    )
)


# ============================================================
# 13. SHAP SUMMARY PLOT
# ============================================================

shap.summary_plot(
    xgb_shap_values,
    shap_sample,
    show=False
)

plt.tight_layout()

shap_summary_path = os.path.join(
    FIGURES_DIR,
    "xgb_shap_summary.png"
)

plt.savefig(
    shap_summary_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {shap_summary_path}"
)


# ============================================================
# 14. SHAP DEPENDENCE PLOT
# ============================================================

top_feature = shap_importance.iloc[0]["Feature"]

print(
    f"\nTop feature for dependence plot: "
    f"{top_feature}"
)

shap.dependence_plot(
    top_feature,
    xgb_shap_values.values,
    shap_sample,
    show=False
)

plt.tight_layout()

dependence_path = os.path.join(
    FIGURES_DIR,
    "xgb_shap_dependence.png"
)

plt.savefig(
    dependence_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {dependence_path}"
)


# ============================================================
# 15. OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("GENERATED OUTPUTS")
print("=" * 60)

print(f"1. {comparison_path}")
print(f"2. {shap_csv_path}")
print(f"3. {shap_summary_path}")
print(f"4. {dependence_path}")