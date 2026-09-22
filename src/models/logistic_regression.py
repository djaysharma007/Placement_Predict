import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from src.data.ingest import load_and_validate_data


def binary_logistic_regression(df):
    """
    Binary classification:
    Predict placement_status using CGPA and aptitude score.
    """

    print("\n" + "=" * 60)
    print("--- PART A: BINARY LOGISTIC REGRESSION ---")
    print("=" * 60)

    feature_cols = [
        "cgpa",
        "aptitude_score"
    ]

    target_col = "placement_status"

    # ------------------------------------------------------------
    # Select required columns
    # ------------------------------------------------------------

    df_binary = df.dropna(
        subset=feature_cols + [target_col]
    ).copy()

    X = df_binary[feature_cols].values
    y_raw = df_binary[target_col].values

    # ------------------------------------------------------------
    # Encode placement status
    # ------------------------------------------------------------

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        y_raw
    )

    print("\nPlacement classes:")

    for index, class_name in enumerate(
        label_encoder.classes_
    ):
        print(
            f"{index} -> {class_name}"
        )

    # ------------------------------------------------------------
    # Train-test split
    # ------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ------------------------------------------------------------
    # Feature scaling
    # ------------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ------------------------------------------------------------
    # Train logistic regression
    # ------------------------------------------------------------

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------

    y_pred = model.predict(
        X_test_scaled
    )

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        f"\nBinary Classification Accuracy: "
        f"{accuracy:.4f}"
    )

    print("\n--- Classification Report ---")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_
        )
    )

    # ------------------------------------------------------------
    # Decision Boundary
    # ------------------------------------------------------------

    plt.figure(figsize=(9, 7))

    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5

    y_min = X[:, 1].min() - 5
    y_max = X[:, 1].max() + 5

    xx, yy = np.meshgrid(
        np.linspace(
            x_min,
            x_max,
            200
        ),
        np.linspace(
            y_min,
            y_max,
            200
        )
    )

    grid_points = np.c_[
        xx.ravel(),
        yy.ravel()
    ]

    grid_scaled = scaler.transform(
        grid_points
    )

    Z = model.predict(
        grid_scaled
    )

    Z = Z.reshape(
        xx.shape
    )

    plt.contourf(
        xx,
        yy,
        Z,
        alpha=0.3
    )

    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        edgecolors="k",
        alpha=0.8
    )

    plt.xlabel("CGPA")
    plt.ylabel("Aptitude Score")

    plt.title(
        "Binary Logistic Regression "
        "Decision Boundary"
    )

    plt.tight_layout()

    output_path = (
        "reports/figures/"
        "logistic_binary_decision_boundary.png"
    )

    plt.savefig(output_path)
    plt.close()

    print(
        f"\n-> Saved binary decision boundary to "
        f"{output_path}"
    )


def multiclass_logistic_regression(df):
    """
    Multiclass classification:
    Predict branch using CGPA and aptitude score.
    """

    print("\n" + "=" * 60)
    print("--- PART B: MULTICLASS LOGISTIC REGRESSION ---")
    print("=" * 60)

    feature_cols = [
        "cgpa",
        "aptitude_score"
    ]

    target_col = "branch"

    # ------------------------------------------------------------
    # Select required columns
    # ------------------------------------------------------------

    df_multi = df.dropna(
        subset=feature_cols + [target_col]
    ).copy()

    X = df_multi[feature_cols].values
    y_raw = df_multi[target_col].values

    # ------------------------------------------------------------
    # Encode branch names
    # ------------------------------------------------------------

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        y_raw
    )

    print("\nBranch classes:")

    for index, class_name in enumerate(
        label_encoder.classes_
    ):
        print(
            f"{index} -> {class_name}"
        )

    # ------------------------------------------------------------
    # Train-test split
    # ------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ------------------------------------------------------------
    # Feature scaling
    # ------------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ------------------------------------------------------------
    # Multinomial Logistic Regression
    # ------------------------------------------------------------

    multinomial_model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000
    )

    multinomial_model.fit(
        X_train_scaled,
        y_train
    )

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------

    y_pred = multinomial_model.predict(
        X_test_scaled
    )

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        f"\nMulticlass Accuracy: "
        f"{accuracy:.4f}"
    )

    print("\n--- Multiclass Classification Report ---")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_
        )
    )

    # ------------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )

    plt.title(
        "Confusion Matrix - "
        "Multiclass Branch Classification"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    output_path = (
        "reports/figures/"
        "logistic_multiclass_confusion_matrix.png"
    )

    plt.savefig(output_path)
    plt.close()

    print(
        f"\n-> Saved multiclass confusion matrix to "
        f"{output_path}"
    )


def run_logistic_regression_experiment():

    # ============================================================
    # LOAD DATA USING EXISTING INGESTION PIPELINE
    # ============================================================

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    df = load_and_validate_data(
        DATA_PATH
    )

    # ============================================================
    # PART A - BINARY CLASSIFICATION
    # ============================================================

    binary_logistic_regression(df)

    # ============================================================
    # PART B - MULTICLASS CLASSIFICATION
    # ============================================================

    multiclass_logistic_regression(df)

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_logistic_regression_experiment()