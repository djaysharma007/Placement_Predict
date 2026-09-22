import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from src.data.ingest import load_and_validate_data


def run_ridge_polynomial_experiment():

    # ============================================================
    # 1. LOAD DATA
    # ============================================================

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    df = load_and_validate_data(DATA_PATH)

    print("\n" + "=" * 60)
    print("--- RIDGE REGRESSION WITH POLYNOMIAL FEATURES ---")
    print("=" * 60)

    # ============================================================
    # 2. SELECT FEATURES AND TARGET
    # ============================================================

    feature_col = "cgpa"
    target_col = "salary_package_lpa"

    # Salary is meaningful for placed students.
    # Remove missing and zero salary values.
    df_clean = df.dropna(
        subset=[feature_col, target_col]
    ).copy()

    df_clean = df_clean[
        df_clean[target_col] > 0
    ]

    X = df_clean[[feature_col]].values
    y = df_clean[target_col].values

    print(f"\nNumber of valid samples: {len(X)}")
    print(f"Input feature: {feature_col}")
    print(f"Target: {target_col}")

    # ============================================================
    # 3. TRAIN-TEST SPLIT
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nTrain samples:", len(X_train))
    print("Test samples :", len(X_test))

    # ============================================================
    # 4. CREATE HIGH-ORDER POLYNOMIAL FEATURES
    # ============================================================

    poly_degree = 15

    poly = PolynomialFeatures(
        degree=poly_degree
    )

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    print(f"\nPolynomial degree: {poly_degree}")
    print(
        f"Polynomial feature dimensions: "
        f"{X_train_poly.shape[1]}"
    )

    # ============================================================
    # 5. STANDARDIZE FEATURES
    # ============================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_poly
    )

    X_test_scaled = scaler.transform(
        X_test_poly
    )

    # ============================================================
    # 6. DEFINE REGULARIZATION PARAMETERS
    # ============================================================

    lambdas = np.logspace(
        -4,
        4,
        100
    )

    train_errors = []
    test_errors = []

    # ============================================================
    # 7. TRAIN RIDGE MODELS
    # ============================================================

    for lam in lambdas:

        ridge_model = Ridge(
            alpha=lam
        )

        ridge_model.fit(
            X_train_scaled,
            y_train
        )

        # Training predictions
        y_train_pred = ridge_model.predict(
            X_train_scaled
        )

        # Testing predictions
        y_test_pred = ridge_model.predict(
            X_test_scaled
        )

        # Calculate MSE
        train_mse = mean_squared_error(
            y_train,
            y_train_pred
        )

        test_mse = mean_squared_error(
            y_test,
            y_test_pred
        )

        train_errors.append(train_mse)
        test_errors.append(test_mse)

    # ============================================================
    # 8. FIND BEST REGULARIZATION PARAMETER
    # ============================================================

    best_index = np.argmin(test_errors)

    best_lambda = lambdas[best_index]
    best_test_error = test_errors[best_index]

    print("\n" + "=" * 60)
    print("--- RIDGE REGULARIZATION RESULTS ---")
    print("=" * 60)

    print(
        f"Best regularization parameter "
        f"(lambda / alpha): {best_lambda:.6f}"
    )

    print(
        f"Minimum test MSE: {best_test_error:.4f}"
    )

    print(
        f"Training MSE at best lambda: "
        f"{train_errors[best_index]:.4f}"
    )

    # ============================================================
    # 9. PLOT TRAINING AND TESTING ERROR
    # ============================================================

    os.makedirs(
        "reports/figures",
        exist_ok=True
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        lambdas,
        train_errors,
        label="Training Error"
    )

    plt.plot(
        lambdas,
        test_errors,
        label="Testing Error",
        linestyle="--"
    )

    plt.xscale("log")

    plt.xlabel(
        "Regularization Parameter (lambda / alpha)"
    )

    plt.ylabel(
        "Mean Squared Error"
    )

    plt.title(
        "Ridge Regularization Path "
        "(Polynomial Degree 15)"
    )

    plt.legend()
    plt.grid(True, which="both")

    plt.tight_layout()

    output_path = (
        "reports/figures/"
        "ridge_regularization_error_curve.png"
    )

    plt.savefig(output_path)
    plt.close()

    print(
        f"\n-> Saved regularization error curve to "
        f"{output_path}"
    )

    # ============================================================
    # 10. TRAIN FINAL RIDGE MODEL
    # ============================================================

    final_model = Ridge(
        alpha=best_lambda
    )

    final_model.fit(
        X_train_scaled,
        y_train
    )

    print("\nFinal Ridge model trained successfully.")


if __name__ == "__main__":
    run_ridge_polynomial_experiment()