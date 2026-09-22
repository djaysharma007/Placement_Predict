import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the validated ingestion function
from ingest import load_and_validate_data


def perform_eda():

    # ============================================================
    # 1. LOAD DATA USING INGESTION PIPELINE
    # ============================================================

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    df = load_and_validate_data(DATA_PATH)

    print("\n" + "=" * 60)
    print("--- 1. DATASET DIMENSIONS ---")
    print("=" * 60)

    print(f"Total Rows (Samples): {df.shape[0]}")
    print(f"Total Columns (Metrics): {df.shape[1]}")


    # ============================================================
    # 2. FEATURE NAMES & DATA TYPES
    # ============================================================

    print("\n" + "=" * 60)
    print("--- 2. FEATURE NAMES & DATA TYPES ---")
    print("=" * 60)

    print(df.dtypes)

    # ============================================================
    # 3. MISSING VALUES & DUPLICATES
    # ============================================================

    print("\n" + "=" * 60)
    print("--- 3. MISSING VALUES & DUPLICATES ---")
    print("=" * 60)

    missing_vals = df.isnull().sum()

    if missing_vals.sum() > 0:
        print("Missing Values per Column:")
        print(missing_vals[missing_vals > 0])
    else:
        print("No missing values found.")

    duplicates = df.duplicated().sum()

    print(f"\nDuplicate Records Count: {duplicates}")


    # ============================================================
    # 4. SUMMARY STATISTICS
    # ============================================================

    print("\n" + "=" * 60)
    print("--- 4. SUMMARY STATISTICS ---")
    print("=" * 60)

    print(df.describe())


    # ============================================================
    # 5. CLASS IMBALANCE ANALYSIS
    # ============================================================

    print("\n" + "=" * 60)
    print("--- 5. CLASS IMBALANCE ANALYSIS ---")
    print("=" * 60)

    if "placement_status" in df.columns:

        class_counts = df["placement_status"].value_counts()

        class_percentages = (
            df["placement_status"]
            .value_counts(normalize=True) * 100
        )

        print("Placement Status Counts:")
        print(class_counts)

        print("\nPlacement Status Percentages:")
        print(class_percentages.round(2))

    else:
        print(
            "Target column 'placement_status' "
            "not found for class imbalance analysis."
        )


    # ============================================================
    # 6. VISUALIZATIONS
    # ============================================================

    print("\n" + "=" * 60)
    print("--- 6. GENERATING VISUALIZATIONS ---")
    print("=" * 60)

    sns.set_theme(style="whitegrid")

    # Create output directory
    os.makedirs("reports/figures", exist_ok=True)

    # ------------------------------------------------------------
    # A. CORRELATION MATRIX HEATMAP
    # ------------------------------------------------------------

    plt.figure(figsize=(14, 10))

    numerical_df = df.select_dtypes(include=[np.number])

    corr_matrix = numerical_df.corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Feature Correlation Matrix Heatmap")
    plt.tight_layout()

    plt.savefig(
        "reports/figures/correlation_heatmap.png",
        dpi=300
    )

    plt.close()

    print(
        "-> Saved correlation heatmap to "
        "reports/figures/correlation_heatmap.png"
    )


    # ------------------------------------------------------------
    # B. CGPA VS SALARY PACKAGE
    # ------------------------------------------------------------

    if (
        "cgpa" in df.columns
        and "salary_package_lpa" in df.columns
    ):

        plt.figure(figsize=(8, 6))

        sns.scatterplot(
            data=df,
            x="cgpa",
            y="salary_package_lpa",
            hue="placement_status",
            alpha=0.7
        )

        plt.title(
            "CGPA vs Salary Package "
            "(Colored by Placement Status)"
        )

        plt.tight_layout()

        plt.savefig(
            "reports/figures/scatter_cgpa_salary.png",
            dpi=300
        )

        plt.close()

        print(
            "-> Saved scatter plot to "
            "reports/figures/scatter_cgpa_salary.png"
        )


    # ------------------------------------------------------------
    # C. PAIR PLOT
    # ------------------------------------------------------------

    try:

        pairplot_cols = [
            "cgpa",
            "backlogs",
            "communication_skill_score",
            "internships_count",
            "salary_package_lpa"
        ]

        valid_pair_cols = [
            col for col in pairplot_cols
            if col in df.columns
        ]

        if len(valid_pair_cols) > 1:

            pp = sns.pairplot(
                df[valid_pair_cols],
                diag_kind="hist",
                corner=True
            )

            pp.fig.suptitle(
                "Pairwise Relationships of Key Numerical Features",
                y=1.02
            )

            pp.savefig(
                "reports/figures/pairplot_features.png",
                dpi=300
            )

            plt.close("all")

            print(
                "-> Saved pair plot to "
                "reports/figures/pairplot_features.png"
            )

    except Exception as e:

        print(f"Skipping pair plot due to: {e}")


    # ------------------------------------------------------------
    # D. OUTLIER DETECTION USING BOXPLOTS
    # ------------------------------------------------------------

    plt.figure(figsize=(14, 10))

    sns.boxplot(
        data=numerical_df,
        orient="h"
    )

    plt.title(
        "Outlier Identification via Boxplots "
        "(Numerical Features)"
    )

    plt.tight_layout()

    plt.savefig(
        "reports/figures/outliers_boxplot.png",
        dpi=300
    )

    plt.close()

    print(
        "-> Saved outlier boxplot to "
        "reports/figures/outliers_boxplot.png"
    )


    # ============================================================
    # COMPLETION MESSAGE
    # ============================================================

    print("\n" + "=" * 60)
    print("EDA EXECUTION COMPLETE")
    print("=" * 60)

    print(
        "Visualizations stored in: reports/figures/"
    )


# ================================================================
# MAIN PROGRAM
# ================================================================

if __name__ == "__main__":
    perform_eda()