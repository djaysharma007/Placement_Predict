import os
import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


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
# 2. SELECT FEATURES AND TARGET
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

print(f"Samples after removing missing values: {len(df)}")


# ============================================================
# 3. CONVERT COLLEGE TIER
# ============================================================

# Convert:
# Tier 1 -> 1
# Tier 2 -> 2
# Tier 3 -> 3

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

# Convert boolean columns to integers

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

print(f"Target mapping: {target_mapping}")


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


print(f"Number of features: {X.shape[1]}")
print(
    f"Target classes: "
    f"{target_labels}"
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
    f"Training samples: "
    f"{len(X_train)}"
)

print(
    f"Testing samples : "
    f"{len(X_test)}"
)


# ============================================================
# 8. DECISION TREE BASELINE
# ============================================================

print("\n" + "=" * 60)
print("DECISION TREE BASELINE")
print("=" * 60)

start_time = time.time()

decision_tree = DecisionTreeClassifier(
    random_state=42
)

decision_tree.fit(
    X_train,
    y_train
)

tree_predictions = decision_tree.predict(
    X_test
)

tree_accuracy = accuracy_score(
    y_test,
    tree_predictions
)

tree_precision = precision_score(
    y_test,
    tree_predictions,
    pos_label=1,
    zero_division=0
)

tree_recall = recall_score(
    y_test,
    tree_predictions,
    pos_label=1,
    zero_division=0
)

tree_f1 = f1_score(
    y_test,
    tree_predictions,
    pos_label=1,
    zero_division=0
)

tree_time = time.time() - start_time


print(
    f"Accuracy       : "
    f"{tree_accuracy:.4f}"
)

print(
    f"Precision      : "
    f"{tree_precision:.4f}"
)

print(
    f"Recall         : "
    f"{tree_recall:.4f}"
)

print(
    f"F1 Score       : "
    f"{tree_f1:.4f}"
)

print(
    f"Training Time  : "
    f"{tree_time:.4f} seconds"
)


# ============================================================
# 9. RANDOM FOREST BASELINE
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST BASELINE")
print("=" * 60)

start_time = time.time()

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(
    X_train,
    y_train
)

forest_predictions = random_forest.predict(
    X_test
)

forest_accuracy = accuracy_score(
    y_test,
    forest_predictions
)

forest_precision = precision_score(
    y_test,
    forest_predictions,
    pos_label=1,
    zero_division=0
)

forest_recall = recall_score(
    y_test,
    forest_predictions,
    pos_label=1,
    zero_division=0
)

forest_f1 = f1_score(
    y_test,
    forest_predictions,
    pos_label=1,
    zero_division=0
)

forest_time = time.time() - start_time


print(
    f"Accuracy       : "
    f"{forest_accuracy:.4f}"
)

print(
    f"Precision      : "
    f"{forest_precision:.4f}"
)

print(
    f"Recall         : "
    f"{forest_recall:.4f}"
)

print(
    f"F1 Score       : "
    f"{forest_f1:.4f}"
)

print(
    f"OOB Score      : "
    f"{random_forest.oob_score_:.4f}"
)

print(
    f"OOB Error      : "
    f"{1 - random_forest.oob_score_:.4f}"
)

print(
    f"Training Time  : "
    f"{forest_time:.4f} seconds"
)


# ============================================================
# 10. NUMBER OF TREES ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("NUMBER OF TREES ANALYSIS")
print("=" * 60)

tree_counts = [
    10,
    25,
    50,
    100,
    200
]

tree_count_accuracies = []

tree_count_times = []


for n_trees in tree_counts:

    print(
        f"Training Random Forest "
        f"with {n_trees} trees..."
    )

    start_time = time.time()

    model = RandomForestClassifier(
        n_estimators=n_trees,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    elapsed_time = (
        time.time() - start_time
    )

    tree_count_accuracies.append(
        accuracy
    )

    tree_count_times.append(
        elapsed_time
    )

    print(
        f"Accuracy: {accuracy:.4f} | "
        f"Time: {elapsed_time:.2f}s"
    )


# Plot number of trees

plt.figure(figsize=(10, 6))

plt.plot(
    tree_counts,
    tree_count_accuracies,
    marker="o"
)

plt.xlabel(
    "Number of Trees"
)

plt.ylabel(
    "Test Accuracy"
)

plt.title(
    "Random Forest Accuracy vs Number of Trees"
)

plt.grid(True)

trees_path = os.path.join(
    FIGURES_DIR,
    "random_forest_number_of_trees.png"
)

plt.savefig(
    trees_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {trees_path}"
)


# ============================================================
# 11. FEATURE SUBSAMPLING ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SUBSAMPLING ANALYSIS")
print("=" * 60)

feature_options = [
    "sqrt",
    "log2",
    None
]

feature_labels = [
    "sqrt",
    "log2",
    "None"
]

feature_accuracies = []


for max_features in feature_options:

    print(
        f"Training Random Forest "
        f"with max_features={max_features}..."
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    feature_accuracies.append(
        accuracy
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )


# Plot feature subsampling

plt.figure(figsize=(10, 6))

plt.bar(
    feature_labels,
    feature_accuracies
)

plt.xlabel(
    "max_features"
)

plt.ylabel(
    "Test Accuracy"
)

plt.title(
    "Random Forest Accuracy vs Feature Subsampling"
)

plt.grid(
    axis="y"
)

feature_path = os.path.join(
    FIGURES_DIR,
    "random_forest_feature_subsampling.png"
)

plt.savefig(
    feature_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {feature_path}"
)


# ============================================================
# 12. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    f"Decision Tree Accuracy : "
    f"{tree_accuracy:.4f}"
)

print(
    f"Random Forest Accuracy : "
    f"{forest_accuracy:.4f}"
)

print(
    f"Decision Tree F1       : "
    f"{tree_f1:.4f}"
)

print(
    f"Random Forest F1       : "
    f"{forest_f1:.4f}"
)

print(
    f"Random Forest OOB Score: "
    f"{random_forest.oob_score_:.4f}"
)

print("\nGenerated figures:")

print(
    f"1. {trees_path}"
)

print(
    f"2. {feature_path}"
)