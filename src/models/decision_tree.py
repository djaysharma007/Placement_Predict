import os
import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


# ============================================================
# 1. LOAD DATA
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
# 3. CONVERT COLLEGE TIER TO NUMERIC
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
# 5. ENCODE TARGET
# ============================================================

if df[TARGET].dtype == "object":

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

else:

    target_labels = sorted(
        df[TARGET].unique()
    )


if df[TARGET].nunique() != 2:

    raise ValueError(
        "placement_status must contain exactly two classes."
    )


# ============================================================
# 6. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]

print(f"Number of features: {X.shape[1]}")
print(f"Target classes: {target_labels}")


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

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# 8. DECISION TREE - GINI
# ============================================================

print("\n" + "=" * 60)
print("DECISION TREE - GINI")
print("=" * 60)

start_time = time.time()

tree_gini = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

tree_gini.fit(
    X_train,
    y_train
)

gini_predictions = tree_gini.predict(
    X_test
)

gini_accuracy = accuracy_score(
    y_test,
    gini_predictions
)

gini_time = time.time() - start_time

print(f"Gini Test Accuracy: {gini_accuracy:.4f}")
print(f"Gini Tree Depth   : {tree_gini.get_depth()}")
print(f"Gini Number Nodes : {tree_gini.tree_.node_count}")
print(f"Training Time     : {gini_time:.4f} seconds")


# ============================================================
# 9. GINI TREE VISUALIZATION
# ============================================================

plt.figure(figsize=(22, 12))

plot_tree(
    tree_gini,
    feature_names=X.columns,
    class_names=[
        str(label)
        for label in target_labels
    ],
    filled=True,
    rounded=True,
    max_depth=4,
    fontsize=7
)

plt.title(
    "Decision Tree using Gini Criterion"
)

plt.tight_layout()

gini_path = os.path.join(
    FIGURES_DIR,
    "decision_tree_gini.png"
)

plt.savefig(
    gini_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {gini_path}")


# ============================================================
# 10. DECISION TREE - ENTROPY
# ============================================================

print("\n" + "=" * 60)
print("DECISION TREE - ENTROPY")
print("=" * 60)

start_time = time.time()

tree_entropy = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

tree_entropy.fit(
    X_train,
    y_train
)

entropy_predictions = tree_entropy.predict(
    X_test
)

entropy_accuracy = accuracy_score(
    y_test,
    entropy_predictions
)

entropy_time = time.time() - start_time

print(
    f"Entropy Test Accuracy: "
    f"{entropy_accuracy:.4f}"
)

print(
    f"Entropy Tree Depth   : "
    f"{tree_entropy.get_depth()}"
)

print(
    f"Entropy Number Nodes : "
    f"{tree_entropy.tree_.node_count}"
)

print(
    f"Training Time        : "
    f"{entropy_time:.4f} seconds"
)


# ============================================================
# 11. ENTROPY TREE VISUALIZATION
# ============================================================

plt.figure(figsize=(22, 12))

plot_tree(
    tree_entropy,
    feature_names=X.columns,
    class_names=[
        str(label)
        for label in target_labels
    ],
    filled=True,
    rounded=True,
    max_depth=4,
    fontsize=7
)

plt.title(
    "Decision Tree using Entropy Criterion"
)

plt.tight_layout()

entropy_path = os.path.join(
    FIGURES_DIR,
    "decision_tree_entropy.png"
)

plt.savefig(
    entropy_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {entropy_path}")


# ============================================================
# 12. COST COMPLEXITY PRUNING
# ============================================================

print("\n" + "=" * 60)
print("COST COMPLEXITY PRUNING")
print("=" * 60)

print("Calculating pruning path...")

path = tree_gini.cost_complexity_pruning_path(
    X_train,
    y_train
)

ccp_alphas = path.ccp_alphas

print(
    f"Available CCP alpha values: "
    f"{len(ccp_alphas)}"
)


# ------------------------------------------------------------
# Reduce the number of alpha values evaluated.
# This avoids training thousands of large trees.
# ------------------------------------------------------------

MAX_ALPHA_TESTS = 30

if len(ccp_alphas) > MAX_ALPHA_TESTS:

    alpha_indices = [
        int(i)
        for i in
        __import__("numpy").linspace(
            0,
            len(ccp_alphas) - 1,
            MAX_ALPHA_TESTS
        )
    ]

    selected_alphas = [
        ccp_alphas[i]
        for i in alpha_indices
    ]

else:

    selected_alphas = ccp_alphas


print(
    f"Testing {len(selected_alphas)} "
    f"representative alpha values..."
)


train_scores = []
test_scores = []
tree_sizes = []


for index, alpha in enumerate(
    selected_alphas,
    start=1
):

    print(
        f"Testing alpha "
        f"{index}/{len(selected_alphas)}: "
        f"{alpha:.8f}"
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=42,
        ccp_alpha=alpha
    )

    model.fit(
        X_train,
        y_train
    )

    train_accuracy = accuracy_score(
        y_train,
        model.predict(X_train)
    )

    test_accuracy = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    train_scores.append(
        train_accuracy
    )

    test_scores.append(
        test_accuracy
    )

    tree_sizes.append(
        model.tree_.node_count
    )


# ============================================================
# 13. FIND BEST PRUNED MODEL
# ============================================================

best_index = max(
    range(len(test_scores)),
    key=lambda i: test_scores[i]
)

best_alpha = selected_alphas[best_index]

best_pruned_accuracy = (
    test_scores[best_index]
)

best_pruned_tree_size = (
    tree_sizes[best_index]
)

print(
    f"\nBest CCP alpha       : "
    f"{best_alpha:.8f}"
)

print(
    f"Best pruned accuracy : "
    f"{best_pruned_accuracy:.4f}"
)

print(
    f"Pruned tree nodes    : "
    f"{best_pruned_tree_size}"
)


# ============================================================
# 14. TRAIN FINAL PRUNED TREE
# ============================================================

pruned_tree = DecisionTreeClassifier(
    criterion="gini",
    random_state=42,
    ccp_alpha=best_alpha
)

pruned_tree.fit(
    X_train,
    y_train
)


# ============================================================
# 15. PLOT CCP RESULTS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    selected_alphas,
    train_scores,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    selected_alphas,
    test_scores,
    marker="o",
    label="Testing Accuracy"
)

plt.xlabel("CCP Alpha")
plt.ylabel("Accuracy")

plt.title(
    "Decision Tree Cost Complexity Pruning"
)

plt.legend()
plt.grid(True)

ccp_path = os.path.join(
    FIGURES_DIR,
    "decision_tree_ccp.png"
)

plt.savefig(
    ccp_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {ccp_path}")


# ============================================================
# 16. TREE DEPTH ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("TREE DEPTH ANALYSIS")
print("=" * 60)

depths = range(1, 16)

depth_train_scores = []
depth_test_scores = []


for depth in depths:

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    train_accuracy = accuracy_score(
        y_train,
        model.predict(X_train)
    )

    test_accuracy = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    depth_train_scores.append(
        train_accuracy
    )

    depth_test_scores.append(
        test_accuracy
    )


# ============================================================
# 17. BEST TREE DEPTH
# ============================================================

best_depth_index = max(
    range(len(depth_test_scores)),
    key=lambda i: depth_test_scores[i]
)

best_depth = list(
    depths
)[best_depth_index]

best_depth_accuracy = (
    depth_test_scores[best_depth_index]
)

print(
    f"Best tree depth     : "
    f"{best_depth}"
)

print(
    f"Best test accuracy  : "
    f"{best_depth_accuracy:.4f}"
)


# ============================================================
# 18. PLOT DEPTH RESULTS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    list(depths),
    depth_train_scores,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    list(depths),
    depth_test_scores,
    marker="o",
    label="Testing Accuracy"
)

plt.xlabel(
    "Maximum Tree Depth"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Decision Tree Accuracy vs Tree Depth"
)

plt.legend()
plt.grid(True)

depth_path = os.path.join(
    FIGURES_DIR,
    "decision_tree_depth.png"
)

plt.savefig(
    depth_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {depth_path}")


# ============================================================
# 19. RESULTS SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"Gini Accuracy       : "
    f"{gini_accuracy:.4f}"
)

print(
    f"Entropy Accuracy    : "
    f"{entropy_accuracy:.4f}"
)

print(
    f"Best CCP Accuracy   : "
    f"{best_pruned_accuracy:.4f}"
)

print(
    f"Best Tree Depth     : "
    f"{best_depth}"
)

print(
    f"Best Depth Accuracy : "
    f"{best_depth_accuracy:.4f}"
)

print("\nGenerated figures:")

print(
    f"1. {gini_path}"
)

print(
    f"2. {entropy_path}"
)

print(
    f"3. {ccp_path}"
)

print(
    f"4. {depth_path}"
)