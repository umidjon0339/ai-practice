"""
LESSON 5 — MODEL SELECTION: Picking the Best Model Honestly
===========================================================
Real-life problem: Your boss asks "which model should we deploy for churn
prediction, and with what settings?" You must answer with EVIDENCE.

Two traps this lesson teaches you to avoid:
  1. A single train/test split can be LUCKY or UNLUCKY.
     -> Cross-validation: test on 5 different splits, average the results.
  2. Hand-tuning settings (hyperparameters) by re-running against the
     test set = slowly overfitting to your test set.
     -> GridSearchCV: systematic, automatic, cross-validated tuning.

NEW IDEAS: cross_val_score, GridSearchCV, hyperparameters
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =====================================================================
# 1. DATA — same telecom churn idea as lesson 3 (numeric-only, for focus)
# =====================================================================
rng = np.random.default_rng(0)
n = 1000
df = pd.DataFrame({
    "monthly_bill": rng.normal(60, 25, n).clip(15, 150),
    "months_subscribed": rng.integers(1, 72, n),
    "support_calls": rng.poisson(1.5, n),
    "data_usage_gb": rng.exponential(20, n),
})
churn_score = (
    0.03 * df["monthly_bill"] - 0.04 * df["months_subscribed"]
    + 0.35 * df["support_calls"] + rng.normal(0, 1, n)
)
y = (churn_score > 0.5).astype(int)
X = df

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"{len(X_train)} customers for training/selection, {len(X_test)} locked away for the final exam\n")

# =====================================================================
# 2. CROSS-VALIDATION — a fairer grade than one split
# =====================================================================
# cv=5 splits the TRAINING data into 5 folds. Train on 4, test on the
# 5th; rotate 5 times. You get 5 scores -> a mean AND a spread.
candidates = {
    "LogisticRegression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression()),
    ]),
    "KNeighbors": Pipeline([
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier()),
    ]),
    "RandomForest": RandomForestClassifier(random_state=42),
}

print("5-fold cross-validation (mean accuracy ± spread):")
for name, candidate in candidates.items():
    scores = cross_val_score(candidate, X_train, y_train, cv=5)
    print(f"  {name:20s} {scores.mean():.1%} ± {scores.std():.1%}   folds: {np.round(scores, 3)}")
print("  -> Notice folds differ! A single split could have misled you.\n")

# =====================================================================
# 3. GRIDSEARCHCV — tune hyperparameters systematically
# =====================================================================
# Hyperparameters = settings YOU choose before training (tree depth,
# number of neighbors...). GridSearchCV tries every combination with
# cross-validation and keeps the best. Here: 3 x 3 = 9 combos x 5 folds
# = 45 trainings, all automatic.
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={
        "n_estimators": [50, 150, 300],
        "max_depth": [3, 6, None],
    },
    cv=5,
)
grid.fit(X_train, y_train)

print("GridSearchCV over RandomForest settings:")
print(f"  best settings: {grid.best_params_}")
print(f"  best cross-val accuracy: {grid.best_score_:.1%}\n")

# =====================================================================
# 4. FINAL EXAM — touch the test set ONCE, at the very end
# =====================================================================
# grid.best_estimator_ is the winning model, already retrained on ALL
# training data. Only NOW do we look at the held-out test set.
final_model = grid.best_estimator_
print(f"Final honest score on never-touched test data: {final_model.score(X_test, y_test):.1%}")
print("\nThis number is what you report to your boss.")

# =====================================================================
# TRY IT YOURSELF:
#  - Add "model__n_neighbors": [3, 5, 11] tuning for KNeighbors — with a
#    Pipeline, grid keys use the step name prefix: "model__<param>".
#  - Run cross_val_score with cv=10. Tighter or wider spread?
#  - Print pd.DataFrame(grid.cv_results_)[["params", "mean_test_score"]]
#    to see EVERY combination's score.
# =====================================================================
