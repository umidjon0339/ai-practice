"""
LESSON 4 — TREES & FORESTS: Loan Default Prediction
===================================================
Real-life problem: You work at a bank. Given an applicant's income, debt,
credit history..., predict whether they will DEFAULT (fail to repay).
Banks run models like this on every single loan application.

NEW IDEAS:
  - DecisionTree: a flowchart of if/else questions the model learns itself
  - Overfitting: a model that memorizes training data but fails on new data
  - RandomForest: hundreds of trees voting together — the workhorse of
    tabular ML (often your best first "serious" model)
  - Feature importance: WHICH facts matter most for the decision?
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================================================================
# 1. THE DATA — 2000 past loan applications
# =====================================================================
rng = np.random.default_rng(7)
n = 2000

df = pd.DataFrame({
    "annual_income": rng.lognormal(10.8, 0.5, n).round(0),      # ~$50k typical
    "loan_amount": rng.lognormal(9.5, 0.7, n).round(0),         # ~$13k typical
    "credit_score": rng.normal(680, 80, n).clip(300, 850).round(0),
    "years_employed": rng.exponential(6, n).clip(0, 40).round(1),
    "existing_debts": rng.integers(0, 6, n),
})

# True risk rule (hidden from the model): high debt-to-income + low
# credit score + short employment => default.
debt_ratio = df["loan_amount"] / df["annual_income"]
risk = (
    3.0 * debt_ratio
    - 0.012 * df["credit_score"]
    - 0.06 * df["years_employed"]
    + 0.25 * df["existing_debts"]
    + rng.normal(0, 0.8, n)
)
df["defaulted"] = (risk > -6.2).astype(int)

X = df.drop(columns="defaulted")
y = df["defaulted"]
print(f"{n} past loans, default rate: {y.mean():.1%}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# =====================================================================
# 2. A SINGLE DECISION TREE — and the overfitting trap
# =====================================================================
# With no depth limit, a tree keeps splitting until it memorizes every
# training example — including the noise. Watch the symptom:
deep_tree = DecisionTreeClassifier(random_state=42)  # no limits!
deep_tree.fit(X_train, y_train)

print("UNLIMITED decision tree (memorizes everything):")
print(f"  accuracy on TRAINING data: {accuracy_score(y_train, deep_tree.predict(X_train)):.1%}")
print(f"  accuracy on TEST data:     {accuracy_score(y_test, deep_tree.predict(X_test)):.1%}")
print("  -> big gap = OVERFITTING. Perfect memory, poor generalization.\n")

# Limiting depth forces the tree to learn general rules only:
small_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
small_tree.fit(X_train, y_train)
print("LIMITED tree (max_depth=4):")
print(f"  accuracy on TRAINING data: {accuracy_score(y_train, small_tree.predict(X_train)):.1%}")
print(f"  accuracy on TEST data:     {accuracy_score(y_test, small_tree.predict(X_test)):.1%}")
print("  -> smaller gap = healthier model.\n")

# =====================================================================
# 3. RANDOM FOREST — many imperfect trees, one strong vote
# =====================================================================
# Each of the 300 trees sees a random subset of loans and features, then
# they vote. Randomness + averaging cancels out individual trees'
# memorization. Note: trees don't need feature scaling — a split like
# "credit_score < 640?" doesn't care about units. One less step!
forest = RandomForestClassifier(n_estimators=300, random_state=42)
forest.fit(X_train, y_train)

print("RANDOM FOREST (300 trees voting):")
print(f"  accuracy on TEST data:     {accuracy_score(y_test, forest.predict(X_test)):.1%}\n")

# =====================================================================
# 4. FEATURE IMPORTANCE — what drives the decision?
# =====================================================================
# Banks are legally required to explain rejections. Feature importance
# shows which inputs the forest actually relies on:
importances = pd.Series(forest.feature_importances_, index=X.columns)
print("What matters most when predicting default:")
for name, imp in importances.sort_values(ascending=False).items():
    bar = "#" * int(imp * 60)
    print(f"  {name:15s} {imp:.2f} {bar}")

# =====================================================================
# 5. DECIDE ON A NEW APPLICATION
# =====================================================================
applicant = pd.DataFrame([{
    "annual_income": 42000, "loan_amount": 30000, "credit_score": 585,
    "years_employed": 0.5, "existing_debts": 4,
}])
p_default = forest.predict_proba(applicant)[0, 1]
print(f"\nNew applicant: $42k income, wants $30k, credit 585, employed 6 months")
print(f"Default risk: {p_default:.1%} -> {'REJECT' if p_default > 0.5 else 'APPROVE'}")

# =====================================================================
# TRY IT YOURSELF:
#  - Plot the small tree's actual flowchart:
#      from sklearn.tree import plot_tree; import matplotlib.pyplot as plt
#      plot_tree(small_tree, feature_names=X.columns, filled=True); plt.show()
#  - Try n_estimators=5 vs 300. More trees = more stable (up to a point).
#  - Compare LogisticRegression on this data. Forest usually wins on
#    tabular data with non-linear rules like debt RATIOS.
# =====================================================================
