"""
LESSON 2 — CLASSIFICATION: Tumor Diagnosis (REAL medical data)
==============================================================
Real-life problem: A hospital measures 30 properties of a breast tumor
(size, texture, symmetry...). Predict: malignant (dangerous) or benign?

"Classification" = predicting a CATEGORY, not a number.

This is REAL data: 569 actual patients (Wisconsin Breast Cancer dataset,
built into scikit-learn).

NEW IDEAS vs lesson 1:
  - LogisticRegression (the go-to first classifier, despite the name!)
  - Feature scaling with StandardScaler
  - Accuracy is NOT enough: precision, recall, confusion matrix
  - predict_proba: models can tell you HOW CONFIDENT they are
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

# =====================================================================
# 1. LOAD REAL DATA
# =====================================================================
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0 = malignant (dangerous), 1 = benign (harmless)

print(f"{X.shape[0]} patients, {X.shape[1]} measurements per tumor")
print(f"Classes: {dict(zip(data.target_names, [sum(y==0), sum(y==1)]))}\n")
print("First 3 patients, first 4 measurements:")
print(X.iloc[:3, :4], "\n")

# =====================================================================
# 2. SPLIT
# =====================================================================
# stratify=y keeps the malignant/benign ratio identical in both splits —
# important for medical data so the test set is representative.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================================
# 3. SCALE THE FEATURES  (new concept!)
# =====================================================================
# Features have wildly different ranges ("mean area" ~ 650, "smoothness"
# ~ 0.1). Many models work much better when all features have a similar
# scale. StandardScaler transforms each feature to mean=0, std=1.
#
# GOLDEN RULE: fit the scaler on TRAIN only, then apply to both.
# Fitting on test data would "leak" information the model shouldn't have.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # learn means/stds + apply
X_test_scaled = scaler.transform(X_test)        # ONLY apply

# =====================================================================
# 4. TRAIN + PREDICT
# =====================================================================
model = LogisticRegression()          # outputs the PROBABILITY of each class
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# =====================================================================
# 5. EVALUATE — and why accuracy alone is dangerous in medicine
# =====================================================================
print(f"Accuracy: {accuracy_score(y_test, y_pred):.1%}\n")

# The confusion matrix shows WHAT KIND of mistakes we make:
#                     predicted malignant | predicted benign
#   truly malignant        GOOD           |  DISASTER (missed cancer!)
#   truly benign           bad (scare)    |  GOOD
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(pd.DataFrame(
    cm,
    index=["truly malignant", "truly benign"],
    columns=["pred. malignant", "pred. benign"],
), "\n")

missed_cancers = cm[0, 1]
false_alarms = cm[1, 0]
print(f"Missed cancers (worst error): {missed_cancers}")
print(f"False alarms (unnecessary scare): {false_alarms}\n")

# precision = "when we say malignant, how often are we right?"
# recall    = "of all real malignant tumors, how many did we catch?"
# In medicine, RECALL on the dangerous class matters most.
print(classification_report(y_test, y_pred, target_names=data.target_names))

# =====================================================================
# 6. CONFIDENCE — classifiers can output probabilities
# =====================================================================
proba = model.predict_proba(X_test_scaled[:5])
print("Model confidence for 5 test patients:")
for i, (p_mal, p_ben) in enumerate(proba):
    verdict = "MALIGNANT" if p_mal > p_ben else "benign"
    print(f"  patient {i}: {verdict:9s}  (malignant: {p_mal:.1%}, benign: {p_ben:.1%})")

# =====================================================================
# TRY IT YOURSELF:
#  - Remove the scaler (fit on raw X_train). LogisticRegression will
#    warn about convergence and accuracy may drop.
#  - A hospital wants to NEVER miss a cancer. Instead of predict(), use
#    predict_proba() and flag malignant whenever p_mal > 0.2 (not 0.5).
#    Count missed cancers vs false alarms now. This tradeoff is a real
#    daily decision in ML for medicine.
# =====================================================================
