"""
LESSON 3 — PREPROCESSING REAL-WORLD MESSY DATA: Customer Churn
==============================================================
Real-life problem: You work at a telecom company. Predict which customers
will CANCEL their subscription ("churn") so you can offer them a discount
before they leave.

Real-world data is MESSY — unlike lessons 1-2:
  - missing values (customer didn't fill in their age)
  - categorical text columns ("fiber", "dsl") — models need numbers!
  - numeric columns on different scales

NEW IDEAS:
  - SimpleImputer   -> fill missing values
  - OneHotEncoder   -> turn categories into numbers
  - ColumnTransformer -> different preprocessing per column
  - Pipeline        -> chain preprocessing + model into ONE object
                       (leak-proof, and you can save/load it as a unit)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# =====================================================================
# 1. SIMULATE A MESSY REAL-WORLD DATASET
# =====================================================================
rng = np.random.default_rng(0)
n = 1000

df = pd.DataFrame({
    "age": rng.integers(18, 80, n).astype(float),
    "monthly_bill": rng.normal(60, 25, n).clip(15, 150).round(2),
    "months_subscribed": rng.integers(1, 72, n),
    "internet_type": rng.choice(["dsl", "fiber", "none"], n, p=[0.4, 0.45, 0.15]),
    "contract": rng.choice(["monthly", "1-year", "2-year"], n, p=[0.55, 0.25, 0.2]),
})

# Realistic churn behavior: expensive bills + monthly contracts + new
# customers churn more. (This rule is what the model must discover.)
churn_score = (
    0.03 * df["monthly_bill"]
    - 0.04 * df["months_subscribed"]
    + np.where(df["contract"] == "monthly", 1.5, 0)
    + np.where(df["internet_type"] == "fiber", 0.5, 0)
    + rng.normal(0, 1, n)
)
df["churned"] = (churn_score > 1.2).astype(int)

# Real data has holes — knock out ~8% of ages and some internet types:
df.loc[rng.choice(n, 80, replace=False), "age"] = np.nan
df.loc[rng.choice(n, 40, replace=False), "internet_type"] = np.nan

print("Messy real-world data (note NaN and text columns):")
print(df.head(8), "\n")
print("Missing values per column:")
print(df.isna().sum(), "\n")
print(f"Churn rate: {df['churned'].mean():.1%}\n")

X = df.drop(columns="churned")
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================================
# 2. BUILD THE PREPROCESSING RECIPE
# =====================================================================
numeric_features = ["age", "monthly_bill", "months_subscribed"]
categorical_features = ["internet_type", "contract"]

# For numbers: fill missing with the median, then scale.
numeric_recipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

# For categories: fill missing with most frequent, then one-hot encode.
# One-hot: "fiber" -> [0,1,0], "dsl" -> [1,0,0] ... one column per category.
categorical_recipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# ColumnTransformer: apply the right recipe to the right columns.
preprocessor = ColumnTransformer([
    ("numbers", numeric_recipe, numeric_features),
    ("categories", categorical_recipe, categorical_features),
])

# =====================================================================
# 3. PIPELINE = preprocessing + model IN ONE OBJECT
# =====================================================================
# When you call .fit(), the pipeline fits the imputer/scaler/encoder on
# TRAINING data only, then trains the model. When you call .predict(),
# it applies the same (already-fitted) transformations. No leakage,
# no manual bookkeeping — this is how professionals ship models.
model = Pipeline([
    ("prep", preprocessor),
    ("classifier", LogisticRegression()),
])

model.fit(X_train, y_train)              # raw messy DataFrame goes in!
y_pred = model.predict(X_test)

print("Churn prediction performance:")
print(classification_report(y_test, y_pred, target_names=["stayed", "churned"]))

# =====================================================================
# 4. USE IT ON A NEW CUSTOMER — even with a missing value!
# =====================================================================
new_customers = pd.DataFrame([
    {"age": 24, "monthly_bill": 95.0, "months_subscribed": 3,
     "internet_type": "fiber", "contract": "monthly"},   # risky profile
    {"age": np.nan, "monthly_bill": 40.0, "months_subscribed": 60,
     "internet_type": "dsl", "contract": "2-year"},      # loyal profile, age unknown
])
probs = model.predict_proba(new_customers)[:, 1]
for i, p in enumerate(probs):
    action = "-> offer retention discount!" if p > 0.5 else "-> no action needed"
    print(f"Customer {i+1}: churn risk {p:.1%} {action}")

# =====================================================================
# TRY IT YOURSELF:
#  - Print pd.DataFrame(model.named_steps['prep'].fit_transform(X_train))
#    to see what the model actually receives after preprocessing.
#  - Change the numeric imputer strategy to "mean". Does anything change?
#  - Feed a customer with internet_type="satellite" (a category never
#    seen in training). handle_unknown="ignore" saves you — remove it
#    (in a copy) and watch it crash. Real production lesson!
# =====================================================================
