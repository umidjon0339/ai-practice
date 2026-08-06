"""
LESSON 1 — REGRESSION: Predicting House Prices
==============================================
Real-life problem: You work for a real-estate agency. Given facts about a
house (size, bedrooms, age, distance to city center), predict its price.

"Regression" = predicting a continuous NUMBER (price, temperature, salary...).

Core scikit-learn workflow (you will use this in EVERY lesson):
    1. Get data into X (features) and y (target)
    2. Split into train/test
    3. model.fit(X_train, y_train)     <- learn
    4. model.predict(X_test)           <- answer
    5. Compare predictions to truth    <- grade
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =====================================================================
# 1. THE DATA
# =====================================================================
# We simulate a realistic dataset of 500 sold houses. In real life this
# would come from pd.read_csv("sold_houses.csv").
rng = np.random.default_rng(42)
n = 500

size_m2 = rng.normal(120, 40, n).clip(30, 300)         # living area
bedrooms = rng.integers(1, 6, n)                       # 1..5 bedrooms
age_years = rng.integers(0, 60, n)                     # building age
dist_center_km = rng.exponential(8, n).clip(0.5, 40)   # distance to center

# The "true" pricing rule of the market (unknown to the model!) + noise:
price = (
    2000 * size_m2          # each m² adds $2000
    + 15000 * bedrooms      # each bedroom adds $15k
    - 1000 * age_years      # older = cheaper
    - 3000 * dist_center_km # farther = cheaper
    + 100000                # base price
    + rng.normal(0, 25000, n)  # real life is noisy
)

df = pd.DataFrame({
    "size_m2": size_m2.round(0),
    "bedrooms": bedrooms,
    "age_years": age_years,
    "dist_center_km": dist_center_km.round(1),
    "price": price.round(0),
})

print("A peek at our dataset (each ROW = one house = one 'sample'):")
print(df.head(), "\n")

# X = the features (what we know).  y = the target (what we predict).
X = df.drop(columns="price")   # ALWAYS 2D: (500 houses, 4 features)
y = df["price"]                # ALWAYS 1D: one answer per house
print(f"X shape: {X.shape}   y shape: {y.shape}\n")

# =====================================================================
# 2. SPLIT — hide 20% of houses from the model to test it honestly later
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training on {len(X_train)} houses, testing on {len(X_test)} unseen houses\n")

# =====================================================================
# 3 + 4. TRAIN, then PREDICT
# =====================================================================
model = LinearRegression()      # finds the best line: price = a*size + b*bedrooms + ...
model.fit(X_train, y_train)     # <- ALL the learning happens here

y_pred = model.predict(X_test)  # predictions for the 100 unseen houses

# =====================================================================
# 5. EVALUATE
# =====================================================================
mae = mean_absolute_error(y_test, y_pred)  # "on average, off by $___"
r2 = r2_score(y_test, y_pred)              # 1.0 = perfect, 0.0 = useless

print("Model performance on UNSEEN houses:")
print(f"  MAE: ${mae:,.0f}  (average prediction error in dollars)")
print(f"  R² : {r2:.3f}     (share of price variation the model explains)\n")

# LinearRegression is interpretable — it literally learned the market rules:
print("What the model learned (compare to the true rule in the code!):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:16s} -> ${coef:+,.0f} per unit")
print(f"  base price       -> ${model.intercept_:,.0f}\n")

# =====================================================================
# 6. USE IT — the whole point: price a NEW house
# =====================================================================
new_house = pd.DataFrame([{
    "size_m2": 150, "bedrooms": 3, "age_years": 10, "dist_center_km": 5.0
}])
predicted = model.predict(new_house)[0]
print(f"New listing: 150m², 3 bed, 10 years old, 5km from center")
print(f"Suggested price: ${predicted:,.0f}")

# =====================================================================
# TRY IT YOURSELF:
#  - Change the new house's size to 300m². Does the price react sensibly?
#  - Increase the noise (rng.normal(0, 100000, n)). Watch R² drop —
#    noisier real-world data = harder to predict.
# =====================================================================
