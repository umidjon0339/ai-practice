import pandas as pd
import pickle

from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# =========================
# 1. Load Built-in Dataset
# =========================

diabetes = load_diabetes()


X = pd.DataFrame(
    diabetes.data,
    columns=diabetes.feature_names
)


y = diabetes.target



print("Dataset shape:")
print(X.shape)



# =========================
# 2. Split Dataset
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# =========================
# 3. Create Model
# =========================

model = LinearRegression()



# =========================
# 4. Train Model
# =========================

model.fit(
    X_train,
    y_train
)



# =========================
# 5. Evaluate Model
# =========================

y_pred = model.predict(
    X_test
)


mse = mean_squared_error(
    y_test,
    y_pred
)


mae = mean_absolute_error(
    y_test,
    y_pred
)


r2 = r2_score(
    y_test,
    y_pred
)


print("\nModel Performance")
print("------------------")

print("MSE:", mse)

print("MAE:", mae)

print("R2 Score:", r2)

# =========================
# 6. Save Model
# =========================

with open(
    "model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


print("\nModel saved successfully!")



