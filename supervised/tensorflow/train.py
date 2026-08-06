import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)



# =========================
# 1. Load Built-in Dataset
# =========================

data = load_diabetes()


X = data.data

y = data.target



print("Dataset shape:")
print(X.shape)



# =========================
# 2. Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# =========================
# 3. Build Linear Regression Model
# =========================

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            units=1,
            input_shape=[10]
        )
    ]
)



# =========================
# 4. Compile Model
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.01
    ),
    loss="mse",
    metrics=["mae"]
)



# Show architecture

model.summary()



# =========================
# 5. Train Model
# =========================

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)



# =========================
# 6. Evaluate Model
# =========================

y_pred = model.predict(
    X_test
)


y_pred = y_pred.flatten()



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
# 7. Save Model
# =========================

model.save(
    "diabetes_model.keras"
)


print("\nModel saved!")