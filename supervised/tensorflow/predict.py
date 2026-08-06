import numpy as np

import tensorflow as tf



# =========================
# Load Model
# =========================

model = tf.keras.models.load_model(
    "diabetes_model.keras"
)



# =========================
# New Patient Data
# =========================

patient = np.array(
[
    [
        0.05,
        0.05,
        0.03,
        0.02,
        -0.01,
        -0.02,
        -0.03,
        0.01,
        0.04,
        0.02
    ]
]
)



# =========================
# Prediction
# =========================

prediction = model.predict(
    patient
)


print(
    "Predicted progression:",
    prediction[0][0]
)