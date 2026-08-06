import pickle
import numpy as np


# =========================
# Load Model
# =========================

with open(
    "model.pkl",
    "rb"
) as file:

    model = pickle.load(file)



# =========================
# New Patient Data
# =========================

patient = np.array(
[
    [
        0.05,   # age
        0.05,   # sex
        0.03,   # bmi
        0.02,   # blood pressure
        -0.01,  # s1
        -0.02,  # s2
        -0.03,  # s3
        0.01,   # s4
        0.04,   # s5
        0.02    # s6
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
    "Predicted diabetes progression:",
    prediction[0]
)