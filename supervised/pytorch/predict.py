import torch
import torch.nn as nn



# =========================
# Model Definition
# =========================

class LinearRegressionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(
            10,
            1
        )


    def forward(self,x):

        return self.linear(x)



# Create model

model = LinearRegressionModel()



# Load weights

model.load_state_dict(
    torch.load("model.pth")
)


model.eval()



# =========================
# New Patient Data
# =========================

patient = torch.tensor(
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
],
dtype=torch.float32
)



# =========================
# Prediction
# =========================

with torch.no_grad():

    prediction = model(patient)



print(
    "Predicted progression:",
    prediction.item()
)