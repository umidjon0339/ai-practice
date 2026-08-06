import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# =========================
# 1. Load Dataset
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



# Convert NumPy -> PyTorch Tensor

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
).reshape(-1,1)



X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).reshape(-1,1)



# =========================
# 3. Create Linear Model
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



model = LinearRegressionModel()



# =========================
# 4. Loss and Optimizer
# =========================

loss_function = nn.MSELoss()


optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)



# =========================
# 5. Training Loop
# =========================

epochs = 1000


for epoch in range(epochs):

    # Prediction

    y_pred = model(X_train)


    # Calculate loss

    loss = loss_function(
        y_pred,
        y_train
    )


    # Reset gradients

    optimizer.zero_grad()


    # Backpropagation

    loss.backward()


    # Update weights

    optimizer.step()



    if epoch % 100 == 0:

        print(
            f"Epoch {epoch}, Loss {loss.item()}"
        )



# =========================
# 6. Evaluation
# =========================

with torch.no_grad():

    predictions = model(X_test)



predictions = predictions.numpy()



mse = mean_squared_error(
    y_test,
    predictions
)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)



print("\nModel Performance")
print("------------------")

print("MSE:", mse)

print("MAE:", mae)

print("R2:", r2)



# =========================
# 7. Save Model
# =========================

torch.save(
    model.state_dict(),
    "model.pth"
)


print("\nModel saved!")