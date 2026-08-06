import pickle
import numpy as np

from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances



# Load dataset again
iris = load_iris()

X = iris.data



# Load scaler

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)



# Load model

with open("model.pkl","rb") as f:
    model = pickle.load(f)



# Scale training data

X_scaled = scaler.transform(X)



# New flower

new_flower = np.array(
    [
        [
            5.9,  # sepal length
            3.0,  # sepal width
            5.1,  # petal length
            1.8   # petal width
        ]
    ]
)



new_scaled = scaler.transform(
    new_flower
)



# Get cluster labels

labels = model.labels_



# Calculate cluster centers

centers = []

for cluster_id in range(3):

    center = X_scaled[
        labels == cluster_id
    ].mean(axis=0)

    centers.append(center)



centers = np.array(centers)



# Find nearest cluster

distance = pairwise_distances(
    new_scaled,
    centers
)


prediction = np.argmin(distance)



print(
    "New flower belongs to cluster:",
    prediction
)