from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import joblib

# Load dataset
iris = load_iris()
X = iris.data

# Create model
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train model
model.fit(X)

# Save model
joblib.dump(model, "kmeans_model.pkl")
print(X)
print("Model trained successfully!")
print("Model saved as kmeans_model.pkl")