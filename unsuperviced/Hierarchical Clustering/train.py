import pickle

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


# Load dataset
iris = load_iris()

X = iris.data


print("Feature names:")
print(iris.feature_names)



# Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



# Create Agglomerative model
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)



# Train clustering
clusters = model.fit_predict(X_scaled)



# Show results

for i in range(10):
    print(
        iris.data[i],
        "Cluster:",
        clusters[i]
    )



# Save model

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)


# Save scaler

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)



print("\nModel saved successfully!")