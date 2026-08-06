import joblib

# Load trained model
model = joblib.load("kmeans_model.pkl")

# New flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

# Predict cluster
cluster = model.predict(new_flower)

print("Predicted Cluster:", cluster[0])