# ==========================================================
# IMPORT REQUIRED LIBRARIES
# ==========================================================

# NumPy is used for numerical operations such as:
# - Arrays
# - Matrix multiplication
# - Mathematical functions
# - Vectorized calculations
import numpy as np

# Load a built-in Breast Cancer dataset
from sklearn.datasets import load_breast_cancer

# Used to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Used to standardize (scale) feature values
from sklearn.preprocessing import StandardScaler


# ==========================================================
# STEP 1: LOAD DATASET
# ==========================================================

# Load the Breast Cancer Wisconsin Dataset.
# It returns a "Bunch" object (similar to a dictionary).
data = load_breast_cancer()

# Feature matrix (X)
#
# X contains all input variables (features).
#
# Each row = one patient
# Each column = one measurement
#
# Shape:
# (569 samples, 30 features)
#
# Example:
# Radius
# Texture
# Perimeter
# Area
# Smoothness
# ...
X = data.data

# Target labels (y)
#
# 0 = Malignant (Cancer)
# 1 = Benign (Not Cancer)
#
# Shape:
# (569,)
y = data.target

# Print dataset information
print("Dataset Shape:", X.shape)

# Print class names
# ['malignant', 'benign']
print("Classes:", data.target_names)


# ==========================================================
# STEP 2: TRAIN-TEST SPLIT
# ==========================================================

# Machine Learning Rule:
#
# Never evaluate a model using the same data it learned from.
#
# Therefore:
#
# Training Set -> Learn patterns
# Test Set     -> Evaluate performance
#
# test_size=0.2
# Means:
# 80% Training
# 20% Testing
#
# random_state=42
# Ensures reproducible results.
#
# Every run gives the same split.
X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)


# ==========================================================
# STEP 3: FEATURE SCALING
# ==========================================================

# Logistic Regression performs much better when
# every feature has a similar scale.
#
# Example before scaling:
#
# Radius      = 15
# Area        = 1200
# Texture     = 11
#
# Area dominates because of its large values.
#
# StandardScaler converts every feature into:
#
# Mean = 0
# Standard Deviation = 1
#
# Formula:
#
# x_scaled = (x - mean) / standard deviation
#
scaler = StandardScaler()

# Learn mean and standard deviation from training data
# then transform training data.
#
# IMPORTANT:
# fit() is only done on training data.
#
# Otherwise information from the test set leaks
# into training (called Data Leakage).
X_train = scaler.fit_transform(X_train)

# Transform testing data using the SAME mean
# and standard deviation learned from training data.
#
# Never call fit_transform() on test data.
X_test = scaler.transform(X_test)


# ==========================================================
# STEP 4: LOGISTIC REGRESSION FROM SCRATCH
# ==========================================================

class LogisticRegressionScratch:

    # ------------------------------------------------------
    # Constructor
    # ------------------------------------------------------
    #
    # learning_rate
    # Controls how big each gradient descent step is.
    #
    # epochs
    # Number of complete passes through the training data.
    #
    def __init__(self, learning_rate=0.01, epochs=1000):

        self.learning_rate = learning_rate
        self.epochs = epochs

    # ------------------------------------------------------
    # Sigmoid Function
    # ------------------------------------------------------
    #
    # Converts any real number into a probability
    # between 0 and 1.
    #
    # Formula:
    #
    #           1
    # -------------------
    # 1 + e^(-z)
    #
    # Example:
    #
    # z = -5 → 0.0067
    #
    # z = 0 → 0.5
    #
    # z = 5 → 0.993
    #
    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    # ------------------------------------------------------
    # Train Model
    # ------------------------------------------------------
    #
    # X = Features
    # y = Labels
    #
    def fit(self, X, y):

        # Number of training samples
        #
        # Example:
        # 455
        #
        # Number of features
        #
        # Example:
        # 30
        #
        samples, features = X.shape

        # Initialize all weights as zero.
        #
        # One weight for every feature.
        #
        # Shape:
        # (30,)
        #
        self.weights = np.zeros(features)

        # Initialize bias to zero.
        self.bias = 0

        # --------------------------------------------------
        # Gradient Descent Loop
        # --------------------------------------------------
        #
        # Repeat many times until
        # the model learns the best weights.
        #
        for epoch in range(self.epochs):

            # ----------------------------------------------
            # Linear Equation
            # ----------------------------------------------
            #
            # z = XW + b
            #
            # X
            # Shape:
            # (455,30)
            #
            # W
            # Shape:
            # (30,)
            #
            # Result:
            # (455,)
            #
            linear = np.dot(X, self.weights) + self.bias

            # Convert linear values into probabilities.
            #
            # Output:
            #
            # 0.91
            # 0.22
            # 0.84
            #
            predictions = self.sigmoid(linear)

            # ----------------------------------------------
            # Compute Gradients
            # ----------------------------------------------
            #
            # Gradient tells us:
            #
            # "How should weights change?"
            #
            # dw
            # Shape:
            # (30,)
            #
            dw = (1 / samples) * np.dot(
                X.T,
                (predictions - y)
            )

            # Gradient of bias.
            db = (1 / samples) * np.sum(
                predictions - y
            )

            # ----------------------------------------------
            # Gradient Descent Update
            # ----------------------------------------------
            #
            # New Weight
            #
            # weight = weight - learning_rate × gradient
            #
            self.weights -= self.learning_rate * dw

            self.bias -= self.learning_rate * db

            # ----------------------------------------------
            # Print Loss Every 100 Epochs
            # ----------------------------------------------
            #
            # Binary Cross Entropy Loss
            #
            # Formula:
            #
            # -(1/m) Σ[
            # y log(p)
            # +
            # (1-y) log(1-p)
            # ]
            #
            if epoch % 100 == 0:

                loss = -(1 / samples) * np.sum(

                    y * np.log(predictions + 1e-9)

                    +

                    (1 - y) * np.log(
                        1 - predictions + 1e-9
                    )

                )

                print(
                    f"Epoch {epoch:4d}  Loss = {loss:.4f}"
                )

    # ------------------------------------------------------
    # Predict Probability
    # ------------------------------------------------------
    #
    # Returns probabilities instead of classes.
    #
    # Example:
    #
    # 0.91
    # 0.23
    # 0.65
    #
    def predict_probability(self, X):

        linear = np.dot(X, self.weights) + self.bias

        return self.sigmoid(linear)

    # ------------------------------------------------------
    # Predict Final Class
    # ------------------------------------------------------
    #
    # Convert probabilities into labels.
    #
    # Rule:
    #
    # probability ≥ 0.5
    #
    # -> Benign (1)
    #
    # probability < 0.5
    #
    # -> Malignant (0)
    #
    def predict(self, X):

        probs = self.predict_probability(X)

        return (probs >= 0.5).astype(int)


# ==========================================================
# STEP 5: CREATE MODEL
# ==========================================================

# Create an object of our Logistic Regression class.
#
# learning_rate = 0.01
# epochs = 1000
#
model = LogisticRegressionScratch(

    learning_rate=0.01,

    epochs=1000

)

# Train the model using training data.
model.fit(X_train, y_train)


# ==========================================================
# STEP 6: MAKE PREDICTIONS
# ==========================================================

# Predict labels for unseen test data.
#
# Output:
#
# [1 0 1 1 0 ...]
#
predictions = model.predict(X_test)


# ==========================================================
# STEP 7: CALCULATE ACCURACY
# ==========================================================

# Compare predicted labels with actual labels.
#
# predictions == y_test
#
# Returns:
#
# True
# False
# True
# True
#
# Mean converts:
#
# True = 1
# False = 0
#
# Example:
#
# 109 correct
# 114 total
#
# Accuracy = 109 / 114
#
accuracy = np.mean(predictions == y_test)

print("\nAccuracy:", accuracy)


# ==========================================================
# STEP 8: DISPLAY FIRST 10 PREDICTIONS
# ==========================================================

print("\nFirst 10 Predictions")

# Display the first 10 test samples.
for i in range(10):

    # Convert numerical prediction into readable text.
    predicted = (
        "Benign"
        if predictions[i] == 1
        else "Malignant"
    )

    # Convert actual label into readable text.
    actual = (
        "Benign"
        if y_test[i] == 1
        else "Malignant"
    )

    # Print prediction vs actual label.
    print(

        f"Sample {i+1}: "

        f"Predicted={predicted:<10} "

        f"Actual={actual}"

    )