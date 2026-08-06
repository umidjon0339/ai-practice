"""
Production-Ready Decision Tree Classifier (CART) Implemented From Scratch.

Author: Machine Learning & Python Engineering
Language: Python 3.12+
Dependencies: NumPy, Pandas, Scikit-Learn (metrics & train_test_split only)
Dataset: UCI Heart Disease Cleveland Dataset
"""

from dataclasses import dataclass
import io
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


@dataclass
class Node:
    """Represents a single node in the CART Decision Tree.

    Attributes:
        feature_index: Index of the feature used for splitting (None for leaf).
        threshold: Split boundary threshold (None for leaf).
        left: Pointer to left child Node (samples <= threshold).
        right: Pointer to right child Node (samples > threshold).
        value: Predicted class label (populated only for leaf nodes).
        depth: Depth level of this node within the tree (root is depth 0).
        gain: Impurity reduction (information gain) achieved by splitting this node.
        n_samples: Number of training samples that reached this node.
    """

    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Optional[int] = None
    depth: int = 0
    gain: float = 0.0
    n_samples: int = 0

    @property
    def is_leaf(self) -> bool:
        """Check if current node is a leaf node."""
        return self.value is not None


class DecisionTreeClassifierScratch:
    """Production-grade Binary Decision Tree Classifier (CART algorithm) implemented

    from scratch using Gini Impurity.

    Attributes:
        max_depth: Maximum depth permitted for the tree.
        min_samples_split: Minimum number of samples required to split an
          internal node.
        min_samples_leaf: Minimum number of samples required at a leaf node.
        root: Root Node of the trained decision tree.
        n_classes_: Number of distinct target classes detected during fit.
        n_features_: Number of input features observed during fit.
        feature_importances_: Array of feature importances (summing to 1.0).
    """

    def __init__(
        self,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> None:
        """Initializes the Decision Tree Classifier with hyperparameter validation.

        Args:
            max_depth: Maximum allowed depth of the tree (None for unlimited).
            min_samples_split: Minimum samples required to attempt a split.
            min_samples_leaf: Minimum samples required in each resulting child
              leaf.

        Raises:
            ValueError: If hyperparameters violate standard boundaries.
        """
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if min_samples_split < 2:
            raise ValueError(
                f"min_samples_split must be >= 2, got {min_samples_split}"
            )
        if min_samples_leaf < 1:
            raise ValueError(
                f"min_samples_leaf must be >= 1, got {min_samples_leaf}"
            )

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.root: Optional[Node] = None
        self.n_classes_: int = 0
        self.n_features_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None

    # ==========================================
    # Core Public API
    # ==========================================

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> "DecisionTreeClassifierScratch":
        """Builds the CART decision tree from training dataset (X, y).

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target class array of shape (n_samples,).

        Returns:
            self: The fitted classifier instance.

        Raises:
            ValueError: If inputs contain invalid shapes, NaNs, or improper values.
        """
        X_arr, y_arr = self._validate_and_clean_inputs(X, y)

        self.n_features_ = X_arr.shape[1]
        self.n_classes_ = len(np.unique(y_arr))

        if self.n_classes_ < 2:
            raise ValueError(
                f"Training data must contain at least 2 target classes. Found {self.n_classes_}."
            )

        # Recursively construct decision tree root
        self.root = self._build_tree(X_arr, y_arr, depth=0)

        # Compute feature importances post-training
        self._calculate_feature_importances()

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predicts target classes for test samples in X.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).

        Raises:
            RuntimeError: If called prior to calling `fit()`.
            ValueError: If input dimensions do not match training data.
        """
        if self.root is None:
            raise RuntimeError(
                "This DecisionTreeClassifierScratch instance is not fitted yet. Call 'fit' first."
            )

        X_arr = self._validate_predict_inputs(X)
        return np.array([self._predict(sample, self.root) for sample in X_arr])

    def count_nodes(self) -> int:
        """Calculates total number of nodes (decision nodes + leaf nodes) in tree."""

        def _count(node: Optional[Node]) -> int:
            if node is None:
                return 0
            return 1 + _count(node.left) + _count(node.right)

        return _count(self.root)

    def tree_depth(self) -> int:
        """Calculates maximum depth achieved by the trained decision tree."""

        def _depth(node: Optional[Node]) -> int:
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self.root)

    def count_leaves(self) -> int:
        """Calculates total number of leaf nodes in the tree."""

        def _count_leaves(node: Optional[Node]) -> int:
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)

        return _count_leaves(self.root)

    def print_tree(self, feature_names: Optional[List[str]] = None) -> None:
        """Prints a visual ASCII representation of the decision tree."""
        if self.root is None:
            print("Tree is empty. Call fit() first.")
            return

        print("Decision Tree Structure:")
        self._print_tree(
            node=self.root, prefix="", is_left=True, feature_names=feature_names
        )

    # ==========================================
    # Internal Recursive Core Methods
    # ==========================================

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int = 0
    ) -> Node:
        """Recursively selects splits and builds nodes of the CART decision tree.

        Args:
            X: Feature subset at current node.
            y: Label subset at current node.
            depth: Current tree depth level.

        Returns:
            Node: Root of the sub-tree constructed.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        is_max_depth = (
            self.max_depth is not None and depth >= self.max_depth
        )
        is_insufficient_samples = n_samples < self.min_samples_split
        is_pure = n_labels == 1

        if is_max_depth or is_insufficient_samples or is_pure:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, depth=depth, n_samples=n_samples)

        # Search for best split across features & thresholds
        best_split_dict = self._best_split(X, y)

        # If no valid split yields impurity reduction meeting criteria
        if (
            not best_split_dict
            or best_split_dict.get("gain", 0.0) <= 1e-7
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, depth=depth, n_samples=n_samples)

        # Build subtrees
        left_child = self._build_tree(
            X[best_split_dict["left_idxs"]],
            y[best_split_dict["left_idxs"]],
            depth + 1,
        )
        right_child = self._build_tree(
            X[best_split_dict["right_idxs"]],
            y[best_split_dict["right_idxs"]],
            depth + 1,
        )

        return Node(
            feature_index=best_split_dict["feature_index"],
            threshold=best_split_dict["threshold"],
            left=left_child,
            right=right_child,
            depth=depth,
            gain=best_split_dict["gain"],
            n_samples=n_samples,
        )

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Finds feature index and threshold that maximize Gini Impurity

        reduction.

        Args:
            X: Sub-feature matrix at current node.
            y: Sub-labels at current node.

        Returns:
            Dict containing best feature_index, threshold, gain, and child indices.
        """
        best_gain = -1.0
        split_dict: Dict[str, Any] = {}
        n_samples, n_features = X.shape

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            # Handle constant feature case
            if np.min(X_column) == np.max(X_column):
                continue

            # Candidate thresholds via sorted unique values midpoints
            unique_vals = np.unique(X_column)
            if len(unique_vals) <= 1:
                continue

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X_column, threshold)

                # Check min_samples_leaf condition
                if (
                    len(left_idxs) < self.min_samples_leaf
                    or len(right_idxs) < self.min_samples_leaf
                ):
                    continue

                gain = self._information_gain(y, left_idxs, right_idxs)

                if gain > best_gain:
                    best_gain = gain
                    split_dict = {
                        "feature_index": feat_idx,
                        "threshold": threshold,
                        "gain": gain,
                        "left_idxs": left_idxs,
                        "right_idxs": right_idxs,
                    }

        return split_dict

    def _split(
        self, X_column: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Splits array indices into left (<= threshold) and right (> threshold)."""
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def _gini(self, y: np.ndarray) -> float:
        """Calculates Gini Impurity for target array y.

        Formula: Gini = 1 - Σ(p_i²)
        """
        n_samples = len(y)
        if n_samples == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / n_samples
        return 1.0 - float(np.sum(probabilities**2))

    def _information_gain(
        self, y: np.ndarray, left_idxs: np.ndarray, right_idxs: np.ndarray
    ) -> float:
        """Calculates weighted Gini Impurity reduction (Information Gain).

        Formula:
        parent_gini - [(N_left / N_total) * Gini_left + (N_right / N_total) * Gini_right]
        """
        parent_gini = self._gini(y)
        n_total = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)

        if n_left == 0 or n_right == 0:
            return 0.0

        left_gini = self._gini(y[left_idxs])
        right_gini = self._gini(y[right_idxs])

        weighted_gini = (n_left / n_total) * left_gini + (
            n_right / n_total
        ) * right_gini
        return parent_gini - weighted_gini

    def _most_common_label(self, y: np.ndarray) -> int:
        """Returns the most frequent target class in y (handles ties deterministically)."""
        if len(y) == 0:
            raise ValueError("Cannot calculate most common label on an empty array.")
        vals, counts = np.unique(y, return_counts=True)
        return int(vals[np.argmax(counts)])

    def _predict(self, x: np.ndarray, node: Node) -> int:
        """Recursively traverses the tree down to a leaf node for sample x."""
        if node.is_leaf:
            return int(node.value)  # type: ignore

        feature_val = x[node.feature_index]
        if feature_val <= node.threshold:  # type: ignore
            return self._predict(x, node.left)  # type: ignore
        return self._predict(x, node.right)  # type: ignore

    def _calculate_feature_importances(self) -> None:
        """Computes feature importances using impurity reduction weighted by

        sample counts.
        """
        importances = np.zeros(self.n_features_, dtype=np.float64)
        total_samples = float(self.root.n_samples) if self.root else 1.0

        def _traverse(node: Optional[Node]) -> None:
            if node is None or node.is_leaf:
                return

            # Impurity reduction weighted by fraction of samples reaching node
            weight = node.n_samples / total_samples
            importances[node.feature_index] += weight * node.gain  # type: ignore

            _traverse(node.left)
            _traverse(node.right)

        _traverse(self.root)

        total_gain = np.sum(importances)
        if total_gain > 0:
            importances /= total_gain

        self.feature_importances_ = importances

    def _print_tree(
        self,
        node: Optional[Node],
        prefix: str,
        is_left: bool,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Internal recursive helper for formatted tree visualization."""
        if node is None:
            return

        connector = "├── True:  " if is_left else "└── False: "

        if node.is_leaf:
            print(f"{prefix}{connector}[Class -> {node.value}] (n={node.n_samples})")
            return

        feat_str = (
            f"Feature '{feature_names[node.feature_index]}'"
            if feature_names and node.feature_index < len(feature_names)
            else f"X[{node.feature_index}]"
        )

        print(
            f"{prefix}{connector}{feat_str} <= {node.threshold:.3f} "
            f"(Gini Gain: {node.gain:.4f}, n={node.n_samples})"
        )

        new_prefix = prefix + ("│   " if is_left else "    ")
        self._print_tree(
            node.left, prefix=new_prefix, is_left=True, feature_names=feature_names
        )
        self._print_tree(
            node.right, prefix=new_prefix, is_left=False, feature_names=feature_names
        )

    # ==========================================
    # Validation Helpers
    # ==========================================

    def _validate_and_clean_inputs(
        self, X: Any, y: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validates and converts X and y into numpy arrays with strict error checks."""
        if X is None or y is None:
            raise ValueError("Inputs X and y cannot be None.")

        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y).ravel()

        if X_arr.size == 0 or y_arr.size == 0:
            raise ValueError("Input dataset X or y is empty.")

        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X_arr.shape}")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Mismatch in sample count: X shape {X_arr.shape} vs y shape {y_arr.shape}"
            )

        if np.isnan(X_arr).any() or np.isnan(y_arr).any():
            raise ValueError("Input array contains NaN or infinite values.")

        return X_arr, y_arr.astype(int)

    def _validate_predict_inputs(self, X: Any) -> np.ndarray:
        """Validates prediction input matrix."""
        if X is None:
            raise ValueError("Prediction input X cannot be None.")

        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()

        X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if X_arr.shape[1] != self.n_features_:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features_}, got {X_arr.shape[1]}"
            )

        if np.isnan(X_arr).any():
            raise ValueError("Input prediction array contains NaN values.")

        return X_arr


# ==============================================================================
# DATA PIPELINE & LOADER
# ==============================================================================


def load_heart_disease_dataset() -> pd.DataFrame:
    """Fetches and cleans the Heart Disease Cleveland dataset from UCI repository.

    Returns:
        pd.DataFrame: Cleaned heart disease dataframe.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]

    print("Fetching Heart Disease Cleveland dataset from UCI repository...")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            csv_content = response.read().decode("utf-8")
        df = pd.read_csv(
            io.StringIO(csv_content), names=column_names, na_values="?"
        )
    except Exception as err:
        print(
            f"Warning: Failed to fetch online dataset ({err}). Generating synthetic Heart Disease dataset..."
        )
        return _generate_synthetic_heart_dataset(column_names)

    # Handle Missing Values (Impute with median)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Binarize Target: 0 = No Disease, 1 = Disease (Original target is 0..4)
    df["target"] = (df["target"] > 0).astype(int)

    print(
        f"Dataset loaded successfully: {df.shape[0]} samples, {df.shape[1]-1} features."
    )
    return df


def _generate_synthetic_heart_dataset(column_names: List[str]) -> pd.DataFrame:
    """Fallback generator for UCI heart disease structure in offline environments."""
    np.random.seed(42)
    n = 300
    data = {
        "age": np.random.randint(29, 77, size=n),
        "sex": np.random.choice([0, 1], size=n),
        "cp": np.random.choice([1, 2, 3, 4], size=n),
        "trestbps": np.random.randint(94, 200, size=n),
        "chol": np.random.randint(126, 564, size=n),
        "fbs": np.random.choice([0, 1], size=n),
        "restecg": np.random.choice([0, 1, 2], size=n),
        "thalach": np.random.randint(71, 202, size=n),
        "exang": np.random.choice([0, 1], size=n),
        "oldpeak": np.round(np.random.uniform(0.0, 6.2, size=n), 1),
        "slope": np.random.choice([1, 2, 3], size=n),
        "ca": np.random.choice([0, 1, 2, 3], size=n),
        "thal": np.random.choice([3.0, 6.0, 7.0], size=n),
        "target": np.random.choice([0, 1], size=n),
    }
    return pd.DataFrame(data, columns=column_names)


# ==============================================================================
# MAIN EXECUTION SCRIPT
# ==============================================================================

if __name__ == "__main__":
    print("================================================================")
    print("   PRODUCTION DECISION TREE CLASSIFIER FROM SCRATCH (CART)    ")
    print("================================================================\n")

    # 1. Load Data
    df = load_heart_disease_dataset()
    X = df.drop(columns=["target"])
    y = df["target"]
    feature_names = list(X.columns)

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Model Training
    clf = DecisionTreeClassifierScratch(
        max_depth=5, min_samples_split=5, min_samples_leaf=2
    )

    print("\nTraining Decision Tree model...")
    start_train = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time_ms = (time.perf_counter() - start_train) * 1000

    # 4. Predictions & Inference Speed
    start_pred = time.perf_counter()
    y_pred = clf.predict(X_test)
    pred_time_ms = (time.perf_counter() - start_pred) * 1000

    # 5. Model Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 45)
    print("             MODEL EVALUATION METRICS        ")
    print("=" * 45)
    print(f"Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision:        {prec:.4f}")
    print(f"Recall:           {rec:.4f}")
    print(f"F1 Score:         {f1:.4f}")
    print(f"Training Time:    {train_time_ms:.2f} ms")
    print(f"Inference Time:   {pred_time_ms:.2f} ms")
    print(f"Tree Depth:       {clf.tree_depth()}")
    print(f"Total Nodes:      {clf.count_nodes()}")
    print(f"Total Leaves:     {clf.count_leaves()}")
    print("\nConfusion Matrix:")
    print("                 Predicted [0]  Predicted [1]")
    print(f"Actual [0] (No)     {cm[0,0]:<12}  {cm[0,1]:<12}")
    print(f"Actual [1] (Yes)    {cm[1,0]:<12}  {cm[1,1]:<12}")

    # 6. Feature Importances
    print("\n" + "=" * 45)
    print("       FEATURE IMPORTANCE RANKINGS (GINI)    ")
    print("=" * 45)
    importances = clf.feature_importances_
    fi_df = (
        pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        )
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    for rank, row in fi_df.iterrows():
        bar = "█" * int(row["Importance"] * 30)
        print(f"{rank+1:2d}. {row['Feature']:<10} | {row['Importance']:.4f} | {bar}")

    # 7. Print Tree Visual Structure
    print("\n" + "=" * 65)
    print("               VISUAL TREE STRUCTURE (ASCII)                 ")
    print("=" * 65)
    clf.print_tree(feature_names=feature_names)

    # 8. Single Sample Patient Inference Test
    print("\n" + "=" * 65)
    print("          SINGLE SAMPLE PATIENT INFERENCE TEST              ")
    print("=" * 65)
    sample_patient = X_test.iloc[0:1]
    sample_pred = clf.predict(sample_patient)[0]
    sample_actual = y_test.iloc[0]

    print(f"Patient Features:\n{sample_patient.to_dict(orient='records')[0]}")
    print(
        f"\nPredicted Diagnosis: {'[1] Heart Disease' if sample_pred == 1 else '[0] Normal'}"
    )
    print(
        f"Actual Diagnosis:    {'[1] Heart Disease' if sample_actual == 1 else '[0] Normal'}"
    )
    print("================================================================\n")