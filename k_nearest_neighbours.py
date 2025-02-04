import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

K = 5

X, Y = make_classification(n_samples=1000, n_classes=2, n_features=10)
df = pd.DataFrame(X)
df["target"] = Y

print(df.shape)  # 1000, 11

# Set column names
df.columns = ["col_" + str(i + 1) for i in range(10)] + ["target"]
print(
    df.columns
)  # ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8', 'col_9', 'col_10', 'target']


class KNN:
    def __init__(self, k=3):
        self.k = k  # Number of neighbors

    def fit(self, X, y):
        """Store training data."""
        self.X_train = X
        self.y_train = y

    def _compute_distances(self, X_test):
        """Compute L2 (Euclidean) distance between test points and training points."""
        distances = np.sqrt(np.sum((self.X_train - X_test) ** 2, axis=1))
        return distances

    def predict(self, X_test):
        """Predict labels for the test set."""
        predictions = []
        for test_point in X_test:
            distances = self._compute_distances(test_point)
            nearest_indices = np.argsort(distances)[: self.k]
            nearest_labels = self.y_train[nearest_indices]

            # Classification: Majority vote
            unique, counts = np.unique(nearest_labels, return_counts=True)
            predicted_label = unique[np.argmax(counts)]
            predictions.append(predicted_label)

        return np.array(predictions)
