import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

X, _ = make_classification(n_samples=1000, n_classes=2, n_features=4)
df = pd.DataFrame(X)
print(df.shape)
df.columns = ["col_" + str(i) for i in range(df.shape[1])]
COLUMNS = df.columns

print(np.random.choice(range(1000), 2))


class KMeansClustering:

    def __init__(self, k, max_iterations):
        self.K = k
        self.max_iterations = max_iterations
        self.cluster_centorids = []
        self.cluster_labels = ["cluster_" + str(i) for i in range(self.K)]

    def _get_random_centroids(self, dataset_size, train):
        random_indices = np.random.choice(range(dataset_size), self.K)
        self.cluster_centorids = [
            train.iloc[index, :].values for index in random_indices
        ]

    def _get_distances(self, X_train, X_centroid):
        diff = X_train - X_centroid
        diff_2 = np.array(diff**2)
        eucledian = np.sum(diff_2, axis=1, keepdims=True)

        return eucledian

    def _get_new_cluster_centroids(self, X_train, cluster_index):
        temp = X_train.copy()
        temp["cluster"] = cluster_index

        new_cluster_centroids = (
            temp.groupby("cluster")[COLUMNS].mean().reset_index(drop=True)
        ).values.tolist()
        return new_cluster_centroids

    def _get_centroid_distance_changes(self, new_centroids):
        change = 0
        for i in range(len(new_centroids)):
            old_centroid_coordinates = np.array(self.cluster_centorids[i])
            new_centroid_coordinates = np.array(new_centroids[i])

            change += np.sqrt(
                np.sum((old_centroid_coordinates - new_centroid_coordinates) ** 2)
            )

        return change

    def fit(self, train, threshold=0.001):
        # Set random centroids
        self._get_random_centroids(train.shape[0], train)

        count_iterations = 0
        total_change = float("inf")

        while (
            count_iterations < self.max_iterations and total_change > self.K * threshold
        ):

            # Get Current Centroid Distances
            current_distances = np.concatenate(
                [
                    self._get_distances(train, centroid)
                    for centroid in self.cluster_centorids
                ],
                axis=1,
            )

            min_distances_cluster_index = np.argmin(current_distances, axis=1)

            # train["cluster"] = min_distances_cluster_index

            new_centroids = self._get_new_cluster_centroids(
                train, min_distances_cluster_index
            )
            total_change = self._get_centroid_distance_changes(new_centroids)
            count_iterations += 1

            self.cluster_centorids = new_centroids

        # get final cluster name
        clusters = [self.cluster_labels[i] for i in min_distances_cluster_index]

        return clusters


k_means = KMeansClustering(k=3, max_iterations=10)
assigned_clusters = k_means.fit(df)

print(np.unique(np.array(assigned_clusters), return_counts=True))
