from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    return [X, len(np.unique(y))]

def test(X, clusters_raw, row_counts, centroids):
    clusters_raw = clusters_raw.flatten().astype(int)
    row_counts = row_counts.flatten().astype(int)
    clusters = [[] for _ in range(len(row_counts))]
    prev_index = 0
    for i in range(len(row_counts)):
        clusters[i] = clusters_raw[prev_index : prev_index + row_counts[i]]
        prev_index += row_counts[i]

    fig, ax = plt.subplots(figsize=(12, 8))

    for _, index in enumerate(clusters):
        point = X[index].T
        ax.scatter(*point)

    for point in centroids:
        ax.scatter(*point, marker="x", color="black", linewidth=2)

    plt.show()
