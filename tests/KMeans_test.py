from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    return [X, len(np.unique(y))]

def test(X, labels, centroids):
    clusters = [[] for _ in range(centroids.shape[0])]
    for i in range(centroids.shape[0]):
        clusters[i] = np.argwhere(labels == i).tolist()

    fig, ax = plt.subplots(figsize=(12, 8))

    for _, index in enumerate(clusters):
        point = X[index].T
        ax.scatter(*point)

    for point in centroids:
        ax.scatter(*point, marker="x", color="black", linewidth=2)

    plt.show()
