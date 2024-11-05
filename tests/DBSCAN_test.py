import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def get_data():
    # Generate synthetic data with 3 clusters
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
    return X

def test(X, y_db):
    y_db = np.array(y_db).flatten()
    # Plot the results
    plt.figure(figsize=(8, 6))

    # Plotting different clusters with different colors
    plt.scatter(X[:, 0], X[:, 1], c=y_db, cmap='viridis', marker='o')
    plt.title('DBSCAN Clustering', fontsize=14)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Highlight noise points (label -1)
    plt.scatter(X[y_db == -1, 0], X[y_db == -1, 1], color='red', s=50, marker='x', label='Noise')

    plt.legend()
    plt.show()
