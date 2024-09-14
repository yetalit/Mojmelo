from mojmelo.KMeans import KMeans
from mojmelo.utils.Matrix import Matrix
from python import Python

def main():
    km_test = Python.import_module("KMeans_test")
    data = km_test.get_data() # X, n_clusters
    k = KMeans(K=data[1], max_iters=150)
    _ = k.predict(Matrix.from_numpy(data[0]))
    clusters_raw, row_counts = k.get_clusters_data()
    km_test.test(data[0], clusters_raw.to_numpy(), row_counts.to_numpy(), k.centroids.to_numpy())
