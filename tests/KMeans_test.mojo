from mojmelo.KMeans import KMeans
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import ids_to_numpy
from python import Python

def main():
    km_test = Python.import_module("KMeans_test")
    data = km_test.get_data() # X, n_clusters
    k = KMeans(K=Int(py=data[1]), max_iters=150)
    labels = k.fit_predict(Matrix.from_numpy(data[0]))
    km_test.test(data[0], ids_to_numpy(labels), k.centroids.to_numpy())
