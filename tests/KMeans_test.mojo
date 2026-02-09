from mojmelo.KMeans import KMeans
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import ids_to_numpy
from python import Python

def main():
    km_test = Python.import_module("KMeans_test")
    data = km_test.get_data() # X, n_clusters
    k = KMeans(k=Int(py=data[1]), max_iters=150)
    X = Matrix.from_numpy(data[0])
    labels = k.fit_predict(X)
    km_test.test((X - X.mean(axis=0)).to_numpy(), ids_to_numpy(labels), k.centroids.to_numpy())
