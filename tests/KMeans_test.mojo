from mojmelo.KMeans import KMeans
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import ids_to_numpy
from std.python import Python

def main() raises:
    var km_test = Python.import_module("KMeans_test")
    var data = km_test.get_data() # X, n_clusters
    var k = KMeans(k=Int(py=data[1]), max_iters=150)
    var labels = k.fit_predict(Matrix.from_numpy(data[0]))
    km_test.test(data[0], ids_to_numpy(labels), k.centroids().to_numpy())
