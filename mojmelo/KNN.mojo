from collections import Dict
from buffer import NDBuffer
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.KDTree import KDTreeResultVector, KDTree
from mojmelo.utils.utils import CVP
from python import PythonObject

struct KNN(CVP):
    var k: Int
    var metric: String
    var kdtree: KDTree
    var y_train: PythonObject

    fn __init__(out self, k: Int = 3, metric: String = 'euc'):
        self.k = k
        self.metric = metric.lower()
        self.kdtree = KDTree(Matrix(0, 0), build=False)
        self.y_train = None

    fn fit(mut self, X: Matrix, y: PythonObject) raises:
        self.kdtree = KDTree(X, self.metric)
        self.y_train = y

    fn predict(self, X: Matrix) raises -> List[String]:
        var y_pred = List[String]()
        for i in range(X.height):
            y_pred.append(self._predict(X[i]))
        return y_pred^

    @always_inline
    fn _predict(self, x: Matrix) raises -> String:
        var kd_results = KDTreeResultVector()
        self.kdtree.n_nearest(NDBuffer[type=DType.float32, rank=1](x.data, x.size), self.k, kd_results)
        # Extract the labels of the k nearest neighbor and return the most common class label
        var k_neighbor_votes = Dict[String, Int]()
        var most_common = String(self.y_train[kd_results[0].idx])
        for i in range(self.k):
            var label = String(self.y_train[kd_results[i].idx])
            if label in k_neighbor_votes:
                k_neighbor_votes[label] += 1
            else:
                k_neighbor_votes[label] = 1
            if k_neighbor_votes[label] > k_neighbor_votes[most_common]:
                most_common = label
        return most_common

    fn __init__(out self, params: Dict[String, String]) raises:
        if 'k' in params:
            self.k = atol(String(params['k']))
        else:
            self.k = 3
        if 'metric' in params:
            self.metric = params['metric'].lower()
        else:
            self.metric = 'euc'
        self.kdtree = KDTree(Matrix(0, 0), build=False)
        self.y_train = None
