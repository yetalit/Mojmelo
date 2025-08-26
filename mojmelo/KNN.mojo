from collections import Dict
from buffer import NDBuffer
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.KDTree import KDTreeResultVector, KDTree
from mojmelo.utils.utils import CVP
from python import PythonObject
from algorithm import parallelize
from sys import num_performance_cores

struct KNN(CVP):
    var k: Int
    var metric: String
    var n_jobs: Int
    var kdtree: KDTree
    var y_train: List[String]

    fn __init__(out self, k: Int = 3, metric: String = 'euc', n_jobs: Int = 0) raises:
        self.k = k
        self.metric = metric.lower()
        self.n_jobs = n_jobs
        self.kdtree = KDTree(Matrix(0, 0), build=False)
        self.y_train = List[String]()

    fn fit(mut self, X: Matrix, y: PythonObject) raises:
        self.kdtree = KDTree(X, self.metric)
        self.y_train = List[String](capacity=len(y))
        self.y_train.resize(len(y), '')
        for i in range(X.height):
            self.y_train[i] = String(y[i])

    fn predict(self, X: Matrix) raises -> List[String]:
        var y_pred = List[String](capacity=X.height)
        y_pred.resize(X.height, '')
        if self.n_jobs == 0:
            for i in range(X.height):
                y_pred[i] = self._predict(X[i])
        else:
            var n_workers = self.n_jobs
            if n_workers == -1:
                n_workers = num_performance_cores()
            @parameter
            fn p(i: Int):
                try:
                    y_pred[i] = self._predict(X[i])
                except:
                    print('Error predicting sample ', i)
            parallelize[p](X.height, n_workers)
        return y_pred^

    @always_inline
    fn _predict(self, x: Matrix) raises -> String:
        var kd_results = KDTreeResultVector()
        self.kdtree.n_nearest(NDBuffer[dtype=DType.float32, rank=1](x.data, x.size), self.k, kd_results)
        # Extract the labels of the k nearest neighbor and return the most common class label
        var k_neighbor_votes = Dict[String, Int]()
        var most_common = self.y_train[kd_results[0].idx]
        for i in range(self.k):
            var label = self.y_train[kd_results[i].idx]
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
        if 'n_jobs' in params:
            self.n_jobs = atol(String(params['n_jobs']))
        else:
            self.n_jobs = 0
        self.kdtree = KDTree(Matrix(0, 0), build=False)
        self.y_train = List[String]()
