from collections import InlinedFixedVector, Dict
from memory import Span
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CVP, squared_euclidean_distance, manhattan_distance, le
from python import PythonObject

struct KNN(CVP):
    var k: Int
    var distance: fn(Matrix, Matrix, Int) raises -> Matrix
    var X_train: Matrix
    var y_train: PythonObject

    fn __init__(out self, k: Int = 3, metric: String = 'euc'):
        self.k = k
        if metric.lower() == 'man':
            self.distance = manhattan_distance
        else:
            self.distance = squared_euclidean_distance
        self.X_train = Matrix(0, 0)
        self.y_train = None

    fn fit(mut self, X: Matrix, y: PythonObject) raises:
        self.X_train = X
        self.y_train = y

    fn predict(self, X: Matrix) raises -> List[String]:
        var y_pred = List[String]()
        for i in range(X.height):
            y_pred.append(self._predict(X[i]))
        return y_pred^

    @always_inline
    fn _predict(self, x: Matrix) raises -> String:
        # Compute distances between x and all examples in the training set
        var distances = self.distance(self.X_train, x, 1)
        var dis_indices = InlinedFixedVector[Int](capacity = distances.size)
        for i in range(distances.size):
            dis_indices[i] = i
        # Sort distances such that first k elements are the smallest
        mojmelo.utils.utils.partition[le](Span[Float32, __origin_of(distances)](ptr= distances.data, length= distances.size), dis_indices, self.k)
        # Extract the labels of the k nearest neighbor and return the most common class label
        var k_neighbor_votes = Dict[String, Int]()
        var most_common = str(self.y_train[dis_indices[0]])
        for i in range(self.k):
            var label = str(self.y_train[dis_indices[i]])
            if label in k_neighbor_votes:
                k_neighbor_votes[label] += 1
            else:
                k_neighbor_votes[label] = 1
            if k_neighbor_votes[label] > k_neighbor_votes[most_common]:
                most_common = label
        return most_common

    fn __init__(out self, params: Dict[String, String]) raises:
        if 'k' in params:
            self.k = atol(params['k'])
        else:
            self.k = 3
        if 'metric' in params:
            if params['metric'].lower() == 'man':
                self.distance = manhattan_distance
            else:
                self.distance = squared_euclidean_distance
        else:
            self.distance = squared_euclidean_distance
        self.X_train = Matrix(0, 0)
        self.y_train = None
