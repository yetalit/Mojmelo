from collections.vector import InlinedFixedVector
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import euclidean_distance, manhattan_distance, le

struct KNN:
    var k: Int
    var distance: fn(Matrix, Matrix) raises -> Float32
    var X_train: Matrix
    var y_train: PythonObject

    fn __init__(inout self, k: Int = 3, metric: String = 'euc'):
        self.k = k
        if metric.lower() == 'man':
            self.distance = manhattan_distance
        else:
            self.distance = euclidean_distance
        self.X_train = Matrix(0, 0)
        self.y_train = None

    fn fit(inout self, X: Matrix, y: PythonObject):
        self.X_train = X
        self.y_train = y

    fn predict(self, X: Matrix) raises -> List[String]:
        var y_pred = List[String]()
        for i in range(X.height):
            y_pred.append(self._predict(X[i]))
        return y_pred^

    fn _predict(self, x: Matrix) raises -> String:
        var distances = Matrix(1, self.X_train.height)
        var dis_indices = InlinedFixedVector[Int](capacity = distances.size)
        # Compute distances between x and all examples in the training set
        for i in range(distances.size):
            dis_indices.append(i)
            distances.data[i] = self.distance(x, self.X_train[i])
        # Sort distances such that first k elements are the smallest
        mojmelo.utils.utils.partition[le](distances.data, dis_indices, self.k, distances.size)
        # Extract the labels of the k nearest neighbor and return the most common class label
        var k_neighbor_votes = Dict[String, Int]()
        var most_common: String = self.y_train[dis_indices[0]]
        for i in range(self.k):
            var label: String = self.y_train[dis_indices[i]]
            if label in k_neighbor_votes:
                k_neighbor_votes[label] += 1
            else:
                k_neighbor_votes[label] = 1
            if k_neighbor_votes[label] > k_neighbor_votes[most_common]:
                most_common = label
        return most_common
