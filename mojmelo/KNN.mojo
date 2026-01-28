from collections import Dict
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.KDTree import KDTreeResultVector, KDTree
from mojmelo.utils.utils import CV
from algorithm import parallelize

struct KNN(CV):
    """Classifier implementing the k-nearest neighbors vote."""
    var k: Int
    """Number of neighbors to use."""
    var metric: String
    """Metric to use for distance computation:
    Euclidean -> 'euc';
    Manhattan -> 'man'.
    """
    var kdtree: KDTree[sort_results=False]
    var y_train: Matrix

    fn __init__(out self, k: Int = 3, metric: String = 'euc') raises:
        self.k = k
        self.metric = metric.lower()
        self.kdtree = KDTree(Matrix(0, 0), build=False)
        self.y_train = Matrix(0, 0)

    fn fit(mut self, X: Matrix, y: Matrix) raises:
        """Fit the k-nearest neighbors classifier from the training dataset."""
        self.kdtree = KDTree(X, self.metric)
        self.y_train = y

    fn predict(mut self, X: Matrix) raises -> Matrix:
        """Predict the class indices for the provided data.

        Returns:
            Class indices for each data sample.
        """
        var y_pred = Matrix(X.height, 1)
        @parameter
        fn p(i: Int):
            try:
                y_pred.data[i] = self._predict(X[i])
            except e:
                print('Error:', e)
        parallelize[p](X.height)
        return y_pred^

    @always_inline
    fn _predict(mut self, x: Matrix) raises -> Float32:
        var kd_results = KDTreeResultVector()
        self.kdtree.n_nearest(Span[Float32](ptr=x.data, length=x.size), self.k, kd_results)
        # Extract the labels of the k nearest neighbor and return the most common class label
        var k_neighbor_votes = Dict[Int, Int]()
        var most_common = Int(self.y_train.data[kd_results[0].idx])
        for i in range(self.k):
            var label = Int(self.y_train.data[kd_results[i].idx])
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
        self.y_train = Matrix(0, 0)
