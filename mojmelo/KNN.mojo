from collections import Dict
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.KDTree import KDTreeResultVector, KDTree
from mojmelo.utils.utils import CV, MODEL_IDS
from algorithm import parallelize

struct KNN(CV, Copyable):
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
    comptime MODEL_ID = 4
    comptime metric_ids: List[String] = ['euc', 'man']

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

    fn save(self, path: String) raises:
        """Save model data necessary for prediction to the specified path."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        with open(_path, "w") as f:
            f.write_bytes(UInt8(Self.MODEL_ID).as_bytes())
            f.write_bytes(UInt32(self.k).as_bytes())
            f.write_bytes(UInt8(materialize[self.metric_ids]().index(self.metric)).as_bytes())
            f.write_bytes(UInt64(self.kdtree.N).as_bytes())
            f.write_bytes(UInt64(self.kdtree.dim).as_bytes())
            var X = Matrix(self.kdtree.N, self.kdtree.dim)
            for i in range(self.kdtree.N):
                X[Int(self.kdtree.ind[i]), unsafe=True] = self.kdtree._data[i, unsafe=True]
            f.write_bytes(Span(ptr=X.data.bitcast[UInt8](), length=4*X.size))
            f.write_bytes(UInt64(self.y_train.size).as_bytes())
            f.write_bytes(Span(ptr=self.y_train.data.bitcast[UInt8](), length=4*self.y_train.size))

    @staticmethod
    fn load(path: String) raises -> Self:
        """Load a saved model from the specified path for prediction."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        var model = Self()
        with open(_path, "r") as f:
            var id = f.read_bytes(1)[0]
            if id < 1 or id > MODEL_IDS.size-1:
                raise Error('Input file with invalid metadata!')
            elif id != Self.MODEL_ID:
                raise Error('Based on the metadata,', _path, 'belongs to', materialize[MODEL_IDS]()[id], 'algorithm!')
            var k = Int(f.read_bytes(4).unsafe_ptr().bitcast[UInt32]()[])
            var metric = materialize[Self.metric_ids]()[f.read_bytes(1)[0]]
            var n_samples = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
            var n_features = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
            var X = Matrix(n_samples, n_features, UnsafePointer[Float32, MutAnyOrigin](f.read_bytes(4 * n_samples * n_features).unsafe_ptr().bitcast[Float32]()))
            var y_size = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
            var y_train = Matrix(y_size, 1, UnsafePointer[Float32, MutAnyOrigin](f.read_bytes(4 * y_size).unsafe_ptr().bitcast[Float32]()))
            model.k = k
            model.metric = metric
            model.fit(X, y_train)
        return model^

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
