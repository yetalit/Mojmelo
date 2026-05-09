from std.collections import Dict
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.KDTree import KDTreeResultVector, KDTree
from mojmelo.utils.utils import CV, MODEL_IDS
from std.algorithm import parallelize

struct KNN[metric: String = 'euc'](CV, Copyable):
    """Classifier implementing the k-nearest neighbors vote.

    Parameters:
        metric: Metric to use for distance computation:
            Euclidean -> 'euc';
            Manhattan -> 'man'.

    """
    var k: Int
    """Number of neighbors to use."""
    var search_depth: Int
    """Current KDTree implementation applies some approximation to its search results.
        Increasing search_depth can lead to more accurate results at the cost of performance."""
    var kdtree: KDTree[sort_results=True, metric=Self.metric]
    var y_train: Matrix
    comptime MODEL_ID = 4
    comptime metric_ids: List[String] = ['euc', 'man']

    def __init__(out self, k: Int = 3, search_depth: Int = 1) raises:
        self.k = k
        self.search_depth = search_depth
        self.kdtree = KDTree[sort_results=True, metric=Self.metric](Matrix(0, 0), build=False)
        self.y_train = Matrix(0, 0)

    def fit(mut self, X: Matrix, y: Matrix) raises:
        """Fit the k-nearest neighbors classifier from the training dataset."""
        self.kdtree = KDTree[sort_results=True, metric=Self.metric](X)
        self.y_train = y

    def predict(mut self, X: Matrix) raises -> Matrix:
        """Predict the class indices for the provided data.

        Returns:
            Class indices for each data sample.
        """
        var y_pred = Matrix(X.height, 1)
        @parameter
        def p(i: Int):
            try:
                y_pred.data[i] = self._predict(X[i])
            except e:
                print('Error:', e)
        parallelize[p](X.height)
        return y_pred^

    @always_inline
    def _predict(mut self, x: Matrix) raises -> Float32:
        var kd_results = KDTreeResultVector()
        self.kdtree.n_nearest(Span(ptr=x.data, length=x.size), self.search_depth * self.k, kd_results)
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
        return Float32(most_common)

    def save(self, path: String) raises:
        """Save model data necessary for prediction to the specified path."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        with open(_path, "w") as f:
            f.write_bytes(UInt8(Self.MODEL_ID).as_bytes())
            f.write_bytes(UInt32(self.k).as_bytes())
            f.write_bytes(UInt8(materialize[self.metric_ids]().index(self.metric)).as_bytes())
            f.write_bytes(UInt32(self.search_depth).as_bytes())
            f.write_bytes(UInt64(self.kdtree.N).as_bytes())
            f.write_bytes(UInt64(self.kdtree.dim).as_bytes())
            var X = Matrix(self.kdtree.N, self.kdtree.dim)
            for i in range(self.kdtree.N):
                X[Int(self.kdtree.ind[i]), unsafe=True] = self.kdtree._data[i, unsafe=True]
            f.write_bytes(Span(ptr=X.data.bitcast[UInt8](), length=4*X.size))
            f.write_bytes(Span(ptr=self.y_train.data.bitcast[UInt8](), length=4*self.y_train.size))

    @staticmethod
    def load[type: UInt8](path: String) raises -> KNN[Self.metric_ids[type]]:
        """Load a saved model from the specified path for prediction."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        var model = KNN[Self.metric_ids[type]]()
        with open(_path, "r") as f:
            var id = f.read_bytes(1)[0]
            if id < 1 or id > UInt8(MODEL_IDS.size-1):
                raise Error('Input file with invalid metadata!')
            elif id != Self.MODEL_ID:
                raise Error('Based on the metadata, ', _path, ' belongs to ', materialize[MODEL_IDS]()[id], ' algorithm!')
            var k = Int(f.read_bytes(4).unsafe_ptr().bitcast[UInt32]()[])
            var metric = f.read_bytes(1)[0]
            if type != metric:
                raise Error('Based on the metadata, ', _path, ' is using ', materialize[Self.metric_ids]()[metric], ' ! Use [type=', metric, ']')
            var search_depth = Int(f.read_bytes(4).unsafe_ptr().bitcast[UInt32]()[])
            var n_samples = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
            var n_features = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
            var X = Matrix(n_samples, n_features, UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(f.read_bytes(4 * n_samples * n_features).unsafe_ptr())))
            var y_train = Matrix(n_samples, 1, UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(f.read_bytes(4 * n_samples).unsafe_ptr())))
            model.k = k
            model.search_depth = search_depth
            model.fit(X, y_train)
        return model^

    def __init__(out self, params: Dict[String, String]) raises:
        if 'k' in params:
            self.k = atol(String(params['k']))
        else:
            self.k = 3
        if 'search_depth' in params:
            self.search_depth = atol(String(params['search_depth']))
        else:
            self.search_depth = 1
        self.kdtree = KDTree[sort_results=True, metric=Self.metric](Matrix(0, 0), build=False)
        self.y_train = Matrix(0, 0)
