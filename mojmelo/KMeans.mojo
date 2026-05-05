from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import euclidean_distance, squared_euclidean_distance, MODEL_IDS
import std.random as random
import std.math as math
from std.algorithm import vectorize, parallelize

struct KMeans(Copyable):
    """K-Means clustering."""
    var k: Int
    """The number of clusters to form as well as the number of centroids to generate."""
    var init: String
    """Method for initialization -> 'kmeans++', 'random'."""
    var n_centroid_init: Int
    """The number of candidate centroids to be initialized."""
    var max_iters: Int
    """Maximum number of iterations of the k-means algorithm for a single run."""
    var converge: String
    """The converge method:
    Change in centroids <= tol -> 'centroid';
    Change in inertia <= tol -> 'inertia';
    Exact change in labels -> 'label'.
    """
    var tol: Float32
    """Relative tolerance value."""
    var labels: List[Int]
    var centroids_: Matrix
    var inertia: Float32
    """Sum of squared distances of samples to their closest cluster center."""
    var X_mean: Matrix
    comptime MODEL_ID = 5

    def __init__(out self, k: Int = 5, init: String = 'kmeans++', n_centroid_init: Int = 1, max_iters: Int = 100, converge: String = 'centroid', tol: Float32 = 1e-4, random_state: Int = 0):
        self.k = k
        self.init = init.lower()
        self.n_centroid_init = n_centroid_init
        self.max_iters = max_iters
        self.converge = converge.lower()
        self.tol = tol

        random.seed(random_state)

        self.labels = List[Int]()
        self.centroids_ = Matrix(0, 0)
        self.inertia = 0.0
        self.X_mean = Matrix(0, 0)

    def fit(mut self, X: Matrix) raises:
        """Compute cluster centers and cluster index for each sample."""
        # Mean centering
        self.X_mean = Matrix.zeros(1, X.width)
        var n_rows = X.height
        var n_cols = X.width
        @parameter
        def p(row: Int):
            var x_ptr = X.data + row * n_cols
            @parameter
            def add_row[simd_width: Int](col: Int) unified {mut}:
                self.X_mean.data.store(col, self.X_mean.data.load[width=simd_width](col) + x_ptr.load[width=simd_width](col))
            vectorize[X.simd_width](n_cols, add_row)
        parallelize[p](n_rows)
        var inv_n_rows = 1.0 / Float32(n_rows)
        @parameter
        def div[simd_width: Int](col: Int) unified {mut}:
            self.X_mean.data.store(col, self.X_mean.data.load[width=simd_width](col) * inv_n_rows)
        vectorize[X.simd_width](n_cols, div)
        var X_ = X - self.X_mean

        self.centroids_ = self._initial_centroids(X_)
        var X_norms = X_.ele_mul(X_).sum(axis=1)
        var C_norms = self.centroids_.ele_mul(self.centroids_).sum(axis=1).T()
        var dist_from_centroids = (X_ * self.centroids_.T()) * -2.0
        dist_from_centroids += X_norms
        dist_from_centroids += C_norms
        self.labels = dist_from_centroids.argmin(axis=1)
        var centroids_old = self.centroids_
        var labels_old = self.labels.copy()
        var inertia_old = self.inertia

        for i in range(self.max_iters):
            self.centroids_ = self._get_centroids(dist_from_centroids, X_)
            self.labels = self._create_labels(dist_from_centroids, X_, X_norms)
            if self._is_converged(dist_from_centroids, centroids_old, labels_old, inertia_old):
                break
            centroids_old = self.centroids_
            labels_old = self.labels.copy()
            inertia_old = self.inertia
            if i == self.max_iters - 1:
                self.inertia = dist_from_centroids.min(axis=1).sum()

    @always_inline
    def _get_centroids(mut self, dist_from_centroids: Matrix, X: Matrix) raises -> Matrix:
        var centroids = Matrix.zeros(self.k, X.width)
        var cluster_sizes = Matrix.zeros(self.k, 1)
        self.inertia = 0.0
        for idx in range(X.height):
            var label = self.labels[idx]
            self.inertia += dist_from_centroids[idx, label]
            cluster_sizes.data[label] += 1.0
            # write directly into centroid row pointer
            var c_ptr = centroids.data + label * X.width
            var x_ptr = X.data + idx * X.width
            @parameter
            def accumulate[simd_width: Int](j: Int) unified {mut}:
                c_ptr.store(j, c_ptr.load[width=simd_width](j) + x_ptr.load[width=simd_width](j))
            vectorize[centroids.simd_width](X.width, accumulate)
        return centroids / cluster_sizes.where(cluster_sizes == 0.0, 1.0, cluster_sizes)

    @always_inline
    def _is_converged(mut self, dist_from_centroids: Matrix, centroids_old: Matrix, 
                    labels_old: List[Int], inertia_old: Float32) raises -> Bool:
        if self.converge == 'centroid':
            if euclidean_distance(centroids_old, self.centroids_) <= self.tol:
                self.inertia = dist_from_centroids.min(axis=1).sum()
                return True
            return False
        if self.converge == 'inertia':
            if abs(inertia_old - self.inertia) <= self.tol:
                self.centroids_ = centroids_old
                self.labels = labels_old.copy()
                return True
            return False
        return labels_old == self.labels

    def _initial_centroids(self, X: Matrix) raises -> Matrix:
        var candidate_centroids = List[Matrix]()
        var inertia_values = Matrix(1, self.n_centroid_init)
        if self.init == 'random':
            for idc in range(self.n_centroid_init):
                candidate_centroids.append(X[Matrix.rand_choice(X.height, self.k, replace=False, seed = False)])
                var dist_from_centroids = Matrix(X.height, self.k)
                for i in range(self.k):
                    # Compute distances to the nearest centroid
                    dist_from_centroids['', i] = squared_euclidean_distance(X, candidate_centroids[idc][i], 1)
                inertia_values.data[idc] = dist_from_centroids.min(axis=1).sum()
        else:
            # Initialize centroids using KMeans++
            self._kmeans_plus_plus(X, candidate_centroids, inertia_values)
        return candidate_centroids[inertia_values.argmin()]

    def predict(self, X: Matrix) raises -> List[Int]:
        """Predict cluster index for each sample.

        Returns:
            List of cluster indices.
        """
        var X_ = X - self.X_mean
        var dist_from_centroids = Matrix(X_.height, self.k)
        for idc in range(self.k):
            dist_from_centroids['', idc] = squared_euclidean_distance(X_, self.centroids_[idc], 1)
        return dist_from_centroids.argmin(axis=1)

    def fit_predict(mut self, X: Matrix) raises -> List[Int]:
        """Compute cluster centers and predict cluster index for each sample.

        Returns:
            List of cluster indices.
        """
        self.fit(X)
        return self.labels.copy()

    def save(self, path: String) raises:
        """Save model data necessary for prediction to the specified path."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        with open(_path, "w") as f:
            f.write_bytes(UInt8(Self.MODEL_ID).as_bytes())
            f.write_bytes(UInt64(self.k).as_bytes())
            f.write_bytes(UInt64(self.centroids_.size).as_bytes())
            f.write_bytes(Span(ptr=self.centroids_.data.bitcast[UInt8](), length=4*self.centroids_.size))
            f.write_bytes(Span(ptr=self.X_mean.data.bitcast[UInt8](), length=4*self.X_mean.size))

    @staticmethod
    def load(path: String) raises -> Self:
        """Load a saved model from the specified path for prediction."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        var model = Self()
        with open(_path, "r") as f:
            var id = f.read_bytes(1)[0]
            if id < 1 or id > UInt8(MODEL_IDS.size-1):
                raise Error('Input file with invalid metadata!')
            elif id != Self.MODEL_ID:
                raise Error('Based on the metadata, ', _path, ' belongs to ', materialize[MODEL_IDS]()[id], ' algorithm!')
            var k = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
            var n_features = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
            model.k = k
            model.centroids_ = Matrix(1, n_features, UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(f.read_bytes(4 * n_features).unsafe_ptr())))
            model.X_mean = Matrix(1, n_features, UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(f.read_bytes(4 * n_features).unsafe_ptr())))
        return model^

    def centroids(self) raises -> Matrix:
        return self.centroids_ + self.X_mean

    def _kmeans_plus_plus(self, X: Matrix, mut candidate_centroids: List[Matrix], mut inertia_values: Matrix) raises:
        for idc in range(self.n_centroid_init):
            candidate_centroids.append(Matrix(self.k, X.width))
            candidate_centroids[idc][0] = X[Int(random.random_ui64(0, UInt64(X.height - 1)))]

            var min_distances = Matrix.full(X.height, 1, math.inf[DType.float32]())

            for i in range(1, self.k):
                # fused squared euclidean
                var dists = Matrix(X.height, 1)
                var c_ptr = candidate_centroids[idc].data + (i - 1) * X.width
                for row in range(X.height):
                    var x_ptr = X.data + row * X.width
                    var acc: Float32 = 0.0
                    @parameter
                    def sq[simd_width: Int](col: Int) unified {mut}:
                        var d = x_ptr.load[width=simd_width](col) - c_ptr.load[width=simd_width](col)
                        acc += (d * d).reduce_add()
                    vectorize[X.simd_width](X.width, sq)
                    dists.data[row] = acc

                for row in range(X.height):
                    if dists.data[row] < min_distances.data[row]:
                        min_distances.data[row] = dists.data[row]
                var total = min_distances.sum()
                var rand_prob = random.random_float64().cast[DType.float32]() * total
                var cumsum: Float32 = 0.0
                for idp in range(X.height):
                    cumsum += min_distances.data[idp]
                    if cumsum >= rand_prob:
                        candidate_centroids[idc][i] = X[idp]
                        break

            var dists_last = Matrix(X.height, 1)
            var c_ptr_last = candidate_centroids[idc].data + (self.k - 1) * X.width
            for row in range(X.height):
                var x_ptr = X.data + row * X.width
                var acc: Float32 = 0.0
                @parameter
                def sq_last[simd_width: Int](col: Int) unified {mut}:
                    var d = x_ptr.load[width=simd_width](col) - c_ptr_last.load[width=simd_width](col)
                    acc += (d * d).reduce_add()
                vectorize[X.simd_width](X.width, sq_last)
                dists_last.data[row] = acc

            for row in range(X.height):
                if dists_last.data[row] < min_distances.data[row]:
                    min_distances.data[row] = dists_last.data[row]
            inertia_values.data[idc] = min_distances.sum()

    @always_inline
    def _create_labels(mut self, dist_from_centroids: Matrix, X: Matrix, X_norms: Matrix) raises -> List[Int]:
        var labels = List[Int](capacity=X.height)
        labels.resize(X.height, 0)
        var C_norms = self.centroids_.ele_mul(self.centroids_).sum(axis=1)
        @parameter
        def p(i: Int):
            var best = 0
            var best_dist: Float32 = math.inf[DType.float32]()

            var x_ptr = X.data + i * X.width

            for k in range(self.k):
                var c_ptr = self.centroids_.data + k * X.width

                var dot: Float32 = 0.0
                @parameter
                def mul[simd_width: Int](j: Int) unified {mut}:
                    dot += (x_ptr.load[width=simd_width](j) *
                            c_ptr.load[width=simd_width](j)).reduce_add()
                vectorize[X.simd_width](X.width, mul)

                var dist = X_norms.data[i] - 2.0 * dot + C_norms.data[k]

                if dist < best_dist:
                    best_dist = dist
                    best = k

            labels[i] = best
        parallelize[p](X.height)

        return labels^
