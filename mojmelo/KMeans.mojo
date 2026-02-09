from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import squared_euclidean_distance, euclidean_distance
import random
import math

struct KMeans:
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
    var centroids: Matrix
    var inertia: Float32
    """Sum of squared distances of samples to their closest cluster center."""
    var X_mean: Matrix

    fn __init__(out self, k: Int = 5, init: String = 'kmeans++', n_centroid_init: Int = 1, max_iters: Int = 100, converge: String = 'centroid', tol: Float32 = 1e-4, random_state: Int = 0):
        self.k = k
        self.init = init.lower()
        self.n_centroid_init = n_centroid_init
        self.max_iters = max_iters
        self.converge = converge.lower()
        self.tol = tol

        random.seed(random_state)

        self.labels = List[Int]()
        self.centroids = Matrix(0, 0)
        self.inertia = 0.0
        self.X_mean = Matrix(0, 0)

    fn fit(mut self, X: Matrix) raises:
        """Compute cluster centers and cluster index for each sample."""
        # Mean centering
        self.X_mean = X.mean(0)
        var X_ = X - self.X_mean

        self.centroids = self._initial_centroids(X_)
        var dist_from_centroids = Matrix(X_.height, self.k)
        self.labels = self._create_labels(dist_from_centroids, X_)
        var centroids_old = self.centroids
        var labels_old = self.labels.copy()
        var inertia_old = self.inertia
        # Optimize clusters
        for i in range(self.max_iters):
            # Calculate new centroids from the clusters
            self.centroids = self._get_centroids(dist_from_centroids, X_)
            # Assign samples to closest centroids (create labels)
            self.labels = self._create_labels(dist_from_centroids, X_)
            # check if clusters have changed
            if self._is_converged(dist_from_centroids, centroids_old, labels_old, inertia_old):
                break
            centroids_old = self.centroids
            labels_old = self.labels.copy()
            inertia_old = self.inertia
            if i == self.max_iters - 1:
                self.inertia = dist_from_centroids.min(axis=1).sum()

    fn _initial_centroids(self, X: Matrix) raises -> Matrix:
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

    fn predict(self, X: Matrix) raises -> List[Int]:
        """Predict cluster index for each sample.

        Returns:
            List of cluster indices.
        """
        var X_ = X - self.X_mean
        var dist_from_centroids = Matrix(X_.height, self.k)
        for idc in range(self.k):
            dist_from_centroids['', idc] = squared_euclidean_distance(X_, self.centroids[idc], 1)
        return dist_from_centroids.argmin(axis=1)

    fn fit_predict(mut self, X: Matrix) raises -> List[Int]:
        """Compute cluster centers and predict cluster index for each sample.

        Returns:
            List of cluster indices.
        """
        self.fit(X)
        return self.labels.copy()

    fn _kmeans_plus_plus(self, X: Matrix, mut candidate_centroids: List[Matrix], mut inertia_values: Matrix) raises:
        for idc in range(self.n_centroid_init):
            # Randomly select the first centroid
            candidate_centroids.append(Matrix(self.k, X.width))
            candidate_centroids[idc][0] = X[Int(random.random_ui64(0, X.height - 1))]

            var dist_from_centroids = Matrix.full(X.height, self.k, math.inf[DType.float32]())

            for i in range(1, self.k):
                # Compute distances to the nearest centroid
                dist_from_centroids['', i-1] = squared_euclidean_distance(X, candidate_centroids[idc][i-1], 1)
                var min_distances = dist_from_centroids.min(axis=1)
                # Select the next centroid with probability proportional to the squared distances
                var probabilities = (min_distances / min_distances.sum()).cumsum()
                # Select the next centroid based on cumulative probabilities
                var rand_prob = random.random_float64().cast[DType.float32]()
                for idp in range(len(probabilities)):
                    if rand_prob < probabilities.data[idp]:
                        candidate_centroids[idc][i] = X[idp]
                        break
            dist_from_centroids['', self.k-1] = squared_euclidean_distance(X, candidate_centroids[idc][self.k-1], 1)
            inertia_values.data[idc] = dist_from_centroids.min(axis=1).sum()

    @always_inline
    fn _create_labels(self, mut dist_from_centroids: Matrix, X: Matrix) raises -> List[Int]:
        # Compute distances to the nearest centroid
        for idc in range(self.k):
            dist_from_centroids['', idc] = squared_euclidean_distance(X, self.centroids[idc], 1)
        return dist_from_centroids.argmin(axis=1)

    @always_inline
    fn _get_centroids(mut self, dist_from_centroids: Matrix, X: Matrix) raises -> Matrix:
        # assign mean value of clusters to centroids
        var centroids = Matrix.zeros(self.k, X.width)
        var cluster_sizes = Matrix.zeros(self.k, 1)
        self.inertia = 0.0
        for idx in range(X.height):
            self.inertia += dist_from_centroids[idx, self.labels[idx]]
            centroids[self.labels[idx]] += X[idx]
            cluster_sizes.data[self.labels[idx]] += 1
        return centroids / cluster_sizes

    @always_inline
    fn _is_converged(mut self, dist_from_centroids: Matrix, centroids_old: Matrix, labels_old: List[Int], inertia_old: Float32) raises -> Bool:
        if self.converge == 'centroid':
            if euclidean_distance(centroids_old, self.centroids, 1).sum() <= self.tol:
                self.inertia = dist_from_centroids.min(axis=1).sum()  
                return True
            return False
        if self.converge == 'inertia':
            if abs(inertia_old - self.inertia) <= self.tol:
                self.centroids = centroids_old
                self.labels = labels_old.copy()
                return True
            return False
        return labels_old == self.labels
