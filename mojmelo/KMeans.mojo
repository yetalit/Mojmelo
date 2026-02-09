from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import squared_euclidean_distance, euclidean_distance
import random
import math

struct KMeans:
    """K-Means clustering."""
    var K: Int
    """The number of clusters to form as well as the number of centroids to generate."""
    var init: String
    """Method for initialization -> 'kmeans++', 'random'."""
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
    var X: Matrix

    fn __init__(out self, K: Int = 5, init: String = 'kmeans++', max_iters: Int = 100, converge: String = 'centroid', tol: Float32 = 1e-4, random_state: Int = 42):
        self.K = K
        self.init = init.lower()
        self.max_iters = max_iters
        self.converge = converge.lower()
        self.tol = tol

        random.seed(random_state)

        self.labels = List[Int]()
        self.centroids = Matrix(0, 0)
        self.inertia = 0.0
        self.X = Matrix(0, 0)

    fn fit(mut self, X: Matrix) raises:
        """Compute cluster centers and cluster index for each sample."""
        self.X = X

        if self.init == 'random':
            self.centroids = X[Matrix.rand_choice(X.height, self.K, replace=False, seed = False)]
        else:
            # Initialize centroids using KMeans++
            self._kmeans_plus_plus()

        var dist_from_centroids = Matrix(self.X.height, self.K, order='f')
        self.labels = self._create_labels(dist_from_centroids)
        var centroids_old = self.centroids
        var labels_old = self.labels.copy()
        var inertia_old = self.inertia
        # Optimize clusters
        for i in range(self.max_iters):
            # Calculate new centroids from the clusters
            self.centroids = self._get_centroids(dist_from_centroids)
            # Assign samples to closest centroids (create labels)
            self.labels = self._create_labels(dist_from_centroids)
            # check if clusters have changed
            if self._is_converged(dist_from_centroids, centroids_old, labels_old, inertia_old):
                break
            centroids_old = self.centroids
            labels_old = self.labels.copy()
            inertia_old = self.inertia
            if i == self.max_iters - 1:
                self.inertia = dist_from_centroids.min(axis=1).sum()

    fn fit_predict(mut self, X: Matrix) raises -> List[Int]:
        """Compute cluster centers and predict cluster index for each sample.

        Returns:
            List of cluster indices.
        """
        self.fit(X)
        return self.labels.copy()

    fn _kmeans_plus_plus(mut self) raises:
        # Randomly select the first centroid
        self.centroids = Matrix(self.K, self.X.width)
        self.centroids[0] = self.X[Int(random.random_ui64(0, self.X.height - 1))]

        var dist_from_centroids = Matrix.full(self.X.height, self.K-1, math.inf[DType.float32](), order='f')

        for i in range(1, self.K):
            # Compute distances to the nearest centroid
            dist_from_centroids['', i-1] = squared_euclidean_distance(self.X, self.centroids[i-1], 1)
            var min_distances = dist_from_centroids.min(axis=1)
            # Select the next centroid with probability proportional to the squared distances
            var probabilities = (min_distances / min_distances.sum()).cumsum()
            # Select the next centroid based on cumulative probabilities
            var rand_prob = random.random_float64().cast[DType.float32]()
            for idp in range(len(probabilities)):
                if rand_prob < probabilities.data[idp]:
                    self.centroids[i] = self.X[idp]
                    break

    @always_inline
    fn _create_labels(self, mut dist_from_centroids: Matrix) raises -> List[Int]:
        # Compute distances to the nearest centroid
        for idc in range(self.K):
            dist_from_centroids['', idc] = squared_euclidean_distance(self.X, self.centroids[idc], 1)
        return dist_from_centroids.argmin(axis=1)

    @always_inline
    fn _get_centroids(mut self, dist_from_centroids: Matrix) raises -> Matrix:
        # assign mean value of clusters to centroids
        var centroids = Matrix.zeros(self.K, self.X.width)
        var cluster_sizes = Matrix.zeros(self.K, 1)
        self.inertia = 0.0
        for idx in range(self.X.height):
            self.inertia += dist_from_centroids[idx, self.labels[idx]]
            centroids[self.labels[idx]] += self.X[idx]
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
