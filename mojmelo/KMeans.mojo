from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import squared_euclidean_distance, euclidean_distance
import random
import math

struct KMeans:
    var K: Int
    var init: String
    var max_iters: Int
    var converge: String
    var tol: Float32
    var seed: Int
    var labels: List[Int]
    var centroids: Matrix
    var dist_from_centroids: Matrix
    var inertia: Float32
    var X: Matrix

    fn __init__(out self, K: Int = 5, init: String = 'kmeans++', max_iters: Int = 100, converge: String = 'centroid', tol: Float32 = 1e-4, random_state: Int = 42):
        self.K = K
        self.init = init.lower()
        self.max_iters = max_iters
        self.converge = converge.lower()
        self.tol = tol
        self.seed = random_state

        self.labels = List[Int]()
        self.centroids = Matrix(0, 0)
        self.dist_from_centroids = Matrix(0, 0)
        self.inertia = 0.0
        self.X = Matrix(0, 0)

    fn predict(mut self, X: Matrix) raises -> List[Int]:
        self.X = X

        if self.init == 'random':
            self.centroids = X[Matrix.rand_choice(X.height, self.K, replace=False, seed = self.seed)]
        else:
            # Initialize centroids using KMeans++
            self._kmeans_plus_plus()

        var centroids_old = self.centroids
        self.dist_from_centroids = Matrix(self.X.height, self.K, order='f')
        self.labels = self._create_labels()
        var labels_old = self.labels
        var inertia_old = self.inertia
        # Optimize clusters
        for _ in range(self.max_iters):
            # Calculate new centroids from the clusters
            self.centroids = self._get_centroids()
            # Assign samples to closest centroids (create labels)
            self.labels = self._create_labels()
            # check if clusters have changed
            if self._is_converged(centroids_old, labels_old, inertia_old):
                break
            centroids_old = self.centroids
            labels_old = self.labels
            inertia_old = self.inertia
        # Classify samples as the index of their clusters
        return self.labels
        
    fn _kmeans_plus_plus(mut self) raises:
        # Randomly select the first centroid
        random.seed(self.seed)
        self.centroids = Matrix(self.K, self.X.width)
        self.centroids[0] = self.X[int(random.random_ui64(0, self.X.height - 1))]

        for i in range(1, self.K):
            # Only consider the centroids that have been initialized
            var dist_from_centroids = Matrix(self.X.height, i, order='f')
            # Compute distances to the nearest centroid
            for idc in range(i):
                dist_from_centroids['', idc] = squared_euclidean_distance(self.X, self.centroids[idc], 1)
            var min_distances = dist_from_centroids.min(axis=1)
            # Select the next centroid with probability proportional to the squared distances
            var probabilities = (min_distances / min_distances.sum()).cumsum()
            # Select the next centroid based on cumulative probabilities
            for idp in range(len(probabilities)):
                if random.random_float64().cast[DType.float32]() < probabilities.data[idp]:
                    self.centroids[i] = self.X[idp]
                    break

    @always_inline
    fn _create_labels(mut self) raises -> List[Int]:
        # Compute distances to the nearest centroid
        for idc in range(self.K):
            self.dist_from_centroids['', idc] = squared_euclidean_distance(self.X, self.centroids[idc], 1)
        return self.dist_from_centroids.argmin_slow(axis=1)

    @always_inline
    fn _get_centroids(mut self) raises -> Matrix:
        # assign mean value of clusters to centroids
        var centroids = Matrix.zeros(self.K, self.X.width)
        var cluster_sizes = Matrix.zeros(self.K, 1)
        self.inertia = 0.0
        for idx in range(self.X.height):
            self.inertia += self.dist_from_centroids[idx, self.labels[idx]]
            centroids[self.labels[idx]] += self.X[idx]
            cluster_sizes.data[self.labels[idx]] += 1
        return centroids / cluster_sizes

    @always_inline
    fn _is_converged(mut self, centroids_old: Matrix, labels_old: List[Int], inertia_old: Float32) raises -> Bool:
        if self.converge == 'centroid':
            if euclidean_distance(centroids_old, self.centroids, 1).sum() <= self.tol:
                self.inertia = 0.0
                for idx in range(self.X.height):
                    self.inertia += self.dist_from_centroids[idx, self.labels[idx]]
                return True
            return False
        if self.converge == 'inertia':
            if abs(inertia_old - self.inertia) <= self.tol:
                self.centroids = centroids_old
                self.labels = labels_old
                return True
            return False
        return labels_old == self.labels
