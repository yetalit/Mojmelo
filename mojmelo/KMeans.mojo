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
    var clusters: List[List[Int]]
    var centroids: Matrix
    var inertia: Float32
    var X: Matrix

    fn __init__(out self, K: Int = 5, init: String = 'kmeans++', max_iters: Int = 100, converge: String = 'centroid', tol: Float32 = 1e-4, random_state: Int = 42):
        self.K = K
        self.init = init.lower()
        self.max_iters = max_iters
        self.converge = converge.lower()
        self.tol = tol
        self.seed = random_state

        # list of sample indices for each cluster
        self.clusters = List[List[Int]](capacity = 0)
        # the centers (mean feature vector) for each cluster
        self.centroids = Matrix(0, 0)
        self.inertia = 0.0
        self.X = Matrix(0, 0)

    fn predict(mut self, X: Matrix) raises -> Matrix:
        self.X = X

        if self.init == 'random':
            self.centroids = X[Matrix.rand_choice(X.height, self.K, replace=False, seed = self.seed)]
        else:
            # Initialize centroids using KMeans++
            self._kmeans_plus_plus()

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            var clusters_old = self.clusters
            self.clusters = self._create_clusters()

            # Calculate new centroids from the clusters
            var centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check if clusters have changed
            if (self.converge == 'label' and self._is_converged(clusters_old)) or (self.converge != 'label' and self._is_converged(centroids_old)):
                break

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)
        
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
    fn _get_cluster_labels(self, clusters: List[List[Int]]) -> Matrix:
        # each sample will get the label of the cluster it was assigned to
        var labels = Matrix(self.X.height, 1)

        for cluster_idx in range(len(clusters)):
            for sample_index in clusters[cluster_idx]:
                labels.data[sample_index[]] = cluster_idx
        return labels^

    @always_inline
    fn _create_clusters(self) raises -> List[List[Int]]:
        # Assign the samples to the closest centroids to create clusters
        var clusters = List[List[Int]](capacity = self.K)
        for _ in range(self.K):
            clusters.append(List[Int]())

        var closest_centroid = self._closest_centroid()

        for idx in range(self.X.height):
            clusters[int(closest_centroid.data[idx])].append(idx)
        return clusters^

    @always_inline
    fn _closest_centroid(self) raises -> Matrix:
        var dist_from_centroids = Matrix(self.X.height, self.K, order='f')
        # Compute distances to the nearest centroid
        for idc in range(self.K):
            dist_from_centroids['', idc] = squared_euclidean_distance(self.X, self.centroids[idc], 1)
        return dist_from_centroids.argmin_slow(axis=1)

    @always_inline
    fn _get_centroids(self, clusters: List[List[Int]]) raises -> Matrix:
        # assign mean value of clusters to centroids
        var centroids = Matrix.zeros(self.K, self.X.width)
        for cluster_idx in range(len(clusters)):
            centroids[cluster_idx] = (self.X[clusters[cluster_idx]]).mean(0)
        return centroids^

    @always_inline
    fn _is_converged(mut self, centroids_old: Matrix) raises -> Bool:
        var old_inertia: Float32 = 0.0
        if self.converge == 'inertia' or self.init == 'random':
            old_inertia = self.inertia
            self.inertia = 0.0
            for idc in range(len(self.clusters)):
                self.inertia += squared_euclidean_distance(self.X[self.clusters[idc]], self.centroids[idc])
        if self.converge == 'centroid':
            return euclidean_distance(centroids_old, self.centroids, 1).sum() <= self.tol
        return abs(old_inertia - self.inertia) <= self.tol
        
    @always_inline
    fn _is_converged(mut self, labels_old: List[List[Int]]) raises -> Bool:
        var old_inertia: Float32 = 0.0
        if self.init == 'random':
            old_inertia = self.inertia
            self.inertia = 0.0
            for idc in range(len(self.clusters)):
                self.inertia += squared_euclidean_distance(self.X[self.clusters[idc]], self.centroids[idc])
        for idc in range(len(self.clusters)):
            if labels_old[idc] != self.clusters[idc]:
                return False
        return True

    fn get_clusters_data(self) raises -> Tuple[Matrix, Matrix]:
        var row_counts = Matrix(1, len(self.clusters))
        for i in range(row_counts.size):
            row_counts.data[i] = len(self.clusters[i])
        var clusters_raw = Matrix(1, self.X.height)
        var pointer = 0
        for cluster in self.clusters:
            for idx in cluster[]:
                clusters_raw.data[pointer] = idx[]
                pointer += 1

        return clusters_raw^, row_counts^
