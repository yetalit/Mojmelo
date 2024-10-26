from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import euclidean_distance
import random
import math

struct KMeans:
    var K: Int
    var init: String
    var max_iters: Int
    var tol: Float32
    var seed: Int
    var clusters: List[List[Int]]
    var centroids: Matrix
    var X: Matrix

    fn __init__(inout self, K: Int = 5, init: String = 'kmeans++', max_iters: Int = 100, tol: Float32 = 1e-4, random_state: Int = 42):
        self.K = K
        self.init = init.lower()
        self.max_iters = max_iters
        self.tol = tol
        self.seed = random_state

        # list of sample indices for each cluster
        self.clusters = List[List[Int]](capacity = 0)
        # the centers (mean feature vector) for each cluster
        self.centroids = Matrix(0, 0)
        self.X = Matrix(0, 0)

    fn predict(inout self, X: Matrix) raises -> Matrix:
        self.X = X

        if self.init == 'kmeans++' or self.init == 'k-means++':
            # Initialize centroids using KMeans++
            self._kmeans_plus_plus()
        else:
            self.centroids = X[Matrix.rand_choice(X.height, self.K, replace=False, seed = self.seed)]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters()

            # Calculate new centroids from the clusters
            var centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)
        
    fn _kmeans_plus_plus(inout self) raises:
        # Randomly select the first centroid
        random.seed(self.seed)
        self.centroids = Matrix(self.K, self.X.width)
        self.centroids[0] = self.X[int(random.random_ui64(0, self.X.height - 1))]

        for i in range(1, self.K):
            var distances = Matrix(self.X.height, 1)
            # Compute distances to the nearest centroid
            for idx in range(self.X.height):
                var min_distance = math.inf[DType.float32]()
                for idc in range(i): # Only consider the centroids that have been initialized
                    var distance = euclidean_distance(self.X[idx], self.centroids[idc])
                    if distance < min_distance:
                        min_distance = distance
                distances.data[idx] = min_distance
            # Select the next centroid with probability proportional to the squared distances
            var probabilities = (distances / distances.sum()).cumsum()
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
        return labels

    @always_inline
    fn _create_clusters(self) raises -> List[List[Int]]:
        # Assign the samples to the closest centroids to create clusters
        var clusters = List[List[Int]](capacity = self.K)
        for _ in range(self.K):
            clusters.append(List[Int]())

        for idx in range(self.X.height):
            clusters[self._closest_centroid(self.X[idx], self.centroids)].append(idx)
        return clusters

    @always_inline
    fn _closest_centroid(self, sample: Matrix, centroids: Matrix) raises -> Int:
        # distance of the current sample to each centroid
        var distances = Matrix(centroids.height, 1)
        for i in range(centroids.height):
            distances.data[i] = euclidean_distance(sample, centroids[i])
        return distances.argmin()

    @always_inline
    fn _get_centroids(self, clusters: List[List[Int]]) raises -> Matrix:
        # assign mean value of clusters to centroids
        var centroids = Matrix.zeros(self.K, self.X.width)
        for cluster_idx in range(len(clusters)):
            centroids[cluster_idx] = (self.X[clusters[cluster_idx]]).mean(0)
        return centroids

    @always_inline
    fn _is_converged(self, centroids_old: Matrix, centroids: Matrix) raises -> Bool:
        # distances between each old and new centroids, fol all centroids
        var distances = Matrix(self.K, 1)
        for i in range(self.K):
            distances.data[i] = euclidean_distance(centroids_old[i], centroids[i])
        return distances.sum() <= self.tol

    fn get_clusters_data(self) -> Tuple[Matrix, Matrix]:
        var row_counts = Matrix(1, len(self.clusters))
        for i in range(row_counts.size):
            row_counts.data[i] = len(self.clusters[i])
        var clusters_raw = Matrix(1, int(row_counts.sum()))
        var pointer = 0
        for l in self.clusters:
            for i in l[]:
                clusters_raw.data[pointer] = i[]
                pointer += 1

        return clusters_raw, row_counts
