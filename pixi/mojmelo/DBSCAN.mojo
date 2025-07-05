from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import euclidean_distance, manhattan_distance

struct DBSCAN:
    var eps: Float32
    var min_samples: Int
    var distance: fn(Matrix, Matrix, Int) raises -> Matrix
    var clusters: List[List[Int]]
    var neighbors: Dict[Int, List[Int]]
    var X: Matrix

    fn __init__(out self, eps: Float32 = 1.0, min_samples: Int = 5, metric: String = 'euc'):
        self.eps = eps
        self.min_samples = min_samples
        if metric.lower() == 'man':
            self.distance = manhattan_distance
        else:
            self.distance = euclidean_distance
        self.clusters = List[List[Int]]()
        self.neighbors = Dict[Int, List[Int]]()
        self.X = Matrix(0, 0)

    @always_inline
    fn _get_neighbors(self, idx: Int) raises -> List[Int]:
        var neighbors = List[Int]()
        var distances = self.distance(self.X, self.X[idx], 1)
        for i in range(distances.size):
            if i != idx and distances.data[i] <= self.eps:
                neighbors.append(i)
        return neighbors^

    fn _expand_cluster(mut self, idx: Int, mut neighbors: List[Int], mut visited_samples: List[Int]) raises -> List[Int]:
        var cluster = List[Int](idx)
        # Iterate through neighbors
        for neighbor_i in neighbors:
            if not neighbor_i in visited_samples:
                visited_samples.append(neighbor_i)
                # Fetch the sample's distant neighbors (neighbors of neighbor)
                self.neighbors[neighbor_i] = self._get_neighbors(neighbor_i)
                # Make sure the neighbor's neighbors are more than min_samples
                # (If this is true the neighbor is a core point)
                if len(self.neighbors[neighbor_i]) >= self.min_samples:
                    # Expand the cluster from the neighbor and Add expanded cluster to this cluster
                    cluster = cluster + self._expand_cluster(neighbor_i, self.neighbors[neighbor_i], visited_samples)
                else:
                    # If the neighbor is not a core point we only add the neighbor point
                    cluster.append(neighbor_i)
        return cluster^

    fn _get_cluster_labels(self) -> Matrix:
        var labels = Matrix.full(self.X.height, 1, -1.0)
        for cluster_i in range(len(self.clusters)):
            for sample_i in self.clusters[cluster_i]:
                labels.data[sample_i] = cluster_i
        return labels^

    fn predict(mut self, X: Matrix) raises -> Matrix:
        self.X = X
        var visited_samples = List[Int]()
        # Iterate through samples and expand clusters from them
        # if they have more neighbors than self.min_samples
        for idx in range(self.X.height):
            if idx in visited_samples:
                continue
            self.neighbors[idx] = self._get_neighbors(idx)
            if len(self.neighbors[idx]) >= self.min_samples:
                # If core point => mark as visited
                visited_samples.append(idx)
                # Sample has more neighbors than self.min_samples => expand cluster from sample and Add cluster to list of clusters
                self.clusters.append(self._expand_cluster(idx, self.neighbors[idx], visited_samples))

        return self._get_cluster_labels()
