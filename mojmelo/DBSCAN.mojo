from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import euclidean_distance
from collections import Dict

struct DBSCAN:
    var eps: Float32
    var min_samples: Int
    var clusters: List[List[Int]]
    var neighbors: Dict[Int, List[Int]]
    var X: Matrix

    fn __init__(inout self, eps: Float32 = 1.0, min_samples: Int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.clusters = List[List[Int]]()
        self.neighbors = Dict[Int, List[Int]]()
        self.X = Matrix(0, 0)

    @always_inline
    fn _get_neighbors(self, idx: Int) raises -> List[Int]:
        var neighbors = List[Int]()
        var distances = euclidean_distance(self.X, self.X[idx], axis=1)
        for i in range(distances.size):
            if i != idx and distances.data[i] <= self.eps:
                neighbors.append(i)
        return neighbors^

    @always_inline
    fn _expand_cluster(inout self, idx: Int, neighbors: List[Int], inout visited_samples: List[Int]) raises -> List[Int]:
        var cluster = List[Int](idx)
        for neighbor_i in neighbors:
            if not neighbor_i[] in visited_samples:
                visited_samples.append(neighbor_i[])
                self.neighbors[neighbor_i[]] = self._get_neighbors(neighbor_i[])
                if len(self.neighbors[neighbor_i[]]) >= self.min_samples:
                    cluster = cluster + self._expand_cluster(neighbor_i[], neighbors[neighbor_i[]], visited_samples)
                else:
                    cluster.append(neighbor_i[])
        return cluster^

    fn _get_cluster_labels(self) -> Matrix:
        var labels = Matrix.full(self.X.height, 1, len(self.clusters))
        for cluster_i in range(len(self.clusters)):
            for sample_i in self.clusters[cluster_i]:
                labels.data[sample_i[]] = cluster_i
        return labels^

    fn predict(inout self, X: Matrix) raises -> Matrix:
        self.X = X
        var visited_samples = List[Int]()
        for idx in range(self.X.height):
            if idx in visited_samples:
                continue
            self.neighbors[idx] = self._get_neighbors(idx)
            if len(self.neighbors[idx]) >= self.min_samples:
                visited_samples.append(idx)
                self.clusters.append(self._expand_cluster(idx, self.neighbors[idx], visited_samples))

        return self._get_cluster_labels()
