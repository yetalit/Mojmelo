from math import sqrt
from mojmelo.utils.Matrix import Matrix

@value
struct DBSCAN:
    var eps: Float32
    var minPts: Int
    var data: Matrix  # Assume each row is a 2D point (x, y)
    var clusters: List[Int]  # Stores cluster assignments for each point

    fn __init__(inout self, eps: Float32, minPts: Int, data: Matrix):
        self.eps = eps
        self.minPts = minPts
        self.data = data
        self.clusters = List[Int]()
        for _ in range(data.height):
            self.clusters.append(-1)

    @always_inline
    fn euclidean_distance(self, idx1: Int, idx2: Int) raises -> Float32:
        var dx = self.data[idx1, 0] - self.data[idx2, 0]
        var dy = self.data[idx1, 1] - self.data[idx2, 1]            
        return sqrt(dx * dx + dy * dy)

    fn get_neighbors(self, idx: Int) raises -> List[Int]:
        var neighbors = List[Int]()
        for i in range(self.data.height):
            if i != idx and self.euclidean_distance(idx, i) <= self.eps:
                neighbors.append(i)
        return neighbors

    fn expand_cluster(inout self, idx: Int, inout neighbors: List[Int], cluster_id: Int) raises:
        self.clusters[idx] = cluster_id
        var i = 0
        while i < len(neighbors):
            var neighbor_idx = neighbors[i]
            if self.clusters[neighbor_idx] == -1:
                self.clusters[neighbor_idx] = cluster_id
                var new_neighbors = self.get_neighbors(neighbor_idx)
                if len(new_neighbors) >= self.minPts:
                    for neighbor in new_neighbors:
                        neighbors.append(neighbor[])
            elif self.clusters[neighbor_idx] == 0: 
                self.clusters[neighbor_idx] = cluster_id
            i += 1

    fn fit(inout self) raises:
        var cluster_id = 0
        for i in range(self.data.height):
            if self.clusters[i] != -1:
                continue
            var neighbors = self.get_neighbors(i)
            if len(neighbors) < self.minPts:
                self.clusters[i] = 0 
            else:
                cluster_id += 1
                self.expand_cluster(i, neighbors, cluster_id)

