from math import sqrt

@always_inline
fn euclidean_distance(p1: Point, p2: Point) -> Float32:
    return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

@value
struct Point:
    var x: Float32
    var y: Float32
    var cluster: Int
    
    fn __init__(inout self, x: Float32, y: Float32):
        self.x = x
        self.y = y
        
        # -1 == unclassified
        self.cluster = -1

@value
struct DBSCAN:
    var eps: Float32
    var minPts: Int
    var data: List[Point]

    fn get_neighbors(self, idx: Int) -> List[Int]:
        var neighbors: List[Int] = List[Int]()
        for i in range(len(self.data)):
            if i != idx and euclidean_distance(self.data[idx], self.data[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    fn expand_cluster(inout self, idx: Int, inout neighbors: List[Int], cluster_id: Int):
        self.data[idx].cluster = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if self.data[neighbor_idx].cluster == -1:
                self.data[neighbor_idx].cluster = cluster_id
                var new_neighbors: List[Int] = self.get_neighbors(neighbor_idx)
                if len(new_neighbors) >= self.minPts:
                    for neighbor in new_neighbors:
                        neighbors.append(neighbor[])
            elif self.data[neighbor_idx].cluster == 0:
                self.data[neighbor_idx].cluster = cluster_id
            i += 1

    fn fit(inout self):
        cluster_id = 0
        for i in range(len(self.data)):
            if self.data[i].cluster != -1:
                continue
            neighbors = self.get_neighbors(i)
            if len(neighbors) < self.minPts:
                self.data[i].cluster = 0
            else:
                cluster_id += 1
                self.expand_cluster(i, neighbors, cluster_id)
        print("Clustering complete!")