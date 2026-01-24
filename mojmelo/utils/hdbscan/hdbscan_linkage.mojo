from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import squared_euclidean_distance
from .hdbscan_tree import arange
import math

fn mst_linkage_core(distance_matrix: Matrix) raises -> Matrix:
    var result = Matrix.zeros(distance_matrix.height - 1, 3)
    var node_labels = arange(0, distance_matrix.height)
    var current_node = 0
    var current_distances = Matrix(1, distance_matrix.height)
    current_distances.fill(math.inf[DType.float32]())
    var current_labels = node_labels.copy()
    for i in range(1, len(node_labels)):
        var label_filter = List[Int]()
        var current_labels_new = List[Scalar[DType.int]]()
        for idx, label in enumerate(current_labels):
            if label != current_node:
                label_filter.append(idx)
                current_labels_new.append(label)
        current_labels = current_labels_new^
        var left = current_distances['', label_filter]
        var right = distance_matrix[current_node][current_labels]
        current_distances = left.where(left.ele_lt(right), left, right)

        var new_node_index = current_distances.argmin()
        var new_node = current_labels[new_node_index]
        result[i - 1, 0] = current_node
        result[i - 1, 1] = new_node.cast[DType.float32]()
        result[i - 1, 2] = current_distances[0, new_node_index]
        current_node = Int(new_node)

    return result^


fn mst_linkage_core_vector(
        raw_data: Matrix,
        core_distances: List[Float32],
        alpha: Float32=1.0) raises -> Matrix:
    var dim = raw_data.height

    var result = Matrix.zeros(dim - 1, 3)
    var in_tree = List[Int8](capacity=dim)
    in_tree.resize(dim, 0)
    var current_node = 0
    var current_distances = List[Float32](capacity=dim)
    current_distances.resize(dim, math.inf[DType.float32]())
    var current_sources = List[Float32](capacity=dim)
    current_sources.resize(dim, 1)

    for i in range(1, dim):
        in_tree[current_node] = 1
        var current_node_core_distance = core_distances[current_node]
        var new_distance = Float32.MAX_FINITE
        var source_node = 0
        var new_node = 0

        for j in range(dim):
            if in_tree[j]:
                continue

            var right_value = current_distances[j]
            var right_source = Int(current_sources[j])

            var left_value = squared_euclidean_distance(raw_data[current_node], raw_data[j])
            var left_source = current_node

            if alpha != 1.0:
                left_value /= alpha

            var core_value = core_distances[j]
            if (current_node_core_distance > right_value or
                    core_value > right_value or
                    left_value > right_value):
                if right_value < new_distance:
                    new_distance = right_value
                    source_node = right_source
                    new_node = j
                continue

            if core_value > current_node_core_distance:
                if core_value > left_value:
                    left_value = core_value
            else:
                if current_node_core_distance > left_value:
                    left_value = current_node_core_distance

            if left_value < right_value:
                current_distances[j] = left_value
                current_sources[j] = left_source
                if left_value < new_distance:
                    new_distance = left_value
                    source_node = left_source
                    new_node = j
            else:
                if right_value < new_distance:
                    new_distance = right_value
                    source_node = right_source
                    new_node = j

        result[i - 1, 0] = source_node
        result[i - 1, 1] = new_node
        result[i - 1, 2] = new_distance
        current_node = new_node

    return result^


struct UnionFind:

    var parent: List[Scalar[DType.int]]
    var size: List[Scalar[DType.int]]
    var next_label: Int

    fn __init__(out self, N: Int):
        self.parent = List[Scalar[DType.int]](capacity=2 * N - 1)
        self.parent.resize(2 * N - 1, -1)
        self.next_label = N
        self.size = List[Scalar[DType.int]](capacity=2*N-1)
        self.size.resize(N, 1)
        self.size.resize(2*N-1, 0)

    fn union(mut self, m: Scalar[DType.int], n: Scalar[DType.int]):
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label

        self.parent[self.next_label] = -1   # <-- MUST set root

        self.size[self.next_label] = self.size[m] + self.size[n]
        self.next_label += 1

    fn fast_find(mut self, var n: Scalar[DType.int]) -> Scalar[DType.int]:
        var root = n

        # find root
        while self.parent[root] != -1:
            root = self.parent[root]

        # path compression
        while n != root:
            var p = self.parent[n]
            self.parent[n] = root
            n = p

        return root


fn label(L: Matrix) raises -> Matrix:
    var N = L.height + 1
    var U = UnionFind(N)

    var result = Matrix.zeros(L.height, 4)
    var out = 0

    for index in range(L.height):
        var a = L[index, 0].cast[DType.int]()
        var b = L[index, 1].cast[DType.int]()
        var delta = L[index, 2]

        var aa = U.fast_find(a)
        var bb = U.fast_find(b)

        if aa == bb:
            continue

        result[out, 0] = aa.cast[DType.float32]()
        result[out, 1] = bb.cast[DType.float32]()
        result[out, 2] = delta
        result[out, 3] = (U.size[aa] + U.size[bb]).cast[DType.float32]()

        U.union(aa, bb)
        out += 1

    return result.load_rows(out)


fn single_linkage(distance_matrix: Matrix) raises -> Matrix:
    var hierarchy = mst_linkage_core(distance_matrix)
    var for_labelling = hierarchy[hierarchy['', 2].argsort()]

    return label(for_labelling)
