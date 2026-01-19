from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import fill_indices_list
from mojmelo.utils.KDTree import KDTree
import math
from algorithm import vectorize
from sys import size_of

@always_inline
fn key(idx: Scalar[DType.int],
                data: UnsafePointer[Float32, MutAnyOrigin],
                dim: Int,
                split_dim: Int) -> Float32:
    return data[idx * dim + split_dim]

@always_inline
fn nth_element(
    var first: UnsafePointer[Scalar[DType.int], MutAnyOrigin],
    nth: UnsafePointer[Scalar[DType.int], MutAnyOrigin],
    var last: UnsafePointer[Scalar[DType.int], MutAnyOrigin],
    var proj: UnsafePointer[Float32, MutAnyOrigin],
    data: UnsafePointer[Float32, MutAnyOrigin],
    dim: Int,
    split_dim: Int):
    for i in range((Int(last) - Int(first))//size_of[DType.int]()):
        proj[i] = key(first[i], data, dim, split_dim)

    while (Int(last) - Int(first))//size_of[DType.int]() > 1:
        var _len = (Int(last) - Int(first))//size_of[DType.int]()
        var mid = _len >> 1

        var a = 0
        var b = mid
        var c = _len - 1

        var pivot_i = (b if proj[b] < proj[c] else (c if proj[a] < proj[c] else a)) if proj[a] < proj[b] else (a if proj[a] < proj[c] else (c if proj[b] < proj[c] else b))

        swap(first[pivot_i], first[_len - 1])
        swap(proj[pivot_i],  proj[_len - 1])

        var pivot_val = proj[_len - 1]
        var pivot_idx = first[_len - 1]

        var store = 0
        for i in range(_len - 1):
            if proj[i] < pivot_val or
               (proj[i] == pivot_val and first[i] < pivot_idx):
                swap(first[i], first[store])
                swap(proj[i],  proj[store])
                store += 1

        swap(first[store], first[_len - 1])
        swap(proj[store],  proj[_len - 1])

        if first + store == nth:
            return
        elif first + store < nth:
            first += store + 1
            proj += store + 1
        else:
            last = first + store

@fieldwise_init
@register_passable("trivial")
struct NodeData:
    var is_leaf: Bool
    var idx_start: Int
    var idx_end: Int
    var radius: Float32

struct KDTreeBoruvka:
    # input
    var data: UnsafePointer[Float32, MutAnyOrigin]
    var kdtree: KDTree
    var n: Int
    var dim: Int
    var leaf_size: Int
    var nodes: List[NodeData]
    var idx_array: List[Scalar[DType.int]]
    var node_bounds: List[Float32]
    var proj_buf: List[Float32]

    @always_inline
    fn __init__(out self, data: Matrix, leaf_size: Int=32) raises:
        self.data = data.data
        self.kdtree = KDTree(data, metric='euc')
        self.n = data.height
        self.dim = data.width
        self.leaf_size = leaf_size
        self.nodes = List[NodeData]()
        self.idx_array = fill_indices_list(self.n)
        self.node_bounds = List[Float32]()
        self.proj_buf = List[Float32]()
        self.build_node(0, 0, self.n)

    @always_inline
    fn left(self, i: Int) -> Int:
        return 2 * i + 1
    
    @always_inline
    fn right(self, i: Int) -> Int:
        return 2 * i + 2

    fn ensure_node(mut self, i: Int):
        if len(self.nodes) <= i:
            self.nodes.resize(i + 1, NodeData(0, 0, 0, 0))
            self.node_bounds.resize((i + 1) * self.dim * 2, 0)

    fn compute_bounds(mut self, node: Int, start: Int, end: Int):
        for d in range(self.dim):
            var mn = math.inf[DType.float32]()
            var mx = -mn

            for i in range(start, end):
                var v = self.data[self.idx_array[i]*self.dim + d]
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v

            self.node_bounds[(node*self.dim + d)*2 + 0] = mn
            self.node_bounds[(node*self.dim + d)*2 + 1] = mx

    fn compute_radius(self, start: Int, end: Int, centroid: UnsafePointer[Float32, MutAnyOrigin]) -> Float32:
        var r2: Float32 = 0.0

        for i in range(start, end):
            var dist2: Float32 = 0.0
            var p = self.data + self.idx_array[i]*self.dim

            @parameter
            fn v[simd_width: Int](idx: Int):
                var t = p.load[width=Matrix.simd_width](idx) - centroid.load[width=Matrix.simd_width](idx)
                dist2 += (t * t).reduce_add()
            vectorize[v, Matrix.simd_width](self.dim)
            r2 = max(r2, math.sqrt(dist2))

        return r2

    fn choose_split_dim(self, start: Int, end: Int) -> Int:
        var best = 0
        var best_spread: Float32 = 0.0

        for d in range(self.dim):
            var mn = math.inf[DType.float32]()
            var mx = -mn

            for i in range(start, end):
                var v = self.data[self.idx_array[i]*self.dim + d]
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v

            if mx - mn > best_spread:
                best_spread = mx - mn
                best = d

        return best

    fn build_node(mut self, node: Int, start: Int, end: Int):
        self.ensure_node(node)
        var nd = self.nodes._data + node
        nd[].idx_start = start
        nd[].idx_end = end

        self.compute_bounds(node, start, end)

        var count = end - start
        if len(self.proj_buf) < count:
            self.proj_buf.resize(count, 0)
        if count <= self.leaf_size:
            nd[].is_leaf = True

            var centroid = List[Float32](capacity=self.dim)
            centroid.resize(self.dim, 0.0)

            for i in range(start, end):
                var p = self.data + self.idx_array[i]*self.dim
                @parameter
                fn v[simd_width: Int](idx: Int):
                    centroid._data.store(idx, centroid._data.load[width=simd_width](idx) + p.load[width=simd_width](idx))
                vectorize[v, Matrix.simd_width](self.dim)

            for d in range(self.dim):
                centroid[d] /= count

            nd[].radius = self.compute_radius(start, end, centroid._data)
            return

        var split_dim = self.choose_split_dim(start, end)
        var mid = (start + end) // 2

        nth_element(
            self.idx_array._data + start,
            self.idx_array._data + mid,
            self.idx_array._data + end,
            self.proj_buf._data,
            self.data,
            self.dim,
            split_dim
        )

        nd[].is_leaf = False

        self.build_node(self.left(node), start, mid)
        self.build_node(self.right(node), mid, end)

        var centroid = List[Float32](capacity=self.dim)
        centroid.resize(self.dim, 0.0)
        for d in range(self.dim):
            centroid[d] =
                0.5 * (self.node_bounds[(node*self.dim + d)*2 + 0] +
                       self.node_bounds[(node*self.dim + d)*2 + 1])

        nd[].radius = self.compute_radius(start, end, centroid._data)

    @always_inline
    fn kdtree_min_rdist_dual(self,
        node1: Int,
        node2: Int,
        node_bounds: UnsafePointer[Float32, MutAnyOrigin],
        dim: Int
    ) -> Float32:
        var dist: Float32 = 0.0

        var b1 = node_bounds + node1 * dim * 2
        var b2 = node_bounds + node2 * dim * 2

        for d in range(dim):
            var a_min = b1[d*2 + 0]
            var a_max = b1[d*2 + 1]
            var b_min = b2[d*2 + 0]
            var b_max = b2[d*2 + 1]

            if a_max < b_min:
                var t = b_min - a_max
                dist += t * t
            elif b_max < a_min:
                var t = a_min - b_max
                dist += t * t

        return dist
