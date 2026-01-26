from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import fill_indices_list
from mojmelo.utils.KDTree import KDTree, KDTreeResultVector
import math
from algorithm import vectorize, parallelize
from sys import size_of
from buffer import NDBuffer

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

        var pivot_i = 0
        if proj[a] < proj[b]:
            pivot_i = b if proj[b] < proj[c] else (c if proj[a] < proj[c] else a)
        else:
            pivot_i = a if proj[a] < proj[c] else (c if proj[b] < proj[c] else b)
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

@always_inline
fn node_pair_lower_bound(
    center1: UnsafePointer[Float32, MutAnyOrigin],
    center2: UnsafePointer[Float32, MutAnyOrigin],
    r1: Float32,
    r2: Float32,
    dim: Int
) -> Float32:
    var dist2: Float32 = 0.0

    @parameter
    fn v[simd_width: Int](k: Int):
        var t = center1.load[width=simd_width](k) - center2.load[width=simd_width](k)
        dist2 += (t * t).reduce_add()

    vectorize[v, Matrix.simd_width](dim)

    var min_possible = dist2 - (r1 + r2) * (r1 + r2)
    if min_possible < 0.0:
        return 0.0

    return min_possible

@fieldwise_init
struct NodeData(Copyable, Movable):
    var is_leaf: Bool
    var idx_start: Int
    var idx_end: Int
    var radius: Float32
    var center: List[Float32]

struct KDTreeBoruvka:
    var data: UnsafePointer[Float32, MutAnyOrigin]
    var kdtree: KDTree[sort_results=True]
    var n: Int
    var dim: Int
    var leaf_size: Int
    var nodes: List[NodeData]
    var core_dist: UnsafePointer[Float32, MutAnyOrigin]
    var build_idx: List[Scalar[DType.int]]       # permuted indices for building
    var proj_buf: List[Float32]

    @always_inline
    fn __init__(out self, data: Matrix, min_samples: Int, leaf_size: Int) raises:
        self.data = data.data
        self.kdtree = KDTree[sort_results=True](data, metric='euc')
        self.n = data.height
        self.dim = data.width
        self.leaf_size = leaf_size
        self.nodes = List[NodeData]()
        self.core_dist = alloc[Float32](self.n)

        # build index array (will be permuted)
        self.build_idx = fill_indices_list(self.n)

        self.proj_buf = List[Float32](capacity=self.n)
        self.proj_buf.resize(self.n, 0.0)

        @parameter
        fn compute_core_dist(p: Int):
            # core_dist must use stable indices
            var kd_results = KDTreeResultVector()
            self.kdtree.n_nearest(
                NDBuffer[dtype=DType.float32, rank=1](self.data + p * self.dim, self.dim),
                min_samples + 1,
                kd_results
            )

            self.core_dist[p] = math.sqrt(kd_results[min_samples].dis)

        parallelize[compute_core_dist](self.n)

        self.build_node(0, 0, self.n)

    @always_inline
    fn __del__(deinit self):
        if self.core_dist:
            self.core_dist.free()

    @always_inline
    fn left(self, i: Int) -> Int:
        return 2 * i + 1
    
    @always_inline
    fn right(self, i: Int) -> Int:
        return 2 * i + 2

    fn ensure_node(mut self, i: Int):
        if len(self.nodes) <= i:
            self.nodes.resize(i + 1, NodeData(0, 0, 0, 0, List[Float32]()))

    fn choose_split_dim(self, start: Int, end: Int, idx: List[Scalar[DType.int]]) -> Int:
        var best = 0
        var best_spread: Float32 = 0.0

        for d in range(self.dim):
            var mn = math.inf[DType.float32]()
            var mx = -mn

            for i in range(start, end):
                var v = self.data[idx[i]*self.dim + d]
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

        var count = end - start

        nd[].center = List[Float32](capacity=self.dim)
        nd[].center.resize(self.dim, 0.0)

        for i in range(start, end):
            var p = self.data + self.build_idx[i] * self.dim
            @parameter
            fn v[simd_width: Int](k: Int):
                nd[].center._data.store(k, nd[].center._data.load[width=simd_width](k) + p.load[width=simd_width](k))
            vectorize[v, Matrix.simd_width](self.dim)

        for d in range(self.dim):
            nd[].center[d] /= count

        var maxd: Float32 = 0.0
        for i in range(start, end):
            var p = self.data + self.build_idx[i] * self.dim
            var d2: Float32 = 0.0

            @parameter
            fn v2[simd_width: Int](k: Int):
                var t = p.load[width=simd_width](k) - nd[].center._data.load[width=simd_width](k)
                d2 += (t * t).reduce_add()
            vectorize[v2, Matrix.simd_width](self.dim)

            if d2 > maxd:
                maxd = d2

        nd[].radius = math.sqrt(maxd)

        if count <= self.leaf_size:
            nd[].is_leaf = True
            return

        nd[].is_leaf = False

        var split_dim = self.choose_split_dim(start, end, self.build_idx)
        var mid = (start + end) // 2

        nth_element(
            self.build_idx._data + start,
            self.build_idx._data + mid,
            self.build_idx._data + end,
            self.proj_buf._data + start,
            self.data,
            self.dim,
            split_dim
        )

        self.build_node(self.left(node), start, mid)
        self.build_node(self.right(node), mid, end)
