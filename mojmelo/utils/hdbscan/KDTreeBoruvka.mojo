from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import fill_indices_list
from mojmelo.utils.KDTree import KDTree, KDTreeResultVector
import std.math as math
from std.algorithm import vectorize, parallelize
from std.sys import size_of
from std.memory import memset_zero

@always_inline
def key(idx: Scalar[DType.int],
        data: UnsafePointer[Float32, MutAnyOrigin],
        dim: Scalar[DType.int],
        split_dim: Scalar[DType.int]) -> Float32:
    return data[idx * dim + split_dim]

@always_inline
def nth_element(
    var first: UnsafePointer[Scalar[DType.int], MutAnyOrigin],
    nth: UnsafePointer[Scalar[DType.int], MutAnyOrigin],
    var last: UnsafePointer[Scalar[DType.int], MutAnyOrigin],
    var proj: UnsafePointer[Float32, MutAnyOrigin],
    data: UnsafePointer[Float32, MutAnyOrigin],
    dim: Scalar[DType.int],
    split_dim: Scalar[DType.int]):
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
def node_pair_lower_bound(
    var center1: UnsafePointer[Float32, MutAnyOrigin],
    var center2: UnsafePointer[Float32, MutAnyOrigin],
    r1: Float32,
    r2: Float32,
    dim: Int
) -> Float32:
    var dist2: Float32 = 0.0

    def v[simd_width: Int](k: Int) {mut}:
        var t = center1.load[width=simd_width](k) - center2.load[width=simd_width](k)
        dist2 += (t * t).reduce_add()

    vectorize[Matrix.simd_width](dim, v)

    var R = r1 + r2
    var lb2 = dist2 - (R * R)

    return lb2 if lb2 > 0.0 else 0.0


# Thin wrapper so nd[].center._data compiles in HDBSCANBoruvka unchanged.
@fieldwise_init
struct CenterPtr(TrivialRegisterPassable):
    var _data: UnsafePointer[Float32, MutAnyOrigin]


@fieldwise_init
struct NodeData(Copyable):
    var is_leaf: Bool
    var idx_start: Int
    var idx_end: Int
    var radius: Float32
    var center: CenterPtr   # points into flat _center_arena


struct KDTreeBoruvka:
    var data: UnsafePointer[Float32, MutAnyOrigin]
    var kdtree: KDTree[sort_results=True]
    var n: Int
    var dim: Int
    var leaf_size: Int
    var nodes: List[NodeData]
    var core_dist: UnsafePointer[Float32, MutAnyOrigin]
    var build_idx: List[Scalar[DType.int]]
    var proj_buf: List[Float32]
    # Single contiguous allocation for ALL node centers: max_nodes × dim floats.
    var _center_arena: UnsafePointer[Float32, MutAnyOrigin]

    @always_inline
    def __init__(out self, data: Matrix, min_samples: Int, leaf_size: Int, search_depth: Int) raises:
        self.data = data.data
        self.kdtree = KDTree[sort_results=True, metric='euc'](data)
        self.n = data.height
        self.dim = data.width
        self.leaf_size = leaf_size
        self.nodes = List[NodeData]()

        # One allocation for all node centers; upper bound on node count is 2n+1.
        var max_nodes = 2 * self.n + 1
        self._center_arena = alloc[Float32](max_nodes * self.dim)
        memset_zero(self._center_arena, max_nodes * self.dim)

        self.core_dist = alloc[Float32](self.n)
        self.build_idx = fill_indices_list(self.n)
        self.proj_buf = List[Float32](capacity=self.n)
        self.proj_buf.resize(self.n, 0.0)

        var k = search_depth * min_samples + 1

        @parameter
        def compute_core_dist(p: Int):
            try:
                var kd_results = KDTreeResultVector()
                self.kdtree.n_nearest(
                    Span(ptr=self.data + p * self.dim, length=self.dim),
                    k,
                    kd_results
                )
                self.core_dist[p] = kd_results[min_samples].dis
            except e:
                print('Error:', e)

        parallelize[compute_core_dist](self.n)

        self.build_node(0, 0, self.n)

    @always_inline
    def __del__(deinit self):
        self.core_dist.free()
        self._center_arena.free()

    @always_inline
    def left(self, i: Int) -> Int:
        return 2 * i + 1

    @always_inline
    def right(self, i: Int) -> Int:
        return 2 * i + 2

    def ensure_node(mut self, i: Int):
        if len(self.nodes) <= i:
            # Placeholder center; overwritten immediately in build_node
            self.nodes.resize(i + 1, NodeData(False, 0, 0, 0.0, CenterPtr(self._center_arena)))

    # Fused single O(n) pass: finds min/max across ALL dims simultaneously.
    def choose_split_dim(self, start: Int, end: Int) -> Scalar[DType.int]:
        var mn = List[Float32](capacity=self.dim)
        var mx = List[Float32](capacity=self.dim)
        mn.resize(self.dim,  math.inf[DType.float32]())
        mx.resize(self.dim, -math.inf[DType.float32]())

        for i in range(start, end):
            var p = self.data + Int(self.build_idx[i]) * self.dim
            for d in range(self.dim):
                var v = p[d]
                if v < mn[d]: mn[d] = v
                if v > mx[d]: mx[d] = v

        var best: Scalar[DType.int] = 0
        var best_spread: Float32 = -1.0
        for d in range(self.dim):
            var s = mx[d] - mn[d]
            if s > best_spread:
                best_spread = s
                best = Scalar[DType.int](d)
        return best

    def build_node(mut self, node: Int, start: Int, end: Int):
        self.ensure_node(node)
        var nd = self.nodes._data + node
        nd[].idx_start = start
        nd[].idx_end = end

        var count = Float32(end - start)

        # Point this node's center at its pre-allocated slot in the arena
        var cptr = self._center_arena + node * self.dim
        nd[].center = CenterPtr(cptr)

        for i in range(start, end):
            var p = self.data + Int(self.build_idx[i]) * self.dim

            def v1[simd_width: Int](k: Int) {read}:
                cptr.store(k, cptr.load[width=simd_width](k) + p.load[width=simd_width](k))
            vectorize[Matrix.simd_width](self.dim, v1)

        for d in range(self.dim):
            cptr[d] /= count

        var maxd: Float32 = 0.0
        for i in range(start, end):
            var p = self.data + Int(self.build_idx[i]) * self.dim
            var d2: Float32 = 0.0

            def v2[simd_width: Int](k: Int) {mut}:
                var t = p.load[width=simd_width](k) - cptr.load[width=simd_width](k)
                d2 += (t * t).reduce_add()
            vectorize[Matrix.simd_width](self.dim, v2)
            if d2 > maxd:
                maxd = d2

        nd[].radius = math.sqrt(maxd)

        if Int(count) <= self.leaf_size:
            nd[].is_leaf = True
            return

        nd[].is_leaf = False

        var split_dim = self.choose_split_dim(start, end)
        var mid = (start + end) // 2

        nth_element(
            self.build_idx._data + start,
            self.build_idx._data + mid,
            self.build_idx._data + end,
            self.proj_buf._data + start,
            self.data,
            Scalar[DType.int](self.dim),
            split_dim
        )

        self.build_node(self.left(node), start, mid)
        self.build_node(self.right(node), mid, end)
