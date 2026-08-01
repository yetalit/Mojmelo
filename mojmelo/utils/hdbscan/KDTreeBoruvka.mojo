from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import fill_indices_list
from mojmelo.utils.KDTree import KDTree, KDTreeResultVector
import std.math as math
from std.algorithm import vectorize
from mojmelo.utils.algorithm import parallelize
from std.sys import size_of
from std.memory import unsafe_memset_zero

@always_inline
def key(idx: Int,
        data: UnsafePointer[Float32, MutUntrackedOrigin],
        dim: Int,
        split_dim: Int) -> Float32:
    return data[unsafe_offset=idx * dim + split_dim]

@always_inline
def nth_element(
    var first: UnsafePointer[Int, MutUntrackedOrigin],
    nth: UnsafePointer[Int, MutUntrackedOrigin],
    var last: UnsafePointer[Int, MutUntrackedOrigin],
    var proj: UnsafePointer[Float32, MutUntrackedOrigin],
    data: UnsafePointer[Float32, MutUntrackedOrigin],
    dim: Int,
    split_dim: Int):
    for i in range((Int(last) - Int(first))//size_of[DType.int]()):
        proj[unsafe_offset=i] = key(first[unsafe_offset=i], data, dim, split_dim)

    while (Int(last) - Int(first))//size_of[DType.int]() > 1:
        var _len = (Int(last) - Int(first))//size_of[DType.int]()
        var mid = _len >> 1

        var a = 0
        var b = mid
        var c = _len - 1

        var pivot_i: Int
        if proj[unsafe_offset=a] < proj[unsafe_offset=b]:
            pivot_i = b if proj[unsafe_offset=b] < proj[unsafe_offset=c] else (c if proj[unsafe_offset=a] < proj[unsafe_offset=c] else a)
        else:
            pivot_i = a if proj[unsafe_offset=a] < proj[unsafe_offset=c] else (c if proj[unsafe_offset=b] < proj[unsafe_offset=c] else b)
        swap(first[unsafe_offset=pivot_i], first[unsafe_offset=_len - 1])
        swap(proj[unsafe_offset=pivot_i],  proj[unsafe_offset=_len - 1])

        var pivot_val = proj[unsafe_offset=_len - 1]
        var pivot_idx = first[unsafe_offset=_len - 1]

        var store = 0
        for i in range(_len - 1):
            if proj[unsafe_offset=i] < pivot_val or
               (proj[unsafe_offset=i] == pivot_val and first[unsafe_offset=i] < pivot_idx):
                swap(first[unsafe_offset=i], first[unsafe_offset=store])
                swap(proj[unsafe_offset=i],  proj[unsafe_offset=store])
                store += 1

        swap(first[unsafe_offset=store], first[unsafe_offset=_len - 1])
        swap(proj[unsafe_offset=store],  proj[unsafe_offset=_len - 1])

        if first.unsafe_offset(store) == nth:
            return
        elif first.unsafe_offset(store) < nth:
            first = first.unsafe_offset(store + 1)
            proj = proj.unsafe_offset(store + 1)
        else:
            last = first.unsafe_offset(store)

@always_inline
def node_pair_lower_bound(
    var center1: UnsafePointer[Float32, MutUntrackedOrigin],
    var center2: UnsafePointer[Float32, MutUntrackedOrigin],
    r1: Float32,
    r2: Float32,
    dim: Int
) -> Float32:
    var dist2: Float32 = 0.0

    def v[simd_width: Int](k: Int) {mut}:
        var t = center1.unsafe_load[width=simd_width](k) - center2.unsafe_load[width=simd_width](k)
        dist2 += (t * t).reduce_add()

    vectorize[Matrix.simd_width](dim, v)

    var R = r1 + r2
    var dist = math.sqrt(dist2) if dist2 > 0.0 else Float32(0.0)
    var lb = dist - R

    return (lb * lb) if lb > 0.0 else Float32(0.0)


# Thin wrapper so nd[].center._data compiles in HDBSCANBoruvka unchanged.
@fieldwise_init
struct CenterPtr(TrivialRegisterPassable):
    var _data: UnsafePointer[Float32, MutUntrackedOrigin]


@fieldwise_init
struct NodeData(Copyable):
    var is_leaf: Bool
    var idx_start: Int
    var idx_end: Int
    var radius: Float32
    var center: CenterPtr   # points into flat _center_arena


struct KDTreeBoruvka:
    var data: UnsafePointer[Float32, MutUntrackedOrigin]
    var kdtree: KDTree[sort_results=True]
    var n: Int
    var dim: Int
    var leaf_size: Int
    var nodes: List[NodeData]
    var core_dist: UnsafePointer[Float32, MutUntrackedOrigin]
    var build_idx: List[Int]
    var proj_buf: List[Float32]
    # Single contiguous allocation for ALL node centers: max_nodes × dim floats.
    var _center_arena: UnsafePointer[Float32, MutUntrackedOrigin]

    @always_inline
    def __init__(out self, data: Matrix, min_samples: Int, leaf_size: Int, search_depth: Int) raises:
        self.data = data.data
        self.kdtree = KDTree[sort_results=True](data, metric='euc')
        self.n = data.height
        self.dim = data.width
        self.leaf_size = leaf_size
        self.nodes = List[NodeData]()

        # One allocation for all node centers; upper bound on node count is 2n+1.
        var max_nodes = 2 * self.n + 1
        self._center_arena = alloc[Float32](max_nodes * self.dim)
        unsafe_memset_zero(self._center_arena, max_nodes * self.dim)

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
                    Span(unsafe_ptr=self.data.unsafe_offset(p * self.dim), length=self.dim),
                    k,
                    kd_results
                )
                self.core_dist[unsafe_offset=p] = kd_results[min_samples].dis
            except e:
                print('Error:', e)

        parallelize[compute_core_dist](self.n)

        self.build_node(0, 0, self.n)

    @always_inline
    def __deinit__(deinit self):
        self.core_dist.unsafe_free()
        self._center_arena.unsafe_free()

    @always_inline
    def left(self, i: Int) -> Int:
        return 2 * i + 1

    @always_inline
    def right(self, i: Int) -> Int:
        return 2 * i + 2

    def ensure_node(mut self, i: Int):
        if len(self.nodes) <= i:
            # Placeholder center; overwritten immediately in build_node
            self.nodes.resize(i + 1, NodeData(True, 0, 0, 0.0, CenterPtr(self._center_arena)))

    # Fused single O(n) pass: finds min/max across ALL dims simultaneously.
    def choose_split_dim(self, start: Int, end: Int) -> Int:
        var mn = List[Float32](capacity=self.dim)
        var mx = List[Float32](capacity=self.dim)
        mn.resize(self.dim,  math.inf[DType.float32]())
        mx.resize(self.dim, -math.inf[DType.float32]())

        for i in range(start, end):
            var p = self.data.unsafe_offset(self.build_idx[i] * self.dim)
            for d in range(self.dim):
                var v = p[unsafe_offset=d]
                if v < mn[d]: mn[d] = v
                if v > mx[d]: mx[d] = v

        var best = 0
        var best_spread: Float32 = -1.0
        for d in range(self.dim):
            var s = mx[d] - mn[d]
            if s > best_spread:
                best_spread = s
                best = d
        return best

    def build_node(mut self, node: Int, start: Int, end: Int):
        self.ensure_node(node)
        var nd = self.nodes._data.unsafe_offset(node)
        nd[].idx_start = start
        nd[].idx_end = end

        var count = Float32(end - start)

        # Point this node's center at its pre-allocated slot in the arena
        var cptr = self._center_arena.unsafe_offset(node * self.dim)
        nd[].center = CenterPtr(cptr)

        for i in range(start, end):
            var p = self.data.unsafe_offset(self.build_idx[i] * self.dim)

            def v1[simd_width: Int](k: Int) {imm}:
                cptr.unsafe_store(k, cptr.unsafe_load[width=simd_width](k) + p.unsafe_load[width=simd_width](k))
            vectorize[Matrix.simd_width](self.dim, v1)

        for d in range(self.dim):
            cptr[unsafe_offset=d] /= count

        var maxd: Float32 = 0.0
        for i in range(start, end):
            var p = self.data.unsafe_offset(self.build_idx[i] * self.dim)
            var d2: Float32 = 0.0

            def v2[simd_width: Int](k: Int) {mut}:
                var t = p.unsafe_load[width=simd_width](k) - cptr.unsafe_load[width=simd_width](k)
                d2 += (t * t).reduce_add()
            vectorize[Matrix.simd_width](self.dim, v2)
            if d2 > maxd:
                maxd = d2

        nd[].radius = math.sqrt(maxd)

        if end - start <= self.leaf_size:
            nd[].is_leaf = True
            return

        nd[].is_leaf = False

        var split_dim = self.choose_split_dim(start, end)
        var mid = (start + end) // 2

        nth_element(
            self.build_idx._data.unsafe_offset(start),
            self.build_idx._data.unsafe_offset(mid),
            self.build_idx._data.unsafe_offset(end),
            self.proj_buf._data.unsafe_offset(start),
            self.data,
            self.dim,
            split_dim
        )

        self.build_node(self.left(node), start, mid)
        self.build_node(self.right(node), mid, end)
