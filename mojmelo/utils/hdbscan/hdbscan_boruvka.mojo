from mojmelo.utils.utils import fill_indices_list
from mojmelo.utils.KDTree import KDTreeResultVector
from .KDTreeBoruvka import KDTreeBoruvka
import math
from algorithm import vectorize, parallelize
from buffer import NDBuffer

struct UnionFind:
    var parent: List[Scalar[DType.int]]
    var rank: List[Scalar[DType.int]]
    
    @always_inline
    fn __init__(out self, size: Int) raises:
        self.parent = fill_indices_list(size)
        self.rank = List[Scalar[DType.int]](capacity=size)
        self.rank.resize(size, 0)

    @always_inline
    fn find(mut self, x: Scalar[DType.int]) -> Scalar[DType.int]:
        var v = x
        while self.parent[v] != v:
            var index = self.parent[v]
            self.parent[v] = self.parent[index]
            v = self.parent[v]
        return v

    @always_inline
    fn unite(mut self, x: Scalar[DType.int], y: Scalar[DType.int]):
        var xr = self.find(x)
        var yr = self.find(y)

        if xr == yr:
            return

        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1


@fieldwise_init
@register_passable("trivial")
struct Edge:
    var u: Scalar[DType.int]
    var v: Scalar[DType.int]
    var w_rdist: Float32

struct HDBSCANBoruvka:
    var tree: UnsafePointer[KDTreeBoruvka, MutAnyOrigin]
    var core_dist: UnsafePointer[Float32, MutAnyOrigin]   # squared rdist
    var n: Int
    var dim: Int
    var min_samples: Int
    var alpha: Float32

    var component_of_point: List[Scalar[DType.int]]
    var component_of_node: List[Scalar[DType.int]]
    var component_remap: List[Scalar[DType.int]]
    var candidate_point: List[Scalar[DType.int]]
    var candidate_neighbor: List[Scalar[DType.int]]
    var candidate_dist: List[Scalar[DType.float32]]
    var bounds: List[Scalar[DType.float32]]
    var u_f: UnionFind
    var edges: List[Edge]
    var num_edges: Int

    @always_inline
    fn __init__(out self, t: UnsafePointer[KDTreeBoruvka, MutAnyOrigin],
                       min_samples: Int=5,
                       alpha: Float32=1.0) raises:
        self.tree = t
        self.core_dist = alloc[Float32](t[].n)
        self.n = t[].n
        self.dim = t[].dim
        self.min_samples = min_samples
        self.alpha = alpha

        self.candidate_point = List[Scalar[DType.int]](capacity=self.n)
        self.candidate_point.resize(self.n, -1)
        self.candidate_neighbor = List[Scalar[DType.int]](capacity=self.n)
        self.candidate_neighbor.resize(self.n, -1)
        self.candidate_dist = List[Scalar[DType.float32]](capacity=self.n)
        self.candidate_dist.resize(self.n, math.inf[DType.float32]())
        self.bounds = List[Scalar[DType.float32]](capacity=len(t[].nodes))
        self.bounds.resize(len(t[].nodes), math.inf[DType.float32]())
        self.u_f = UnionFind(self.n)
        self.edges = List[Edge](capacity=self.tree[].n - 1)
        self.edges.resize(self.tree[].n - 1, Edge(0, 0, 0))
        self.num_edges = 0
        self.component_of_point = fill_indices_list(self.n)
        self.component_of_node = fill_indices_list(len(t[].nodes))
        self.component_remap = List[Scalar[DType.int]](capacity=self.n)
        self.component_remap.resize(self.n, -1)

        @parameter
        fn compute_core_dist(p: Int):
            # request min_samples + 1 neighbors
            var kd_results = KDTreeResultVector()
            self.tree[].kdtree.n_nearest(NDBuffer[dtype=DType.float32, rank=1](self.tree[].data + p * self.dim, self.dim), self.min_samples + 1, kd_results)

            var count = 0
            for i in range(len(kd_results)):
                if kd_results[i].idx == p:
                    continue
                count += 1
                if count == self.min_samples:
                    self.core_dist[p] = kd_results[i].dis
                    return

            # safety fallback (should never happen)
            self.core_dist[p] = math.inf[DType.float32]()
        parallelize[compute_core_dist](self.n)

        @parameter
        fn v[simd_width: Int](idx: Int):
            self.component_of_node._data.store(idx, -self.component_of_node._data.load[width=simd_width](idx) - 1)
        vectorize[v, hdbscan_tree.simd_width](len(t[].nodes))

    @always_inline
    fn __del__(deinit self):
        if self.core_dist:
            self.core_dist.free()

    @always_inline
    fn mr_rdist(self, var d_rdist: Float32, p: Scalar[DType.int], q: Scalar[DType.int]) -> Float32:
        if self.alpha!=1.0:
            d_rdist /= self.alpha*self.alpha
        return max(max(d_rdist, self.core_dist[p]), self.core_dist[q])

    @always_inline
    fn propagate_bounds(mut self, var node: Int):
        while node>0:
            var parent=(node-1)//2
            var l=self.tree[].left(parent)
            var r=self.tree[].right(parent)
            var nb = max(self.bounds[l], self.bounds[r])
            if nb < self.bounds[parent]:
                self.bounds[parent]=nb
                node=parent
            else:
                break

    fn update_components_and_nodes(mut self) raises -> Int:
        # Consume candidate edges and union components
        for c in range(self.n):
            var p = self.candidate_point[c]
            var q = self.candidate_neighbor[c]

            if p < 0 or q < 0:
                continue

            var cp = self.u_f.find(p)
            var cq = self.u_f.find(q)

            if cp == cq:
                # stale candidate
                self.candidate_dist[c] = math.inf[DType.float32]()
                self.candidate_point[c] = -1
                self.candidate_neighbor[c] = -1
                continue

            # add edge to MST
            self.edges[self.num_edges] = Edge(p, q, self.candidate_dist[c])
            self.num_edges += 1

            self.u_f.unite(cp, cq)

            # reset candidate slot
            self.candidate_dist[c] = math.inf[DType.float32]()
            self.candidate_point[c] = -1
            self.candidate_neighbor[c] = -1

            if self.num_edges == self.n - 1:
                break

        # Recompute component_of_point
        @parameter
        fn update_point(i: Int):
            self.component_of_point[i] = self.u_f.find(i)
        parallelize[update_point](self.n)

        # Compact component ids (important!)
        # Ensures components are dense in [0, k)
        var next_id = 0

        for i in range(self.n):
            var c = self.component_of_point[i]
            if self.component_remap[c] == -1:
                self.component_remap[c] = next_id
                next_id += 1
            self.component_of_point[i] = self.component_remap[c]

        # reset remap for next iteration
        for i in range(next_id):
            self.component_remap[i] = -1

        var num_components = next_id

        # Recompute component_of_node (bottom-up)
        for ni in range(len(self.tree[].nodes) - 1, -1, -1):
            var nd = UnsafePointer(to=self.tree[].nodes[ni])

            if nd[].is_leaf:
                var start = nd[].idx_start
                var end = nd[].idx_end
                var c = self.component_of_point[
                    self.tree[].idx_array[start]
                ]

                var same = True
                for i in range(start + 1, end):
                    if self.component_of_point[self.tree[].idx_array[i]] != c:
                        same = False
                        break

                self.component_of_node[ni] = c if same else -1

            else:
                var l = self.tree[].left(ni)
                var r = self.tree[].right(ni)

                var cl = self.component_of_node[l]
                var cr = self.component_of_node[r]

                self.component_of_node[ni] = cl if cl == cr else -1

        # Reset bounds for next iteration
        NDBuffer[dtype=DType.float32, rank=1](self.bounds, len(self.bounds)).fill(math.inf[DType.float32]())

        return num_components

    fn dual_tree_traversal(mut self, node1: Int, node2: Int):
        var nd = self.tree[].kdtree_min_rdist_dual(
            node1,node2, self.tree[].node_bounds._data, self.dim)

        if nd >= self.bounds[node1]:
            return

        var c1 = self.component_of_node[node1]
        var c2 = self.component_of_node[node2]
        if c1==c2 and c1>=0:
            return

        var n1 = UnsafePointer(to=self.tree[].nodes[node1])
        var n2 = UnsafePointer(to=self.tree[].nodes[node2])

        if n1[].is_leaf and n2[].is_leaf:
            var new_upper: Float32=0.0
            var new_lower= math.inf[DType.float32]()

            for i in range(n1[].idx_start, n1[].idx_end):
                var p=self.tree[].idx_array[i]
                var cp=self.component_of_point[p]
                if self.core_dist[p] > self.candidate_dist[cp]:
                    continue

                var xp = self.tree[].data + p*self.dim

                for j in range(n2[].idx_start, n2[].idx_end):
                    var q=self.tree[].idx_array[j]
                    if self.component_of_point[q]==cp or self.core_dist[q] > self.candidate_dist[cp]:
                        continue

                    var xq = self.tree[].data + q*self.dim

                    var d: Float32=0.0
                    @parameter
                    fn v[simd_width: Int](idx: Int):
                        d += ((xp.load[width=simd_width](idx) - xq.load[width=simd_width](idx)) ** 2).reduce_add()
                    vectorize[v, mojmelo.utils.Matrix.Matrix.simd_width](self.dim)
                    var mr = self.mr_rdist(d,p,q)
                    if mr < self.candidate_dist[cp]:
                        self.candidate_dist[cp]=mr
                        self.candidate_point[cp]=p
                        self.candidate_neighbor[cp]=q

                new_upper = max(new_upper, self.candidate_dist[cp])
                new_lower = min(new_lower, self.candidate_dist[cp])

            var newb = min(
                new_upper,
                new_lower + 2.0*(n1[].radius*n1[].radius)
            )
            if newb < self.bounds[node1]:
                self.bounds[node1]=newb
                self.propagate_bounds(node1)
            return

        if not n2[].is_leaf and
           (n1[].is_leaf or n2[].radius > n1[].radius):
            self.dual_tree_traversal(node1, self.tree[].left(node2))
            self.dual_tree_traversal(node1, self.tree[].right(node2))
        else:
            self.dual_tree_traversal(self.tree[].left(node1), node2)
            self.dual_tree_traversal(self.tree[].right(node1), node2)

    fn spanning_tree(mut self) raises -> List[Edge]:
        var num_components = self.tree[].n
        while num_components > 1:
            self.dual_tree_traversal(0, 0)
            num_components = self.update_components_and_nodes()

        return self.edges[:self.num_edges].copy()
