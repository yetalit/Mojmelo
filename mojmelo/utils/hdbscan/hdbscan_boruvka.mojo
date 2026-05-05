from mojmelo.utils.utils import fill_indices_list
from mojmelo.utils.Matrix import Matrix
from .KDTreeBoruvka import KDTreeBoruvka, node_pair_lower_bound
import std.math as math
from std.algorithm import vectorize, parallelize
from std.memory import memset

struct UnionFind:
    var parent: List[Scalar[DType.int]]
    var rank: List[Scalar[DType.int]]

    @always_inline
    def __init__(out self, size: Int) raises:
        self.parent = fill_indices_list(size)
        self.rank = List[Scalar[DType.int]](capacity=size)
        self.rank.resize(size, 0)

    @always_inline
    def find(mut self, x: Scalar[DType.int]) -> Scalar[DType.int]:
        var v = x
        while self.parent[v] != v:
            var index = self.parent[v]
            self.parent[v] = self.parent[index]
            v = self.parent[v]
        return v

    @always_inline
    def unite(mut self, x: Scalar[DType.int], y: Scalar[DType.int]):
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


struct HDBSCANBoruvka:
    var tree: UnsafePointer[KDTreeBoruvka, MutAnyOrigin]
    var n: Int
    var dim: Int
    var min_samples: Int
    var alpha: Float32
    var num_components: Int

    # Per-point candidate edge (indexed by point)
    var candidate_point: List[Scalar[DType.int]]
    var candidate_neighbor: List[Scalar[DType.int]]
    var candidate_dist: List[Scalar[DType.float32]]

    # Per-component best bound — key for pruning in traversal
    # component_bound[c] = best mutual-reachability distance found so far
    # for any edge leaving component c this round.
    var component_bound: List[Scalar[DType.float32]]

    var u_f: UnionFind
    # u_f_finds[i] = find(i) result after last update_components; used to
    # identify component roots without re-running find() in hot loops.
    var u_f_finds: List[Int]

    var edges: Matrix
    var num_edges: Int

    # Component membership arrays updated each Borůvka round
    var component_of_point: List[Scalar[DType.int]]
    var component_of_node: List[Scalar[DType.int]]
    # Temporary remap buffer, kept alive across rounds to avoid re-alloc
    var component_remap: List[Scalar[DType.int]]

    @always_inline
    def __init__(out self,
                 t: UnsafePointer[KDTreeBoruvka, MutAnyOrigin],
                 min_samples: Int = 5,
                 alpha: Float32 = 1.0) raises:
        self.tree = t
        self.n = t[].n
        self.dim = t[].dim
        self.min_samples = min_samples
        self.alpha = alpha
        self.num_components = self.n

        self.candidate_point = List[Scalar[DType.int]](capacity=self.n)
        self.candidate_point.resize(self.n, -1)
        self.candidate_neighbor = List[Scalar[DType.int]](capacity=self.n)
        self.candidate_neighbor.resize(self.n, -1)
        self.candidate_dist = List[Scalar[DType.float32]](capacity=self.n)
        self.candidate_dist.resize(self.n, math.inf[DType.float32]())

        self.component_bound = List[Scalar[DType.float32]](capacity=self.n)
        self.component_bound.resize(self.n, math.inf[DType.float32]())

        self.u_f = UnionFind(self.n)
        self.u_f_finds = List[Int](capacity=self.n)
        self.u_f_finds.resize(self.n, 0)

        self.edges = Matrix(self.tree[].n - 1, 3)
        self.num_edges = 0

        self.component_of_point = fill_indices_list(self.n)
        self.component_of_node = List[Scalar[DType.int]](capacity=len(t[].nodes))
        self.component_of_node.resize(len(t[].nodes), -1)
        self.component_remap = List[Scalar[DType.int]](capacity=self.n)
        self.component_remap.resize(self.n, -1)

    # ------------------------------------------------------------------ #
    #  Mutual reachability distance (squared, deferred sqrt to edge emit) #
    # ------------------------------------------------------------------ #

    @always_inline
    def mr_rdist(self, var d2: Float32,
                 p: Scalar[DType.int],
                 q: Scalar[DType.int]) -> Float32:
        if self.alpha != 1.0:
            d2 /= (self.alpha * self.alpha)
        return max(max(d2, self.tree[].core_dist[p]),
                   self.tree[].core_dist[q])

    # ------------------------------------------------------------------ #
    #  Component + node label update (one call per Borůvka round)         #
    # ------------------------------------------------------------------ #

    def update_components_and_nodes(mut self) raises:
        # --- 1. Refresh point→component labels in parallel ---
        @parameter
        def update_point(i: Int):
            self.component_of_point[i] = self.u_f.find(Scalar[DType.int](i))
            self.u_f_finds[i] = Int(self.component_of_point[i])
        parallelize[update_point](self.n)

        # --- 2. Compact component IDs to [0, num_components) ---
        var next_id: Scalar[DType.int] = 0
        for i in range(self.n):
            var c = self.component_of_point[i]
            if self.component_remap[c] == -1:
                self.component_remap[c] = next_id
                next_id += 1
            self.component_of_point[i] = self.component_remap[c]

        memset(self.component_remap.unsafe_ptr(), -1, self.n)
        self.num_components = Int(next_id)

        # --- 3. Propagate component labels up the node tree (bottom-up) ---
        # A node gets a definite component label only when ALL points below
        # it belong to the same component.  -1 means "mixed / unknown".
        for ni in range(len(self.tree[].nodes) - 1, -1, -1):
            var nd = UnsafePointer(to=self.tree[].nodes[ni])

            if nd[].is_leaf:
                var start = nd[].idx_start
                var end   = nd[].idx_end
                var c = self.component_of_point[start]
                var same = True
                for i in range(start + 1, end):
                    if self.component_of_point[i] != c:
                        same = False
                        break
                self.component_of_node[ni] = c if same else -1
            else:
                var l  = self.tree[].left(ni)
                var r  = self.tree[].right(ni)
                var cl = self.component_of_node[l]
                var cr = self.component_of_node[r]
                self.component_of_node[ni] = cl if cl == cr else -1

    # -------------------------------- #
    #  Single-tree query for one point #
    # -------------------------------- #

    def _query_single(mut self,
                      node: Int,
                      point_idx: Int,
                      point_component: Int,
                      heap_dist: UnsafePointer[Float32, MutAnyOrigin],
                      heap_nbr: UnsafePointer[Scalar[DType.int], MutAnyOrigin],
                      core_p: Float32,
                      comp_bound: UnsafePointer[Float32, MutAnyOrigin]) raises:
        var nd = UnsafePointer(to=self.tree[].nodes[node])

        # --- Pruning case 1: node is entirely one component (same as query) ---
        if self.component_of_node[node] == Scalar[DType.int](point_component):
            return

        # --- Pruning case 2: node lower-bound can't beat current heap top ---
        var xp = self.tree[].data + Scalar[DType.int](point_idx) * Scalar[DType.int](self.dim)
        var lb2 = node_pair_lower_bound(
            xp,
            nd[].center._data,
            0.0,
            nd[].radius,
            self.dim
        )
        # Apply core distance floor: effective lb for mutual reachability
        var lb_mr = max(max(lb2, core_p), Float32(0.0))
        if lb_mr >= heap_dist[0]:
            return
        # Also prune against the shared per-component bound
        if lb_mr >= comp_bound[0]:
            return

        # --- Leaf: examine every point in the node ---
        if nd[].is_leaf:
            for i in range(nd[].idx_start, nd[].idx_end):
                var q = Scalar[DType.int](i)
                if self.component_of_point[i] == Scalar[DType.int](point_component):
                    continue

                # Skip q if its core distance alone can't improve comp_bound
                if self.tree[].core_dist[q] >= comp_bound[0]:
                    continue

                var xq = self.tree[].data + q * Scalar[DType.int](self.dim)
                var d2: Float32 = 0.0
                @parameter
                def v[simd_width: Int](k: Int) unified {mut}:
                    var t = xp.load[width=simd_width](k) - xq.load[width=simd_width](k)
                    d2 += (t * t).reduce_add()
                vectorize[Matrix.simd_width](self.dim, v)

                var mr = self.mr_rdist(d2, Scalar[DType.int](point_idx), q)

                if mr < heap_dist[0]:
                    heap_dist[0] = mr
                    heap_nbr[0]  = q
                    # Also record query point so merge_components can read it
                    self.candidate_point[point_idx] = Scalar[DType.int](point_idx)
                    # Tighten the shared component bound immediately so that
                    # other points in the same component benefit from this find.
                    if mr < comp_bound[0]:
                        comp_bound[0] = mr
            return

        # --- Internal node: recurse into closer child first ---
        var l = self.tree[].left(node)
        var r = self.tree[].right(node)
        var ndl = UnsafePointer(to=self.tree[].nodes[l])
        var ndr = UnsafePointer(to=self.tree[].nodes[r])

        var lb2l = node_pair_lower_bound(xp, ndl[].center._data, 0.0, ndl[].radius, self.dim)
        var lb2r = node_pair_lower_bound(xp, ndr[].center._data, 0.0, ndr[].radius, self.dim)

        if lb2l <= lb2r:
            self._query_single(l, point_idx, point_component, heap_dist, heap_nbr, core_p, comp_bound)
            self._query_single(r, point_idx, point_component, heap_dist, heap_nbr, core_p, comp_bound)
        else:
            self._query_single(r, point_idx, point_component, heap_dist, heap_nbr, core_p, comp_bound)
            self._query_single(l, point_idx, point_component, heap_dist, heap_nbr, core_p, comp_bound)

    # ------------------------------------------------------------------ #
    #  Parallel Borůvka query — one task per point, sharing component      #
    #  bounds so that early finds prune later queries in the same round.   #
    # ------------------------------------------------------------------ #

    def boruvka_query(mut self) raises:
        # Reset per-point candidates
        memset(self.candidate_point.unsafe_ptr(),    -1, self.n)
        memset(self.candidate_neighbor.unsafe_ptr(), -1, self.n)
        Span[Float32, origin_of(self.candidate_dist)](
            ptr=self.candidate_dist.unsafe_ptr(), length=self.n
        ).fill(math.inf[DType.float32]())

        # Reset per-component bounds (indexed by compacted component ID)
        Span[Float32, origin_of(self.component_bound)](
            ptr=self.component_bound.unsafe_ptr(), length=self.n
        ).fill(math.inf[DType.float32]())

        # One task per point.  Each point holds a private heap_dist / heap_nbr
        # slot (its own candidate_dist[i] / candidate_neighbor[i]) but shares
        # component_bound[component] with all other points in the same component 
        @parameter
        def query_point(i: Int):
            var comp = Int(self.component_of_point[i])
            var heap_dist = self.candidate_dist.unsafe_ptr() + i
            var heap_nbr  = self.candidate_neighbor.unsafe_ptr() + i
            var comp_bnd  = self.component_bound.unsafe_ptr() + comp
            try:
                self._query_single(
                    0,
                    i,
                    comp,
                    heap_dist,
                    heap_nbr,
                    self.tree[].core_dist[i],
                    comp_bnd
                )
            except:
                print()
        parallelize[query_point](self.n)

    # ------------------------------------------------------------------ #
    #  Edge collection and union-find merge (sequential, after parallel    #
    #  query)                                                              #
    # ------------------------------------------------------------------ #

    def merge_components(mut self) raises -> Int:
        # For each component, find the point with the best (lowest) candidate
        # distance, then emit one edge per component pair.
        # component_bound[comp] already holds the per-component minimum
        # distance from boruvka_query — we just need to find which point i
        # achieved it.

        # best_i[comp] = point index that achieved component_bound[comp]
        var best_i   = List[Scalar[DType.int]](capacity=self.n)
        var best_nbr = List[Scalar[DType.int]](capacity=self.n)
        best_i.resize(self.n, -1)
        best_nbr.resize(self.n, -1)

        for i in range(self.n):
            # candidate_point[i] is set to i only when a neighbor was found
            if self.candidate_point[i] < 0:
                continue
            var comp = Int(self.component_of_point[i])
            if best_i[comp] < 0 or self.candidate_dist[i] < self.candidate_dist[Int(best_i[comp])]:
                best_i[comp]   = Scalar[DType.int](i)
                best_nbr[comp] = self.candidate_neighbor[i]

        var edges_added = 0
        for comp in range(self.n):
            var i = best_i[comp]
            if i < 0:
                continue
            var q = best_nbr[comp]
            if q < 0:
                continue

            var cp = self.u_f.find(i)
            var cq = self.u_f.find(q)
            if cp == cq:
                continue

            var d = math.sqrt(self.candidate_dist[Int(i)])

            self.edges[self.num_edges, 0] = i.cast[DType.float32]()
            self.edges[self.num_edges, 1] = q.cast[DType.float32]()
            self.edges[self.num_edges, 2] = d
            self.num_edges += 1
            edges_added += 1

            self.u_f.unite(i, q)

        return edges_added

    def spanning_tree(mut self) raises -> Matrix:
        self.num_edges = 0

        while True:
            self.update_components_and_nodes()

            if self.num_components <= 1:
                break

            self.boruvka_query()

            var edges_added = self.merge_components()

            if edges_added == 0:
                break

        return self.edges.load_rows(self.num_edges)
