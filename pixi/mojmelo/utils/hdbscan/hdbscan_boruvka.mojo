from mojmelo.utils.utils import fill_indices_list
from mojmelo.utils.Matrix import Matrix
from .KDTreeBoruvka import KDTreeBoruvka, node_pair_lower_bound
import math
from algorithm import vectorize, parallelize
from memory import memset

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


struct HDBSCANBoruvka:
    var tree: UnsafePointer[KDTreeBoruvka, MutAnyOrigin]
    var n: Int
    var dim: Int
    var min_samples: Int
    var alpha: Float32
    var num_components: Int
    var component_of_point: List[Scalar[DType.int]]
    var component_of_node: List[Scalar[DType.int]]
    var component_remap: List[Scalar[DType.int]]
    var candidate_point: List[Scalar[DType.int]]
    var candidate_neighbor: List[Scalar[DType.int]]
    var candidate_dist: List[Scalar[DType.float32]]
    var u_f: UnionFind
    var u_f_finds: List[Int]
    var edges: Matrix
    var num_edges: Int

    @always_inline
    fn __init__(out self, t: UnsafePointer[KDTreeBoruvka, MutAnyOrigin],
                       min_samples: Int=5,
                       alpha: Float32=1.0) raises:
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

    @always_inline
    fn mr_rdist(self, var d2: Float32, p: Scalar[DType.int], q: Scalar[DType.int]) -> Float32:
        if self.alpha != 1.0:
            d2 /= (self.alpha * self.alpha)

        return max(max(d2, self.tree[].core_dist[p]), self.tree[].core_dist[q])

    fn update_components_and_nodes(mut self) raises -> Int:
        @parameter
        fn update_point(i: Int):
            self.component_of_point[i] = self.u_f.find(i)
            self.u_f_finds[i] = Int(self.component_of_point[i])
        parallelize[update_point](self.n)

        var next_id = 0
        for i in range(self.n):
            var c = self.component_of_point[i]
            if self.component_remap[c] == -1:
                self.component_remap[c] = next_id
                next_id += 1
            self.component_of_point[i] = self.component_remap[c]

        memset(self.component_remap.unsafe_ptr(), -1, self.n)

        var num_components = next_id

        for ni in range(len(self.tree[].nodes) - 1, -1, -1):
            var nd = UnsafePointer(to=self.tree[].nodes[ni])

            if nd[].is_leaf:
                var start = nd[].idx_start
                var end = nd[].idx_end
                var c = self.component_of_point[start]

                var same = True
                for i in range(start + 1, end):
                    if self.component_of_point[i] != c:
                        same = False
                        break

                self.component_of_node[ni] = c if same else -1

            else:
                var l = self.tree[].left(ni)
                var r = self.tree[].right(ni)

                var cl = self.component_of_node[l]
                var cr = self.component_of_node[r]

                self.component_of_node[ni] = cl if cl == cr else -1

        self.num_components = num_components
        return num_components

    fn dual_tree_traversal(mut self, node1: Int, node2: Int) raises:
        var nd1 = UnsafePointer(to=self.tree[].nodes[node1])
        var nd2 = UnsafePointer(to=self.tree[].nodes[node2])

        # self-node case
        if node1 == node2:
            if nd1[].is_leaf:
                for i in range(nd1[].idx_start, nd1[].idx_end):
                    var p = i
                    var cp = self.u_f_finds[p]

                    for j in range(i + 1, nd1[].idx_end):
                        var q = j
                        var cq = self.u_f_finds[q]

                        if cp == cq:
                            continue

                        var xp = self.tree[].data + p * self.dim
                        var xq = self.tree[].data + q * self.dim

                        var d2: Float32 = 0.0
                        @parameter
                        fn v1[simd_width: Int](k: Int) unified {mut}:
                            var t = xp.load[width=simd_width](k) - xq.load[width=simd_width](k)
                            d2 += (t * t).reduce_add()
                        vectorize[Matrix.simd_width](self.dim, v1)

                        var mr = self.mr_rdist(d2, p, q)

                        if mr < self.candidate_dist[cp]:
                            self.candidate_dist[cp] = mr
                            self.candidate_point[cp] = p
                            self.candidate_neighbor[cp] = q

                        if mr < self.candidate_dist[cq]:
                            self.candidate_dist[cq] = mr
                            self.candidate_point[cq] = q
                            self.candidate_neighbor[cq] = p
                return

            var l = self.tree[].left(node1)
            var r = self.tree[].right(node1)
            self.dual_tree_traversal(l, l)
            self.dual_tree_traversal(l, r)
            self.dual_tree_traversal(r, r)
            return

        # leaf–leaf
        if nd1[].is_leaf and nd2[].is_leaf:
            var lb2 = node_pair_lower_bound(
                nd1[].center._data,
                nd2[].center._data,
                nd1[].radius,
                nd2[].radius,
                self.dim
            )
            for i in range(nd1[].idx_start, nd1[].idx_end):
                var p = i
                var cp = self.u_f_finds[p]

                for j in range(nd2[].idx_start, nd2[].idx_end):
                    var q = j
                    if p == q:
                        continue

                    var cq = self.u_f_finds[q]
                    if cp == cq:
                        continue

                    # lower-bound pruning
                    if lb2 >= self.candidate_dist[min(cp, cq)]:
                        return

                    var xp = self.tree[].data + p * self.dim
                    var xq = self.tree[].data + q * self.dim

                    var d2: Float32 = 0.0
                    @parameter
                    fn v2[simd_width: Int](k: Int) unified {mut}:
                        var t = xp.load[width=simd_width](k) - xq.load[width=simd_width](k)
                        d2 += (t * t).reduce_add()
                    vectorize[Matrix.simd_width](self.dim, v2)

                    var mr = self.mr_rdist(d2, p, q)

                    if mr < self.candidate_dist[cp]:
                        self.candidate_dist[cp] = mr
                        self.candidate_point[cp] = p
                        self.candidate_neighbor[cp] = q

                    if mr < self.candidate_dist[cq]:
                        self.candidate_dist[cq] = mr
                        self.candidate_point[cq] = q
                        self.candidate_neighbor[cq] = p
            return

        # recurse
        if nd2[].is_leaf or (not nd1[].is_leaf and nd1[].radius >= nd2[].radius):
            self.dual_tree_traversal(self.tree[].left(node1), node2)
            self.dual_tree_traversal(self.tree[].right(node1), node2)
        else:
            self.dual_tree_traversal(node1, self.tree[].left(node2))
            self.dual_tree_traversal(node1, self.tree[].right(node2))


    fn spanning_tree(mut self) raises -> Matrix:
        self.num_edges = 0

        while True:

            _ = self.update_components_and_nodes()
            if self.num_components <= 1:
                break

            memset(self.candidate_point.unsafe_ptr(), -1, self.n)
            memset(self.candidate_neighbor.unsafe_ptr(), -1, self.n)
            Span[Float32, origin_of(self.candidate_dist)](ptr=self.candidate_dist.unsafe_ptr(), length=self.n).fill(math.inf[DType.float32]())

            self.dual_tree_traversal(0, 0)

            var edges_added = 0

            for i in range(self.n):

                if self.u_f.find(i) != i:
                    continue

                var p = self.candidate_point[i]
                if p < 0:
                    continue

                var q = self.candidate_neighbor[i]

                var cp = self.u_f.find(p)
                var cq = self.u_f.find(q)
                if cp == cq:
                    continue

                var d = math.sqrt(self.candidate_dist[i])

                self.edges[self.num_edges, 0] = p.cast[DType.float32]()
                self.edges[self.num_edges, 1] = q.cast[DType.float32]()
                self.edges[self.num_edges, 2] = d
                self.num_edges += 1
                edges_added += 1

                self.u_f.unite(p, q)

            if edges_added == 0:
                break

        return self.edges.load_rows(self.num_edges)
