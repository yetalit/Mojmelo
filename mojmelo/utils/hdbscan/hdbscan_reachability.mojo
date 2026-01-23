from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import squared_euclidean_distance
import math
from algorithm import vectorize, parallelize
from utils.numerics import isfinite
from mojmelo.utils.KDTree import KDTree, KDTreeResultVector
from buffer import NDBuffer

struct LILMatrix:
    var height: Int
    var width: Int
    var rows: List[List[Int]]
    var data: List[List[Float32]]

    @always_inline
    fn __init__(out self, height: Int, width: Int):
        self.height = height
        self.width = width
        self.rows = List[List[Int]](capacity=height)
        self.rows.resize(height, List[Int](capacity=width))
        self.data = List[List[Float32]](capacity=height)
        self.data.resize(height, List[Float32](capacity=width))

    fn nonzero(self) -> Tuple[List[Int], List[Int]]:
        var nz_row = List[Int]()
        var nz_col = List[Int]()
        for r in range(self.height):
            nz_row.resize(len(nz_row) + len(self.data[r]), r)
            nz_col += self.rows[r].copy()

        return nz_row^, nz_col^

    fn tocsr(self) -> CSRMatrix:
        var nz_data = List[Float32]()
        var nz_col = List[Int]()
        var size = 0
        var indptr = [0]
        for r in range(self.height):
            nz_data += self.data[r].copy()
            nz_col += self.rows[r].copy()
            size += len(self.data[r])
            indptr.append(size)

        return CSRMatrix(nz_data^, nz_col^, indptr^)

@fieldwise_init
struct CSRMatrix:
    var data: List[Float32]
    var indices: List[Int]
    var indptr: List[Int]


fn mutual_reachability(var distance_matrix: Matrix, var min_points: Int=5, alpha: Float32=1.0) raises -> Matrix:
    var size = distance_matrix.height
    min_points = min(size - 1, min_points)
    var core_distances = Matrix(1, distance_matrix.width)
    @parameter
    fn cmp_float(a: Scalar[DType.float32], b: Scalar[DType.float32]) -> Bool:
        return a < b
    for c in range(distance_matrix.width):
        var column = distance_matrix['', c]
        partition[cmp_float](
                Span[
                    Scalar[DType.float32],
                    origin_of(MutAnyOrigin),
                ](ptr=column.data, length=len(column)), min_points)
        core_distances.data[c] = column.data[min_points]

    if alpha != 1.0:
        distance_matrix /= alpha

    var stage1_T = core_distances.where(core_distances.ele_gt(distance_matrix),
                      core_distances, distance_matrix).T()
    var result = core_distances.where(core_distances.ele_gt(stage1_T),
                      core_distances.T(), stage1_T).T()
    return result^


fn sparse_mutual_reachability(var lil_matrix: LILMatrix, min_points: Int=5,
                                alpha: Float32=1.0, max_dist: Float32=0) -> CSRMatrix:
    var result = LILMatrix(lil_matrix.height, lil_matrix.width)
    var core_distance = List[Float32](capacity=lil_matrix.height)
    core_distance.resize(lil_matrix.height, 0)

    @parameter
    fn cmp_float(a: Scalar[DType.float32], b: Scalar[DType.float32]) -> Bool:
        return a < b

    for i in range(lil_matrix.height):
        var sorted_row_data = lil_matrix.data[i].copy()
        sort[cmp_float](
                Span[
                    Scalar[DType.float32],
                    origin_of(sorted_row_data),
                ](ptr=sorted_row_data.unsafe_ptr(), length=len(sorted_row_data)))

        if min_points - 1 < len(sorted_row_data):
            core_distance[i] = sorted_row_data[min_points - 1]
        else:
            core_distance[i] = math.inf[DType.float32]()

    if alpha != 1.0:
        for data in lil_matrix.data:
            @parameter
            fn v[simd_width: Int](idx: Int):
                data._data.store(idx, data._data.load[width=simd_width](idx) / alpha)
            vectorize[v, Matrix.simd_width](len(data))

    var nz = lil_matrix.nonzero()
    var nz_row_data = nz[0].copy()
    var nz_col_data = nz[1].copy()

    var current_list = nz_row_data[0]
    var latest_size = 0
    for n in range(len(nz_row_data)):
        var i = nz_row_data[n]
        var j = nz_col_data[n]

        if i > current_list:
            current_list = i
            latest_size += n - latest_size

        var mr_dist = max(core_distance[i], max(core_distance[j], lil_matrix.data[i][n - latest_size]))
        if isfinite(mr_dist):
            result.rows[i].append(j)
            result.data[i].append(mr_dist)
        elif max_dist > 0:
            result.rows[i].append(j)
            result.data[i].append(max_dist)

    return result.tocsr()


fn kdtree_mutual_reachability(X: Matrix, var distance_matrix: Matrix, var min_points: Int=5,
                               alpha: Float32=1.0) raises -> Matrix:
    var dim = distance_matrix.height
    min_points = min(dim - 1, min_points)

    var kdtree = KDTree(X, metric='euc')
    var core_distances = Matrix(X.height, 1)
    @parameter
    fn compute_core_dist(p: Int):
        # request min_samples + 1 neighbors
        var kd_results = KDTreeResultVector()
        kdtree.n_nearest(NDBuffer[dtype=DType.float32, rank=1](X.data + p * X.width, X.width), min_points + 1, kd_results)

        var count = 0
        for i in range(len(kd_results)):
            if kd_results[i].idx == p:
                continue
            count += 1
            if count == min_points:
                core_distances.data[p] = kd_results[i].dis
                return

        # safety fallback (should never happen)
        core_distances.data[p] = math.inf[DType.float32]()
    parallelize[compute_core_dist](X.height)

    if alpha != 1.0:
        distance_matrix = distance_matrix / alpha

    var stage1_T = core_distances.where(core_distances.ele_gt(distance_matrix),
                      core_distances, distance_matrix).T()
    var result = core_distances.where(core_distances.ele_gt(stage1_T),
                      core_distances.T(), stage1_T).T()
    return result^


fn mutual_reachability_from_pdist(core_distances: Matrix, mut dists: Matrix, dim: Int):
    var result_pos = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            if core_distances.data[i] > core_distances.data[j]:
                if core_distances.data[i] > dists.data[result_pos]:
                    dists.data[result_pos] = core_distances.data[i]

            else:
                if core_distances.data[j] > dists.data[result_pos]:
                    dists.data[result_pos] = core_distances.data[j]

            result_pos += 1

@always_inline
fn start_index(i: Int, n: Int) -> Int:
    return i * (2 * n - i - 1) // 2

fn kdtree_pdist_mutual_reachability(X: Matrix, var min_points: Int=5, alpha: Float32=1.0) raises -> Matrix:

    var dim = X.height
    min_points = min(dim - 1, min_points)

    var kdtree = KDTree(X, metric='euc')
    var core_distances = Matrix(X.height, 1)
    @parameter
    fn compute_core_dist(p: Int):
        # request min_samples + 1 neighbors
        var kd_results = KDTreeResultVector()
        kdtree.n_nearest(NDBuffer[dtype=DType.float32, rank=1](X.data + p * X.width, X.width), min_points + 1, kd_results)

        var count = 0
        for i in range(len(kd_results)):
            if kd_results[i].idx == p:
                continue
            count += 1
            if count == min_points:
                core_distances.data[p] = kd_results[i].dis
                return

        # safety fallback (should never happen)
        core_distances.data[p] = math.inf[DType.float32]()
    parallelize[compute_core_dist](X.height)

    var dists = Matrix(X.height*(X.height-1)//2, 1)
    @parameter
    fn pdist(i: Int):
        try:
            var xi = X[i]
            var k = start_index(i, X.height)

            for j in range(i+1, X.height):
                var xj = X[j]
                dists.data[k] = squared_euclidean_distance(xi, xj)
                k += 1
        except e:
            print('Error:', e)
    parallelize[pdist](X.height)


    if alpha != 1.0:
        dists /= alpha

    mutual_reachability_from_pdist(core_distances, dists, dim)

    return dists^
