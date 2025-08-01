from memory import memcpy, UnsafePointer
import math
from mojmelo.utils.Matrix import Matrix
from python import Python, PythonObject
from algorithm import parallelize, elementwise
from sys import simdwidthof
from utils import IndexList

# Cross Validation y as Matrix
trait CVM:
    fn __init__(out self, params: Dict[String, String]) raises:
        ...
    fn fit(mut self, X: Matrix, y: Matrix) raises:
        ...
    fn predict(self, X: Matrix) raises -> Matrix:
        ...

# Cross Validation y as PythonObject
trait CVP:
    fn __init__(out self, params: Dict[String, String]) raises:
        ...
    fn fit(mut self, X: Matrix, y: PythonObject) raises:
        ...
    fn predict(self, X: Matrix) raises -> List[String]:
        ...

fn cov_value(x_mean_diff: Matrix, y_mean_diff: Matrix) raises -> Float32:
    return (y_mean_diff.ele_mul(x_mean_diff)).sum() / (x_mean_diff.size - 1)

fn complete_orthonormal_basis(X: Matrix, full_size: Int) raises -> Matrix:
    if X.width == full_size:
        return X
    P = Matrix.eye(X.height, X.order) - X * X.T()  # projection onto orthogonal complement
    Q, _ = P.qr()
    return X.concatenate(Q.load_columns(full_size - X.width), axis=1)

# ===-----------------------------------------------------------------------===#
# argn
# ===-----------------------------------------------------------------------===#

fn argn[is_max: Bool](input: Matrix, output: Matrix):
    alias simd_width = simdwidthof[DType.float32]()
    var axis_size = input.size
    var input_stride = input.size
    alias output_stride = 1
    alias chunk_size = 1
    alias parallel_size = 1

    @__copy_capture(
        axis_size, chunk_size, output_stride, input_stride, parallel_size
    )
    
    @parameter
    @always_inline
    fn cmpeq[
        type: DType, simd_width: Int
    ](a: SIMD[type, simd_width], b: SIMD[type, simd_width]) -> SIMD[
        DType.bool, simd_width
    ]:
        @parameter
        if is_max:
            return a <= b
        else:
            return a >= b

    @parameter
    @always_inline
    fn cmp[
        type: DType, simd_width: Int
    ](a: SIMD[type, simd_width], b: SIMD[type, simd_width]) -> SIMD[
        DType.bool, simd_width
    ]:
        @parameter
        if is_max:
            return a < b
        else:
            return a > b

    # iterate over flattened axes
    alias start = 0
    alias end = 1
    for i in range(start, end):
        var input_offset = i * input_stride
        var output_offset = i * output_stride
        var input_dim_ptr = input.data.offset(input_offset)
        var output_dim_ptr = output.data.offset(output_offset)
        var global_val: Float32

        # initialize limits
        @parameter
        if is_max:
            global_val = Float32.MIN
        else:
            global_val = Float32.MAX

        # initialize vector of maximal/minimal values
        var global_values: SIMD[DType.float32, simd_width]
        if axis_size < simd_width:
            global_values = global_val
        else:
            global_values = input_dim_ptr.load[width=simd_width]()

        # iterate over values evenly divisible by simd_width
        var indices = math.iota[DType.float32, simd_width]()
        var global_indices = indices
        var last_simd_index = math.align_down(axis_size, simd_width)
        for j in range(simd_width, last_simd_index, simd_width):
            var curr_values = input_dim_ptr.load[width=simd_width](j)
            indices += simd_width

            var mask = cmpeq(curr_values, global_values)
            global_indices = mask.select(global_indices, indices)
            global_values = mask.select(global_values, curr_values)

        @parameter
        if is_max:
            global_val = global_values.reduce_max()
        else:
            global_val = global_values.reduce_min()

        # Check trailing indices.
        var idx = Float32(0)
        var found_min: Bool = False
        for j in range(last_simd_index, axis_size, 1):
            var elem = input_dim_ptr.load(j)
            if cmp(global_val, elem):
                global_val = elem
                idx = j
                found_min = True

        # handle the case where min wasn't in trailing values
        if not found_min:
            var matching = global_values == global_val
            var min_indices = matching.select(
                global_indices, Float32.MAX
            )
            idx = min_indices.reduce_min()
        output_dim_ptr[] = idx

# ===----------------------------------------------------------------------===#

@always_inline
fn euclidean_distance(x1: Matrix, x2: Matrix) raises -> Float32:
    return math.sqrt(((x1 - x2) ** 2).sum())

@always_inline
fn euclidean_distance(x1: Matrix, x2: Matrix, axis: Int) raises -> Matrix:
    return (((x1 - x2) ** 2).sum(axis)).sqrt()

@always_inline
fn squared_euclidean_distance(x1: Matrix, x2: Matrix) raises -> Float32:
    return ((x1 - x2) ** 2).sum()

@always_inline
fn squared_euclidean_distance(x1: Matrix, x2: Matrix, axis: Int) raises -> Matrix:
    return ((x1 - x2) ** 2).sum(axis)

@always_inline
fn manhattan_distance(x1: Matrix, x2: Matrix) raises -> Float32:
    return (x1 - x2).abs().sum()

@always_inline
fn manhattan_distance(x1: Matrix, x2: Matrix, axis: Int) raises -> Matrix:
    return (x1 - x2).abs().sum(axis)

@always_inline
fn add[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return a + b

@always_inline
fn sub[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return a - b

@always_inline
fn mul[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return a * b

@always_inline
fn div[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return a / b

@always_inline
fn partial_simd_load[width: Int](data: UnsafePointer[Float32], offset: Int, size: Int) -> SIMD[DType.float32, width]:
    var nelts = size - offset
    if nelts >= width:
        return data.load[width=width](offset)
    var point = data + offset
    var simd = SIMD[DType.float32, width]()
    for i in range(0, nelts):
        simd[i] = point[i]
    return simd

@always_inline
fn sigmoid(z: Matrix) -> Matrix:
    return 1 / (1 + (-z).exp())

@always_inline
fn normal_distr(x: Matrix, mean: Matrix, _var: Matrix) raises -> Matrix:
    return (-((x - mean) ** 2) / (2 * _var)).exp() / (2 * math.pi * _var).sqrt()

@always_inline
fn unit_step(z: Matrix) -> Matrix:
    return z.where(z >= 0.0, 1.0, 0.0)

@always_inline
fn sign(z: Matrix) -> Matrix:
    var mat = Matrix(z.height, z.width, order= z.order)
    if mat.size < 147456:
        for i in range(mat.size):
            if z.data[i] > 0.0:
                mat.data[i] = 1.0
            elif z.data[i] < 0.0:
                mat.data[i] = -1.0
            else:
                mat.data[i] = 0.0
    else:
        @parameter
        fn p(i: Int):
            if z.data[i] > 0.0:
                mat.data[i] = 1.0
            elif z.data[i] < 0.0:
                mat.data[i] = -1.0
            else:
                mat.data[i] = 0.0
        parallelize[p](mat.size)
    return mat^

@always_inline
fn ReLu(z: Matrix) -> Matrix:
    return z.where(z > 0.0, z, 0.0)

fn polynomial_kernel(params: Tuple[Float32, Int], X: Matrix, Z: Matrix) raises -> Matrix:
    return (params[0] + X * Z.T()) ** params[1] #(c + X.y)^degree

fn gaussian_kernel(params: Tuple[Float32, Int], X: Matrix, Z: Matrix) raises -> Matrix:
    var sq_dist = Matrix(X.height, Z.height, order= X.order)
    for i in range(sq_dist.height):  # Loop over each sample in X
        sq_dist[i] = ((X[i] - Z) ** 2).sum(axis=1)
    return (-sq_dist * params[0]).exp() # e^-(1/ σ2) ||X-y|| ^2

@always_inline
fn mse(y: Matrix, y_pred: Matrix) raises -> Float32:
    return ((y - y_pred) ** 2).mean()

@always_inline
fn cross_entropy(y: Matrix, y_pred: Matrix) raises -> Float32:
    return -(y.ele_mul((y_pred + 1e-15).log()) + (1.0 - y).ele_mul((1.0 - y_pred + 1e-15).log())).mean()

fn r2_score(y: Matrix, y_pred: Matrix) raises -> Float32:
    return 1.0 - (((y_pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())

fn accuracy_score(y: Matrix, y_pred: Matrix) raises -> Float32:
    var correct_count: Float32 = 0.0
    for i in range(y.size):
        if y.data[i] == y_pred.data[i]:
            correct_count += 1.0
    return correct_count / y.size

fn accuracy_score(y: List[String], y_pred: List[String]) raises -> Float32:
    var correct_count: Float32 = 0.0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            correct_count += 1.0
    return correct_count / len(y)

fn accuracy_score(y: PythonObject, y_pred: Matrix) raises -> Float32:
    var correct_count: Float32 = 0.0
    for i in range(y_pred.size):
        if y[i] == y_pred.data[i]:
            correct_count += 1.0
    return correct_count / y_pred.size

fn accuracy_score(y: PythonObject, y_pred: List[String]) raises -> Float32:
    var correct_count: Float32 = 0.0
    for i in range(len(y_pred)):
        if String(y[i]) == y_pred[i]:
            correct_count += 1.0
    return correct_count / len(y_pred)

@always_inline
fn entropy(y: Matrix) raises -> Float32:
    var histogram = y.bincount()
    var size = Float32(y.size)
    var _sum: Float32 = 0.0
    for i in range(len(histogram)):
        var p: Float32 = histogram[i] / size
        if p > 0 and p != 1.0:
            _sum += p * math.log2(p)
    return -_sum

@always_inline
fn entropy_precompute(size: Float32, histogram: List[Int]) raises -> Float32:
    var _sum: Float32 = 0.0
    for i in range(len(histogram)):
        var p: Float32 = histogram[i] / size
        if p > 0 and p != 1.0:
            _sum += p * math.log2(p)
    return -_sum

@always_inline
fn gini(y: Matrix) raises -> Float32:
    var histogram = y.bincount()
    var size = Float32(y.size)
    var _sum: Float32 = 0.0
    for i in range(len(histogram)):
        _sum += (histogram[i] / size) ** 2
    return 1 - _sum

@always_inline
fn gini_precompute(size: Float32, histogram: List[Int]) raises -> Float32:
    var _sum: Float32 = 0.0
    for i in range(len(histogram)):
        _sum += (histogram[i] / size) ** 2
    return 1 - _sum

@always_inline
fn mse_loss(y: Matrix) raises -> Float32:
    if len(y) == 0:
        return 0.0
    return ((y - y.mean()) ** 2).mean()

@always_inline
fn mse_loss_precompute(size: Int, sum: Float32, sum_sq: Float32) raises -> Float32:
    if size == 0:
        return 0.0
    return sum_sq / size - (sum / size) ** 2


@always_inline
fn mse_g(true: Matrix, score: Matrix) raises -> Matrix:
    return score - true
@always_inline
fn mse_h(score: Matrix) raises -> Matrix:
    return Matrix.ones(score.height, 1, order=score.order)

@always_inline
fn log_g(true: Matrix, score: Matrix) raises -> Matrix:
    return sigmoid(score) - true
@always_inline
fn log_h(score: Matrix) raises -> Matrix:
    var pred = sigmoid(score)
    return pred.ele_mul(1 - pred)


fn fill_indices(N: Int) raises -> UnsafePointer[Scalar[DType.index]]:
    var indices = UnsafePointer[Scalar[DType.index]].alloc(N)
    @parameter
    fn fill_indices_iota[width: Int, rank: Int](offset: IndexList[rank]):
        indices.store(offset[0], math.iota[DType.index, width](offset[0]))

    elementwise[fill_indices_iota, simdwidthof[DType.index](), target="cpu"](
        N
    )
    return indices

fn fill_indices_list(N: Int) raises -> List[Scalar[DType.index]]:
    var indices = UnsafePointer[Scalar[DType.index]].alloc(N)
    @parameter
    fn fill_indices_iota[width: Int, rank: Int](offset: IndexList[rank]):
        indices.store(offset[0], math.iota[DType.index, width](offset[0]))

    elementwise[fill_indices_iota, simdwidthof[DType.index](), target="cpu"](
        N
    )
    var list = List[Scalar[DType.index]](unsafe_uninit_length=N)
    list.data = indices
    return list^

fn l_to_numpy(list: List[String]) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var np_arr = np.empty(len(list), dtype='object')
    for i in range(len(list)):
        np_arr[i] = list[i]
    return np_arr^

fn ids_to_numpy(list: List[Scalar[DType.index]]) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var np_arr = np.empty(len(list), dtype='int')
    memcpy(np_arr.__array_interface__['data'][0].unsafe_get_as_pointer[DType.index](), list.data, len(list))
    return np_arr^

fn ids_to_numpy(list: List[Int]) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var np_arr = np.empty(len(list), dtype='int')
    memcpy(np_arr.__array_interface__['data'][0].unsafe_get_as_pointer[DType.index](), list.data.bitcast[Scalar[DType.index]](), len(list))
    return np_arr^

fn cartesian_product(lists: List[List[String]]) -> List[List[String]]:
    var result = List[List[String]]()
    if not lists:
        result.append(List[String]())
        return result^

    first, rest = lists[0], lists[1:]
    var rest_product = cartesian_product(rest)

    # Create the Cartesian product
    for item in first:
        for prod in rest_product:
            result.append(List[String](item) + prod)

    return result^
