from collections import InlinedFixedVector, Dict
from memory import memcpy, UnsafePointer, Span
import math
from mojmelo.utils.Matrix import Matrix
from python import Python, PythonObject
from algorithm import parallelize

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

@always_inline
fn eliminate(r1: Matrix, mut r2: Matrix, col: Int, target: Int = 0) raises:
    var fac = (r2.data[col] - target) / r1.data[col]
    r2 -= fac * r1

fn gauss_jordan(owned a: Matrix) raises -> Matrix:
    for i in range(a.height):
        if a.data[i * a.width + i] == 0:
            for j in range(i+1, a.height):
                if a.data[i * a.width + j] != 0:
                    a[i], a[j] = a[j], a[i]
                    break
            else:
                raise Error("Error: Matrix is not invertible!")
        for j in range(i + 1, a.height):
            eliminate(a[i], a[j], i)
    for i in range(a.height - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            eliminate(a[i], a[j], i)
    for i in range(a.height):
        eliminate(a[i], a[i], i, target=1)
    return a^

fn cov_value(x: Matrix, y: Matrix) raises -> Float32:
    return ((y - y.mean()).ele_mul(x - x.mean())).sum() / (x.size - 1)

# ===----------------------------------------------------------------------===#
# partition
# ===----------------------------------------------------------------------===#

@value
struct _SortWrapper[type: CollectionElement](CollectionElement):
    var data: type

    @implicit
    fn __init__(out self, data: type):
        self.data = data

    fn __init__(out self, *, other: Self):
        self.data = other.data


@always_inline
fn _partition[
    type: CollectionElement,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing [_] -> Bool,
](span: Span[type, origin], mut indices: InlinedFixedVector[Int]) -> Int:
    var size = len(span)
    if size <= 1:
        return 0

    var array = span.unsafe_ptr()
    var pivot = size // 2

    var pivot_value = array[pivot]

    var left = 0
    var right = size - 2

    swap(array[pivot], array[size - 1])
    indices[pivot], indices[size - 1] = indices[size - 1], indices[pivot]

    while left < right:
        if cmp_fn(array[left], pivot_value):
            left += 1
        elif not cmp_fn(array[right], pivot_value):
            right -= 1
        else:
            swap(array[left], array[right])
            indices[left], indices[right] = indices[right], indices[left]

    if cmp_fn(array[right], pivot_value):
        right += 1
    swap(array[size - 1], array[right])
    indices[size - 1], indices[right] = indices[right], indices[size - 1]
    
    return right


fn _partition[
    type: CollectionElement,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing [_] -> Bool,
](owned span: Span[type, origin], mut indices: InlinedFixedVector[Int], owned k: Int):
    while True:
        var pivot = _partition[cmp_fn](span, indices)
        if pivot == k:
            return
        elif k < pivot:
            span._len = pivot
            span = span[:pivot]
        else:
            span._data += pivot + 1
            span._len -= pivot + 1
            k -= pivot + 1


fn partition[
    lifetime: MutableOrigin, //,
    cmp_fn: fn (Float32, Float32) capturing [_] -> Bool,
](span: Span[Float32, lifetime], mut indices: InlinedFixedVector[Int], k: Int):
    """Partition the input buffer inplace such that first k elements are the
    largest (or smallest if cmp_fn is < operator) elements.
    The ordering of the first k elements is undefined.

    Parameters:
        lifetime: Lifetime of span.
        cmp_fn: Comparison functor of (type, type) capturing -> Bool type.
    """

    @parameter
    fn _cmp_fn(lhs: _SortWrapper[Float32], rhs: _SortWrapper[Float32]) -> Bool:
        return cmp_fn(lhs.data, rhs.data)

    _partition[_cmp_fn](span, indices, k)

# ===----------------------------------------------------------------------===#

@always_inline
fn euclidean_distance(x1: Matrix, x2: Matrix) raises -> Float32:
    return math.sqrt(((x1 - x2) ** 2).sum())

@always_inline
fn euclidean_distance(x1: Matrix, x2: Matrix, axis: Int) raises -> Matrix:
    return (((x1 - x2) ** 2).sum(axis)).sqrt()

@always_inline
fn manhattan_distance(x1: Matrix, x2: Matrix) raises -> Float32:
    return (x1 - x2).abs().sum()

@always_inline
fn manhattan_distance(x1: Matrix, x2: Matrix, axis: Int) raises -> Matrix:
    return (x1 - x2).abs().sum(axis)

@always_inline
fn lt(lhs: Float32, rhs:Float32) capturing -> Bool:
    return lhs < rhs

@always_inline
fn le(lhs: Float32, rhs:Float32) capturing -> Bool:
    return lhs <= rhs

@always_inline
fn gt(lhs: Float32, rhs:Float32) capturing -> Bool:
    return lhs > rhs

@always_inline
fn ge(lhs: Float32, rhs:Float32) capturing -> Bool:
    return lhs >= rhs

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

@always_inline
fn polynomial_kernel(params: Tuple[Float32, Int], X: Matrix, Z: Matrix) raises -> Matrix:
    return (params[0] + X * Z.T()) ** params[1] #(c + X.y)^degree

@always_inline
fn gaussian_kernel(params: Tuple[Float32, Int], X: Matrix, Z: Matrix) raises -> Matrix:
    var sq_dist = Matrix(X.height, Z.height, order= X.order)
    for i in range(sq_dist.height):  # Loop over each sample in X
        for j in range(sq_dist.width):  # Loop over each sample in Z
            sq_dist[i, j] = ((X[i] - Z[j]) ** 2).sum()
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
        if str(y[i]) == y_pred[i]:
            correct_count += 1.0
    return correct_count / len(y_pred)

@always_inline
fn entropy(y: Matrix) raises -> Float32:
    var histogram = y.bincount()
    var size = Float32(y.size)
    var _sum: Float32 = 0.0
    for i in range(histogram.capacity):
        var p: Float32 = histogram[i] / size
        if p > 0 and p != 1.0:
            _sum += p * math.log2(p)
    return -_sum

@always_inline
fn gini(y: Matrix) raises -> Float32:
    var histogram = y.bincount()
    var size = Float32(y.size)
    var _sum: Float32 = 0.0
    for i in range(histogram.capacity):
        _sum += (histogram[i] / size) ** 2
    return 1 - _sum

@always_inline
fn mse_loss(y: Matrix) raises -> Float32:
    if len(y) == 0:
        return 0.0
    return ((y - y.mean()) ** 2).mean()

@always_inline
fn mse_g(true: Matrix, score: Matrix) raises -> Matrix:
    return score - true
@always_inline
fn mse_h(score: Matrix) raises -> Matrix:
    return Matrix.ones(score.height, 1)

@always_inline
fn log_g(true: Matrix, score: Matrix) raises -> Matrix:
    return sigmoid(score) - true
@always_inline
fn log_h(score: Matrix) raises -> Matrix:
    var pred = sigmoid(score)
    return pred.ele_mul(1 - pred)


fn l_to_numpy(list: List[String]) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var np_arr = np.empty(len(list), dtype='object')
    for i in range(len(list)):
        np_arr[i] = list[i]
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
            result.append(List[String](item[]) + prod[])

    return result^
