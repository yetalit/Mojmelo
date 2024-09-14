from collections.vector import InlinedFixedVector
import math
from mojmelo.utils.Matrix import Matrix
from python import Python, PythonObject
from sys import bitwidthof
from bit import count_leading_zeros
from utils import Span

fn eliminate(r1: Matrix, inout r2: Matrix, col: Int, target: Int = 0):
    var fac = (r2.data[col] - target) / r1.data[col]
    for i in range(r2.size):
        r2.data[i] -= fac * r1.data[i]

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

fn cov_value(x: Matrix, y: Matrix) -> Float32:
    var mean_x = x.mean()
    var mean_y = y.mean()

    var sub_x = x - mean_x
    var sub_y = y - mean_y

    var sum_value: Float32 = 0.0
    for i in range(x.size):
        sum_value += sub_y.data[i] * sub_x.data[i]
    return sum_value / (x.size - 1)

# ===----------------------------------------------------------------------===#
# partition
# ===----------------------------------------------------------------------===#

@always_inline
fn _estimate_initial_height(size: Int) -> Int:
    # Compute the log2 of the size rounded upward.
    var log2 = int(
        (bitwidthof[DType.index]() - 1) ^ count_leading_zeros(size | 1)
    )
    # The number 1.3 was chosen by experimenting the max stack size for random
    # input. This also depends on insertion_sort_threshold
    return max(2, int(math.ceil(1.3 * log2)))

@value
struct _SortWrapper[type: CollectionElement](CollectionElement):
    var data: type

    fn __init__(inout self, *, other: Self):
        self.data = other.data


@always_inline
fn _partition[
    type: CollectionElement,
    lifetime: MutableLifetime, //,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](span: Span[type, lifetime], inout indices: InlinedFixedVector[Int]) -> Int:
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
    lifetime: MutableLifetime, //,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](owned span: Span[type, lifetime], inout indices: InlinedFixedVector[Int], owned k: Int):
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
    lifetime: MutableLifetime, //,
    cmp_fn: fn (Float32, Float32) capturing -> Bool,
](span: Span[Float32, lifetime], inout indices: InlinedFixedVector[Int], k: Int):
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

fn euclidean_distance(x1: Matrix, x2: Matrix) raises -> Float32:
    return math.sqrt(((x1 - x2) ** 2).sum())

fn manhattan_distance(x1: Matrix, x2: Matrix) raises -> Float32:
    return (x1 - x2).abs().sum()

fn lt(lhs: Float32, rhs:Float32) capturing -> Bool:
    return lhs < rhs

fn le(lhs: Float32, rhs:Float32) capturing -> Bool:
    return lhs <= rhs

fn gt(lhs: Float32, rhs:Float32) capturing -> Bool:
    return lhs > rhs

fn ge(lhs: Float32, rhs:Float32) capturing -> Bool:
    return lhs >= rhs

fn sigmoid(z: Matrix) -> Matrix:
    return 1 / (1 + (-z).exp())

fn normal_distr(x: Matrix, mean: Matrix, _var: Matrix) raises -> Matrix:
    return (-((x - mean) ** 2) / (2 * _var)).exp() / (2 * math.pi * _var).sqrt()

fn unit_step(z: Matrix) -> Matrix:
    return z.where(z >= 0.0, 1.0, 0.0)

fn sign(z: Matrix) -> Matrix:
    var mat = Matrix(z.height, z.width)
    for i in range(mat.size):
        if z.data[i] > 0.0:
            mat.data[i] = 1.0
        elif z.data[i] < 0.0:
            mat.data[i] = -1.0
        else:
            mat.data[i] = 0.0
    return mat^

fn ReLu(z: Matrix) -> Matrix:
    return z.where(z > 0.0, z, 0.0)

fn polynomial_kernel(params: Tuple[Float32, Int], X: Matrix, Z: Matrix) raises -> Matrix:
    return (params[0] + X * Z.T()) ** params[1] #(c + X.y)^degree

fn gaussian_kernel(params: Tuple[Float32, Int], X: Matrix, Z: Matrix) raises -> Matrix:
    var sq_dist = Matrix(X.height, Z.height)
    for i in range(sq_dist.height):  # Loop over each sample in X
        for j in range(sq_dist.width):  # Loop over each sample in Z
            sq_dist.data[i * sq_dist.width + j] = ((X[i] - Z[j]) ** 2).sum()
    return (-sq_dist / (params[0] ** 2)).exp() # e^-(1/ σ2) ||X-y|| ^2

fn mse(y: Matrix, y_pred: Matrix) raises -> Float32:
    return ((y - y_pred) ** 2).mean()

fn r2_score(y: Matrix, y_pred: Matrix) raises -> Float32:
    return 1.0 - (((y_pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())

fn accuracy_score(y: Matrix, y_pred: Matrix, zero_to_negone: Bool = False) -> Float32:
    if zero_to_negone:
        return accuracy_score(y.where(y <= 0.0, -1.0, 1.0), y_pred)
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

fn entropy(y: Matrix) -> Float32:
    var histogram = y.bincount()
    var size = Float32(y.size)
    var _sum: Float32 = 0.0
    for i in range(histogram.capacity):
        var p: Float32 = histogram[i] / size
        if p > 0 and p != 1.0:
            _sum += p * math.log2(p)
    return -_sum

fn gini(y: Matrix) -> Float32:
    var histogram = y.bincount()
    var size = Float32(y.size)
    var _sum: Float32 = 0.0
    for i in range(histogram.capacity):
        _sum += (histogram[i] / size) ** 2
    return 1 - _sum

fn mse_loss(y: Matrix) -> Float32:
    if len(y) == 0:
        return 0.0
    return ((y - y.mean()) ** 2).mean()


fn mse_g(true: Matrix, score: Matrix) raises -> Matrix:
    return score - true
fn mse_h(score: Matrix) -> Matrix:
    return Matrix.ones(score.height, 1)

fn log_g(true: Matrix, score: Matrix) raises -> Matrix:
    return sigmoid(score) - true
fn log_h(score: Matrix) -> Matrix:
    var pred = sigmoid(score)
    return pred.ele_mul(1 - pred)


fn l_to_numpy(list: List[String]) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var np_arr = np.empty(len(list), dtype='object')
    for i in range(len(list)):
        np_arr[i] = list[i]
    return np_arr^
