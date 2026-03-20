from std.memory import memcpy
from std.memory._nonnull import NonNullUnsafePointer
import std.math as math
from mojmelo.utils.Matrix import Matrix
from std.python import Python, PythonObject
from std.algorithm import parallelize, elementwise, vectorize
from std.sys import simd_width_of
from std.utils import IndexList

# Cross Validation trait
trait CV(ImplicitlyDestructible):
    def __init__(out self, params: Dict[String, String]) raises:
        ...
    def fit(mut self, X: Matrix, y: Matrix) raises:
        ...
    def predict(mut self, X: Matrix) raises -> Matrix:
        ...


comptime MODEL_IDS: InlineArray[String, 13] = ['',
                                            'Linear Regression',
                                            'Polynomial Regression',
                                            'Logistic Regression',
                                            'KNN',
                                            'KMeans',
                                            'SVM',
                                            'GaussianNB',
                                            'MultinomialNB',
                                            'Decision Tree',
                                            'Random Forest',
                                            'GBDT',
                                            'PCA'
                                            ]

# ===-----------------------------------------------------------------------===#
# argn
# ===-----------------------------------------------------------------------===#

def argn[is_max: Bool](input: Matrix, output: Matrix):
    comptime simd_width = simd_width_of[DType.float32]()
    var axis_size = input.size
    var input_stride = input.size
    comptime output_stride = 1
    comptime chunk_size = 1
    comptime parallel_size = 1

    @__copy_capture(
        axis_size, chunk_size, output_stride, input_stride, parallel_size
    )

    @parameter
    @always_inline
    def cmpeq[
        type: DType, simd_width: Int
    ](a: SIMD[type, simd_width], b: SIMD[type, simd_width]) -> SIMD[
        DType.bool, simd_width
    ]:
        comptime if is_max:
            return a.le(b)
        else:
            return a.ge(b)

    @parameter
    @always_inline
    def cmp[
        type: DType, simd_width: Int
    ](a: SIMD[type, simd_width], b: SIMD[type, simd_width]) -> SIMD[
        DType.bool, simd_width
    ]:
        comptime if is_max:
            return a.lt(b)
        else:
            return a.gt(b)

    # iterate over flattened axes
    comptime start = 0
    comptime end = 1
    for i in range(start, end):
        var input_offset = i * input_stride
        var output_offset = i * output_stride
        var input_dim_ptr = input.data + input_offset
        var output_dim_ptr = output.data + output_offset
        var global_val: Float32

        # initialize limits
        comptime if is_max:
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
            indices += Float32(simd_width)

            var mask = cmpeq(curr_values, global_values)
            global_indices = mask.select(global_indices, indices)
            global_values = mask.select(global_values, curr_values)

        comptime if is_max:
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
                idx = Float32(j)
                found_min = True

        # handle the case where min wasn't in trailing values
        if not found_min:
            var matching = global_values.eq(global_val)
            var min_indices = matching.select(
                global_indices, Float32.MAX
            )
            idx = min_indices.reduce_min()
        output_dim_ptr[] = idx

# ===----------------------------------------------------------------------===#

@always_inline
def euclidean_distance(x1: Matrix, x2: Matrix) raises -> Float32:
    return math.sqrt(((x1 - x2) ** 2).sum())

@always_inline
def euclidean_distance(x1: Matrix, x2: Matrix, axis: Int) raises -> Matrix:
    return (((x1 - x2) ** 2).sum(axis)).sqrt()

@always_inline
def squared_euclidean_distance(x1: Matrix, x2: Matrix) raises -> Float32:
    return ((x1 - x2) ** 2).sum()

@always_inline
def squared_euclidean_distance(x1: Matrix, x2: Matrix, axis: Int) raises -> Matrix:
    return ((x1 - x2) ** 2).sum(axis)

@always_inline
def manhattan_distance(x1: Matrix, x2: Matrix) raises -> Float32:
    return (x1 - x2).abs().sum()

@always_inline
def manhattan_distance(x1: Matrix, x2: Matrix, axis: Int) raises -> Matrix:
    return (x1 - x2).abs().sum(axis)

@always_inline
def add[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return a + b

@always_inline
def sub[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return a - b

@always_inline
def mul[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return a * b

@always_inline
def div[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return a / b

@always_inline
def eq[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
    return a.eq(b)

@always_inline
def ne[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
    return a.ne(b)

@always_inline
def gt[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
    return a.gt(b)

@always_inline
def ge[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
    return a.ge(b)

@always_inline
def lt[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
    return a.lt(b)

@always_inline
def le[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
    return a.le(b)

@always_inline
def partial_simd_load[width: Int](data: UnsafePointer[Float32, MutAnyOrigin], offset: Int, size: Int) -> SIMD[DType.float32, width]:
    var nelts = size - offset
    if nelts >= width:
        return data.load[width=width](offset)
    var point = data + offset
    var simd = SIMD[DType.float32, width]()
    for i in range(0, nelts):
        simd[i] = point[i]
    return simd

@always_inline
def sigmoid(z: Matrix) raises -> Matrix:
    var z_exp = z.exp()
    return z.where(z >= 0,
                    1 / (1 + (-z).exp()),
                    z_exp / (1 + z_exp))

@always_inline
def normal_distr(x: Matrix, mean: Matrix, _var: Matrix) raises -> Matrix:
    return (-((x - mean) ** 2) / (2 * _var)).exp() / (2 * math.pi * _var).sqrt()

@always_inline
def unit_step(z: Matrix) -> Matrix:
    return z.where(z >= 0.0, 1.0, 0.0)

@always_inline
def sign(z: Matrix) -> Matrix:
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
        def p(i: Int):
            if z.data[i] > 0.0:
                mat.data[i] = 1.0
            elif z.data[i] < 0.0:
                mat.data[i] = -1.0
            else:
                mat.data[i] = 0.0
        parallelize[p](mat.size)
    return mat^

@always_inline
def mse(y: Matrix, y_pred: Matrix) raises -> Float32:
    """Mean Squared Error.

    Returns:
        The error.
    """
    return ((y - y_pred) ** 2).mean()

@always_inline
def cross_entropy(y: Matrix, y_pred: Matrix) raises -> Float32:
    """Binary Cross Entropy.

    Returns:
        The loss.
    """
    return -(y.ele_mul((y_pred + 1e-15).log()) + (1.0 - y).ele_mul((1.0 - y_pred + 1e-15).log())).mean()

def r2_score(y: Matrix, y_pred: Matrix) raises -> Float32:
    """Coefficient of determination.

    Returns:
        The score.
    """
    return 1.0 - (((y_pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())

def accuracy_score(y: Matrix, y_pred: Matrix) raises -> Float32:
    """Accuracy classification score.

    Returns:
        The score.
    """
    var correct_count = 0
    var y_data = y.data
    var y_pred_data = y_pred.data
    @parameter
    def compare[simd_width: Int](idx: Int) unified {mut}:
        correct_count += y_data.load[width=simd_width](idx).eq(y_pred_data.load[width=simd_width](idx)).reduce_bit_count()
    vectorize[y_pred.simd_width](len(y), compare)
    return Float32(correct_count) / Float32(len(y))

@always_inline
def entropy(y: Matrix, weights: Matrix, size: Float32) raises -> Float32:
    var histogram = y.bincount() if weights.size == 0 else y.bincount(weights)
    var _sum: Float32 = 0.0
    for i in range(len(histogram)):
        var p = Float32(histogram[i]) / size
        if p > 0 and p != 1.0:
            _sum += p * math.log2(p)
    return -_sum

@always_inline
def entropy_precompute(size: Float32, histogram: List[Int]) raises -> Float32:
    var _sum: Float32 = 0.0
    for i in range(len(histogram)):
        var p = Float32(histogram[i]) / size
        if p > 0 and p != 1.0:
            _sum += p * math.log2(p)
    return -_sum

@always_inline
def gini(y: Matrix, weights: Matrix, size: Float32) raises -> Float32:
    var histogram = y.bincount() if weights.size == 0 else y.bincount(weights)
    var _sum: Float32 = 0.0
    for i in range(len(histogram)):
        _sum += (Float32(histogram[i]) / size) ** 2
    return 1 - _sum

@always_inline
def gini_precompute(size: Float32, histogram: List[Int]) raises -> Float32:
    var _sum: Float32 = 0.0
    for i in range(len(histogram)):
        _sum += (Float32(histogram[i]) / size) ** 2
    return 1 - _sum

@always_inline
def mse_loss(y: Matrix, weights: Matrix, size: Float32) raises -> Float32:
    if len(y) == 0:
        return 0.0
    if weights.size == 0:
        return ((y - y.mean()) ** 2).mean()
    return ((y - y.mean_weighted(weights, size)) ** 2).mean_weighted(weights, size)

@always_inline
def mse_loss_precompute(size: Float32, sum: Float32, sum_sq: Float32) raises -> Float32:
    if size == 0:
        return 0.0
    return sum_sq / size - (sum / size) ** 2


@always_inline
def mse_g(true: Matrix, score: Matrix) raises -> Matrix:
    return score - true
@always_inline
def mse_h(score: Matrix) raises -> Matrix:
    return Matrix.ones(score.height, 1, order=score.order)

@always_inline
def log_g(true: Matrix, score: Matrix) raises -> Matrix:
    return sigmoid(score) - true
@always_inline
def log_h(score: Matrix) raises -> Matrix:
    var pred = sigmoid(score)
    return pred.ele_mul(1 - pred)

@always_inline
def softmax_link(score: Matrix) raises -> Matrix:
    var exp_score = (score - score.max(axis=1)).exp()  # for stability
    return exp_score / exp_score.sum(axis=1)
@always_inline
def softmax_g(true: Matrix, score: Matrix) raises -> Matrix:
    var g = softmax_link(score)
    g.set_per_row(true, g.get_per_row(true) - 1)  # derivative of softmax + CE
    return g^
@always_inline
def softmax_h(score: Matrix) raises -> Matrix:
    var prob = softmax_link(score)
    return prob.ele_mul(1 - prob)


@always_inline
def findInterval(intervals: List[Tuple[Float32, Float32]], x: Float32) -> Int:
    var left = 0
    var right = len(intervals) - 1

    while left <= right:
        var mid = left + (right - left) // 2

        if x < intervals[mid][0]:
            right = mid - 1
        elif x >= intervals[mid][1]:
            left = mid + 1
        else:
            return mid  # x is within the interval

    return -1  # not found

@always_inline
def fill_indices(N: Int) raises -> UnsafePointer[Scalar[DType.int], MutExternalOrigin]:
    """Generates indices from 0 to N.

    Returns:
        The pointer to indices.
    """
    var indices = alloc[Scalar[DType.int]](N)
    @parameter
    def fill_indices_iota[width: Int, rank: Int, alignment: Int = 1](offset: IndexList[rank]):
        indices.store(offset[0], math.iota[DType.int, width](Scalar[DType.int](offset[0])))

    elementwise[fill_indices_iota, simd_width_of[DType.int](), target="cpu"](
        N
    )
    return indices

@always_inline
def fill_indices_list(N: Int) raises -> List[Scalar[DType.int]]:
    """Generates indices from 0 to N.

    Returns:
        The list of indices.
    """
    var indices = alloc[Scalar[DType.int]](N)
    @parameter
    def fill_indices_iota[width: Int, rank: Int, alignment: Int = 1](offset: IndexList[rank]):
        indices.store(offset[0], math.iota[DType.int, width](Scalar[DType.int](offset[0])))

    elementwise[fill_indices_iota, simd_width_of[DType.int](), target="cpu"](
        N
    )
    var list = List[Scalar[DType.int]](unsafe_uninit_length=N)
    list._data = NonNullUnsafePointer(unsafe_from_nullable=indices)
    return list^

@always_inline
def cast[src: DType, des: DType, width: Int](data: UnsafePointer[Scalar[src], MutAnyOrigin], size: Int) -> UnsafePointer[Scalar[des], MutExternalOrigin]:
    var ptr = alloc[Scalar[des]](size)
    if size < 262144:
        @parameter
        def matrix_vectorize[simd_width: Int](idx: Int) unified {mut}:
            ptr.store(idx, data.load[width=simd_width](idx).cast[des]())
        vectorize[width](size, matrix_vectorize)
    else:
        var n_vects = Int(math.ceil(size / width))
        @parameter
        def matrix_vectorize_parallelize(i: Int):
            var idx = i * width
            ptr.store(idx, data.load[width=width](idx).cast[des]())
        parallelize[matrix_vectorize_parallelize](n_vects)
    return ptr

def ids_to_numpy(list: List[Int]) raises -> PythonObject:
    """Converts list of indices to numpy array.

    Returns:
        The numpy array.
    """
    var np = Python.import_module("numpy")
    var np_arr = np.empty(len(list), dtype='int')
    memcpy(dest=np_arr.__array_interface__['data'][0].unsafe_get_as_pointer[DType.int](), src=list._data.bitcast[Scalar[DType.int]](), count=len(list))
    return np_arr^

def ids_to_numpy(list: List[Scalar[DType.int]]) raises -> PythonObject:
    """Converts list of indices to numpy array.

    Returns:
        The numpy array.
    """
    var np = Python.import_module("numpy")
    var np_arr = np.empty(len(list), dtype='int')
    memcpy(dest=np_arr.__array_interface__['data'][0].unsafe_get_as_pointer[DType.int](), src=list._data.bitcast[Scalar[DType.int]](), count=len(list))
    return np_arr^

def cartesian_product(lists: List[List[String]]) -> List[List[String]]:
    var result = List[List[String]]()
    if not lists:
        result.append(List[String]())
        return result^

    var first = lists[0].copy()
    var rest = lists[1:].copy()
    var rest_product = cartesian_product(List[List[String]](rest))

    # Create the Cartesian product
    for item in first:
        for prod in rest_product:
            result.append([item] + prod.copy())

    return result^
