from std.memory import unsafe_memcpy, Layout
from std import math
from mojmelo.linalg.Matrix import Matrix
from std.python import Python, PythonObject
from std.algorithm import vectorize
from mojmelo.utils.algorithm import parallelize
from std.sys import simd_width_of
from mojmelo.linalg.utils import sub, mul, div

# Cross Validation trait
trait CV(Deinitable):
    def __init__(out self, params: Dict[String, String]) raises:
        ...
    def fit(mut self, X: Matrix, y: Matrix) raises:
        ...
    def predict(mut self, X: Matrix) raises -> Matrix:
        ...


comptime MODEL_IDS: Array[String, 14] = ['',
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
    'PCA',
    'BernoulliNB'
                                        ]

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
def sigmoid(z: Matrix) raises -> Matrix:
    var z_exp = z.exp()
    return z.where(z >= 0,
                    1 / (1 + (-z).exp()),
                    z_exp._elemwise_matrix[div](1 + z_exp))

@always_inline
def normal_distr(x: Matrix, mean: Matrix, _var: Matrix) raises -> Matrix:
    return (-((x - mean) ** 2) / (2 * _var)).exp() / (2.0 * math.pi * _var).sqrt()

@always_inline
def unit_step(z: Matrix) -> Matrix:
    return z.where(z >= 0.0, 1.0, 0.0)

@always_inline
def sign(z: Matrix) -> Matrix:
    var mat = Matrix(z.height, z.width, order= z.order)
    if mat.size < 147456:
        for i in range(mat.size):
            if z.data[unsafe_offset=i] > 0.0:
                mat.data[unsafe_offset=i] = 1.0
            elif z.data[unsafe_offset=i] < 0.0:
                mat.data[unsafe_offset=i] = -1.0
            else:
                mat.data[unsafe_offset=i] = 0.0
    else:
        @__parameter
        def p(i: Int):
            if z.data[unsafe_offset=i] > 0.0:
                mat.data[unsafe_offset=i] = 1.0
            elif z.data[unsafe_offset=i] < 0.0:
                mat.data[unsafe_offset=i] = -1.0
            else:
                mat.data[unsafe_offset=i] = 0.0
        parallelize[p](mat.size)
    return mat^

@always_inline
def mse(y: Matrix, y_pred: Matrix) raises -> Float32:
    """Mean Squared Error.

    Returns:
        The error.
    """
    return ((y._elemwise_matrix[sub](y_pred)) ** 2).mean()

@always_inline
def cross_entropy(y: Matrix, y_pred: Matrix) raises -> Float32:
    """Binary Cross Entropy.

    Returns:
        The loss.
    """
    return -(y._elemwise_matrix[mul]((y_pred + 1e-15).log()) + (1.0 - y)._elemwise_matrix[mul]((1.0 - y_pred + 1e-15).log())).mean()

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

    def compare[simd_width: Int](idx: Int) {mut correct_count, y, y_pred}:
        correct_count += y.data.unsafe_load[width=simd_width](idx).eq(y_pred.data.unsafe_load[width=simd_width](idx)).reduce_bit_count()
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
def fill_indices(N: Int) raises -> Pointer[Int, MutUntrackedOrigin]:
    """Generates indices from 0 to N.

    Returns:
        The pointer to indices.
    """
    var indices = alloc(Layout[Int](count=N)).unsafe_leak()

    def fill_indices_iota[width: Int](idx: Int) {imm}:
        indices.unsafe_store(idx, math.iota[DType.int, width](idx))

    vectorize[simd_width_of[DType.int]()](N, fill_indices_iota)
    return indices

@always_inline
def fill_indices_list(N: Int) raises -> List[Int]:
    """Generates indices from 0 to N.

    Returns:
        The list of indices.
    """
    var list = List[Int](unsafe_uninit_length=N)
    list._data = fill_indices(N)
    return list^

def ids_to_numpy(list: List[Int]) raises -> PythonObject:
    """Converts list of indices to numpy array.

    Returns:
        The numpy array.
    """
    var np = Python.import_module("numpy")
    var np_arr = np.empty(len(list), dtype='int')
    unsafe_memcpy(dest=np_arr.__array_interface__['data'][0].unsafe_get_as_pointer[DType.int](), src=list._data.unsafe_bitcast[Int](), count=len(list))
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
            result.append(List([item]) + prod.copy())

    return result^
