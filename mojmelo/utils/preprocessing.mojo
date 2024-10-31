from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CVM, CVP, cartesian_product, ids_to_numpy
from collections import Dict
from python import Python, PythonObject
import math
import time

fn normalize(data: Matrix, norm: String = 'l2') raises -> Tuple[Matrix, Matrix]:
    var z = Matrix(data.height, data.width, order= data.order)
    var values = Matrix(data.height, 1)
    if norm.lower() == 'l1':
        if data.height == 1 or data.width == 1:
            values.fill(data.abs().sum())
        else:
            for i in range(values.height):
                values.data[i] = data[i].abs().sum()
    else:
        if data.height == 1 or data.width == 1:
            values.fill(data.norm())
        else:
            for i in range(values.height):
                values.data[i] = data[i].norm()

    for i in range(z.height):
        if values.data[i] != 0.0:
            z[i] = data[i] / values.data[i]
        else:
            z[i].fill_zero()

    return z^, values^

fn normalize(data: Matrix, values: Matrix, norm: String = 'l2') raises -> Matrix:
    var z = Matrix(data.height, data.width, order= data.order)
    if norm.lower() == 'l1':
        if data.height == 1 or data.width == 1:
            values.fill(data.abs().sum())
        else:
            for i in range(values.height):
                values.data[i] = data[i].abs().sum()
    else:
        if data.height == 1 or data.width == 1:
            values.fill(data.norm())
        else:
            for i in range(values.height):
                values.data[i] = data[i].norm()

    for i in range(z.height):
        if values.data[i] != 0.0:
            z[i] = data[i] / values.data[i]
        else:
            z[i].fill_zero()

    return z^

fn inv_normalize(z: Matrix, values: Matrix) raises -> Matrix:
    return z.ele_mul(values)

fn MinMaxScaler(data: Matrix) raises -> Tuple[Matrix, Matrix, Matrix]:
    var x_min = data.min(0)
    var x_max = data.max(0)
    # normalize data
    var div = x_max - x_min
    return (data - x_min) / div.where(div == 0.0, 1.0, div), x_min^, x_max^

fn MinMaxScaler(data: Matrix, x_min: Matrix, x_max: Matrix) raises -> Matrix:
    # normalize data
    var div = x_max - x_min
    return (data - x_min) / div.where(div == 0.0, 1.0, div)

fn inv_MinMaxScaler(z: Matrix, x_min: Matrix, x_max: Matrix) raises -> Matrix:
    var div = x_max - x_min
    return z.ele_mul(div.where(div == 0.0, 1.0, div)) + x_min

fn StandardScaler(data: Matrix) raises -> Tuple[Matrix, Matrix, Matrix]:
    var mu = data.mean_slow0()
    var sigma = data.std_slow(0, mu)
    # standardize data
    return (data - mu) / sigma.where(sigma == 0.0, 1.0, sigma), mu^, sigma^

fn StandardScaler(data: Matrix, mu: Matrix, sigma: Matrix) raises -> Matrix:
    # standardize data
    return (data - mu) / sigma.where(sigma == 0.0, 1.0, sigma)

fn inv_StandardScaler(z: Matrix, mu: Matrix, sigma: Matrix) raises -> Matrix:
    return z.ele_mul(sigma.where(sigma == 0.0, 1.0, sigma)) + mu

fn train_test_split(X: Matrix, y: Matrix, *, test_size: Float16 = 0.5, train_size: Float16 = 0.0) raises -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    var test_ratio = test_size if train_size <= 0.0 else 1.0 - train_size
    var ids = Matrix.rand_choice(X.height, X.height, False)
    var split_i = int(X.height - (test_ratio * X.height))
    return X[ids[:split_i]], X[ids[split_i:]], y[ids[:split_i]], y[ids[split_i:]]

fn train_test_split(X: Matrix, y: Matrix, *, random_state: Int, test_size: Float16 = 0.5, train_size: Float16 = 0.0) raises -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    var test_ratio = test_size if train_size <= 0.0 else 1.0 - train_size
    var ids = Matrix.rand_choice(X.height, X.height, False, random_state)
    var split_i = int(X.height - (test_ratio * X.height))
    return X[ids[:split_i]], X[ids[split_i:]], y[ids[:split_i]], y[ids[split_i:]]

@value
struct SplittedPO:
    var train: PythonObject
    var test: PythonObject
    fn __init__(inout self, train: PythonObject, test: PythonObject):
        self.train = train
        self.test = test

fn train_test_split(X: Matrix, y: PythonObject, *, test_size: Float16 = 0.5, train_size: Float16 = 0.0) raises -> Tuple[Matrix, Matrix, SplittedPO]:
    var test_ratio = test_size if train_size <= 0.0 else 1.0 - train_size
    var ids = Matrix.rand_choice(X.height, X.height, False)
    var split_i = int(X.height - (test_ratio * X.height))
    return X[ids[:split_i]], X[ids[split_i:]], SplittedPO(y[ids_to_numpy(ids[:split_i])], y[ids_to_numpy(ids[split_i:])])

fn train_test_split(X: Matrix, y: PythonObject, *, random_state: Int, test_size: Float16 = 0.5, train_size: Float16 = 0.0) raises -> Tuple[Matrix, Matrix, SplittedPO]:
    var test_ratio = test_size if train_size <= 0.0 else 1.0 - train_size
    var ids = Matrix.rand_choice(X.height, X.height, False, random_state)
    var split_i = int(X.height - (test_ratio * X.height))
    return X[ids[:split_i]], X[ids[split_i:]], SplittedPO(y[ids_to_numpy(ids[:split_i])], y[ids_to_numpy(ids[split_i:])])

fn KFold[m_type: CVM](inout model: m_type, X: Matrix, y: Matrix, scoring: fn(Matrix, Matrix) raises -> Float32, n_splits: Int = 5) raises -> Float32:
    var test_ratio = 1 / n_splits
    var ids = Matrix.rand_choice(X.height, X.height, False)
    var test_count = int(test_ratio * X.height)
    var start_of_test = 0
    var total_score: Float32 = 0.0
    for _ in range(n_splits):
        var end_of_test = min(start_of_test + test_count, X.height)
        model.fit(X[ids[end_of_test:] + ids[:start_of_test]], y[ids[end_of_test:] + ids[:start_of_test]])
        y_pred = model.predict(X[ids[start_of_test:end_of_test]])
        total_score += scoring(y[ids[start_of_test:end_of_test]], y_pred)
        start_of_test += test_count
    return total_score / n_splits

fn KFold[m_type: CVP](inout model: m_type, X: Matrix, y: PythonObject, scoring: fn(PythonObject, List[String]) raises -> Float32, n_splits: Int = 5) raises -> Float32:
    var test_ratio = 1 / n_splits
    var ids = Matrix.rand_choice(X.height, X.height, False)
    var test_count = int(test_ratio * X.height)
    var start_of_test = 0
    var total_score: Float32 = 0.0
    for _ in range(n_splits):
        var end_of_test = min(start_of_test + test_count, X.height)
        model.fit(X[ids[end_of_test:] + ids[:start_of_test]], y[ids_to_numpy(ids[end_of_test:] + ids[:start_of_test])])
        y_pred = model.predict(X[ids[start_of_test:end_of_test]])
        total_score += scoring(y[ids_to_numpy(ids[start_of_test:end_of_test])], y_pred)
        start_of_test += test_count
    return total_score / n_splits

fn GridSearchCV[m_type: CVM](X: Matrix, y: Matrix, param_grid: Dict[String, List[String]],
                            scoring: fn(Matrix, Matrix) raises -> Float32, neg_score: Bool = False, cv: Int = 5) raises -> Tuple[Dict[String, String], Float32]:
    var best_params = Dict[String, String]()
    var best_score = -math.inf[DType.float32]()
    var values = List[List[String]]()
    for i in range(len(param_grid)):
        values.append(List[String]())
        values[i] = param_grid._entries[i].value().value
    var combinations = cartesian_product(values)
    for comb in combinations:
        var current_params = Dict[String, String]()
        var j = 0
        for key in param_grid.keys():
            current_params[key[]] = comb[][j]
            j += 1
        var model = m_type(current_params)
        var score = KFold[m_type](model, X, y, scoring, cv)
        if neg_score:
            score *= -1
        if score > best_score:
            best_score = score
            best_params = current_params
    if neg_score:
        best_score *= -1
    return best_params^, best_score

fn GridSearchCV[m_type: CVP](X: Matrix, y: PythonObject, param_grid: Dict[String, List[String]],
                            scoring: fn(PythonObject, List[String]) raises -> Float32, neg_score: Bool = False, cv: Int = 5) raises -> Tuple[Dict[String, String], Float32]:
    var best_params = Dict[String, String]()
    var best_score = -math.inf[DType.float32]()
    var values = List[List[String]]()
    for i in range(len(param_grid)):
        values.append(List[String]())
        values[i] = param_grid._entries[i].value().value
    var combinations = cartesian_product(values)
    for comb in combinations:
        var current_params = Dict[String, String]()
        var j = 0
        for key in param_grid.keys():
            current_params[key[]] = comb[][j]
            j += 1
        var model = m_type(current_params)
        var score = KFold[m_type](model, X, y, scoring, cv)
        if neg_score:
            score *= -1
        if score > best_score:
            best_score = score
            best_params = current_params
    if neg_score:
        best_score *= -1
    return best_params^, best_score
