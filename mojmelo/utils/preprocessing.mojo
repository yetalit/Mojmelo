from mojmelo.utils.Matrix import Matrix
from python import Python, PythonObject
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

fn train_test_split(X: Matrix, y: Matrix, test_size: Float16 = 0.5, random_state: Int = time.perf_counter_ns()) raises -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    var ids = Matrix.rand_choice(X.height, X.height, False, random_state)
    var split_i = int(X.height - (test_size * X.height))
    return X[ids[:split_i]], X[ids[split_i:]], y[ids[:split_i]], y[ids[split_i:]]

@value
struct SplittedPO:
    var train: PythonObject
    var test: PythonObject
    fn __init__(inout self, train: PythonObject, test: PythonObject):
        self.train = train
        self.test = test

fn train_test_split(X: Matrix, y: PythonObject, test_size: Float16 = 0.5, random_state: Int = time.perf_counter_ns()) raises -> Tuple[Matrix, Matrix, SplittedPO]:
    var np = Python.import_module("numpy")
    var ids = Matrix.rand_choice(X.height, X.height, False, random_state)
    var split_i = int(X.height - (test_size * X.height))
    var y_train = np.empty(split_i, dtype='object')
    var y_test = np.empty(X.height - split_i, dtype='object')
    for i in range(split_i):
        y_train[i] = y[ids[i]]
    for i in range(split_i, X.height):
        y_test[i - split_i] = y[ids[i]]
    return X[ids[:split_i]], X[ids[split_i:]], SplittedPO(y_train, y_test)
