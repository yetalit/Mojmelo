from mojmelo.utils.Matrix import Matrix
from python import Python, PythonObject
import time

fn normalize(data: Matrix, norm: String = 'l2') raises -> Tuple[Matrix, Matrix]:
    var z = Matrix(data.height, data.width)
    var values = Matrix(data.height, 1)
    if norm.lower() == 'l1':
        if data.height == 1 or data.width == 1:
            var val = data.abs().sum()
            for i in range(values.height):
                values.data[i] = val
        else:
            for i in range(values.height):
                values.data[i] = data[i].abs().sum()
    else:
        if data.height == 1 or data.width == 1:
            var val = data.norm()
            for i in range(values.height):
                values.data[i] = val
        else:
            for i in range(values.height):
                values.data[i] = data[i].norm()

    for i in range(z.height):
        if values.data[i] != 0.0:
            z[i] = data[i] / values.data[i]
        else:
            z[i] = Matrix.zeros(1, z.width)

    return z^, values^

fn normalize(data: Matrix, values: Matrix, norm: String = 'l2') raises -> Matrix:
    var z = Matrix(data.height, data.width)
    if norm.lower() == 'l1':
        if data.height == 1 or data.width == 1:
            var val = data.abs().sum()
            for i in range(values.height):
                values.data[i] = val
        else:
            for i in range(values.height):
                values.data[i] = data[i].abs().sum()
    else:
        if data.height == 1 or data.width == 1:
            var val = data.norm()
            for i in range(values.height):
                values.data[i] = val
        else:
            for i in range(values.height):
                values.data[i] = data[i].norm()

    for i in range(z.height):
        if values.data[i] != 0.0:
            z[i] = data[i] / values.data[i]
        else:
            z[i] = Matrix.zeros(1, z.width)

    return z^

fn inv_normalize(z: Matrix, values: Matrix) raises -> Matrix:
    var data = Matrix(z.height, z.width)
    for i in range(data.height):
        data[i] *= values.data[i]
    return data^

fn MinMaxScaler(data: Matrix) raises -> Tuple[Matrix, Matrix, Matrix]:
    var x_min = data.min(0)
    var x_max = data.max(0)
    var x = Matrix(data.height, data.width)
    # normalize data
    var div = x_max - x_min
    for i in range(x.width):
        if div.data[i] != 0.0:
            x['', i] = (data['', i] - x_min.data[i]) / div.data[i]
        else:
            x['', i] = data['', i]
    return x^, x_min^, x_max^

fn MinMaxScaler(data: Matrix, x_min: Matrix, x_max: Matrix) raises -> Matrix:
    var x = Matrix(data.height, data.width)
    # normalize data
    var div = x_max - x_min
    for i in range(x.width):
        if div.data[i] != 0.0:
            x['', i] = (data['', i] - x_min.data[i]) / div.data[i]
        else:
            x['', i] = data['', i]
    return x^

fn inv_MinMaxScaler(z: Matrix, x_min: Matrix, x_max: Matrix) raises -> Matrix:
    var div = x_max - x_min
    var mat = z.ele_mul(div.where(div == 0.0, 1.0, div))
    for i in range(mat.width):
        mat['', i] += x_min.data[i]
    return mat^

fn StandardScaler(data: Matrix) raises -> Tuple[Matrix, Matrix, Matrix]:
    var mu = data.mean_slow0()
    var sigma = data.std_slow(0, mu)
    var x = Matrix(data.height, data.width)
    # standardize data
    for i in range(x.width):
        if sigma.data[i] != 0.0:
            x['', i] = (data['', i] - mu.data[i]) / sigma.data[i]
        else:
            x['', i] = data['', i]
    return x^, mu^, sigma^

fn StandardScaler(data: Matrix, mu: Matrix, sigma: Matrix) raises -> Matrix:
    var x = Matrix(data.height, data.width)
    # standardize data
    for i in range(x.width):
        if sigma.data[i] != 0.0:
            x['', i] = (data['', i] - mu.data[i]) / sigma.data[i]
        else:
            x['', i] = data['', i]
    return x^

fn inv_StandardScaler(z: Matrix, mu: Matrix, sigma: Matrix) raises -> Matrix:
    var mat = z.ele_mul(sigma.where(sigma == 0.0, 1.0, sigma))
    for i in range(mat.width):
        mat['', i] += mu.data[i]
    return mat^

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
