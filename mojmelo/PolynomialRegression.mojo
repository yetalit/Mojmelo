from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CV, sign, mse
from collections import Dict
import math
import time

@value
struct PolyRegression(CV):
    var degree: Int
    var lr: Float32
    var n_iters: Int
    var penalty: String
    var reg_alpha: Float32
    var l1_ratio: Float32
    var tol: Float32
    var batch_size: Int
    var random_state: Int
    var weights: Matrix
    var bias: Float32

    fn __init__(inout self, degree: Int = 2, learning_rate: Float32 = 0.01, n_iters: Int = 1000, penalty: String = 'l2', reg_alpha: Float32 = 0.0, l1_ratio: Float32 = -1.0,
                tol: Float32 = 0.0, batch_size: Int = 0, random_state: Int = -1):
        self.degree = degree
        self.lr = learning_rate
        self.n_iters = n_iters
        self.penalty = penalty.lower()
        self.reg_alpha = reg_alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights = Matrix(0, 0)
        self.bias = 0.0

    fn _polynomial_features(self, X: Matrix) -> List[Matrix]:
        var X_poly = List[Matrix]()
        for d in range(2, self.degree + 1):
            X_poly.append(X ** d)
        return X_poly^

    fn fit(inout self, X: Matrix, y: Matrix) raises:
        var X_poly = self._polynomial_features(X)
        # init parameters
        self.weights = Matrix.zeros(X.width, self.degree, order='f')
        self.bias = 0.0

        var X_T = Matrix(0, 0)
        var X_poly_T = List[Matrix]()
        if self.batch_size <= 0:
            X_T = X.T()
            for i in range(1, self.degree):
                X_poly_T.append(X_poly[i - 1].T())

        var l1_lambda = self.reg_alpha
        var l2_lambda = self.reg_alpha
        if self.l1_ratio >= 0.0:
            # Elastic net regularization
            l1_lambda *= self.l1_ratio
            l2_lambda *= 1.0 - self.l1_ratio
        else:
            if self.penalty == 'l2':
                l1_lambda = 0.0
            else:
                l2_lambda = 0.0

        var prev_cost = math.inf[DType.float32]()
        # gradient descent
        for _ in range(self.n_iters):
            var y_predicted = X * self.weights['', 0] + self.bias
            for i in range(1, self.degree):
                y_predicted += X_poly[i - 1] * self.weights['', i]

            if self.tol > 0.0:
                var cost = mse(y, y_predicted)
                if abs(prev_cost - cost) <= self.tol:
                    break
                prev_cost = cost
            
            if self.batch_size > 0:
                var ids: List[Int]
                if self.random_state != -1:
                    ids = Matrix.rand_choice(X.height, X.height, False, self.random_state)
                else:
                    ids = Matrix.rand_choice(X.height, X.height, False)
                var dw = Matrix(X.width, self.degree, order='f')
                # Iterate over mini-batches
                for start_idx in range(0, X.height, self.batch_size):
                    var end_idx = min(start_idx + self.batch_size, X.height)
                    var batch_indices = ids[start_idx:end_idx]
                    
                    var X_batch = X[batch_indices]
                    var X_poly_batch = List[Matrix]()
                    for i in range(1, self.degree):
                        X_poly_batch.append(X_poly[i - 1][batch_indices])
                    var y_batch = y[batch_indices]

                    var y_batch_predicted = X_batch * self.weights['', 0] + self.bias
                    for i in range(1, self.degree):
                        y_batch_predicted += X_poly[i - 1][batch_indices] * self.weights['', i]
                    # compute gradients and update parameters
                    dw['', 0] = (X_batch.T() * (y_batch_predicted - y_batch)) / len(y_batch)
                    for i in range(1, self.degree):
                        dw['', i] = (X_poly_batch[i - 1].T() * (y_batch_predicted - y_batch)) / len(y_batch)
                    if l1_lambda > 0.0:
                        # L1 regularization
                        dw += l1_lambda * sign(self.weights)
                    if l2_lambda > 0.0:
                        # L2 regularization
                        dw += l2_lambda * self.weights
                    var db = (y_batch_predicted - y_batch).sum() / len(y_batch)
                    self.weights['', 0] -= self.lr * dw['', 0]
                    self.bias -= self.lr * db
                    for i in range(1, self.degree):
                        self.weights['', i] -= self.lr * dw['', i]
            else:
                # compute gradients and update parameters
                var dw = Matrix(X.width, self.degree, order='f')
                dw['', 0] = ((X_T * (y_predicted - y)) / X.height)
                for i in range(1, self.degree):
                    dw['', i] = ((X_poly_T[i - 1] * (y_predicted - y)) / X.height)
                if l1_lambda > 0.0:
                    # L1 regularization
                    dw += l1_lambda * sign(self.weights)
                if l2_lambda > 0.0:
                    # L2 regularization
                    dw += l2_lambda * self.weights
                var db = ((y_predicted - y).sum() / X.height)
                self.weights['', 0] -= self.lr * dw['', 0]
                self.bias -= self.lr * db
                for i in range(1, self.degree):
                    self.weights['', i] -= self.lr * dw['', i]

    fn predict(self, X: Matrix) raises -> Matrix:
        var X_poly = self._polynomial_features(X)
        var y_predicted: Matrix = X * self.weights['', 0] + self.bias
        for i in range(1, self.degree):
            y_predicted += X_poly[i - 1] * self.weights['', i]
        return y_predicted^

    fn set_param(inout self, p_name: String, p_val: String) raises:
        if p_name == 'degree':
            self.degree = atol(p_val)
        elif p_name == 'learning_rate':
            self.lr = atof(p_val).cast[DType.float32]()
        elif p_name == 'n_iters':
            self.n_iters = atol(p_val)
        elif p_name == 'penalty':
            self.penalty = p_val
        elif p_name == 'reg_alpha':
            self.reg_alpha = atof(p_val).cast[DType.float32]()
        elif p_name == 'l1_ratio':
            self.l1_ratio = atof(p_val).cast[DType.float32]()
        elif p_name == 'tol':
            self.tol = atof(p_val).cast[DType.float32]()
        elif p_name == 'batch_size':
            self.batch_size = atol(p_val)
        elif p_name == 'random_state':
            self.random_state = atol(p_val)

    fn set_params_from_dict(inout self, params: Dict[String, String]) raises:
        for key in params.keys():
            self.set_param(key[], params[key[]])
