from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import sign
import math

struct PolyRegression:
    var degree: Int
    var lr: Float32
    var n_iters: Int
    var penalty: String
    var reg_alpha: Float32
    var l1_ratio: Float32
    var tol: Float32
    var weights: Matrix
    var bias: Float32

    fn __init__(inout self, degree: Int = 2, learning_rate: Float32 = 0.01, n_iters: Int = 1000, penalty: String = 'l2', reg_alpha: Float32 = 0.0, l1_ratio: Float32 = -1.0, tol: Float32 = 0.0):
        self.degree = degree
        self.lr = learning_rate
        self.n_iters = n_iters
        self.penalty = penalty.lower()
        self.reg_alpha = reg_alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
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

        var X_T = X.T()
        var X_poly_T = List[Matrix]()
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
        # gradient descent
        for _ in range(self.n_iters):
            var y_predicted = X * self.weights['', 0] + self.bias
            for i in range(1, self.degree):
                y_predicted += X_poly[i - 1] * self.weights['', i]
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

            if self.tol > 0.0 and dw.norm() <= self.tol and abs(db) <= self.tol:
                break

    fn predict(self, X: Matrix) raises -> Matrix:
        var X_poly = self._polynomial_features(X)
        var y_predicted: Matrix = X * self.weights['', 0] + self.bias
        for i in range(1, self.degree):
            y_predicted += X_poly[i - 1] * self.weights['', i]
        return y_predicted^
