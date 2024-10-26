from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import sign

struct LinearRegression:
    var lr: Float32
    var n_iters: Int
    var penalty: String
    var reg_alpha: Float32
    var l1_ratio: Float32
    var tol: Float32
    var weights: Matrix
    var bias: Float32

    fn __init__(inout self, learning_rate: Float32 = 0.001, n_iters: Int = 1000, penalty: String = 'l2', reg_alpha: Float32 = 0.0, l1_ratio: Float32 = -1.0, tol: Float32 = 0.0):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.penalty = penalty.lower()
        self.reg_alpha = reg_alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.weights = Matrix(0, 0)
        self.bias = 0.0

    fn fit(inout self, X: Matrix, y: Matrix) raises:
        # init parameters
        self.weights = Matrix.zeros(X.width, 1)
        self.bias = 0.0
        
        var X_T = X.T()

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
            var y_predicted = X * self.weights + self.bias
            # compute gradients and update parameters
            var dw = ((X_T * (y_predicted - y)) / X.height)
            if l1_lambda > 0.0:
                # L1 regularization
                dw += l1_lambda * sign(self.weights)
            if l2_lambda > 0.0:
                # L2 regularization
                dw += l2_lambda * self.weights
            var db = ((y_predicted - y).sum() / X.height)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if self.tol > 0.0 and dw.norm() <= self.tol and abs(db) <= self.tol:
                break

    fn predict(self, X: Matrix) raises -> Matrix:
        return X * self.weights + self.bias
