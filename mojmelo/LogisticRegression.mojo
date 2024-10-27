from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import sigmoid, sign, cross_entropy
import math
import time

struct LogisticRegression:
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

    fn __init__(inout self, learning_rate: Float32 = 0.001, n_iters: Int = 1000, penalty: String = 'l2', reg_alpha: Float32 = 0.0, l1_ratio: Float32 = -1.0,
                tol: Float32 = 0.0, batch_size: Int = 0, random_state: Int = -1):
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

    fn fit(inout self, X: Matrix, y: Matrix) raises:
        # init parameters
        self.weights = Matrix.zeros(X.width, 1)
        self.bias = 0.0

        var X_T = Matrix(0, 0)
        if self.batch_size <= 0:
            X_T = X.T()

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
            # approximate y with sigmoid function
            var y_predicted = sigmoid(X * self.weights + self.bias)

            if self.tol > 0.0:
                var cost = cross_entropy(y, y_predicted)
                if abs(prev_cost - cost) <= self.tol:
                    break
                prev_cost = cost

            if self.batch_size > 0:
                var ids: List[Int]
                if self.random_state != -1:
                    ids = Matrix.rand_choice(X.height, X.height, False, self.random_state)
                else:
                    ids = Matrix.rand_choice(X.height, X.height, False)
                # Iterate over mini-batches
                for start_idx in range(0, X.height, self.batch_size):
                    var end_idx = min(start_idx + self.batch_size, X.height)
                    var batch_indices = ids[start_idx:end_idx]
                    
                    var X_batch = X[batch_indices]
                    var y_batch = y[batch_indices]

                    var y_batch_predicted = sigmoid(X_batch * self.weights + self.bias)
                    # compute gradients and update parameters
                    var dw = ((X_batch.T() * (y_batch_predicted - y_batch)) / len(y_batch))
                    if l1_lambda > 0.0:
                        # L1 regularization
                        dw += l1_lambda * sign(self.weights)
                    if l2_lambda > 0.0:
                        # L2 regularization
                        dw += l2_lambda * self.weights
                    var db = ((y_batch_predicted - y_batch).sum() / len(y_batch))
                    self.weights -= self.lr * dw
                    self.bias -= self.lr * db
            else:
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

    fn predict(self, X: Matrix) raises -> Matrix:
        var y_predicted = sigmoid(X * self.weights + self.bias)
        return y_predicted.where(y_predicted > 0.5, 1.0, 0.0)
