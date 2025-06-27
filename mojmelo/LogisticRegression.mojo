from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CVM, sigmoid, sign, cross_entropy
import math
import time

struct LogisticRegression(CVM):
    var lr: Float32
    var n_iters: Int
    var method: String
    var penalty: String
    var reg_alpha: Float32
    var l1_ratio: Float32
    var tol: Float32
    var batch_size: Int
    var random_state: Int
    var weights: Matrix
    var bias: Float32

    fn __init__(out self, learning_rate: Float32 = 0.001, n_iters: Int = 1000, method: String = 'gradient', penalty: String = 'l2', reg_alpha: Float32 = 0.0, l1_ratio: Float32 = -1.0,
                tol: Float32 = 0.0, batch_size: Int = 0, random_state: Int = -1):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.method = method.lower()
        self.penalty = penalty.lower()
        self.reg_alpha = reg_alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights = Matrix(0, 0)
        self.bias = 0.0

    fn fit(mut self, X: Matrix, y: Matrix) raises:
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
        var num_b_iters = X.height // self.batch_size if self.batch_size > 0 else 0
        var _reg = (1e-5 + l2_lambda) * Matrix.eye(X.width)
        for _ in range(self.n_iters):
            if self.batch_size > 0:
                var ids: List[Int]
                if self.random_state != -1:
                    ids = Matrix.rand_choice(X.height, X.height, False, self.random_state)
                else:
                    ids = Matrix.rand_choice(X.height, X.height, False)
                var cost: Float32 = 0.0
                # Iterate over mini-batches
                for start_idx in range(0, X.height, self.batch_size):
                    var batch_indices = ids[start_idx:start_idx + self.batch_size]
                    
                    var X_batch = X[batch_indices]
                    var y_batch = y[batch_indices]

                    var y_batch_predicted = sigmoid(X_batch * self.weights + self.bias)
                    if self.tol > 0.0:
                        cost += cross_entropy(y_batch, y_batch_predicted) / num_b_iters
                    var dw = (X_batch.T() * (y_batch_predicted - y_batch)) / len(y_batch)
                    if self.method == 'newton':
                        var H = (X_batch.T() * X_batch.ele_mul(y_batch_predicted.ele_mul(1.0 - y_batch_predicted))) / len(y_batch)
                        # Add regularization to Hessian for L2 only
                        # Update weights using Newton's method
                        self.weights -= Matrix.solve((H + _reg), dw)
                    else:
                        # gradient descent
                        if l1_lambda > 0.0:
                            # L1 regularization
                            dw += l1_lambda * sign(self.weights)
                        if l2_lambda > 0.0:
                            # L2 regularization
                            dw += l2_lambda * self.weights
                        
                        self.weights -= self.lr * dw
                    
                    var db = ((y_batch_predicted - y_batch).sum() / len(y_batch))
                    self.bias -= self.lr * db
                if self.tol > 0.0:
                    if abs(prev_cost - cost) <= self.tol:
                        break
                    prev_cost = cost
            else:
                # approximate y with sigmoid function
                var y_predicted = sigmoid(X * self.weights + self.bias)

                if self.tol > 0.0:
                    var cost = cross_entropy(y, y_predicted)
                    if abs(prev_cost - cost) <= self.tol:
                        break
                    prev_cost = cost

                var dw = ((X_T * (y_predicted - y)) / X.height)
                if self.method == 'newton':
                    var H = (X_T * X.ele_mul(y_predicted.ele_mul(1.0 - y_predicted))) / X.height
                    # Add regularization to Hessian for L2 only
                    # Update weights using Newton's method
                    self.weights -= Matrix.solve((H + _reg), dw)
                else:
                    # gradient descent
                    if l1_lambda > 0.0:
                        # L1 regularization
                        dw += l1_lambda * sign(self.weights)
                    if l2_lambda > 0.0:
                        # L2 regularization
                        dw += l2_lambda * self.weights
                    
                    self.weights -= self.lr * dw
                
                var db = ((y_predicted - y).sum() / X.height)
                self.bias -= self.lr * db

    fn predict(self, X: Matrix) raises -> Matrix:
        var y_predicted = sigmoid(X * self.weights + self.bias)
        return y_predicted.where(y_predicted > 0.5, 1.0, 0.0)

    fn __init__(out self, params: Dict[String, String]) raises:
        if 'learning_rate' in params:
            self.lr = atof(String(params['learning_rate'])).cast[DType.float32]()
        else:
            self.lr = 0.01
        if 'n_iters' in params:
            self.n_iters = atol(String(params['n_iters']))
        else:
            self.n_iters = 1000
        if 'method' in params:
            self.method = params['method']
        else:
            self.method = 'gradient'
        if 'penalty' in params:
            self.penalty = params['penalty']
        else:
            self.penalty = 'l2'
        if 'reg_alpha' in params:
            self.reg_alpha = atof(String(params['reg_alpha'])).cast[DType.float32]()
        else:
            self.reg_alpha = 0.0
        if 'l1_ratio' in params:
            self.l1_ratio = atof(String(params['l1_ratio'])).cast[DType.float32]()
        else:
            self.l1_ratio = -1.0
        if 'tol' in params:
            self.tol = atof(String(params['tol'])).cast[DType.float32]()
        else:
            self.tol = 0.0
        if 'batch_size' in params:
            self.batch_size = atol(String(params['batch_size']))
        else:
            self.batch_size = 0
        if 'random_state' in params:
            self.random_state = atol(String(params['random_state']))
        else:
            self.random_state = -1
        self.weights = Matrix(0, 0)
        self.bias = 0.0
