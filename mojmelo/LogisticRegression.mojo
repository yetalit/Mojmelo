from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import sigmoid

struct LogisticRegression:
    var lr: Float32
    var n_iters: Int
    var weights: Matrix
    var bias: Float32

    fn __init__(inout self, learning_rate: Float32 = 0.001, n_iters: Int = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = Matrix(0, 0)
        self.bias = 0.0

    fn fit(inout self, X: Matrix, y: Matrix) raises:
        # init parameters
        self.weights = Matrix.zeros(X.width, 1)
        self.bias = 0.0
        # gradient descent
        for _ in range(self.n_iters):
            # approximate y with sigmoid function
            var y_predicted: Matrix = sigmoid(X * self.weights + self.bias)
            # compute gradients and update parameters
            self.weights -= self.lr * ((X.T() * (y_predicted - y)) / X.height)
            self.bias -= self.lr * ((y_predicted - y).sum() / X.height)

    fn predict(self, X: Matrix) raises -> Matrix:
        var y_predicted: Matrix = sigmoid(X * self.weights + self.bias)
        for i in range(y_predicted.size):
            y_predicted.data[i] = 1.0 if y_predicted.data[i] > 0.5 else 0.0
        return y_predicted^
