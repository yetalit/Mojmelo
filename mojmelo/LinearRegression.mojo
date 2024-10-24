from mojmelo.utils.Matrix import Matrix

struct LinearRegression:
    var lr: Float32
    var n_iters: Int
    var tol: Float32
    var weights: Matrix
    var bias: Float32

    fn __init__(inout self, learning_rate: Float32 = 0.001, n_iters: Int = 1000, tol: Float32 = 0.0):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.tol = tol
        self.weights = Matrix(0, 0)
        self.bias = 0.0

    fn fit(inout self, X: Matrix, y: Matrix) raises:
        # init parameters
        self.weights = Matrix.zeros(X.width, 1)
        self.bias = 0.0
        
        var X_T = X.T()
        # gradient descent
        for _ in range(self.n_iters):
            var y_predicted = X * self.weights + self.bias
            # compute gradients and update parameters
            var dw = ((X_T * (y_predicted - y)) / X.height)
            var db = ((y_predicted - y).sum() / X.height)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if self.tol > 0.0 and dw.norm() <= self.tol and abs(db) <= self.tol:
                break

    fn predict(self, X: Matrix) raises -> Matrix:
        return X * self.weights + self.bias
