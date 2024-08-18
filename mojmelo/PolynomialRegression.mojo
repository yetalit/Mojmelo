from mojmelo.utils.Matrix import Matrix

struct PolyRegression:
    var degree: Int
    var lr: Float32
    var n_iters: Int
    var weights: Matrix
    var bias: Float32

    fn __init__(inout self, degree: Int = 2, learning_rate: Float32 = 0.01, n_iters: Int = 1000):
        self.degree = degree
        self.lr = learning_rate
        self.n_iters = n_iters
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
        self.weights = Matrix.zeros(X.width, self.degree)
        self.bias = 0.0
        # gradient descent
        for _ in range(self.n_iters):
            var y_predicted: Matrix = X * self.weights['', 0] + self.bias
            for i in range(1, self.degree):
                y_predicted += X_poly[i - 1] * self.weights['', i]
            # compute gradients and update parameters
            self.weights['', 0] -= self.lr * ((X.T() * (y_predicted - y)) / X.height)
            self.bias -= self.lr * ((y_predicted - y).sum() / X.height)
            for i in range(1, self.degree):
                self.weights['', i] -= self.lr * ((X_poly[i - 1].T() * (y_predicted - y)) / X.height)

    fn predict(self, X: Matrix) raises -> Matrix:
        var X_poly = self._polynomial_features(X)
        var y_predicted: Matrix = X * self.weights['', 0] + self.bias
        for i in range(1, self.degree):
            y_predicted += X_poly[i - 1] * self.weights['', i]
        return y_predicted^
