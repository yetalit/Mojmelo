from mojmelo.utils.Matrix import Matrix

struct Perceptron:
    var lr: Float32
    var n_iters: Int
    var weights: Matrix
    var bias: Float32

    fn __init__(inout self, learning_rate: Float32 = 0.01, n_iters: Int = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = Matrix(0, 0)
        self.bias = 0.0

    fn fit(inout self, X: Matrix, y: Matrix) raises:
        # init parameters
        self.weights = Matrix.zeros(X.width, 1)
        self.bias = 0.0

        for _ in range(self.n_iters):
            for i in range(X.height):
                # Perceptron update rule
                var update = self.lr * (y.data[i] - self.predict(X[i])[0, 0])
                
                self.weights += update * X[i].reshape(X.width, 1)
                self.bias += update

    fn predict(self, X: Matrix) raises -> Matrix:
        var y_predicted: Matrix = X * self.weights + self.bias
        # Unit Step function activation
        for i in range(y_predicted.size):
            y_predicted.data[i] = 1.0 if y_predicted.data[i] >= 0.0 else 0.0
        return y_predicted^
