from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import sign, polynomial_kernel, gaussian_kernel

struct SVM_Primal:
    var lr: Float32
    var lambda_param: Float32
    var n_iters: Int
    var weights: Matrix
    var bias: Float32

    fn __init__(inout self, learning_rate: Float32 = 0.001, lambda_param: Float32 = 0.01, n_iters: Int = 1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = Matrix(0, 0)
        self.bias = 0.0

    fn fit(inout self, X: Matrix, y: Matrix, class_zero: Bool = False) raises:
        if class_zero:
            self.fit(X, y.where(y <= 0.0, -1.0, 1.0))
            return

        self.weights = Matrix.zeros(X.width, 1)
        self.bias = 0.0

        for _ in range(self.n_iters):
            for i in range(X.height):
                if y.data[i] * ((X[i] * self.weights)[0, 0] - self.bias) >= 1:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (
                        2 * self.lambda_param * self.weights - (X[i] * y.data[i]).reshape(X.width, 1)
                    )
                    self.bias -= self.lr * y.data[i]

    fn predict(self, X: Matrix) raises -> Matrix:
        return sign(X * self.weights - self.bias)


struct SVM_Dual:
    var learning_rate: Float32
    var epoches: Int
    var C: Float32
    var kernel: fn(Tuple[Float32, Int], Matrix, Matrix) raises -> Matrix
    var degree: Int
    var sigma: Float32
    var k_params: Tuple[Float32, Int]
    var alpha: Matrix
    var bias: Float32
    var X: Matrix
    var y: Matrix

    fn __init__(inout self, learning_rate: Float32 = 0.001, n_iters: Int = 1000, C: Float32 = 1.0, kernel: String = 'poly', degree: Int = 2, sigma: Float32 = 0.1):
        self.learning_rate = learning_rate
        self.epoches = n_iters
        self.C = C
        if kernel.lower() == 'poly':
            self.k_params = (C, degree)
            self.kernel = polynomial_kernel
        else:
            self.k_params = (sigma, 0)
            self.kernel = gaussian_kernel
        self.degree = degree
        self.sigma = sigma
        self.alpha = Matrix(0, 0)
        self.bias = 0.0
        self.X = Matrix(0, 0)
        self.y = Matrix(0, 0)

    fn fit(inout self, X: Matrix, y: Matrix, class_zero: Bool = False) raises:
        self.X = X
        if class_zero:
            self.y = y.where(y <= 0.0, -1.0, 1.0)
        else:
            self.y = y
        self.alpha = Matrix.random(X.height, 1)
        self.bias = 0.0
        var ones = Matrix.ones(X.height, 1) 

        var y_mul_kernal = self.y.outer(self.y).ele_mul(self.kernel(self.k_params, X, X)) # yi yj K(xi, xj)

        var alpha_index = List[Int]()

        for i in range(self.epoches):
            self.alpha += self.learning_rate * (ones - y_mul_kernal * self.alpha) # α = α + η*(1 – yk ∑ αj yj K(xj, xk)) to maximize
            for j in range(self.alpha.size):
                # 0<α<C
                if self.alpha.data[j] > self.C:
                    self.alpha.data[j] = self.C
                elif self.alpha.data[j] < 0.0:
                    self.alpha.data[j] = 0.0
                else:
                    if i == self.epoches - 1:
                        alpha_index.append(j)

        # for intercept b, we will only consider α which are 0<α<C 
        self.bias = (self.y[alpha_index] - (self.alpha.ele_mul(self.y).reshape(1, self.y.height) * self.kernel(self.k_params, X, X[alpha_index])).reshape(len(alpha_index), 1)).mean() # avgC≤αi≤0{ yi – ∑αjyj K(xj, xi) }
    
    fn predict(self, X: Matrix) raises -> Matrix:
        return sign(self.alpha.ele_mul(self.y).reshape(1, self.y.height) * self.kernel(self.k_params, self.X, X) + self.bias).reshape(X.height, 1)
