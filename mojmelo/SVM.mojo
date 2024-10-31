from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CVM, sign, polynomial_kernel, gaussian_kernel
from collections import Dict

struct SVM_Primal(CVM):
    var lr: Float32
    var lambda_param: Float32
    var n_iters: Int
    var class_zero: Bool
    var weights: Matrix
    var bias: Float32

    fn __init__(inout self, learning_rate: Float32 = 0.001, lambda_param: Float32 = 0.01, n_iters: Int = 1000, class_zero: Bool = False):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.class_zero = class_zero
        self.weights = Matrix(0, 0)
        self.bias = 0.0

    fn _fit(inout self, X: Matrix, y: Matrix) raises:
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

    fn fit(inout self, X: Matrix, y: Matrix) raises:
        if self.class_zero:
            self._fit(X, y.where(y <= 0.0, -1.0, 1.0))
        else:
            self._fit(X, y)

    fn predict(self, X: Matrix) raises -> Matrix:
        if self.class_zero:
            var y_predicted = sign(X * self.weights - self.bias)
            return y_predicted.where(y_predicted < 0.0, 0.0, 1.0)
        return sign(X * self.weights - self.bias)

    fn __init__(inout self, params: Dict[String, String]) raises:
        if 'learning_rate' in params:
            self.lr = atof(params['learning_rate']).cast[DType.float32]()
        else:
            self.lr = 0.001
        if 'lambda_param' in params:
            self.lambda_param = atof(params['lambda_param']).cast[DType.float32]()
        else:
            self.lambda_param = 0.01
        if 'n_iters' in params:
            self.n_iters = atol(params['n_iters'])
        else:
            self.n_iters = 1000
        if 'class_zero' in params:
            if params['class_zero'] == 'True':
                self.class_zero = True
            else:
                self.class_zero = False
        else:
            self.class_zero = False
        self.weights = Matrix(0, 0)
        self.bias = 0.0


struct SVM_Dual(CVM):
    var lr: Float32
    var epoches: Int
    var C: Float32
    var kernel: String
    var kernel_func: fn(Tuple[Float32, Int], Matrix, Matrix) raises -> Matrix
    var degree: Int
    var gamma: Float32
    var class_zero: Bool
    var k_params: Tuple[Float32, Int]
    var alpha: Matrix
    var bias: Float32
    var X: Matrix
    var y: Matrix

    fn __init__(inout self, learning_rate: Float32 = 0.001, n_iters: Int = 1000, C: Float32 = 1.0, kernel: String = 'poly', degree: Int = 2, gamma: Float32 = -1.0, class_zero: Bool = False):
        self.lr = learning_rate
        self.epoches = n_iters
        self.C = C
        self.kernel = kernel.lower()
        if self.kernel == 'poly':
            self.k_params = (C, degree)
            self.kernel_func = polynomial_kernel
        else:
            self.k_params = (gamma, 0)
            self.kernel_func = gaussian_kernel
        self.degree = degree
        self.gamma = gamma
        self.class_zero = class_zero
        self.alpha = Matrix(0, 0)
        self.bias = 0.0
        self.X = Matrix(0, 0)
        self.y = Matrix(0, 0)

    fn fit(inout self, X: Matrix, y: Matrix) raises:
        self.X = X
        if self.kernel == 'rbf' and self.gamma < 0.0:
            self.gamma = 1.0 / (self.X.width * self.X._var())
            self.k_params = (self.gamma, 0)
        if self.class_zero:
            self.y = y.where(y <= 0.0, -1.0, 1.0)
        else:
            self.y = y
        self.alpha = Matrix.zeros(X.height, 1)
        self.bias = 0.0
        var ones = Matrix.ones(X.height, 1) 

        var y_mul_kernal = self.y.outer(self.y).ele_mul(self.kernel_func(self.k_params, X, X)) # yi yj K(xi, xj)

        var alpha_index = List[Int]()

        for i in range(self.epoches):
            self.alpha += self.lr * (ones - y_mul_kernal * self.alpha) # α = α + η*(1 – yk ∑ αj yj K(xj, xk)) to maximize
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
        self.bias = (self.y[alpha_index] - (self.alpha.ele_mul(self.y).reshape(1, self.y.height) * self.kernel_func(self.k_params, X, X[alpha_index])).reshape(len(alpha_index), 1)).mean() # avgC≤αi≤0{ yi – ∑αjyj K(xj, xi) }
    
    fn predict(self, X: Matrix) raises -> Matrix:
        if self.class_zero:
            var y_predicted = sign(self.alpha.ele_mul(self.y).reshape(1, self.y.height) * self.kernel_func(self.k_params, self.X, X) + self.bias).reshape(X.height, 1)
            return y_predicted.where(y_predicted < 0.0, 0.0, 1.0)
        return sign(self.alpha.ele_mul(self.y).reshape(1, self.y.height) * self.kernel_func(self.k_params, self.X, X) + self.bias).reshape(X.height, 1)

    fn __init__(inout self, params: Dict[String, String]) raises:
        if 'learning_rate' in params:
            self.lr = atof(params['learning_rate']).cast[DType.float32]()
        else:
            self.lr = 0.001
        if 'n_iters' in params:
            self.epoches = atol(params['n_iters'])
        else:
            self.epoches = 1000
        if 'C' in params:
            self.C = atof(params['C']).cast[DType.float32]()
        else:
            self.C = 1.0
        if 'kernel' in params:
            self.kernel = params['kernel']
        else:
            self.kernel = 'poly'
        if 'degree' in params:
            self.degree = atol(params['degree'])
        else:
            self.degree = 2
        if 'gamma' in params:
            self.gamma = atof(params['gamma']).cast[DType.float32]()
        else:
            self.gamma = -1.0
        if self.kernel == 'poly':
            self.k_params = (self.C, self.degree)
            self.kernel_func = polynomial_kernel
        else:
            self.k_params = (self.gamma, 0)
            self.kernel_func = gaussian_kernel
        if 'class_zero' in params:
            if params['class_zero'] == 'True':
                self.class_zero = True
            else:
                self.class_zero = False
        else:
            self.class_zero = False
        self.alpha = Matrix(0, 0)
        self.bias = 0.0
        self.X = Matrix(0, 0)
        self.y = Matrix(0, 0)
