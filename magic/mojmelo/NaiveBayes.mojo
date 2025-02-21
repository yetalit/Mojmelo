from collections import InlinedFixedVector, Dict
import math
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CVP, normal_distr
from python import PythonObject

struct GaussianNB:
    var _classes: List[String]
    var _mean: Matrix
    var _var: Matrix
    var _priors: InlinedFixedVector[Float32]

    fn __init__(out self):
        self._classes = List[String]()
        self._mean = Matrix(0, 0)
        self._var = Matrix(0, 0)
        self._priors = InlinedFixedVector[Float32](capacity = 0)

    fn fit(mut self, X: Matrix, y: PythonObject) raises:
        var n_samples = Float32(X.height)
        var _class_freq: List[Int]
        self._classes, _class_freq = Matrix.unique(y)

        # calculate mean, var, and prior for each class
        self._mean = Matrix.zeros(len(self._classes), X.width)
        self._var = Matrix.zeros(len(self._classes), X.width)
        self._priors = InlinedFixedVector[Float32](capacity = len(self._classes))

        for i in range(len(self._classes)):
            var X_c = Matrix(_class_freq[i], X.width)
            var pointer: Int = 0
            for j in range(X.height):
                if String(y[j]) == self._classes[i]:
                    X_c[pointer] = X[j]
                    pointer += 1
            self._mean[i] = X_c.mean(0)
            self._var[i] = X_c._var(0, self._mean[i])
            self._priors[i] = X_c.height / n_samples

    fn predict(self, X: Matrix) raises -> List[String]:
        var y_pred = List[String]()
        for i in range(X.height):
            y_pred.append(self._predict(X[i]))
        return y_pred^

    @always_inline
    fn _predict(self, x: Matrix) raises -> String:
        var max_posterior = math.log(self._priors[0]) + self._pdf(0, x).log().sum()
        var argmax = 0
        # calculate posterior probability for each class
        for i in range(1, len(self._classes)):
            var current_posterior = math.log(self._priors[i]) + self._pdf(i, x).log().sum()
            if current_posterior > max_posterior:
                max_posterior = current_posterior
                argmax = i
        # return class with highest posterior probability
        return self._classes[argmax]

    # Probability Density Function
    @always_inline
    fn _pdf(self, class_idx: Int, x: Matrix) raises -> Matrix:
        return normal_distr(x, self._mean[class_idx], self._var[class_idx])

struct MultinomialNB:
    var _alpha: Float32
    var _classes: List[String]
    var _class_probs: Matrix
    var _priors: InlinedFixedVector[Float32]

    fn __init__(out self, alpha: Float32 = 0.0):
        self._alpha = alpha
        self._classes = List[String]()
        self._class_probs = Matrix(0, 0)
        self._priors = InlinedFixedVector[Float32](capacity = 0)

    fn fit(mut self, X: Matrix, y: PythonObject) raises:
        var n_samples = Float32(X.height)
        var _class_freq: List[Int]
        self._classes, _class_freq = Matrix.unique(y)

        # calculate feature probabilities and prior for each class
        self._class_probs = Matrix.zeros(len(self._classes), X.width)
        self._priors = InlinedFixedVector[Float32](capacity = len(self._classes))

        for i in range(X.height):
            self._class_probs[self._classes.index(String(y[i]))] += X[i]
        for i in range(len(self._classes)):
            var c_histogram = self._class_probs[i] + self._alpha
            self._class_probs[i] = c_histogram / c_histogram.sum()
            self._priors[i] = _class_freq[i] / n_samples

    fn predict(self, X: Matrix) raises -> List[String]:
        var y_pred = List[String]()
        for i in range(X.height):
            y_pred.append(self._predict(X[i]))
        return y_pred^

    @always_inline
    fn _predict(self, x: Matrix) raises -> String:
        var max_posterior = math.log(self._priors[0]) + self._class_probs[0].log().ele_mul(x).sum()
        var argmax = 0
        # calculate posterior probability for each class
        for i in range(1, len(self._classes)):
            var current_posterior = math.log(self._priors[i]) + self._class_probs[i].log().ele_mul(x).sum()
            if current_posterior > max_posterior:
                max_posterior = current_posterior
                argmax = i
        # return class with highest posterior probability
        return self._classes[argmax]

    fn __init__(out self, params: Dict[String, String]) raises:
        if '_alpha' in params:
            self._alpha = atof(String(params['_alpha'])).cast[DType.float32]()
        else:
            self._alpha = 0.0
        self._classes = List[String]()
        self._class_probs = Matrix(0, 0)
        self._priors = InlinedFixedVector[Float32](capacity = 0)
