from collections.vector import InlinedFixedVector
import math
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import normal_distr

struct GaussianNB:
    var _classes: List[String]
    var _mean: Matrix
    var _var: Matrix
    var _priors: InlinedFixedVector[Float32]

    fn __init__(inout self):
        self._classes = List[String]()
        self._mean = Matrix(0, 0)
        self._var = Matrix(0, 0)
        self._priors = InlinedFixedVector[Float32](capacity = 0)

    fn fit(inout self, X: Matrix, y: PythonObject) raises:
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
                if str(y[j]) == self._classes[i]:
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

    fn _predict(self, x: Matrix) raises -> String:
        var posteriors = Matrix(1, len(self._classes))
        # calculate posterior probability for each class
        for i in range(len(self._classes)):
            posteriors.data[i] = math.log(self._priors[i]) + self._pdf(i, x).log().sum()
        # return class with highest posterior probability
        return self._classes[posteriors.argmax()]

    # Probability Density Function
    fn _pdf(self, class_idx: Int, x: Matrix) raises -> Matrix:
        return normal_distr(x, self._mean[class_idx], self._var[class_idx])

struct MultinomialNB:
    var _alpha: Int
    var _classes: List[String]
    var _class_probs: Matrix
    var _priors: InlinedFixedVector[Float32]

    fn __init__(inout self, alpha: Int = 0):
        self._alpha = alpha
        self._classes = List[String]()
        self._class_probs = Matrix(0, 0)
        self._priors = InlinedFixedVector[Float32](capacity = 0)

    fn fit(inout self, X: Matrix, y: PythonObject) raises:
        var n_samples = Float32(X.height)
        var _class_freq: List[Int]
        self._classes, _class_freq = Matrix.unique(y)

        # calculate feature probabilities and prior for each class
        self._class_probs = Matrix.zeros(len(self._classes), X.width)
        self._priors = InlinedFixedVector[Float32](capacity = len(self._classes))

        for i in range(X.height):
            self._class_probs[self._classes.index(y[i])] += X[i]
        for i in range(len(self._classes)):
            var c_histogram = self._class_probs[i] + self._alpha
            self._class_probs[i] = c_histogram / c_histogram.sum()
            self._priors[i] = _class_freq[i] / n_samples

    fn predict(self, X: Matrix) raises -> List[String]:
        var y_pred = List[String]()
        for i in range(X.height):
            y_pred.append(self._predict(X[i]))
        return y_pred^

    fn _predict(self, x: Matrix) -> String:
        var posteriors = Matrix(1, len(self._classes))
        # calculate posterior probability for each class
        for i in range(len(self._classes)):
            posteriors.data[i] = self._priors[i] * self._pdf(i, x)
        # return class with highest posterior probability
        return self._classes[posteriors.argmax()]

    # Probability Density Function
    fn _pdf(self, class_idx: Int, x: Matrix) -> Float32:
        var result: Float32 = 1.0
        for i in range(x.width):
            result *= self._class_probs.data[(class_idx * self._class_probs.width) + i] ** x.data[i]
        return result
