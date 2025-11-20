import math
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CV, normal_distr
from algorithm import parallelize

struct GaussianNB:
    """Gaussian Naive Bayes (GaussianNB)."""
    var var_smoothing: Float32
    """Portion of the largest variance of all features that is added to variances for calculation stability."""
    var _classes: List[Int]
    var _mean: Matrix
    var _var: Matrix
    var _priors: List[Float32]

    fn __init__(out self, var_smoothing: Float32 = 1e-8):
        self.var_smoothing = var_smoothing
        self._classes = List[Int]()
        self._mean = Matrix(0, 0)
        self._var = Matrix(0, 0)
        self._priors = List[Float32]()

    fn fit(mut self, X: Matrix, y: Matrix) raises:
        """Fit Gaussian Naive Bayes."""
        var n_samples = Float32(X.height)
        var y_indices = y.unique()
        self._classes.clear()
        for i in range(len(y_indices)):
            self._classes.append(i)

        # calculate mean, var, and prior for each class
        self._mean = Matrix.zeros(len(self._classes), X.width)
        self._var = Matrix.zeros(len(self._classes), X.width)
        self._priors = List[Float32](capacity=len(self._classes))
        self._priors.resize(len(self._classes), 0.0)

        for i in range(len(self._classes)):
            var X_c = X[y_indices[i]]
            self._mean[i] = X_c.mean(0)
            self._var[i] = X_c._var(0, self._mean[i]) + self.var_smoothing
            self._priors[i] = X_c.height / n_samples

    fn predict(self, X: Matrix) raises -> Matrix:
        """Predict class for X.
        
        Returns:
            The predicted classes.
        """
        var posteriors = Matrix(X.height, len(self._classes))
        for i in range(len(self._classes)):
            # calculate posterior probability for each class
            posteriors['', i] = math.log(self._priors[i]) + self._pdf(i, X).log().sum(axis=1)
        var y_pred = Matrix(X.height, 1)
        @parameter
        fn p(i: Int):
            # return class with highest posterior probability
            y_pred.data[i] = self._classes[posteriors[i, unsafe=True].argmax()]
        parallelize[p](X.height)
        return y_pred^

    # Probability Density Function
    @always_inline
    fn _pdf(self, class_idx: Int, X: Matrix) raises -> Matrix:
        return normal_distr(X, self._mean[class_idx], self._var[class_idx])

struct MultinomialNB(CV):
    """Naive Bayes classifier for multinomial models."""
    var alpha: Float32
    """Additive smoothing parameter."""
    var _classes: List[Int]
    var _class_probs: Matrix
    var _priors: List[Float32]

    fn __init__(out self, alpha: Float32 = 0.0):
        self.alpha = alpha
        self._classes = List[Int]()
        self._class_probs = Matrix(0, 0)
        self._priors = List[Float32]()

    fn fit(mut self, X: Matrix, y: Matrix) raises:
        """Fit Naive Bayes classifier."""
        var n_samples = Float32(X.height)
        var y_indices = y.unique()
        self._classes.clear()
        for i in range(len(y_indices)):
            self._classes.append(i)

        # calculate feature probabilities and prior for each class
        self._class_probs = Matrix.zeros(len(self._classes), X.width)
        self._priors = List[Float32](capacity=len(self._classes))
        self._priors.resize(len(self._classes), 0.0)

        for i in range(len(self._classes)):
            var c_histogram = X[y_indices[i]].sum(axis=0) + self.alpha
            self._class_probs[i] = c_histogram / c_histogram.sum()
            self._priors[i] = len(y_indices[i]) / n_samples

    fn predict(self, X: Matrix) raises -> Matrix:
       """Predict class for X.

        Returns:
            The predicted classes.
        """
        var posteriors = Matrix(X.height, len(self._classes))
        for i in range(len(self._classes)):
            # calculate posterior probability for each class
            posteriors['', i] = math.log(self._priors[i]) + self._class_probs[i].log().ele_mul(X).sum(axis=1)
        var y_pred = Matrix(X.height, 1)
        @parameter
        fn p(i: Int):
            # return class with highest posterior probability
            y_pred.data[i] = self._classes[posteriors[i, unsafe=True].argmax()]
        parallelize[p](X.height)
        return y_pred^

    fn __init__(out self, params: Dict[String, String]) raises:
        if 'alpha' in params:
            self.alpha = atof(String(params['alpha'])).cast[DType.float32]()
        else:
            self.alpha = 0.0
        self._classes = List[Int]()
        self._class_probs = Matrix(0, 0)
        self._priors = List[Float32]()
