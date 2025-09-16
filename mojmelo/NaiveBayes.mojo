import math
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CVP, normal_distr
from algorithm import parallelize
from python import PythonObject

struct GaussianNB:
    """Gaussian Naive Bayes (GaussianNB)."""
    var var_smoothing: Float32
    """Portion of the largest variance of all features that is added to variances for calculation stability."""
    var _classes: List[String]
    var _mean: Matrix
    var _var: Matrix
    var _priors: List[Float32]

    fn __init__(out self, var_smoothing: Float32 = 1e-8):
        self.var_smoothing = var_smoothing
        self._classes = List[String]()
        self._mean = Matrix(0, 0)
        self._var = Matrix(0, 0)
        self._priors = List[Float32]()

    fn fit(mut self, X: Matrix, y: PythonObject) raises:
        """Fit Gaussian Naive Bayes."""
        var n_samples = Float32(X.height)
        var _class_freq: List[Int]
        self._classes, _class_freq = Matrix.unique(y)
        
        var classes_ids = Dict[String, Int]()
        for i in range(len(self._classes)):
            classes_ids[self._classes[i]] = i
        var y_indices = List[List[Int]](capacity=len(self._classes))
        y_indices.resize(len(self._classes), List[Int]())
        for i in range(len(y)):
            y_indices[classes_ids[String(y[i])]].append(i)

        # calculate mean, var, and prior for each class
        self._mean = Matrix.zeros(len(self._classes), X.width)
        self._var = Matrix.zeros(len(self._classes), X.width)
        self._priors = List[Float32](capacity=len(self._classes))
        self._priors.resize(len(self._classes), 0.0)

        @parameter
        fn p(i: Int):
            try:
                var X_c = X[y_indices[i]]
                self._mean[i] = X_c.mean(0)
                self._var[i] = X_c._var(0, self._mean[i]) + self.var_smoothing
                self._priors[i] = X_c.height / n_samples
            except e:
                print('Error:', e)
        parallelize[p](len(self._classes))

    fn predict(self, X: Matrix) raises -> List[String]:
        """Predict class for X.
        
        Returns:
            The predicted classes.
        """
        var posteriors = Matrix(X.height, len(self._classes))
        @parameter
        fn p1(i: Int):
            try:
                # calculate posterior probability for each class
                posteriors['', i] = math.log(self._priors[i]) + self._pdf(i, X).log().sum(axis=1)
            except e:
                print('Error:', e)
        parallelize[p1](len(self._classes))
        var y_pred = List[String](capacity=X.height)
        y_pred.resize(X.height, '')
        @parameter
        fn p2(i: Int):
            # return class with highest posterior probability
            y_pred[i] = self._classes[posteriors[i, unsafe=True].argmax()]
        parallelize[p2](X.height)
        return y_pred^

    # Probability Density Function
    @always_inline
    fn _pdf(self, class_idx: Int, X: Matrix) raises -> Matrix:
        return normal_distr(X, self._mean[class_idx], self._var[class_idx])

struct MultinomialNB:
    """Naive Bayes classifier for multinomial models."""
    var alpha: Float32
    """Additive smoothing parameter."""
    var _classes: List[String]
    var _class_probs: Matrix
    var _priors: List[Float32]

    fn __init__(out self, alpha: Float32 = 0.0):
        self.alpha = alpha
        self._classes = List[String]()
        self._class_probs = Matrix(0, 0)
        self._priors = List[Float32]()

    fn fit(mut self, X: Matrix, y: PythonObject) raises:
        """Fit Naive Bayes classifier."""
        var n_samples = Float32(X.height)
        var _class_freq: List[Int]
        self._classes, _class_freq = Matrix.unique(y)

        var classes_ids = Dict[String, Int]()
        for i in range(len(self._classes)):
            classes_ids[self._classes[i]] = i

        # calculate feature probabilities and prior for each class
        self._class_probs = Matrix.zeros(len(self._classes), X.width)
        self._priors = List[Float32](capacity=len(self._classes))
        self._priors.resize(len(self._classes), 0.0)

        for i in range(X.height):
            self._class_probs[classes_ids[String(y[i])]] += X[i]
        @parameter
        fn p(i: Int):
            try:
                var c_histogram = self._class_probs[i] + self.alpha
                self._class_probs[i] = c_histogram / c_histogram.sum()
                self._priors[i] = _class_freq[i] / n_samples
            except e:
                print('Error:', e)
        parallelize[p](len(self._classes))

    fn predict(self, X: Matrix) raises -> List[String]:
       """Predict class for X.

        Returns:
            The predicted classes.
        """
        var posteriors = Matrix(X.height, len(self._classes))
        @parameter
        fn p1(i: Int):
            try:
                # calculate posterior probability for each class
                posteriors['', i] = math.log(self._priors[i]) + self._class_probs[i].log().ele_mul(X).sum(axis=1)
            except e:
                print('Error:', e)
        parallelize[p1](len(self._classes))
        var y_pred = List[String](capacity=X.height)
        y_pred.resize(X.height, '')
        @parameter
        fn p2(i: Int):
            # return class with highest posterior probability
            y_pred[i] = self._classes[posteriors[i, unsafe=True].argmax()]
        parallelize[p2](X.height)
        return y_pred^

    fn __init__(out self, params: Dict[String, String]) raises:
        if 'alpha' in params:
            self.alpha = atof(String(params['alpha'])).cast[DType.float32]()
        else:
            self.alpha = 0.0
        self._classes = List[String]()
        self._class_probs = Matrix(0, 0)
        self._priors = List[Float32]()
