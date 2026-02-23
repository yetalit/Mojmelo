import math
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CV, normal_distr, MODEL_IDS
from algorithm import parallelize
from sys import size_of
from memory import memcpy

struct GaussianNB(Copyable):
    """Gaussian Naive Bayes (GaussianNB)."""
    var var_smoothing: Float32
    """Portion of the largest variance of all features that is added to variances for calculation stability."""
    var _classes: List[Int]
    var _mean: Matrix
    var _var: Matrix
    var _priors: List[Float32]
    comptime MODEL_ID = 7

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

    fn save(self, path: String) raises:
        """Save model data necessary for prediction to the specified path."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        with open(_path, "w") as f:
            f.write_bytes(UInt8(Self.MODEL_ID).as_bytes())
            f.write_bytes(UInt64(len(self._classes)).as_bytes())
            f.write_bytes(Span(ptr=self._classes._data.bitcast[UInt8](), length=size_of[DType.int]()*len(self._classes)))
            f.write_bytes(UInt64(self._mean.width).as_bytes())
            f.write_bytes(Span(ptr=self._mean.data.bitcast[UInt8](), length=4*self._mean.size))
            f.write_bytes(Span(ptr=self._var.data.bitcast[UInt8](), length=4*self._var.size))
            f.write_bytes(Span(ptr=self._priors._data.bitcast[UInt8](), length=4*len(self._priors)))

    @staticmethod
    fn load(path: String) raises -> Self:
        """Load a saved model from the specified path for prediction."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        var model = Self()
        with open(_path, "r") as f:
            var id = f.read_bytes(1)[0]
            if id < 1 or id > MODEL_IDS.size-1:
                raise Error('Input file with invalid metadata!')
            elif id != Self.MODEL_ID:
                raise Error('Based on the metadata,', _path, 'belongs to', materialize[MODEL_IDS]()[id], 'algorithm!')
            var n_classes = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
            model._classes = List[Int](capacity=n_classes)
            model._classes.resize(n_classes, 0)
            memcpy(dest=model._classes._data, src=f.read_bytes(size_of[DType.int]()*n_classes).unsafe_ptr().bitcast[Int](), count=n_classes)
            var X_width = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
            model._mean = Matrix(n_classes, X_width, UnsafePointer[Float32, MutAnyOrigin](f.read_bytes(4*n_classes*X_width).unsafe_ptr().bitcast[Float32]()))
            model._var = Matrix(n_classes, X_width, UnsafePointer[Float32, MutAnyOrigin](f.read_bytes(4*n_classes*X_width).unsafe_ptr().bitcast[Float32]()))
            model._priors = List[Float32](capacity=n_classes)
            model._priors.resize(n_classes, 0)
            memcpy(dest=model._priors._data, src=f.read_bytes(4*n_classes).unsafe_ptr().bitcast[Float32](), count=n_classes)
        return model^

struct MultinomialNB(CV, Copyable):
    """Naive Bayes classifier for multinomial models."""
    var alpha: Float32
    """Additive smoothing parameter."""
    var _classes: List[Int]
    var _class_probs: Matrix
    var _priors: List[Float32]
    comptime MODEL_ID = 8

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

    fn save(self, path: String) raises:
        """Save model data necessary for prediction to the specified path."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        with open(_path, "w") as f:
            f.write_bytes(UInt8(Self.MODEL_ID).as_bytes())
            f.write_bytes(UInt64(len(self._classes)).as_bytes())
            f.write_bytes(Span(ptr=self._classes._data.bitcast[UInt8](), length=size_of[DType.int]()*len(self._classes)))
            f.write_bytes(UInt64(self._class_probs.width).as_bytes())
            f.write_bytes(Span(ptr=self._class_probs.data.bitcast[UInt8](), length=4*self._class_probs.size))
            f.write_bytes(Span(ptr=self._priors._data.bitcast[UInt8](), length=4*len(self._priors)))

    @staticmethod
    fn load(path: String) raises -> Self:
        """Load a saved model from the specified path for prediction."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        var model = Self()
        with open(_path, "r") as f:
            var id = f.read_bytes(1)[0]
            if id < 1 or id > MODEL_IDS.size-1:
                raise Error('Input file with invalid metadata!')
            elif id != Self.MODEL_ID:
                raise Error('Based on the metadata,', _path, 'belongs to', materialize[MODEL_IDS]()[id], 'algorithm!')
            var n_classes = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
            model._classes = List[Int](capacity=n_classes)
            model._classes.resize(n_classes, 0)
            memcpy(dest=model._classes._data, src=f.read_bytes(size_of[DType.int]()*n_classes).unsafe_ptr().bitcast[Int](), count=n_classes)
            var X_width = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
            model._class_probs = Matrix(n_classes, X_width, UnsafePointer[Float32, MutAnyOrigin](f.read_bytes(4*n_classes*X_width).unsafe_ptr().bitcast[Float32]()))
            model._priors = List[Float32](capacity=n_classes)
            model._priors.resize(n_classes, 0)
            memcpy(dest=model._priors._data, src=f.read_bytes(4*n_classes).unsafe_ptr().bitcast[Float32](), count=n_classes)
        return model^

    fn __init__(out self, params: Dict[String, String]) raises:
        if 'alpha' in params:
            self.alpha = atof(String(params['alpha'])).cast[DType.float32]()
        else:
            self.alpha = 0.0
        self._classes = List[Int]()
        self._class_probs = Matrix(0, 0)
        self._priors = List[Float32]()
