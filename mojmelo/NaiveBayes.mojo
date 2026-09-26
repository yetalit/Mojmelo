from std import math
from mojmelo.linalg.Matrix import Matrix
from mojmelo.utils.utils import CV, normal_distr, MODEL_IDS
from mojmelo.utils.algorithm import parallelize
from std.sys import size_of
from std.memory import unsafe_memcpy

struct GaussianNB(CV, Copyable):
    """Gaussian Naive Bayes (GaussianNB).

    Assumes the likelihood of each feature, conditioned on the class, follows
    a Gaussian distribution. Suited for continuous, real-valued features.
    """
    var var_smoothing: Float32
    """Portion of the largest variance of all features that is added to variances for calculation stability."""
    var _classes: List[Int]
    var _mean: Matrix
    var _var: Matrix
    var _priors: List[Float32]
    comptime MODEL_ID = 7

    def __init__(out self, var_smoothing: Float32 = 1e-8):
        self.var_smoothing = var_smoothing
        self._classes = List[Int]()
        self._mean = Matrix(0, 0)
        self._var = Matrix(0, 0)
        self._priors = List[Float32]()

    def fit(mut self, X: Matrix, y: Matrix) raises:
        """Fit Gaussian Naive Bayes.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples, 1), encoded as contiguous
                non-negative integers starting at 0.
        """
        if self.var_smoothing < 0.0:
            raise Error('GaussianNB: var_smoothing must be non-negative!')
        if X.height == 0:
            raise Error('GaussianNB.fit: X must contain at least one sample!')
        if X.height != y.height:
            raise Error('GaussianNB.fit: X and y must have the same number of samples!')

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
            if X_c.height == 0:
                # No training samples were seen for this label; keep it
                # harmless (zero prior means it can never win at predict time)
                self._mean[i] = Matrix.zeros(1, X.width)
                self._var[i] = Matrix.zeros(1, X.width) + self.var_smoothing
                self._priors[i] = 0.0
                continue
            self._mean[i] = X_c.mean(0)
            self._var[i] = X_c._var(0, self._mean[i]) + self.var_smoothing
            self._priors[i] = Float32(X_c.height) / n_samples

    def predict(self, X: Matrix) raises -> Matrix:
        """Predict class for X.

        Returns:
            The predicted classes.
        """
        if len(self._classes) == 0:
            raise Error('GaussianNB.predict: model is not fitted. Call fit() before predict()!')
        var posteriors = Matrix(X.height, len(self._classes))
        for i in range(len(self._classes)):
            # calculate posterior probability for each class
            posteriors['', i] = math.log(self._priors[i] + 1e-300) + self._pdf(i, X).log().sum(axis=1)
        var y_pred = Matrix(X.height, 1)
        @__parameter
        def p(i: Int):
            # return class with highest posterior probability
            y_pred.data[unsafe_offset=i] = Float32(self._classes[posteriors[i, unsafe=True].argmax()])
        parallelize[p](X.height)
        return y_pred^

    # Probability Density Function
    @always_inline
    def _pdf(self, class_idx: Int, X: Matrix) raises -> Matrix:
        return normal_distr(X, self._mean[class_idx], self._var[class_idx])

    def save(self, path: String) raises:
        """Save model data necessary for prediction to the specified path."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        with open(_path, "w") as f:
            f.write_bytes(UInt8(Self.MODEL_ID).as_bytes())
            f.write_bytes(UInt64(len(self._classes)).as_bytes())
            f.write_bytes(Span(unsafe_ptr=self._classes._data.unsafe_bitcast[UInt8](), length=size_of[DType.int]()*len(self._classes)))
            f.write_bytes(UInt64(self._mean.width).as_bytes())
            f.write_bytes(Span(unsafe_ptr=self._mean.data.unsafe_bitcast[UInt8](), length=4*self._mean.size))
            f.write_bytes(Span(unsafe_ptr=self._var.data.unsafe_bitcast[UInt8](), length=4*self._var.size))
            f.write_bytes(Span(unsafe_ptr=self._priors._data.unsafe_bitcast[UInt8](), length=4*len(self._priors)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Load a saved model from the specified path for prediction."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        var model = Self()
        with open(_path, "r") as f:
            var id = f.read_bytes(1)[0]
            if id < 1 or id > UInt8(MODEL_IDS.length-1):
                raise Error('Input file with invalid metadata!')
            elif id != Self.MODEL_ID:
                raise Error('Based on the metadata, ', _path, ' belongs to ', materialize[MODEL_IDS]()[id], ' algorithm!')
            var n_classes = Int(f.read_bytes(8).unsafe_ptr().unsafe_bitcast[UInt64]()[])
            if n_classes <= 0:
                raise Error('GaussianNB.load: corrupted model file (invalid class count)!')
            model._classes = List[Int](capacity=n_classes)
            model._classes.resize(n_classes, 0)
            unsafe_memcpy(dest=model._classes._data, src=f.read_bytes(size_of[DType.int]()*n_classes).unsafe_ptr().unsafe_bitcast[Int](), count=n_classes)
            var X_width = Int(f.read_bytes(8).unsafe_ptr().unsafe_bitcast[UInt64]()[])
            if X_width <= 0:
                raise Error('GaussianNB.load: corrupted model file (invalid feature count)!')
            var _mean = f.read_bytes(4*n_classes*X_width)
            model._mean = Matrix(n_classes, X_width, Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_mean.unsafe_ptr())))
            _ = _mean
            var _var = f.read_bytes(4*n_classes*X_width)
            model._var = Matrix(n_classes, X_width, Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_var.unsafe_ptr())))
            _ = _var
            model._priors = List[Float32](capacity=n_classes)
            model._priors.resize(n_classes, 0)
            unsafe_memcpy(dest=model._priors._data, src=f.read_bytes(4*n_classes).unsafe_ptr().unsafe_bitcast[Float32](), count=n_classes)
        return model^

    def __init__(out self, params: Dict[String, String]) raises:
        """Construct from a hyperparameter dictionary."""
        if 'var_smoothing' in params:
            self.var_smoothing = atof(String(params['var_smoothing'])).cast[DType.float32]()
        else:
            self.var_smoothing = 1e-8
        self._classes = List[Int]()
        self._mean = Matrix(0, 0)
        self._var = Matrix(0, 0)
        self._priors = List[Float32]()

struct MultinomialNB(CV, Copyable):
    """Naive Bayes classifier for multinomial models.

    Suited for discrete count features.
    """
    var alpha: Float32
    """Additive (Laplace/Lidstone) smoothing parameter. Must be non-negative."""
    var _classes: List[Int]
    var _class_probs: Matrix
    var _priors: List[Float32]
    comptime MODEL_ID = 8

    def __init__(out self, alpha: Float32 = 0.0):
        self.alpha = alpha
        self._classes = List[Int]()
        self._class_probs = Matrix(0, 0)
        self._priors = List[Float32]()

    def fit(mut self, X: Matrix, y: Matrix) raises:
        """Fit Naive Bayes classifier.

        Args:
            X: Training features of shape (n_samples, n_features). Expected to
                be non-negative counts.
            y: Training labels of shape (n_samples, 1), encoded as contiguous
                non-negative integers starting at 0.
        """
        if self.alpha < 0.0:
            raise Error('MultinomialNB: alpha must be non-negative!')
        if X.height == 0:
            raise Error('MultinomialNB.fit: X must contain at least one sample!')
        if X.height != y.height:
            raise Error('MultinomialNB.fit: X and y must have the same number of samples!')

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
            if len(y_indices[i]) == 0:
                # No training samples for this label: leave it with a zero
                # prior (so it can never be predicted) and a valid, uniform
                # feature distribution.
                self._class_probs[i] = Matrix.full(1, X.width, 1.0 / Float32(X.width))
                self._priors[i] = 0.0
                continue
            var c_histogram = X[y_indices[i]].sum(axis=0) + self.alpha
            var total = c_histogram.sum()
            self._class_probs[i] = c_histogram / total if total > 0.0 else Matrix.full(1, X.width, 1.0 / Float32(X.width))
            self._priors[i] = Float32(len(y_indices[i])) / n_samples

    def predict(self, X: Matrix) raises -> Matrix:
        """Predict class for X.

        Returns:
            The predicted classes.
        """
        if len(self._classes) == 0:
            raise Error('MultinomialNB.predict: model is not fitted. Call fit() before predict()!')
        var posteriors = Matrix(X.height, len(self._classes))
        for i in range(len(self._classes)):
            # calculate posterior probability for each class
            posteriors['', i] = math.log(self._priors[i] + 1e-300) + self._class_probs[i].log().ele_mul(X).sum(axis=1)
        var y_pred = Matrix(X.height, 1)
        @__parameter
        def p(i: Int):
            # return class with highest posterior probability
            y_pred.data[unsafe_offset=i] = Float32(self._classes[posteriors[i, unsafe=True].argmax()])
        parallelize[p](X.height)
        return y_pred^

    def save(self, path: String) raises:
        """Save model data necessary for prediction to the specified path."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        with open(_path, "w") as f:
            f.write_bytes(UInt8(Self.MODEL_ID).as_bytes())
            f.write_bytes(UInt64(len(self._classes)).as_bytes())
            f.write_bytes(Span(unsafe_ptr=self._classes._data.unsafe_bitcast[UInt8](), length=size_of[DType.int]()*len(self._classes)))
            f.write_bytes(UInt64(self._class_probs.width).as_bytes())
            f.write_bytes(Span(unsafe_ptr=self._class_probs.data.unsafe_bitcast[UInt8](), length=4*self._class_probs.size))
            f.write_bytes(Span(unsafe_ptr=self._priors._data.unsafe_bitcast[UInt8](), length=4*len(self._priors)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Load a saved model from the specified path for prediction."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        var model = Self()
        with open(_path, "r") as f:
            var id = f.read_bytes(1)[0]
            if id < 1 or id > UInt8(MODEL_IDS.length-1):
                raise Error('Input file with invalid metadata!')
            elif id != Self.MODEL_ID:
                raise Error('Based on the metadata, ', _path, ' belongs to ', materialize[MODEL_IDS]()[id], ' algorithm!')
            var n_classes = Int(f.read_bytes(8).unsafe_ptr().unsafe_bitcast[UInt64]()[])
            if n_classes <= 0:
                raise Error('MultinomialNB.load: corrupted model file (invalid class count)!')
            model._classes = List[Int](capacity=n_classes)
            model._classes.resize(n_classes, 0)
            unsafe_memcpy(dest=model._classes._data, src=f.read_bytes(size_of[DType.int]()*n_classes).unsafe_ptr().unsafe_bitcast[Int](), count=n_classes)
            var X_width = Int(f.read_bytes(8).unsafe_ptr().unsafe_bitcast[UInt64]()[])
            if X_width <= 0:
                raise Error('MultinomialNB.load: corrupted model file (invalid feature count)!')
            var _class_probs = f.read_bytes(4*n_classes*X_width)
            model._class_probs = Matrix(n_classes, X_width, Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_class_probs.unsafe_ptr())))
            _ = _class_probs
            model._priors = List[Float32](capacity=n_classes)
            model._priors.resize(n_classes, 0)
            unsafe_memcpy(dest=model._priors._data, src=f.read_bytes(4*n_classes).unsafe_ptr().unsafe_bitcast[Float32](), count=n_classes)
        return model^

    def __init__(out self, params: Dict[String, String]) raises:
        """Construct from a hyperparameter dictionary."""
        if 'alpha' in params:
            self.alpha = atof(String(params['alpha'])).cast[DType.float32]()
        else:
            self.alpha = 0.0
        self._classes = List[Int]()
        self._class_probs = Matrix(0, 0)
        self._priors = List[Float32]()

struct BernoulliNB(CV, Copyable):
    """Naive Bayes classifier for multivariate Bernoulli models.

    Suited for discrete, binary/boolean features. Each feature is binarized
    against `binarize` (if it isn't already boolean) before being modelled
    with an independent Bernoulli distribution per class.
    """
    var alpha: Float32
    """Additive (Laplace/Lidstone) smoothing parameter. Must be non-negative."""
    var binarize: Float32
    """Threshold for binarizing features: values strictly greater than this become 1, others become 0."""
    var _classes: List[Int]
    var _feature_probs: Matrix
    var _priors: List[Float32]
    comptime MODEL_ID = 13

    def __init__(out self, alpha: Float32 = 0.0, binarize: Float32 = 0.0):
        self.alpha = alpha
        self.binarize = binarize
        self._classes = List[Int]()
        self._feature_probs = Matrix(0, 0)
        self._priors = List[Float32]()

    def fit(mut self, X: Matrix, y: Matrix) raises:
        """Fit Bernoulli Naive Bayes classifier.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples, 1), encoded as contiguous
                non-negative integers starting at 0.
        """
        if self.alpha < 0.0:
            raise Error('BernoulliNB: alpha must be non-negative!')
        if X.height == 0:
            raise Error('BernoulliNB.fit: X must contain at least one sample!')
        if X.height != y.height:
            raise Error('BernoulliNB.fit: X and y must have the same number of samples!')

        var n_samples = Float32(X.height)
        var X_bin = X.where(X > self.binarize, 1.0, 0.0)
        var y_indices = y.unique()
        self._classes.clear()
        for i in range(len(y_indices)):
            self._classes.append(i)

        # calculate per-feature Bernoulli probability and prior for each class
        self._feature_probs = Matrix.zeros(len(self._classes), X.width)
        self._priors = List[Float32](capacity=len(self._classes))
        self._priors.resize(len(self._classes), 0.0)

        for i in range(len(self._classes)):
            var n_c = Float32(len(y_indices[i]))
            if n_c == 0.0:
                # No training samples for this label: leave it with a zero
                # prior (so it can never be predicted) and a valid probability (0.5).
                self._feature_probs[i] = Matrix.full(1, X.width, 0.5)
                self._priors[i] = 0.0
                continue
            var counts = X_bin[y_indices[i]].sum(axis=0)
            self._feature_probs[i] = (counts + self.alpha) / (n_c + 2.0 * self.alpha)
            self._priors[i] = n_c / n_samples

    def predict(self, X: Matrix) raises -> Matrix:
        """Predict class for X.

        Returns:
            The predicted classes.
        """
        if len(self._classes) == 0:
            raise Error('BernoulliNB.predict: model is not fitted. Call fit() before predict()!')
        var X_bin = X.where(X > self.binarize, 1.0, 0.0)
        var posteriors = Matrix(X.height, len(self._classes))
        for i in range(len(self._classes)):
            # log P(class) + sum_j [ x_j * log(p_j) + (1 - x_j) * log(1 - p_j) ]
            var log_p = self._feature_probs[i].log()
            var log_not_p = (1.0 - self._feature_probs[i]).log()
            posteriors['', i] = math.log(self._priors[i] + 1e-300) + (log_p - log_not_p).ele_mul(X_bin).sum(axis=1) + log_not_p.sum()
        var y_pred = Matrix(X.height, 1)
        @__parameter
        def p(i: Int):
            # return class with highest posterior probability
            y_pred.data[unsafe_offset=i] = Float32(self._classes[posteriors[i, unsafe=True].argmax()])
        parallelize[p](X.height)
        return y_pred^

    def save(self, path: String) raises:
        """Save model data necessary for prediction to the specified path."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        with open(_path, "w") as f:
            f.write_bytes(UInt8(Self.MODEL_ID).as_bytes())
            f.write_bytes(self.binarize.as_bytes())
            f.write_bytes(UInt64(len(self._classes)).as_bytes())
            f.write_bytes(Span(unsafe_ptr=self._classes._data.unsafe_bitcast[UInt8](), length=size_of[DType.int]()*len(self._classes)))
            f.write_bytes(UInt64(self._feature_probs.width).as_bytes())
            f.write_bytes(Span(unsafe_ptr=self._feature_probs.data.unsafe_bitcast[UInt8](), length=4*self._feature_probs.size))
            f.write_bytes(Span(unsafe_ptr=self._priors._data.unsafe_bitcast[UInt8](), length=4*len(self._priors)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Load a saved model from the specified path for prediction."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        var model = Self()
        with open(_path, "r") as f:
            var id = f.read_bytes(1)[0]
            if id < 1 or id > UInt8(MODEL_IDS.length-1):
                raise Error('Input file with invalid metadata!')
            elif id != Self.MODEL_ID:
                raise Error('Based on the metadata, ', _path, ' belongs to ', materialize[MODEL_IDS]()[id], ' algorithm!')
            model.binarize = f.read_bytes(4).unsafe_ptr().unsafe_bitcast[Float32]()[]
            var n_classes = Int(f.read_bytes(8).unsafe_ptr().unsafe_bitcast[UInt64]()[])
            if n_classes <= 0:
                raise Error('BernoulliNB.load: corrupted model file (invalid class count)!')
            model._classes = List[Int](capacity=n_classes)
            model._classes.resize(n_classes, 0)
            unsafe_memcpy(dest=model._classes._data, src=f.read_bytes(size_of[DType.int]()*n_classes).unsafe_ptr().unsafe_bitcast[Int](), count=n_classes)
            var X_width = Int(f.read_bytes(8).unsafe_ptr().unsafe_bitcast[UInt64]()[])
            if X_width <= 0:
                raise Error('BernoulliNB.load: corrupted model file (invalid feature count)!')
            var _feature_probs = f.read_bytes(4*n_classes*X_width)
            model._feature_probs = Matrix(n_classes, X_width, Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_feature_probs.unsafe_ptr())))
            _ = _feature_probs
            model._priors = List[Float32](capacity=n_classes)
            model._priors.resize(n_classes, 0)
            unsafe_memcpy(dest=model._priors._data, src=f.read_bytes(4*n_classes).unsafe_ptr().unsafe_bitcast[Float32](), count=n_classes)
        return model^

    def __init__(out self, params: Dict[String, String]) raises:
        """Construct from a hyperparameter dictionary."""
        if 'alpha' in params:
            self.alpha = atof(String(params['alpha'])).cast[DType.float32]()
        else:
            self.alpha = 0.0
        if 'binarize' in params:
            self.binarize = atof(String(params['binarize'])).cast[DType.float32]()
        else:
            self.binarize = 0.0
        self._classes = List[Int]()
        self._feature_probs = Matrix(0, 0)
        self._priors = List[Float32]()
