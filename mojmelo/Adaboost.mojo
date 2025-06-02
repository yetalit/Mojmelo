from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CVM, sign
import math
from collections import Dict

# Decision stump used as weak classifier
@value
struct DecisionStump:
    var polarity: Int
    var feature_idx: Int
    var threshold: Float32
    var alpha: Float32

    @always_inline
    fn __init__(out self):
        self.polarity = 1
        self.feature_idx = -1
        self.threshold = -math.inf[DType.float32]()
        self.alpha = 0.0

    @always_inline
    fn predict(self, X: Matrix) raises -> Matrix:
        var X_column = X['', self.feature_idx]
        var predictions = Matrix.ones(X.height, 1)
        var indices: List[Int]
        if self.polarity == 1:
            indices = X_column.argwhere_l(X_column < self.threshold)
        else:
            indices = X_column.argwhere_l(X_column > self.threshold)
        for index in indices:
            predictions.data[index[]] = -1.0
        return predictions^


struct Adaboost(CVM):
    var n_clf: Int
    var class_zero: Bool
    var clfs: List[DecisionStump]

    fn __init__(out self, n_clf: Int = 5, class_zero: Bool = False):
        self.n_clf = n_clf
        self.class_zero = class_zero
        self.clfs = List[DecisionStump]()

    fn _fit(mut self, X: Matrix, y: Matrix) raises:
        # Initialize weights to 1/N
        var w = Matrix.full(X.height, 1, Float32(1) / X.height)

        self.clfs = List[DecisionStump]()

        # Iterate through classifiers
        for _ in range(self.n_clf):
            var clf = DecisionStump()
            var min_error = math.inf[DType.float32]()

            # greedy search to find best threshold and feature
            for feature_i in range(X.width):
                var X_column = X['', feature_i]
                var thresholds = X_column.uniquef()

                for i_t in range(len(thresholds)):
                    # predict with polarity 1
                    var p = 1
                    var predictions = Matrix.ones(X.height, 1)
                    var indices = X_column.argwhere_l(X_column < thresholds.data[i_t])
                    for index in indices:
                        predictions.data[index[]] = -1.0

                    # Error = sum of weights of misclassified samples
                    var error: Float32 = 0.0
                    for i in range(predictions.size):
                        if (y.data[i] != predictions.data[i]):
                            error += w.data[i]

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # store the best configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = thresholds.data[i_t]
                        clf.feature_idx = feature_i
                        min_error = error

            # calculate alpha
            var EPS: Float32 = 1e-10
            clf.alpha = 0.5 * math.log((1.0 - min_error + EPS) / (min_error + EPS))

            # calculate predictions and update weights
            w = w.ele_mul((-clf.alpha * y.ele_mul(clf.predict(X))).exp())
            # Normalize to one
            w /= w.sum()

            # Save classifier
            self.clfs.append(clf)

    fn fit(mut self, X: Matrix, y: Matrix) raises:
        if self.class_zero:
            self._fit(X, y.where(y <= 0.0, -1.0, 1.0))
        else:
            self._fit(X, y)

    fn predict(self, X: Matrix) raises -> Matrix:
        var clf_preds = Matrix(X.height, self.n_clf)
        for clf_i in range(self.n_clf):
            clf_preds['', clf_i] = self.clfs[clf_i].alpha * self.clfs[clf_i].predict(X)
        if self.class_zero:
            var y_predicted = sign(clf_preds.sum(1))
            return y_predicted.where(y_predicted < 0.0, 0.0, 1.0)
        return sign(clf_preds.sum(1))

    fn __init__(out self, params: Dict[String, String]) raises:
        if 'n_clf' in params:
            self.n_clf = atol(String(params['n_clf']))
        else:
            self.n_clf = 5
        if 'class_zero' in params:
            if params['class_zero'] == 'True':
                self.class_zero = True
            else:
                self.class_zero = False
        else:
            self.class_zero = False
        self.clfs = List[DecisionStump]()
