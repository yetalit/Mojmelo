from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import sign
import math

# Decision stump used as weak classifier
@value
struct DecisionStump:
    var polarity: Int
    var feature_idx: Int
    var threshold: Float32
    var alpha: Float32
    fn __init__(inout self):
        self.polarity = 1
        self.feature_idx = -1
        self.threshold = -math.inf[DType.float32]()
        self.alpha = 0.0

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


struct Adaboost:
    var n_clf: Int
    var clfs: List[DecisionStump]

    fn __init__(inout self, n_clf: Int = 5):
        self.n_clf = n_clf
        self.clfs = List[DecisionStump]()

    fn fit(inout self, X: Matrix, y: Matrix, class_zero: Bool = False) raises:
        if class_zero:
            self.fit(X, y.where(y <= 0.0, -1.0, 1.0))
            return
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

                for threshold in thresholds:
                    # predict with polarity 1
                    var p = 1
                    var predictions = Matrix.ones(X.height, 1)
                    var indices = X_column.argwhere_l(X_column < threshold[])
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
                        clf.threshold = threshold[]
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

    fn predict(self, X: Matrix) raises -> Matrix:
        var clf_preds = Matrix(X.height, self.n_clf)
        for clf_i in range(self.n_clf):
            clf_preds['', clf_i] = self.clfs[clf_i].alpha * self.clfs[clf_i].predict(X)
        return sign(clf_preds.sum(1))
