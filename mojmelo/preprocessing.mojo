from mojmelo.linalg.Matrix import Matrix
from mojmelo.linalg.utils import _check_finite
from mojmelo.utils.utils import CV, cartesian_product
from mojmelo.utils.algorithm import parallelize
from std.sys import num_performance_cores
from std.python import Python, PythonObject
from std.math import ceil
from std import random

# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _check_nonempty(data: Matrix, fn_name: String) raises:
    if data.height == 0 or data.width == 0:
        raise Error(fn_name + ': input matrix is empty!')

def _check_feature_range(feature_range: Tuple[Float32, Float32], fn_name: String) raises:
    if not (feature_range[0] < feature_range[1]):
        raise Error(fn_name + ': feature_range must satisfy min < max!')

def _check_stats(data: Matrix, a: Matrix, b: Matrix, fn_name: String) raises:
    if a.width != data.width or b.width != data.width:
        raise Error(fn_name + ': statistics width does not match data width!')

def _check_xy(X: Matrix, y: Matrix, fn_name: String) raises:
    if X.height == 0:
        raise Error(fn_name + ': X is empty')
    if X.height != y.height:
        raise Error(fn_name + ': X and y have different numbers of rows!')

def _check_n_splits(n_splits: Int, n_samples: Int, fn_name: String) raises:
    if n_splits < 2 or n_splits > n_samples:
        raise Error(fn_name + ': n_splits must be between 2 and the number of samples!')

def _nonzero(div: Matrix) raises -> Matrix:
    """Replace zero scale factors with 1 so constant features do not divide by zero."""
    return div.where(div == 0.0, 1.0, div)

# ---------------------------------------------------------------------------
# Normalization / scaling
# ---------------------------------------------------------------------------

def normalize(data: Matrix, norm: String = 'l2') raises -> Tuple[Matrix, Matrix]:
    """Scale input vectors individually to unit norm (vector length).

    Args:
        data: Data.
        norm: The norm to use -> 'l2', 'l1'.

    Returns:
        Normalized data, norms.
    """
    var norm_kind = norm.lower()
    if norm_kind != 'l1' and norm_kind != 'l2':
        raise Error("normalize: norm must be 'l1' or 'l2', got '" + norm + "'!")
    _check_nonempty(data, 'normalize')
    _check_finite(data, 'normalize')

    var z = Matrix(data.height, data.width, order= data.order)
    var norms = Matrix(data.height, 1)
    # One flag per row: failures inside parallel workers must not be swallowed.
    var row_failed = List[Bool](capacity=data.height)
    row_failed.resize(data.height, False)

    if norm_kind == 'l1':
        if data.height == 1 or data.width == 1:
            norms.fill(data.abs().sum())
        else:
            norms = data.abs().sum(axis=1)
    else:
        if data.height == 1 or data.width == 1:
            norms.fill(data.norm())
        else:
            @__parameter
            def compute_norm(i: Int):
                try:
                    norms.data[unsafe_offset=i] = data[i].norm()
                except e:
                    row_failed[i] = True
            if norms.height < 768:
                for i in range(norms.height):
                    compute_norm(i)
            else:
                parallelize[compute_norm](norms.height)

    for i in range(len(row_failed)):
        if row_failed[i]:
            raise Error('normalize: failed to compute the norm of row ' + String(i))

    @__parameter
    def scale_row(i: Int):
        try:
            if norms.data[unsafe_offset=i] != 0.0:
                z[i] = data[i] / norms.data[unsafe_offset=i]
            else:
                # A zero norm means the whole row is zero.
                z[i] = data[i]
        except e:
            row_failed[i] = True
    if z.height < 768:
        for i in range(z.height):
            scale_row(i)
    else:
        parallelize[scale_row](z.height)

    for i in range(len(row_failed)):
        if row_failed[i]:
            raise Error('normalize: failed to scale row ' + String(i))

    return z^, norms^

def inv_normalize(z: Matrix, norms: Matrix) raises -> Matrix:
    """Reproduce normalized data given its norms.

    Args:
        z: Normalized data.
        norms: Norms.

    Returns:
        Original data.
    """
    if z.height != norms.height:
        raise Error('inv_normalize: z and norms have different numbers of rows!')
    return z.ele_mul(norms)

def MinMaxScaler(data: Matrix, feature_range: Tuple[Float32, Float32] = (0, 1)) raises -> Tuple[Matrix, Matrix, Matrix]:
    """Transform features by scaling each feature to a given range.

    Args:
        data: Data.
        feature_range: Desired range of transformed data.

    Returns:
        Scaled data, data_min, data_max.
    """
    _check_feature_range(feature_range, 'MinMaxScaler')
    _check_nonempty(data, 'MinMaxScaler')
    _check_finite(data, 'MinMaxScaler')
    var x_min = data.min(0)
    var x_max = data.max(0)
    # normalize then scale data
    var div = x_max - x_min
    return ((data - x_min) / _nonzero(div)) * (feature_range[1] - feature_range[0]) + feature_range[0], x_min^, x_max^

def MinMaxScaler(data: Matrix, x_min: Matrix, x_max: Matrix, feature_range: Tuple[Float32, Float32] = (0, 1)) raises -> Matrix:
    """Transform features by scaling each feature to a given range, data_min and data_max.

    Args:
        data: Data.
        x_min: Per feature minimum seen in the data.
        x_max: Per feature maximum seen in the data.
        feature_range: Desired range of transformed data.

    Returns:
        Scaled data.
    """
    _check_feature_range(feature_range, 'MinMaxScaler')
    _check_nonempty(data, 'MinMaxScaler')
    _check_finite(data, 'MinMaxScaler')
    _check_stats(data, x_min, x_max, 'MinMaxScaler')
    # normalize then scale data
    var div = x_max - x_min
    return ((data - x_min) / _nonzero(div)) * (feature_range[1] - feature_range[0]) + feature_range[0]

def inv_MinMaxScaler(z: Matrix, x_min: Matrix, x_max: Matrix, feature_range: Tuple[Float32, Float32] = (0, 1)) raises -> Matrix:
    """Reproduce scaled data given its range, data_min and data_max.

    Args:
        z: Scaled data.
        x_min: Per feature minimum seen in the data.
        x_max: Per feature maximum seen in the data.
        feature_range: Desired range of transformed data.

    Returns:
        Original data.
    """
    _check_feature_range(feature_range, 'inv_MinMaxScaler')
    _check_stats(z, x_min, x_max, 'inv_MinMaxScaler')
    var div = x_max - x_min
    return ((z - feature_range[0]) / (feature_range[1] - feature_range[0])).ele_mul(_nonzero(div)) + x_min

def StandardScaler(data: Matrix) raises -> Tuple[Matrix, Matrix, Matrix]:
    """Standardize features by removing the mean and scaling to unit variance.

    Args:
        data: Data.

    Returns:
        Scaled data, mean, standard deviation.
    """
    _check_nonempty(data, 'StandardScaler')
    _check_finite(data, 'StandardScaler')
    var mu = data.mean_slow0()
    var sigma = data.std_slow(0, mu)
    # standardize data
    return (data - mu) / _nonzero(sigma), mu^, sigma^

def StandardScaler(data: Matrix, mu: Matrix, sigma: Matrix) raises -> Matrix:
    """Standardize features by removing the mean and scaling to unit variance given mean and standard deviation.

    Args:
        data: Data.
        mu: Mean.
        sigma: Standard Deviation.

    Returns:
        Scaled data.
    """
    _check_nonempty(data, 'StandardScaler')
    _check_finite(data, 'StandardScaler')
    _check_stats(data, mu, sigma, 'StandardScaler')
    # standardize data
    return (data - mu) / _nonzero(sigma)

def inv_StandardScaler(z: Matrix, mu: Matrix, sigma: Matrix) raises -> Matrix:
    """Reproduce scaled data given its mean and standard deviation.

    Args:
        z: Scaled data.
        mu: Mean.
        sigma: Standard Deviation.

    Returns:
        Original data.
    """
    _check_stats(z, mu, sigma, 'inv_StandardScaler')
    return z.ele_mul(_nonzero(sigma)) + mu

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def _train_count(n_samples: Int, test_size: Float32, train_size: Float32) raises -> Int:
    """Number of training rows. Both sides are guaranteed to get at least one row."""
    if n_samples < 2:
        raise Error('train_test_split: need at least 2 samples!')
    var test_ratio: Float64
    if train_size > 0.0:
        if train_size >= 1.0:
            raise Error('train_test_split: train_size must be in (0, 1)!')
        test_ratio = 1.0 - Float64(train_size)
    else:
        if not (test_size > 0.0 and test_size < 1.0):
            raise Error('train_test_split: test_size must be in (0, 1)!')
        test_ratio = Float64(test_size)
    var raw = test_ratio * Float64(n_samples)
    var n_test = Int(ceil(raw - 1e-6 * raw))
    if n_test < 1:
        n_test = 1
    if n_test > n_samples - 1:
        n_test = n_samples - 1
    return n_samples - n_test

def train_test_split(X: Matrix, y: Matrix, *, test_size: Float32 = 0.5, train_size: Float32 = 0.0) raises -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    """Split matrices into random train and test subsets."""
    _check_xy(X, y, 'train_test_split')
    var split_i = _train_count(X.height, test_size, train_size)
    var ids = Matrix.rand_choice(X.height, X.height, False)
    var split_train = List[Int](ids[:split_i])
    var split_test = List[Int](ids[split_i:])
    return X[split_train], X[split_test], y[split_train], y[split_test]

def train_test_split(X: Matrix, y: Matrix, *, random_state: Int, test_size: Float32 = 0.5, train_size: Float32 = 0.0) raises -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    """Split matrices into random train and test subsets."""
    _check_xy(X, y, 'train_test_split')
    var split_i = _train_count(X.height, test_size, train_size)
    random.seed(random_state)
    var ids = Matrix.rand_choice(X.height, X.height, False, seed = False)
    var split_train = List[Int](ids[:split_i])
    var split_test = List[Int](ids[split_i:])
    return X[split_train], X[split_test], y[split_train], y[split_test]

# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

struct LabelEncoder:
    """Encode target labels with value between 0 and n_classes-1.
    This transformer can be used to encode target values from numpy, and not the input X.
    """
    var str_to_index: Dict[String, Int]
    var index_to_str: Dict[Int, String]

    def __init__(out self):
        self.str_to_index = Dict[String, Int]()
        self.index_to_str = Dict[Int, String]()

    def fit_transform(mut self, y: PythonObject) raises -> Matrix:
        """Fit label encoder and return encoded labels.

        Args:
            y: Targets Python object.

        Returns:
            Encoded labels.
        """
        var n = len(y)
        if n == 0:
            raise Error('LabelEncoder.fit_transform: y is empty!')
        self.str_to_index = Dict[String, Int]()
        self.index_to_str = Dict[Int, String]()
        var y_encoded = Matrix(n, 1)
        var latest_index = 0
        for i in range(n):
            var str_ = String(y[i])
            if not (str_ in self.str_to_index):
                self.str_to_index[str_] = latest_index
                self.index_to_str[latest_index] = str_
                latest_index += 1
            y_encoded.data[unsafe_offset=i] = Float32(self.str_to_index[str_])
        return y_encoded^

    def transform(self, y: PythonObject) raises -> Matrix:
        """Return encoded labels based on fitted encoder.

        Args:
            y: Targets Python object.

        Returns:
            Encoded labels.
        """
        if len(self.str_to_index) == 0:
            raise Error('LabelEncoder.transform: encoder is not fitted!')
        var n = len(y)
        var y_encoded = Matrix(n, 1)
        for i in range(n):
            var str_ = String(y[i])
            if not (str_ in self.str_to_index):
                raise Error("LabelEncoder.transform: unseen label '" + str_ + "'!")
            y_encoded.data[unsafe_offset=i] = Float32(self.str_to_index[str_])
        return y_encoded^

    def inverse_transform(self, y: Matrix) raises -> PythonObject:
        """Transform labels back to original encoding.

        Args:
            y: Encoded targets.

        Returns:
            Original targets Python object.
        """
        if len(self.index_to_str) == 0:
            raise Error('LabelEncoder.inverse_transform: encoder is not fitted!')
        var np = Python.import_module("numpy")
        var np_arr = np.empty(y.size, dtype='object')
        for i in range(y.size):
            var v = y.data[unsafe_offset=i]
            if v != v or v < 0.0:
                raise Error('LabelEncoder.inverse_transform: invalid encoded label at position ' + String(i))
            var idx = Int(v + 0.5)
            if not (idx in self.index_to_str):
                raise Error('LabelEncoder.inverse_transform: unknown class index ' + String(idx))
            np_arr[i] = self.index_to_str[idx]
        return np_arr^

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _make_ids(n: Int, random_state: Int) raises -> List[Int]:
    if random_state >= 0:
        random.seed(random_state)
        return Matrix.rand_choice(n, n, False, seed = False)
    return Matrix.rand_choice(n, n, False)

def _cross_val_score[m_type: CV](mut model: m_type, X: Matrix, y: Matrix, scoring: def(Matrix, Matrix) thin raises -> Float32, ids: List[Int], n_splits: Int) raises -> Float32:
    """Mean score over n_splits folds of the given (already shuffled) row order.
    Fold sizes differ by at most one, so every sample is tested exactly once."""
    var base = X.height // n_splits
    var rem = X.height % n_splits
    var start = 0
    var mean_score: Float32 = 0.0
    for k in range(n_splits):
        var end = start + base + (1 if k < rem else 0)
        var train_ids = List[Int](ids[end:]) + List[Int](ids[:start])
        var test_ids = List[Int](ids[start:end])
        model.fit(X[train_ids], y[train_ids])
        var y_pred = model.predict(X[test_ids])
        mean_score += scoring(y[test_ids], y_pred) / Float32(n_splits)
        start = end
    return mean_score

def KFold[m_type: CV](mut model: m_type, X: Matrix, y: Matrix, scoring: def(Matrix, Matrix) thin raises -> Float32, n_splits: Int = 5, random_state: Int = -1) raises -> Float32:
    """K-Fold cross-validator.

    Parameters:
        m_type: Model type.

    Args:
        model: Model.
        X: Samples.
        y: Targets.
        scoring: Scoring function.
        n_splits: Number of folds.
        random_state: Controls the shuffling applied to the data. A negative value means "time-seeded".

    Returns:
        Score.
    """
    _check_xy(X, y, 'KFold')
    _check_n_splits(n_splits, X.height, 'KFold')
    var ids = _make_ids(X.height, random_state)
    return _cross_val_score(model, X, y, scoring, ids, n_splits)

def GridSearchCV[m_type: CV](X: Matrix, y: Matrix, param_grid: Dict[String, List[String]],
                scoring: def(Matrix, Matrix) thin raises -> Float32, neg_score: Bool = False, n_jobs: Int = 0, cv: Int = 5, random_state: Int = -1) raises -> Tuple[Dict[String, String], Float32]:
    """Exhaustive search over specified parameter values for an estimator.

    Every candidate is evaluated on the same folds, so scores are comparable.
    If any candidate fails to fit or score, the search raises.

    Parameters:
        m_type: Model type.

    Args:
        X: Samples.
        y: Targets.
        param_grid: Dictionary with parameters names as keys and lists of parameter settings to try as values.
        scoring: Scoring function.
        neg_score: Invert the scoring results when finding the best params.
        n_jobs: Number of jobs to run in parallel. `-1` means using all processors, `0` runs serially.
        cv: Number of folds in a KFold.
        random_state: Controls the shuffling applied to the data. A negative value means "time-seeded".

    Returns:
        Best parameters, best score.
    """
    _check_xy(X, y, 'GridSearchCV')
    _check_n_splits(cv, X.height, 'GridSearchCV')
    if n_jobs < -1:
        raise Error('GridSearchCV: n_jobs must be -1 or >= 0!')
    if len(param_grid) == 0:
        raise Error('GridSearchCV: param_grid is empty!')

    # Snapshot the key order once so params and combinations always line up.
    var keys = List[String]()
    var dict_values = List[List[String]]()
    for key in param_grid.keys():
        if len(param_grid[key]) == 0:
            raise Error("GridSearchCV: parameter '" + key + "' has no values to try!")
        keys.append(key.copy())
        dict_values.append(param_grid[key].copy())
    var combinations = cartesian_product(dict_values)
    var n = len(combinations)
    if n == 0:
        raise Error('GridSearchCV: no parameter combinations to evaluate!')

    # Build every candidate up front (cheap, serial); only fitting is parallel.
    var params = List[Dict[String, String]](capacity=n)
    params.resize(n, Dict[String, String]())
    for i in range(n):
        for j in range(len(keys)):
            params[i][keys[j]] = combinations[i][j]

    # One shuffle shared by all candidates.
    var ids = _make_ids(X.height, random_state)

    var scores = List[Float32](capacity=n)
    scores.resize(n, 0.0)
    var failed = List[Bool](capacity=n)
    failed.resize(n, False)
    var errors = List[String](capacity=n)
    errors.resize(n, String(''))

    @__parameter
    def evaluate(i: Int):
        try:
            var model = m_type(params[i])
            var score = _cross_val_score(model, X, y, scoring, ids, cv)
            if neg_score:
                score = -score
            scores[i] = score
        except e:
            failed[i] = True
            errors[i] = String(e)

    if n_jobs == 0:
        for i in range(n):
            evaluate(i)
            if failed[i]:
                break
    else:
        var n_workers = n_jobs
        if n_workers == -1:
            n_workers = num_performance_cores()
        if n_workers > n:
            n_workers = n
        parallelize[evaluate](n, n_workers)

    for i in range(n):
        if failed[i]:
            raise Error('GridSearchCV: candidate ' + String(i) + ' failed: ' + errors[i])

    # First best wins on ties; NaN scores are never selected.
    var best = -1
    for i in range(n):
        var s = scores[i]
        if s != s:
            continue
        if best == -1 or s > scores[best]:
            best = i
    if best == -1:
        raise Error('GridSearchCV: every candidate produced a NaN score!')

    var best_score = scores[best]
    if neg_score:
        best_score = -best_score
    var best_params = params[best].copy()
    return best_params^, best_score
