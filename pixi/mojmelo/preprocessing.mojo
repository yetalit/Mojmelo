from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CV, cartesian_product
from std.algorithm import parallelize
from std.sys import num_performance_cores
from std.python import Python, PythonObject
import std.time as time
import std.random as random

def normalize(data: Matrix, norm: String = 'l2') raises -> Tuple[Matrix, Matrix]:
    """Scale input vectors individually to unit norm (vector length).

    Args:
        data: Data.
        norm: The norm to use -> 'l2', 'l1'.

    Returns:
        Normalized data, norms.
    """
    var z = Matrix(data.height, data.width, order= data.order)
    var norms = Matrix(data.height, 1)
    if norm.lower() == 'l1':
        if data.height == 1 or data.width == 1:
            norms.fill(data.abs().sum())
        else:
            norms = data.abs().sum(axis=1)
    else:
        if data.height == 1 or data.width == 1:
            norms.fill(data.norm())
        else:
            if norms.height < 768:
                for i in range(norms.height):
                    norms.data[i] = data[i].norm()
            else:
                @parameter
                def p1(i: Int):
                    try:
                        norms.data[i] = data[i].norm()
                    except e:
                        print('Error:', e)
                parallelize[p1](norms.height)

    @parameter
    def p2(i: Int):
        try:
            if norms.data[i] != 0.0:
                z[i] = data[i] / norms.data[i]
            else:
                z[i].fill_zero()
        except e:
            print('Error:', e)
    parallelize[p2](z.height)

    return z^, norms^

def inv_normalize(z: Matrix, norms: Matrix) raises -> Matrix:
    """Reproduce normalized data given its norms.

    Args:
        z: Normalized data.
        norms: Norms.

    Returns:
        Original data.
    """
    return z.ele_mul(norms)

def MinMaxScaler(data: Matrix, feature_range: Tuple[Float32, Float32] = (0, 1)) raises -> Tuple[Matrix, Matrix, Matrix]:
    """Transform features by scaling each feature to a given range.
    
    Args:
        data: Data.
        feature_range: Desired range of transformed data.

    Returns:
        Scaled data, data_min, data_max.
    """
    var x_min = data.min(0)
    var x_max = data.max(0)
    # normalize then scale data
    var div = x_max - x_min
    return ((data - x_min) / div.where(div == 0.0, 1.0, div)) * (feature_range[1] - feature_range[0]) + feature_range[0], x_min^, x_max^

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
    # normalize then scale data
    var div = x_max - x_min
    return ((data - x_min) / div.where(div == 0.0, 1.0, div)) * (feature_range[1] - feature_range[0]) + feature_range[0]

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
    var div = x_max - x_min
    return ((z - feature_range[0]) / (feature_range[1] - feature_range[0])).ele_mul(div.where(div == 0.0, 1.0, div)) + x_min

def StandardScaler(data: Matrix) raises -> Tuple[Matrix, Matrix, Matrix]:
    """Standardize features by removing the mean and scaling to unit variance.
    
    Args:
        data: Data.

    Returns:
        Scaled data, mean, standard deviation.
    """
    var mu = data.mean_slow0()
    var sigma = data.std_slow(0, mu)
    # standardize data
    return (data - mu) / sigma.where(sigma == 0.0, 1.0, sigma), mu^, sigma^

def StandardScaler(data: Matrix, mu: Matrix, sigma: Matrix) raises -> Matrix:
    """Standardize features by removing the mean and scaling to unit variance given mean and standard deviation.
    
    Args:
        data: Data.
        mu: Mean.
        sigma: Standard Deviation.

    Returns:
        Scaled data.
    """
    # standardize data
    return (data - mu) / sigma.where(sigma == 0.0, 1.0, sigma)

def inv_StandardScaler(z: Matrix, mu: Matrix, sigma: Matrix) raises -> Matrix:
    """Reproduce scaled data given its mean and standard deviation.

    Args:
        z: Scaled data.
        mu: Mean.
        sigma: Standard Deviation.

    Returns:
        Original data.
    """
    return z.ele_mul(sigma.where(sigma == 0.0, 1.0, sigma)) + mu

def train_test_split(X: Matrix, y: Matrix, *, test_size: Float32 = 0.5, train_size: Float32 = 0.0) raises -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    """Split matrices into random train and test subsets."""
    var test_ratio = test_size if train_size <= 0.0 else 1.0 - train_size
    var ids = Matrix.rand_choice(X.height, X.height, False)
    var split_i = Int(Float32(X.height) - (test_ratio * Float32(X.height)))
    var split_train = List[Scalar[DType.int]](ids[:split_i])
    var split_test = List[Scalar[DType.int]](ids[split_i:])
    return X[split_train], X[split_test], y[split_train], y[split_test]

def train_test_split(X: Matrix, y: Matrix, *, random_state: Int, test_size: Float32 = 0.5, train_size: Float32 = 0.0) raises -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    """Split matrices into random train and test subsets."""
    var test_ratio = test_size if train_size <= 0.0 else 1.0 - train_size
    random.seed(random_state)
    var ids = Matrix.rand_choice(X.height, X.height, False, seed = False)
    var split_i = Int(Float32(X.height) - (test_ratio * Float32(X.height)))
    var split_train = List[Scalar[DType.int]](ids[:split_i])
    var split_test = List[Scalar[DType.int]](ids[split_i:])
    return X[split_train], X[split_test], y[split_train], y[split_test]

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
        self.str_to_index = Dict[String, Int]()
        self.index_to_str = Dict[Int, String]() 
        var y_encoded = Matrix(len(y), 1)
        var latest_index = 0
        for i in range(len(y)):
            var str_ = String(y[i])
            if not (str_ in self.str_to_index):
                self.str_to_index[str_] = latest_index
                self.index_to_str[latest_index] = str_
                latest_index += 1
            y_encoded.data[i] = Float32(self.str_to_index[str_])
        return y_encoded^

    def transform(self, y: PythonObject) raises -> Matrix:
        """Return encoded labels based on fitted encoder.

        Args:
            y: Targets Python object.

        Returns:
            Encoded labels.
        """
        var y_encoded = Matrix(len(y), 1)
        for i in range(len(y)):
            y_encoded.data[i] = Float32(self.str_to_index[String(y[i])])
        return y_encoded^

    def inverse_transform(self, y: Matrix) raises -> PythonObject:
        """Transform labels back to original encoding.
            
        Args:
            y: Encoded targets.

        Returns:
            Original targets Python object.
        """
        var np = Python.import_module("numpy")
        var np_arr = np.empty(len(y), dtype='object')
        for i in range(len(y)):
            np_arr[i] = self.index_to_str[Int(y.data[i])]
        return np_arr^

def KFold[m_type: CV](mut model: m_type, X: Matrix, y: Matrix, scoring: fn(Matrix, Matrix) raises -> Float32, n_splits: Int = 5) raises -> Float32:
    """K-Fold cross-validator.

    Parameters:
        m_type: Model type.

    Args:
        model: Model.
        X: Samples.
        y: Targets.
        scoring: Scoring function.
        n_splits: Number of folds.

    Returns:
        Score.
    """
    var ids = Matrix.rand_choice(X.height, X.height, False)
    var test_count = Int((1 / n_splits) * X.height)
    var start_of_test = 0
    var mean_score: Float32 = 0.0
    for _ in range(n_splits):
        var end_of_test = min(start_of_test + test_count, X.height)
        var train_ids = List[Scalar[DType.int]](ids[end_of_test:]) + List[Scalar[DType.int]](ids[:start_of_test])
        model.fit(X[train_ids], y[train_ids])
        var test_ids = List[Scalar[DType.int]](ids[start_of_test:end_of_test])
        y_pred = model.predict(X[test_ids])
        mean_score += scoring(y[test_ids], y_pred) / Float32(n_splits)
        start_of_test += test_count
    return mean_score

def GridSearchCV[m_type: CV](X: Matrix, y: Matrix, param_grid: Dict[String, List[String]],
                            scoring: fn(Matrix, Matrix) raises -> Float32, neg_score: Bool = False, n_jobs: Int = 0, cv: Int = 5) raises -> Tuple[Dict[String, String], Float32]:
    """Exhaustive search over specified parameter values for an estimator.

    Parameters:
        m_type: Model type.

    Args:
        X: Samples.
        y: Targets.
        param_grid: Dictionary with parameters names as keys and lists of parameter settings to try as values.
        scoring: Scoring function.
        neg_score: Invert the scoring results when finding the best params.
        n_jobs: Number of jobs to run in parallel. `-1` means using all processors.
        cv: Number of folds in a KFold.

    Returns:
        Best parameters.
    """
    var dict_values = List[List[String]]()
    for key in param_grid.keys():
        dict_values.append(param_grid[key].copy())
    var combinations = cartesian_product(dict_values)
    var scores = Matrix(1, len(combinations))
    var params = List[Dict[String, String]](capacity=len(combinations))
    params.resize(len(combinations), Dict[String, String]())
    if n_jobs == 0:
        for i in range(len(combinations)):
            var j = 0
            for key in param_grid.keys():
                params[i][key] = combinations[i][j]
                j += 1
            var model = m_type(params[i])
            var score = KFold(model, X, y, scoring, cv)
            if neg_score:
                score *= -1
            scores.data[i] = score
    else:
        var n_workers = n_jobs
        if n_workers == -1:
            n_workers = num_performance_cores()
        @parameter
        def p(i: Int):
            params[i] = Dict[String, String]()
            var j = 0
            for key in param_grid.keys():
                params[i][key] = combinations[i][j]
                j += 1
            try:
                var model = m_type(params[i])
                var score = KFold(model, X, y, scoring, cv)
                if neg_score:
                    score *= -1
                scores.data[i] = score
            except e:
                print('Error:', e)
        parallelize[p](len(combinations), n_workers)
    var best_score = scores.max()
    var best = -1
    for i in range(len(scores)):
        if scores.data[i] == best_score:
            best = i
            break
    var best_params = params[best].copy()
    if neg_score:
        best_score *= -1
    return best_params^, best_score
