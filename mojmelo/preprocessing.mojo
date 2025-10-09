from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CVM, CVP, cartesian_product, ids_to_numpy
from algorithm import parallelize
from sys import num_performance_cores
from memory import UnsafePointer
from python import Python, PythonObject
import time
import random

fn normalize(data: Matrix, norm: String = 'l2') raises -> Tuple[Matrix, Matrix]:
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
                fn p1(i: Int):
                    try:
                        norms.data[i] = data[i].norm()
                    except e:
                        print('Error:', e)
                parallelize[p1](norms.height)

    @parameter
    fn p2(i: Int):
        try:
            if norms.data[i] != 0.0:
                z[i] = data[i] / norms.data[i]
            else:
                z[i].fill_zero()
        except e:
            print('Error:', e)
    parallelize[p2](z.height)

    return z^, norms^

fn inv_normalize(z: Matrix, norms: Matrix) raises -> Matrix:
    """Reproduce normalized data given its norms.

    Args:
        z: Normalized data.
        norms: Norms.

    Returns:
        Original data.
    """
    return z.ele_mul(norms)

fn MinMaxScaler(data: Matrix, feature_range: Tuple[Int, Int] = (0, 1)) raises -> Tuple[Matrix, Matrix, Matrix]:
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

fn MinMaxScaler(data: Matrix, x_min: Matrix, x_max: Matrix, feature_range: Tuple[Int, Int] = (0, 1)) raises -> Matrix:
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

fn inv_MinMaxScaler(z: Matrix, x_min: Matrix, x_max: Matrix, feature_range: Tuple[Int, Int] = (0, 1)) raises -> Matrix:
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

fn StandardScaler(data: Matrix) raises -> Tuple[Matrix, Matrix, Matrix]:
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

fn StandardScaler(data: Matrix, mu: Matrix, sigma: Matrix) raises -> Matrix:
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

fn inv_StandardScaler(z: Matrix, mu: Matrix, sigma: Matrix) raises -> Matrix:
    """Reproduce scaled data given its mean and standard deviation.

    Args:
        z: Scaled data.
        mu: Mean.
        sigma: Standard Deviation.

    Returns:
        Original data.
    """
    return z.ele_mul(sigma.where(sigma == 0.0, 1.0, sigma)) + mu

fn train_test_split(X: Matrix, y: Matrix, *, test_size: Float16 = 0.5, train_size: Float16 = 0.0) raises -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    """Split matrices into random train and test subsets."""
    var test_ratio = test_size if train_size <= 0.0 else 1.0 - train_size
    var ids = Matrix.rand_choice(X.height, X.height, False)
    var split_i = Int(X.height - (test_ratio * X.height))
    return X[ids[:split_i]], X[ids[split_i:]], y[ids[:split_i]], y[ids[split_i:]]

fn train_test_split(X: Matrix, y: Matrix, *, random_state: Int, test_size: Float16 = 0.5, train_size: Float16 = 0.0) raises -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    """Split matrices into random train and test subsets."""
    var test_ratio = test_size if train_size <= 0.0 else 1.0 - train_size
    random.seed(random_state)
    var ids = Matrix.rand_choice(X.height, X.height, False, seed = False)
    var split_i = Int(X.height - (test_ratio * X.height))
    return X[ids[:split_i]], X[ids[split_i:]], y[ids[:split_i]], y[ids[split_i:]]

@fieldwise_init
struct SplittedPO(Copyable, Movable, ImplicitlyCopyable):
    var train: PythonObject
    var test: PythonObject

fn train_test_split(X: Matrix, y: PythonObject, *, test_size: Float16 = 0.5, train_size: Float16 = 0.0) raises -> Tuple[Matrix, Matrix, SplittedPO]:
    """Split matrix and python object into random train and test subsets."""
    var test_ratio = test_size if train_size <= 0.0 else 1.0 - train_size
    var ids = Matrix.rand_choice(X.height, X.height, False)
    var split_i = Int(X.height - (test_ratio * X.height))
    return X[ids[:split_i]], X[ids[split_i:]], SplittedPO(y[ids_to_numpy(ids[:split_i])], y[ids_to_numpy(ids[split_i:])])

fn train_test_split(X: Matrix, y: PythonObject, *, random_state: Int, test_size: Float16 = 0.5, train_size: Float16 = 0.0) raises -> Tuple[Matrix, Matrix, SplittedPO]:
    """Split matrix and python object into random train and test subsets."""
    var test_ratio = test_size if train_size <= 0.0 else 1.0 - train_size
    random.seed(random_state)
    var ids = Matrix.rand_choice(X.height, X.height, False, seed = False)
    var split_i = Int(X.height - (test_ratio * X.height))
    return X[ids[:split_i]], X[ids[split_i:]], SplittedPO(y[ids_to_numpy(ids[:split_i])], y[ids_to_numpy(ids[split_i:])])

fn KFold[m_type: CVM](mut model: m_type, X: Matrix, y: Matrix, scoring: fn(Matrix, Matrix) raises -> Float32, n_splits: Int = 5) raises -> Float32:
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
        model.fit(X[ids[end_of_test:] + ids[:start_of_test]], y[ids[end_of_test:] + ids[:start_of_test]])
        y_pred = model.predict(X[ids[start_of_test:end_of_test]])
        mean_score += scoring(y[ids[start_of_test:end_of_test]], y_pred) / n_splits
        start_of_test += test_count
    return mean_score

fn KFold[m_type: CVP](mut model: m_type, X: Matrix, y: PythonObject, scoring: fn(PythonObject, List[String]) raises -> Float32, n_splits: Int = 5) raises -> Float32:
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
        model.fit(X[ids[end_of_test:] + ids[:start_of_test]], y[ids_to_numpy(ids[end_of_test:] + ids[:start_of_test])])
        y_pred = model.predict(X[ids[start_of_test:end_of_test]])
        mean_score += scoring(y[ids_to_numpy(ids[start_of_test:end_of_test])], y_pred) / n_splits
        start_of_test += test_count
    return mean_score

fn GridSearchCV[m_type: CVM](X: Matrix, y: Matrix, param_grid: Dict[String, List[String]],
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
    var dic_values = List[List[String]]()
    for i in range(len(param_grid)):
        dic_values.append(List[String]())
        dic_values[i] = param_grid._entries[i].value().value.copy()
    var combinations = cartesian_product(dic_values)
    var scores = Matrix(1, len(combinations))
    var params = UnsafePointer[Dict[String, String]].alloc(len(combinations))
    if n_jobs == 0:
        for i in range(len(combinations)):
            params[i] = Dict[String, String]()
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
        fn p(i: Int):
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
            except:
                print('Error: Failed to perform KFold!')
        parallelize[p](len(combinations), n_workers)
    var best_score = scores.max()
    var best = -1
    for i in range(len(scores)):
        if scores.data[i] == best_score:
            best = i
            break
    var best_params = params[best].copy()
    params.free()
    if neg_score:
        best_score *= -1
    return best_params^, best_score

fn GridSearchCV[m_type: CVP](X: Matrix, y: PythonObject, param_grid: Dict[String, List[String]],
                            scoring: fn(PythonObject, List[String]) raises -> Float32, neg_score: Bool = False, n_jobs: Int = 0, cv: Int = 5) raises -> Tuple[Dict[String, String], Float32]:
    """Exhaustive search over specified parameter values for an estimator.

    Parameters:
        m_type: Model type.

    Args:
        X: Samples.
        y: Targets.
        param_grid: Dictionary with parameters names as keys and lists of parameter settings to try as values.
        scoring: Scoring function.
        neg_score: Invert the scoring results when finding the best params. `-1` means using all processors.
        n_jobs: Number of jobs to run in parallel.
        cv: Number of folds in a KFold.

    Returns:
        Best parameters.
    """
    var dic_values = List[List[String]]()
    for i in range(len(param_grid)):
        dic_values.append(List[String]())
        dic_values[i] = param_grid._entries[i].value().value.copy()
    var combinations = cartesian_product(dic_values)
    var scores = Matrix(1, len(combinations))
    var params = UnsafePointer[Dict[String, String]].alloc(len(combinations))
    if n_jobs == 0:
        for i in range(len(combinations)):
            params[i] = Dict[String, String]()
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
        fn p(i: Int):
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
            except:
                print('Error: Failed to perform KFold!')
        parallelize[p](len(combinations), n_workers)
    var best_score = scores.max()
    var best = -1
    for i in range(len(scores)):
        if scores.data[i] == best_score:
            best = i
            break
    var best_params = params[best].copy()
    params.free()
    if neg_score:
        best_score *= -1
    return best_params^, best_score
