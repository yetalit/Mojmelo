from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CV, entropy, entropy_precompute, gini, gini_precompute, mse_loss, mse_loss_precompute
from algorithm import parallelize
import math
import random

struct Node(Copyable, Movable):
    var feature: Int
    var threshold: Float32
    var left: UnsafePointer[Node, MutAnyOrigin]
    var right: UnsafePointer[Node, MutAnyOrigin]
    var value: Float32

    fn __init__(
        out self, feature: Int = -1, threshold: Float32 = 0.0, left: UnsafePointer[Node, MutAnyOrigin] = UnsafePointer[Node, MutAnyOrigin](), right: UnsafePointer[Node, MutAnyOrigin] = UnsafePointer[Node, MutAnyOrigin](), value: Float32 = math.inf[DType.float32]()
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @always_inline
    fn is_leaf_node(self) -> Bool:
        return self.feature == -1

    fn __str__(self) -> String:
        if self.is_leaf_node():
            return '{' + String(self.value) + '}'
        return '<' + String(self.feature) + ': ' + String(self.threshold) + '>'

struct DecisionTree(CV, Copyable, Movable, ImplicitlyCopyable):
    """A decision tree supporting both classification and regression."""
    var criterion: String
    """The function to measure the quality of a split:
    For classification -> 'entropy', 'gini';
    For regression -> 'mse'.
    """
    var loss_func: fn(Matrix, Matrix, Float32) raises -> Float32
    var c_func: fn(Float32, List[Int]) raises -> Float32
    var r_func: fn(Int, Float32, Float32) raises -> Float32
    var min_samples_split: Int
    """The minimum number of samples required to split an internal node."""
    var max_depth: Int
    """The maximum depth of the tree."""
    var n_feats: Int
    """The number of features to consider when looking for the best split."""
    var root: UnsafePointer[Node, MutAnyOrigin]
    
    fn __init__(out self, criterion: String = 'gini', min_samples_split: Int = 2, max_depth: Int = 100, n_feats: Int = -1, random_state: Int = 42):
        self.criterion = criterion.lower()
        if self.criterion == 'gini':
            self.loss_func = gini
            self.c_func = gini_precompute
            self.r_func = mse_loss_precompute
        elif self.criterion == 'mse':
            self.loss_func = mse_loss
            self.r_func = mse_loss_precompute
            self.c_func = entropy_precompute
        else:
            self.loss_func = entropy
            self.c_func = entropy_precompute
            self.r_func = mse_loss_precompute
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        if random_state != -1:
            random.seed(random_state)
        self.root = UnsafePointer[Node, MutAnyOrigin]()

    fn __init__(out self, params: Dict[String, String]) raises:
        if 'criterion' in params:
            self.criterion = params['criterion'].lower()
        else:
            self.criterion = 'gini'
        if self.criterion == 'gini':
            self.loss_func = gini
            self.c_func = gini_precompute
            self.r_func = mse_loss_precompute
        elif self.criterion == 'mse':
            self.loss_func = mse_loss
            self.r_func = mse_loss_precompute
            self.c_func = entropy_precompute
        else:
            self.loss_func = entropy
            self.c_func = entropy_precompute
            self.r_func = mse_loss_precompute
        if 'min_samples_split' in params:
            self.min_samples_split = atol(String(params['min_samples_split']))
        else:
            self.min_samples_split = 2
        if 'max_depth' in params:
            self.max_depth = atol(String(params['max_depth']))
        else:
            self.max_depth = 100
        if 'n_feats' in params:
            self.n_feats = atol(String(params['n_feats']))
        else:
            self.n_feats = -1
        if 'random_state' in params:
            var seed = atol(String(params['random_state']))
            if seed != -1:
                random.seed(seed)
        else:
            random.seed(42)
        self.root = UnsafePointer[Node, MutAnyOrigin]()

    fn _moveinit_(mut self, mut existing: Self):
        self.criterion = existing.criterion
        self.loss_func = existing.loss_func
        self.min_samples_split = existing.min_samples_split
        self.max_depth = existing.max_depth
        self.n_feats = existing.n_feats
        self.root = existing.root
        existing.criterion = ''
        existing.min_samples_split = existing.max_depth = existing.n_feats = 0
        existing.root = UnsafePointer[Node, MutAnyOrigin]()

    fn __del__(deinit self):
        if self.root:
            delTree(self.root)

    fn fit(mut self, X: Matrix, y: Matrix) raises:
        """Build a decision tree from the training set."""
        self.n_feats = X.width if self.n_feats < 1 else min(self.n_feats, X.width)
        if y.width != 1:
            self.root = self._grow_tree(X.asorder('f'), y.reshape(y.size, 1))
        else:
            self.root = self._grow_tree(X.asorder('f'), y)

    fn fit_weighted(mut self, X: Matrix, y_with_weights: Matrix) raises:
        """Build a decision tree from a weighted training set."""
        self.n_feats = X.width if self.n_feats < 1 else min(self.n_feats, X.width)
        self.root = self._grow_tree(X.asorder('f'), y_with_weights)

    fn predict(self, X: Matrix) raises -> Matrix:
        """Predict class or regression value for X.
        
        Returns:
            The predicted values.
        """
        var y_predicted = Matrix(X.height, 1)
        @parameter
        fn p(i: Int):
            y_predicted.data[i] = _traverse_tree(X[i, unsafe=True], self.root)
        parallelize[p](X.height)
        return y_predicted^

    fn _grow_tree(self, X: Matrix, y: Matrix, depth: Int = 0) raises -> UnsafePointer[Node, MutAnyOrigin]:
        var _y = y['', 0]
        var weights = Matrix(0, 0)
        if y.width == 2:
            weights = y['', 1]
        var unique_targets: Int
        var freq = List[List[Int]]()
        if self.criterion == 'mse':
            unique_targets = _y.is_uniquef()
        else:
            freq = _y.unique() if y.width == 1 else _y.unique(weights)
            unique_targets = len(freq)

        var new_node = alloc[Node](1)
        # stopping criteria
        if (
            depth >= self.max_depth
            or unique_targets == 1
            or X.height < self.min_samples_split
        ):
            new_node.init_pointee_move(Node(value = set_value(_y, weights, freq, self.criterion)))
            return new_node

        var feat_idxs = Matrix.rand_choice(X.width, self.n_feats, False, seed = False)

        # greedily select the best split according to information gain
        var best_feat: Int
        var best_thresh: Float32
        best_feat, best_thresh = _best_criteria(X, y, _y, weights, feat_idxs, self.loss_func, self.c_func, self.r_func, self.criterion)
        # grow the children that result from the split
        var left_right_idxs = _split(X['', best_feat], best_thresh)
        var left_idxs = left_right_idxs[0].copy()
        var right_idxs = left_right_idxs[1].copy()
        var left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        var right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        new_node.init_pointee_move(Node(best_feat, best_thresh, left, right))
        return new_node

fn set_value(y: Matrix, weights: Matrix, freq: List[List[Int]], criterion: String) raises -> Float32:
    if criterion == 'mse':
        if weights.size == 0:
            return y.mean()
        return y.mean_weighted(weights, weights.sum())
    var max_val: Int = 0
    var most_common: Int = 0
    for i in range(len(freq)):
        if len(freq[i]) > max_val:
            max_val = len(freq[i])
            most_common = i
    return Float32(most_common)

fn _best_criteria(X: Matrix, y: Matrix, _y: Matrix, weights: Matrix, feat_idxs: List[Scalar[DType.int]], loss_func: fn(Matrix, Matrix, Float32) raises -> Float32, c_precompute: fn(Float32, List[Int]) raises -> Float32, r_precompute: fn(Int, Float32, Float32) raises -> Float32, criterion: String) raises -> Tuple[Int, Float32]:
    var total_samples = len(_y) if y.width == 1 else weights.sum()
    var parent_loss = loss_func(_y, weights, total_samples)
    var max_gains = Matrix(1, len(feat_idxs))
    max_gains.fill(-math.inf[DType.float32]())
    var best_thresholds = Matrix(1, len(feat_idxs))
    if criterion != 'mse':
        var num_classes = Int(_y.max() + 1)  # assuming y is 0-indexed
        @parameter
        fn p_c(idx: Int):
            try:
                var column = X['', Int(feat_idxs[idx]), unsafe=True]
                var left_histogram = List[Int](capacity=num_classes)
                left_histogram.resize(num_classes, 0)
                var right_histogram = _y.bincount() if weights.size == 0 else _y.bincount(weights)
                var sorted_indices = column.argsort_inplace()
                var y_sorted = y[sorted_indices]
                var n_left: Float32 = 0.0
                for step in range(1, len(_y)):
                    var c = y_sorted.data[step - 1]
                    if y_sorted.width == 1:
                        n_left += 1
                        left_histogram[Int(c)] += 1
                        right_histogram[Int(c)] -= 1
                    else:
                        var weight = Int(y_sorted[step - 1, 1])
                        n_left += weight
                        left_histogram[Int(c)] += weight
                        right_histogram[Int(c)] -= weight

                    if column.data[step] == column.data[step - 1]:
                        continue  # skip redundant thresholds
                    
                    var n_right = total_samples - n_left

                    var child_loss = (n_left / total_samples) * c_precompute(n_left, left_histogram) + (n_right / total_samples) * c_precompute(n_right, right_histogram)
                    var ig = parent_loss - child_loss
                    if ig > max_gains.data[idx]:
                        max_gains.data[idx] = ig
                        best_thresholds.data[idx] = (column.data[step] + column.data[step - 1]) / 2.0  # midpoint
            except e:
                print('Error:', e)
        parallelize[p_c](len(feat_idxs))
    else:
        var sum_total = _y.sum() if y.width == 1 else _y.ele_mul(weights).sum()
        var sum_sq_total = _y.ele_mul(_y).sum() if y.width == 1 else (_y.ele_mul(_y).ele_mul(weights)).sum()
        @parameter
        fn p_r(idx: Int):
            try:
                var column = X['', Int(feat_idxs[idx]), unsafe=True]
                var sorted_indices = column.argsort_inplace()
                var y_sorted = y[sorted_indices]

                var left_sum: Float32 = 0.0
                var left_sum_sq: Float32 = 0.0
                var n_left: Float32 = 0.0
                for step in range(1, len(_y)):
                    var yi = y_sorted.data[step - 1]
                    if y_sorted.width == 1:
                        n_left += 1
                        left_sum += yi
                        left_sum_sq += yi * yi
                    else:
                        var weight = y_sorted[step - 1, 1]
                        n_left += weight
                        left_sum += yi * weight
                        left_sum_sq += yi * yi * weight

                    if column.data[step] == column.data[step - 1]:
                        continue  # skip redundant thresholds
                    
                    var n_right = total_samples - n_left

                    var child_loss = (n_left / total_samples) * r_precompute(Int(n_left), left_sum, left_sum_sq) + (n_right / total_samples) * r_precompute(Int(n_right), sum_total - left_sum, sum_sq_total - left_sum_sq)
                    var ig = parent_loss - child_loss
                    if ig > max_gains.data[idx]:
                        max_gains.data[idx] = ig
                        best_thresholds.data[idx] = (column.data[step] + column.data[step - 1]) / 2.0  # midpoint
            except e:
                print('Error:', e)
        parallelize[p_r](len(feat_idxs))
    
    var feat_idx = max_gains.argmax()
    return Int(feat_idxs[feat_idx]), best_thresholds.data[feat_idx]

@always_inline
fn _split(X_column: Matrix, split_thresh: Float32) -> Tuple[List[Int], List[Int]]:
    return X_column.argwhere_l(X_column <= split_thresh), X_column.argwhere_l(X_column > split_thresh)

fn _traverse_tree(x: Matrix, node: UnsafePointer[Node, MutAnyOrigin]) -> Float32:
    if node[].is_leaf_node():
        return node[].value

    if x.data[node[].feature] <= node[].threshold:
        return _traverse_tree(x, node[].left)
    return _traverse_tree(x, node[].right)

fn delTree(node: UnsafePointer[Node, MutAnyOrigin]):
    if node[].left:
        delTree(node[].left)
    if node[].right:
        delTree(node[].right)
    node.free()
