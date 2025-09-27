from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CVM, entropy, entropy_precompute, gini, gini_precompute, mse_loss, mse_loss_precompute
from algorithm import parallelize
import math
import random

struct Node(Copyable, Movable):
    var feature: Int
    var threshold: Float32
    var left: UnsafePointer[Node]
    var right: UnsafePointer[Node]
    var value: Float32

    fn __init__(
        out self, feature: Int = -1, threshold: Float32 = 0.0, left: UnsafePointer[Node] = UnsafePointer[Node](), right: UnsafePointer[Node] = UnsafePointer[Node](), value: Float32 = math.inf[DType.float32]()
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

struct DecisionTree(CVM, Copyable, Movable, ImplicitlyCopyable):
    """A decision tree supporting both classification and regression."""
    var criterion: String
    """The function to measure the quality of a split:
    For classification -> 'entropy', 'gini';
    For regression -> 'mse'.
    """
    var loss_func: fn(Matrix) raises -> Float32
    var c_func: fn(Float32, List[Int]) raises -> Float32
    var r_func: fn(Int, Float32, Float32) raises -> Float32
    var min_samples_split: Int
    """The minimum number of samples required to split an internal node."""
    var max_depth: Int
    """The maximum depth of the tree."""
    var n_feats: Int
    """The number of features to consider when looking for the best split."""
    var root: UnsafePointer[Node]
    
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
        self.root = UnsafePointer[Node]()

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
        self.root = UnsafePointer[Node]()

    fn _moveinit_(mut self, mut existing: Self):
        self.criterion = existing.criterion
        self.loss_func = existing.loss_func
        self.min_samples_split = existing.min_samples_split
        self.max_depth = existing.max_depth
        self.n_feats = existing.n_feats
        self.root = existing.root
        existing.criterion = ''
        existing.min_samples_split = existing.max_depth = existing.n_feats = 0
        existing.root = UnsafePointer[Node]()

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

    fn _grow_tree(self, X: Matrix, y: Matrix, depth: Int = 0) raises -> UnsafePointer[Node]:
        var unique_targets: Int
        var freq = Dict[Int, Int]()
        if self.criterion == 'mse':
            unique_targets = y.is_uniquef()
        else:
            freq = y.unique()
            unique_targets = len(freq)

        var new_node = UnsafePointer[Node].alloc(1)
        # stopping criteria
        if (
            depth >= self.max_depth
            or unique_targets == 1
            or X.height < self.min_samples_split
        ):
            new_node.init_pointee_move(Node(value = set_value(y, freq, self.criterion)))
            return new_node

        var feat_idxs = Matrix.rand_choice(X.width, self.n_feats, False, seed = False)

        # greedily select the best split according to information gain
        var best_feat: Int
        var best_thresh: Float32
        best_feat, best_thresh = _best_criteria(X, y, feat_idxs, self.loss_func, self.c_func, self.r_func, self.criterion)
        # grow the children that result from the split
        var left_right_idxs = _split(X['', best_feat], best_thresh)
        var left_idxs = left_right_idxs[0].copy()
        var right_idxs = left_right_idxs[1].copy()
        var left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        var right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        new_node.init_pointee_move(Node(best_feat, best_thresh, left, right))
        return new_node

fn set_value(y: Matrix, freq: Dict[Int, Int], criterion: String) raises -> Float32:
    if criterion == 'mse':
        return y.mean()
    var max_val: Int = 0
    var most_common: Int = 0
    for k in freq.keys():
        if freq[k] > max_val:
            max_val = freq[k]
            most_common = k
    return Float32(most_common)

fn _best_criteria(X: Matrix, y: Matrix, feat_idxs: List[Scalar[DType.int]], loss_func: fn(Matrix) raises -> Float32, c_precompute: fn(Float32, List[Int]) raises -> Float32, r_precompute: fn(Int, Float32, Float32) raises -> Float32, criterion: String) raises -> Tuple[Int, Float32]:
    var parent_loss = loss_func(y)
    var max_gains = Matrix(1, len(feat_idxs))
    max_gains.fill(-math.inf[DType.float32]())
    var best_thresholds = Matrix(1, len(feat_idxs))
    if criterion != 'mse':
        var num_classes = Int(y.max() + 1)  # assuming y is 0-indexed
        @parameter
        fn p_c(idx: Int):
            try:
                var column = X['', Int(feat_idxs[idx]), unsafe=True]
                var sorted_indices = column.argsort_inplace()
                var y_sorted = y[sorted_indices]
                var left_histogram = List[Int](capacity=num_classes)
                left_histogram.resize(num_classes, 0)
                var right_histogram = y_sorted.bincount()

                for step in range(1, len(y)):
                    var c = y_sorted.data[step - 1]
                    left_histogram[Int(c)] += 1
                    right_histogram[Int(c)] -= 1

                    if column.data[step] == column.data[step - 1]:
                        continue  # skip redundant thresholds
                    
                    var n_left = Float32(step)
                    var n_right = Float32(len(y) - step)

                    var child_loss = (n_left / len(y)) * c_precompute(n_left, left_histogram) + (n_right / len(y)) * c_precompute(n_right, right_histogram)
                    var ig = parent_loss - child_loss
                    if ig > max_gains.data[idx]:
                        max_gains.data[idx] = ig
                        best_thresholds.data[idx] = (column.data[step] + column.data[step - 1]) / 2.0  # midpoint
            except e:
                print('Error:', e)
        parallelize[p_c](len(feat_idxs))
    else:
        var sum_total = y.sum()
        var sum_sq_total = (y ** 2).sum()
        @parameter
        fn p_r(idx: Int):
            try:
                var column = X['', Int(feat_idxs[idx]), unsafe=True]
                var sorted_indices = column.argsort_inplace()
                var y_sorted = y[sorted_indices]

                var left_sum: Float32 = 0.0
                var left_sum_sq: Float32 = 0.0
                var right_sum = sum_total
                var right_sum_sq = sum_sq_total

                for step in range(1, len(y)):
                    var yi = y_sorted.data[step - 1]
                    left_sum += yi
                    left_sum_sq += yi ** 2
                    right_sum -= yi
                    right_sum_sq -= yi ** 2

                    if column.data[step] == column.data[step - 1]:
                        continue  # skip redundant thresholds
                    
                    var n_left = step
                    var n_right = len(y) - step

                    var child_loss = (Float32(n_left) / len(y)) * r_precompute(n_left, left_sum, left_sum_sq) + (Float32(n_right) / len(y)) * r_precompute(n_right, right_sum, right_sum_sq)
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

fn _traverse_tree(x: Matrix, node: UnsafePointer[Node]) -> Float32:
    if node[].is_leaf_node():
        return node[].value

    if x.data[node[].feature] <= node[].threshold:
        return _traverse_tree(x, node[].left)
    return _traverse_tree(x, node[].right)

fn delTree(node: UnsafePointer[Node]):
    if node[].left:
        delTree(node[].left)
    if node[].right:
        delTree(node[].right)
    node.free()
