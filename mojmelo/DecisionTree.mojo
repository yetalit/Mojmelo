from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CVM, entropy, gini, mse_loss
from memory import UnsafePointer
from collections import Dict
import math

@value
struct Node:
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

struct DecisionTree(CVM):
    var criterion: String
    var loss_func: fn(Matrix) raises -> Float32
    var min_samples_split: Int
    var max_depth: Int
    var n_feats: Int
    var threshold_precision: Float32
    var root: UnsafePointer[Node]
    
    fn __init__(out self, criterion: String = 'gini', min_samples_split: Int = 2, max_depth: Int = 100, n_feats: Int = -1, threshold_precision: Float32 = 0.001):
        self.criterion = criterion.lower()
        if self.criterion == 'gini':
            self.loss_func = gini
        elif self.criterion == 'mse':
            self.loss_func = mse_loss
        else:
            self.loss_func = entropy
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.threshold_precision = threshold_precision
        self.root = UnsafePointer[Node]()

    fn __init__(out self, params: Dict[String, String]) raises:
        if 'criterion' in params:
            self.criterion = params['criterion'].lower()
        else:
            self.criterion = 'gini'
        if self.criterion == 'gini':
            self.loss_func = gini
        elif self.criterion == 'mse':
            self.loss_func = mse_loss
        else:
            self.loss_func = entropy
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
        if 'threshold_precision' in params:
            self.threshold_precision = atof(String(params['threshold_precision'])).cast[DType.float32]()
        else:
            self.threshold_precision = 0.001
        self.root = UnsafePointer[Node]()

    fn _moveinit_(mut self, mut existing: Self):
        self.criterion = existing.criterion
        self.loss_func = existing.loss_func
        self.min_samples_split = existing.min_samples_split
        self.max_depth = existing.max_depth
        self.n_feats = existing.n_feats
        self.threshold_precision = existing.threshold_precision
        self.root = existing.root
        existing.criterion = ''
        existing.min_samples_split = existing.max_depth = existing.n_feats = 0
        existing.root = UnsafePointer[Node]()

    fn __del__(owned self):
        if self.root:
            delTree(self.root)

    fn fit(mut self, X: Matrix, y: Matrix) raises:
        self.n_feats = X.width if self.n_feats == -1 else min(self.n_feats, X.width)
        self.root = self._grow_tree(X, y)

    fn predict(self, X: Matrix) raises -> Matrix:
        var y_predicted = Matrix(X.height, 1)
        for i in range(X.height):
            y_predicted.data[i] = _traverse_tree(X[i], self.root)
        return y_predicted

    fn _grow_tree(self, X: Matrix, y: Matrix, depth: Int = 0) raises -> UnsafePointer[Node]:
        var unique_targets = 0
        var freq = Dict[Int, Int]()
        var freqf = List[Float32]()
        if self.criterion == 'mse':
            freqf = y.uniquef(self.threshold_precision)
            unique_targets = len(freqf)
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

        var feat_idxs = Matrix.rand_choice(X.width, self.n_feats, False)

        # greedily select the best split according to information gain
        var best_feat: Int
        var best_thresh: Float32
        best_feat, best_thresh = _best_criteria(X, y, feat_idxs, self.threshold_precision, self.loss_func)
        # grow the children that result from the split
        var left_idxs: List[Int]
        var right_idxs: List[Int]
        left_idxs, right_idxs = _split(X['', best_feat], best_thresh)
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
        if freq[k[]] > max_val:
            max_val = freq[k[]]
            most_common = k[]
    return Float32(most_common)

fn _best_criteria(X: Matrix, y: Matrix, feat_idxs: List[Int], threshold_precision: Float32, loss_func: fn(Matrix) raises -> Float32) raises -> Tuple[Int, Float32]:
    var parent_loss = loss_func(y)
    var split_idx = feat_idxs[0]
    var split_thresh = X[0, split_idx]
    var best_gain = -math.inf[DType.float32]()
    for feat_idx in feat_idxs:
        var X_column = X['', feat_idx[]]
        var thresholds = X_column.uniquef(threshold_precision)
        for threshold in thresholds:
            var gain = _information_gain(parent_loss, y, X_column, threshold[], loss_func)
            if gain > best_gain:
                best_gain = gain
                split_idx = feat_idx[]
                split_thresh = threshold[]

    return split_idx, split_thresh

@always_inline
fn _information_gain(parent_loss: Float32, y: Matrix, X_column: Matrix, split_thresh: Float32, loss_func: fn(Matrix) raises -> Float32) raises -> Float32:
    # generate split
    var left_idxs: List[Int]
    var right_idxs: List[Int]
    left_idxs, right_idxs = _split(X_column, split_thresh)

    if len(left_idxs) == 0 or len(right_idxs) == 0:
        return 0.0

    # compute the weighted avg. of the loss for the children
    var child_loss = (len(left_idxs) / Float32(y.size)) * loss_func(y[left_idxs]) + (len(right_idxs) / Float32(y.size)) * loss_func(y[right_idxs])
    # information gain is difference in loss before vs. after split
    return parent_loss - child_loss

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
