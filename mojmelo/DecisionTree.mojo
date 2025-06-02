from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CVM, entropy, gini, mse_loss, lt
from memory import UnsafePointer
from collections import Dict
from algorithm import parallelize
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

@value
struct DecisionTree(CVM):
    var criterion: String
    var loss_func: fn(Matrix) raises -> Float32
    var min_samples_split: Int
    var max_depth: Int
    var n_feats: Int
    var n_bins: Int
    var root: UnsafePointer[Node]
    
    fn __init__(out self, criterion: String = 'gini', min_samples_split: Int = 2, max_depth: Int = 100, n_feats: Int = -1, n_bins: Int = 0):
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
        self.n_bins = n_bins
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
        if 'n_bins' in params:
            self.n_bins = atol(String(params['n_bins']))
        else:
            self.n_bins = 0
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

    fn __del__(owned self):
        if self.root:
            delTree(self.root)

    fn fit(mut self, X: Matrix, y: Matrix) raises:
        self.n_feats = X.width if self.n_feats == -1 else min(self.n_feats, X.width)
        var X_sorted = X.asorder('f')
        @parameter
        fn p(i: Int):
            var X_column = X_sorted['', i, unsafe=True]
            sort[lt](Span[Float32, __origin_of(X_column)](ptr= X_column.data, length= X_column.size))
        parallelize[p](X_sorted.width)
        self.root = self._grow_tree(X_sorted, y)

    fn predict(self, X: Matrix) raises -> Matrix:
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
            unique_targets = len(y.uniquef())
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
        best_feat, best_thresh = _best_criteria(X, y, feat_idxs, self.n_bins, self.loss_func)
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

fn _best_criteria(X: Matrix, y: Matrix, feat_idxs: List[Int], n_bins: Int, loss_func: fn(Matrix) raises -> Float32) raises -> Tuple[Int, Float32]:
    var parent_loss = loss_func(y)
    var max_gains = Matrix(1, len(feat_idxs))
    var best_thresholds = Matrix(1, len(feat_idxs))
    var columns = List[Matrix](capacity=len(feat_idxs))
    columns.resize(len(feat_idxs), Matrix(0, 0))
    var thresholds_list = List[Matrix](capacity=len(feat_idxs))
    thresholds_list.resize(len(feat_idxs), Matrix(0, 0))

    @parameter
    fn prepare(i: Int):
        columns[i] = X['', feat_idxs[i], unsafe=True]
        var vals: Matrix
        try:
            if n_bins <= 0:
                vals = columns[i].uniquef()
                if vals.size == 1:
                    vals = vals.concatenate(vals, axis=0)
            else:
                vals = Matrix.linspace(columns[i].data[0], columns[i].data[columns[i].height - 1], n_bins + 1).T()
                if columns[i].data[0] == columns[i].data[columns[i].height - 1]:
                    vals = vals.load_rows(2)
            thresholds_list[i] = (vals.load_rows(vals.size - 1) + vals[True, 1, 0]) / 2
        except e:
            print('Error:', e)
    parallelize[prepare](len(feat_idxs))

    for i in range(len(feat_idxs)):
        var gains = Matrix(len(thresholds_list[i]), 1)
        @parameter
        fn p(i_t: Int):
            try:
                gains.data[i_t] = _information_gain(parent_loss, y, columns[i], thresholds_list[i].data[i_t], loss_func)
            except:
                print('Error: Failed to calculate information gain!')
        parallelize[p](len(thresholds_list[i]))
        var max_gain = gains.argmax()
        max_gains.data[i] = gains.data[max_gain]
        best_thresholds.data[i] = thresholds_list[i].data[max_gain]
    
    var feat_idx = max_gains.argmax()
    return feat_idxs[feat_idx], best_thresholds.data[feat_idx]

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
