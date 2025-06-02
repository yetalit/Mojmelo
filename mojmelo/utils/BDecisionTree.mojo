from mojmelo.DecisionTree import Node
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import lt
from memory import UnsafePointer
from algorithm import parallelize
import math

@value
struct BDecisionTree:
    var min_samples_split: Int
    var max_depth: Int
    var reg_lambda: Float32 
    var gamma: Float32
    var root: UnsafePointer[Node]
    
    fn __init__(out self, min_samples_split: Int = 10, max_depth: Int = 3, reg_lambda: Float32 = 1.0, gamma: Float32 = 0.0):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.root = UnsafePointer[Node]()

    fn _moveinit_(mut self, mut existing: Self):
        self.min_samples_split = existing.min_samples_split
        self.max_depth = existing.max_depth
        self.reg_lambda = existing.reg_lambda
        self.gamma = existing.gamma
        self.root = existing.root
        existing.min_samples_split = existing.max_depth = 0
        existing.reg_lambda = existing.gamma = 0.0
        existing.root = UnsafePointer[Node]()

    fn __del__(owned self):
        if self.root:
            delTree(self.root)

    fn fit(mut self, X: Matrix, g: Matrix, h: Matrix) raises:
        var X_sorted = X.asorder('f')
        @parameter
        fn p(i: Int):
            var X_column = X_sorted['', i, unsafe=True]
            sort[lt](Span[Float32, __origin_of(X_column)](ptr= X_column.data, length= X_column.size))
        parallelize[p](X_sorted.width)
        self.root = self._grow_tree(X_sorted, g, h)

    fn predict(self, X: Matrix) raises -> Matrix:
        var y_predicted = Matrix(X.height, 1)
        @parameter
        fn p(i: Int):
            y_predicted.data[i] = _traverse_tree(X[i, unsafe=True], self.root)
        parallelize[p](X.height)
        return y_predicted^

    fn _grow_tree(self, X: Matrix, g: Matrix, h: Matrix, depth: Int = 0) raises -> UnsafePointer[Node]:
        var new_node = UnsafePointer[Node].alloc(1)
        # stopping criteria
        if (
            depth >= self.max_depth
            or X.height < self.min_samples_split
        ):
            new_node.init_pointee_move(Node(value = leaf_score(self.reg_lambda, g, h)))
            return new_node

        var feat_idxs = Matrix.rand_choice(X.width, X.width, False)

        # greedily select the best split according to information gain
        var best_feat: Int
        var best_thresh: Float32
        var best_gain: Float32
        best_feat, best_thresh, best_gain = _best_criteria(self.reg_lambda, X, g, h, feat_idxs)
        if best_gain <= self.gamma:
            # The best gain is less than gamma
            new_node.init_pointee_move(Node(value = leaf_score(self.reg_lambda, g, h)))
            return new_node
        
        # grow the children that result from the split
        var left_idxs: List[Int]
        var right_idxs: List[Int]
        left_idxs, right_idxs = _split(X['', best_feat], best_thresh)
        var left = self._grow_tree(X[left_idxs], g[left_idxs], h[left_idxs], depth + 1)
        var right = self._grow_tree(X[right_idxs], g[right_idxs], h[right_idxs], depth + 1)
        new_node.init_pointee_move(Node(best_feat, best_thresh, left, right))
        return new_node

@always_inline
fn leaf_score(reg_lambda: Float32, g: Matrix, h: Matrix) raises -> Float32:
    '''
    Given the gradient and hessian of a tree leaf,
    return the prediction (score) at this leaf.
    The score is -G/(H+λ).
    '''
    return -g.sum() / (h.sum() + reg_lambda)

@always_inline
fn leaf_loss(reg_lambda: Float32, g: Matrix, h: Matrix) raises -> Float32:
    '''
    Given the gradient and hessian of a tree leaf,
    return the minimized loss at this leaf.
    The minimized loss is -0.5*G^2/(H+λ).
    .'''
    return -0.5 * (g.sum() ** 2) / (h.sum() + reg_lambda)

fn _best_criteria(reg_lambda: Float32, X: Matrix, g: Matrix, h: Matrix, feat_idxs: List[Int]) raises -> Tuple[Int, Float32, Float32]:
    var parent_loss = leaf_loss(reg_lambda, g, h)
    var max_gains = Matrix(1, len(feat_idxs))
    var best_thresholds = Matrix(1, len(feat_idxs))
    var columns = List[Matrix](capacity=len(feat_idxs))
    columns.resize(len(feat_idxs), Matrix(0, 0))
    var thresholds_list = List[Matrix](capacity=len(feat_idxs))
    thresholds_list.resize(len(feat_idxs), Matrix(0, 0))

    @parameter
    fn prepare(i: Int):
        columns[i] = X['', feat_idxs[i], unsafe=True]
        var unique_vals = columns[i].uniquef()
        if unique_vals.size == 1:
            thresholds_list[i] = unique_vals^
        else:
            try:
                thresholds_list[i] = (unique_vals.load_rows(unique_vals.size - 1) + unique_vals[True, 1, 0]) / 2
            except:
                print('Error: Loading values failed!')
    parallelize[prepare](len(feat_idxs))

    for i in range(len(feat_idxs)):
        var gains = Matrix(len(thresholds_list[i]), 1)
        @parameter
        fn p(i_t: Int):
            try:
                gains.data[i_t] = _information_gain(parent_loss, reg_lambda, g, h, columns[i], thresholds_list[i].data[i_t])
            except:
                print('Error: Failed to calculate information gain!')
        parallelize[p](len(thresholds_list[i]))
        var max_gain = gains.argmax()
        max_gains.data[i] = gains.data[max_gain]
        best_thresholds.data[i] = thresholds_list[i].data[max_gain]
    
    var feat_idx = max_gains.argmax()
    return feat_idxs[feat_idx], best_thresholds.data[feat_idx], max_gains.data[feat_idx]

@always_inline
fn _information_gain(parent_loss: Float32, reg_lambda: Float32, g: Matrix, h: Matrix, X_column: Matrix, split_thresh: Float32) raises -> Float32:
    # generate split
    var left_idxs: List[Int]
    var right_idxs: List[Int]
    left_idxs, right_idxs = _split(X_column, split_thresh)

    if len(left_idxs) == 0 or len(right_idxs) == 0:
        return 0.0

    # compute the the minimized loss of the children
    var child_loss = leaf_loss(reg_lambda, g[left_idxs], h[left_idxs]) + leaf_loss(reg_lambda, g[right_idxs], h[right_idxs])
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
