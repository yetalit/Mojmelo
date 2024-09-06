from mojmelo.DecisionTree import Node
from mojmelo.utils.Matrix import Matrix
import math

@value
struct BDecisionTree:
    var min_samples_split: Int
    var max_depth: Int
    var reg_lambda: Float32 
    var gamma: Float32
    var root: UnsafePointer[Node]
    
    fn __init__(inout self, min_samples_split: Int = 10, max_depth: Int = 3, reg_lambda: Float32 = 1.0, gamma: Float32 = 0.0):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.root = UnsafePointer[Node]()

    fn __del__(owned self):
        if self.root:
            delTree(self.root)

    fn fit(inout self, X: Matrix, g: Matrix, h: Matrix) raises:
        self.root = self._grow_tree(X, g, h)

    fn predict(self, X: Matrix) raises -> Matrix:
        var y_predicted = Matrix(X.height, 1)
        for i in range(X.height):
            y_predicted.data[i] = _traverse_tree(X[i], self.root)
        return y_predicted

    fn _grow_tree(self, X: Matrix, g: Matrix, h: Matrix, depth: Int = 0) raises -> UnsafePointer[Node]:
        var new_node = UnsafePointer[Node].alloc(1)
        # stopping criteria
        if (
            depth >= self.max_depth
            or X.height < self.min_samples_split
        ):
            new_node[] = Node(value = leaf_score(self.reg_lambda, g, h))
            return new_node

        var feat_idxs = Matrix.rand_choice(X.width, X.width, False)

        # greedily select the best split according to information gain
        var best_feat: Int
        var best_thresh: Float32
        var best_gain: Float32
        best_feat, best_thresh, best_gain = _best_criteria(self.reg_lambda, X, g, h, feat_idxs)
        if best_gain <= self.gamma:
            # The best gain is less than gamma
            new_node[] = Node(value = leaf_score(self.reg_lambda, g, h))
            return new_node
        
        # grow the children that result from the split
        var left_idxs: List[Int]
        var right_idxs: List[Int]
        left_idxs, right_idxs = _split(X['', best_feat], best_thresh)
        var left = self._grow_tree(X[left_idxs], g[left_idxs], h[left_idxs], depth + 1)
        var right = self._grow_tree(X[right_idxs], g[right_idxs], h[right_idxs], depth + 1)
        new_node[] = Node(best_feat, best_thresh, left, right)
        return new_node

fn leaf_score(reg_lambda: Float32, g: Matrix, h: Matrix) -> Float32:
    '''
    Given the gradient and hessian of a tree leaf,
    return the prediction (score) at this leaf.
    The score is -G/(H+λ).
    '''
    return -g.sum() / (h.sum() + reg_lambda)

fn leaf_loss(reg_lambda: Float32, g: Matrix, h: Matrix) -> Float32:
    '''
    Given the gradient and hessian of a tree leaf,
    return the minimized loss at this leaf.
    The minimized loss is -0.5*G^2/(H+λ).
    .'''
    return -0.5 * (g.sum() ** 2) / (h.sum() + reg_lambda)

fn _best_criteria(reg_lambda: Float32, X: Matrix, g: Matrix, h: Matrix, feat_idxs: List[Int]) raises -> Tuple[Int, Float32, Float32]:
    var split_idx = feat_idxs[0]
    var split_thresh = X[0, split_idx]
    var best_gain = -math.inf[DType.float32]()
    for feat_idx in feat_idxs:
        var X_column = X['', feat_idx[]]
        var thresholds = X_column.uniquef()
        for threshold in thresholds:
            var gain = _information_gain(reg_lambda, g, h, X_column, threshold[])
            if gain > best_gain:
                best_gain = gain
                split_idx = feat_idx[]
                split_thresh = threshold[]

    return split_idx, split_thresh, best_gain

fn _information_gain(reg_lambda: Float32, g: Matrix, h: Matrix, X_column: Matrix, split_thresh: Float32) raises -> Float32:
    var parent_loss = leaf_loss(reg_lambda, g, h)

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
