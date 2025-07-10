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
        self.root = self._grow_tree(X, g, h)

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

@always_inline
fn leaf_loss_precompute(reg_lambda: Float32, g_sum: Float32, h_sum: Float32) raises -> Float32:
    return -0.5 * (g_sum ** 2) / (h_sum + reg_lambda)

fn _best_criteria(reg_lambda: Float32, X: Matrix, g: Matrix, h: Matrix, feat_idxs: List[Scalar[DType.index]]) raises -> Tuple[Int, Float32, Float32]:
    var total_g_sum = g.sum()
    var total_h_sum = h.sum()
    var parent_loss = leaf_loss_precompute(reg_lambda, total_g_sum, total_h_sum)
    var max_gains = Matrix(1, len(feat_idxs))
    max_gains.fill(-math.inf[DType.float32]())
    var best_thresholds = Matrix(1, len(feat_idxs))

    @parameter
    fn p(idx: Int):
        try:
            var column = X['', feat_idxs[idx].value, unsafe=True]
            var sorted_indices = column.argsort()
            column = column[sorted_indices]
            var g_sorted = g[sorted_indices]
            var h_sorted = h[sorted_indices]

            var left_g_sum: Float32 = 0.0
            var left_h_sum: Float32 = 0.0
            var right_g_sum = total_g_sum
            var right_h_sum = total_h_sum

            for step in range(1, X.height):
                var gi = g_sorted.data[step - 1]
                var hi = h_sorted.data[step - 1]
                left_g_sum += gi
                left_h_sum += hi
                right_g_sum -= gi
                right_h_sum -= hi

                if column.data[step] == column.data[step - 1]:
                    continue  # skip redundant thresholds

                var child_loss = leaf_loss_precompute(reg_lambda, left_g_sum, left_h_sum) + leaf_loss_precompute(reg_lambda, right_g_sum, right_h_sum)
                var ig = parent_loss - child_loss
                if ig > max_gains.data[idx]:
                    max_gains.data[idx] = ig
                    best_thresholds.data[idx] = (column.data[step] + column.data[step - 1]) / 2.0  # midpoint
        except e:
            print('Error:', e)
    parallelize[p](len(feat_idxs))
    
    var feat_idx = max_gains.argmax()
    return feat_idxs[feat_idx].value, best_thresholds.data[feat_idx], max_gains.data[feat_idx]

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
