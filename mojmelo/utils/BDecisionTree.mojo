from mojmelo.DecisionTree import Node
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import findInterval
from algorithm import parallelize
import math

struct BDecisionTree(Copyable, Movable, ImplicitlyCopyable):
    var min_samples_split: Int
    var max_depth: Int
    var reg_lambda: Float32
    var reg_alpha: Float32
    var gamma: Float32
    var n_bins: Int
    var root: UnsafePointer[Node, MutAnyOrigin]

    fn __init__(out self, min_samples_split: Int = 10, max_depth: Int = 3, reg_lambda: Float32 = 1.0, reg_alpha: Float32 = 0.0, gamma: Float32 = 0.0, n_bins: Int = 0):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.n_bins = n_bins if n_bins >= 2 else 0
        self.root = UnsafePointer[Node, MutAnyOrigin]()

    fn _moveinit_(mut self, mut existing: Self):
        self.min_samples_split = existing.min_samples_split
        self.max_depth = existing.max_depth
        self.reg_lambda = existing.reg_lambda
        self.reg_alpha = existing.reg_alpha
        self.gamma = existing.gamma
        self.n_bins = existing.n_bins
        self.root = existing.root
        existing.min_samples_split = existing.max_depth = 0
        existing.reg_lambda = existing.reg_alpha = existing.gamma = 0.0
        existing.root = UnsafePointer[Node, MutAnyOrigin]()

    fn __del__(deinit self):
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

    fn _grow_tree(self, X: Matrix, g: Matrix, h: Matrix, depth: Int = 0) raises -> UnsafePointer[Node, MutAnyOrigin]:
        var new_node = alloc[Node](1)
        # stopping criteria
        if (
            depth >= self.max_depth
            or X.height < self.min_samples_split
        ):
            new_node.init_pointee_move(Node(value = leaf_score(self.reg_lambda, self.reg_alpha, g, h)))
            return new_node

        var feat_idxs = Matrix.rand_choice(X.width, X.width, False)

        # greedily select the best split according to information gain
        var best_feat: Int
        var best_thresh: Float32
        var best_gain: Float32
        best_feat, best_thresh, best_gain = _best_criteria(self.reg_lambda, self.reg_alpha, X, g, h, feat_idxs, self.n_bins)
        if best_gain <= self.gamma:
            # The best gain is less than gamma
            new_node.init_pointee_move(Node(value = leaf_score(self.reg_lambda, self.reg_alpha, g, h)))
            return new_node
        
        # grow the children that result from the split
        var left_right_idxs = _split(X['', best_feat], best_thresh)
        var left_idxs = left_right_idxs[0].copy()
        var right_idxs = left_right_idxs[1].copy()
        var left = self._grow_tree(X[left_idxs], g[left_idxs], h[left_idxs], depth + 1)
        var right = self._grow_tree(X[right_idxs], g[right_idxs], h[right_idxs], depth + 1)
        new_node.init_pointee_move(Node(best_feat, best_thresh, left, right))
        return new_node

@always_inline
fn leaf_score(reg_lambda: Float32, reg_alpha: Float32, g: Matrix, h: Matrix) raises -> Float32:
    var g_sum = g.sum()
    return (-g_sum / (h.sum() + reg_lambda)) - reg_alpha * math.copysign(1, g_sum)

@always_inline
fn leaf_score_precompute(reg_lambda: Float32, reg_alpha: Float32, g_sum: Float32, h_sum: Float32) raises -> Float32:
    return (-g_sum / (h_sum + reg_lambda)) - reg_alpha * math.copysign(1, g_sum)

@always_inline
fn leaf_loss(reg_lambda: Float32, reg_alpha: Float32, g: Matrix, h: Matrix) raises -> Float32:
    var g_sum = g.sum()
    var h_sum = h.sum()
    return (-0.5 * (g_sum ** 2) / (h_sum + reg_lambda)) + reg_alpha * abs(leaf_score_precompute(reg_lambda, reg_alpha, g_sum, h_sum))

@always_inline
fn leaf_loss_precompute(reg_lambda: Float32, reg_alpha: Float32, g_sum: Float32, h_sum: Float32) raises -> Float32:
    return (-0.5 * (g_sum ** 2) / (h_sum + reg_lambda)) + reg_alpha * abs(leaf_score_precompute(reg_lambda, reg_alpha, g_sum, h_sum))

fn _best_criteria(reg_lambda: Float32, reg_alpha: Float32, X: Matrix, g: Matrix, h: Matrix, feat_idxs: List[Scalar[DType.int]], n_bins: Int) raises -> Tuple[Int, Float32, Float32]:
    var total_g_sum = g.sum()
    var total_h_sum = h.sum()
    var parent_loss = leaf_loss_precompute(reg_lambda, reg_alpha, total_g_sum, total_h_sum)
    var max_gains = Matrix(1, len(feat_idxs))
    max_gains.fill(-math.inf[DType.float32]())
    var best_thresholds = Matrix(1, len(feat_idxs))

    @parameter
    fn p(idx: Int):
        try:
            var column = X['', Int(feat_idxs[idx]), unsafe=True]
            if n_bins < 2 or len(column) < n_bins:
                var sorted_indices = column.argsort_inplace()
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

                    var child_loss = leaf_loss_precompute(reg_lambda, reg_alpha, left_g_sum, left_h_sum) + leaf_loss_precompute(reg_lambda, reg_alpha, right_g_sum, right_h_sum)
                    var ig = parent_loss - child_loss
                    if ig > max_gains.data[idx]:
                        max_gains.data[idx] = ig
                        best_thresholds.data[idx] = (column.data[step] + column.data[step - 1]) / 2.0  # midpoint
            else:
                var start = column.min()
                var end = column.max()
                if start != end:
                    var bins = Matrix.linspace(start, end, n_bins+1)
                    var intervals = List[Tuple[Float32, Float32]]()
                    for bin_i in range(1, len(bins)):
                        intervals.append((bins.data[bin_i-1], bins.data[bin_i]))

                    var g_per_interval = Matrix.zeros(len(column), len(intervals))
                    var h_per_interval = Matrix.zeros(len(column), len(intervals))
                    @parameter
                    fn find_interval(i: Int):
                        try:
                            var interval = findInterval(intervals, column.data[i])
                            g_per_interval[i, interval] = g.data[i]
                            h_per_interval[i, interval] = h.data[i]
                        except e:
                            print('Error:', e)
                    parallelize[find_interval](len(column))
                    var g_sum = g_per_interval.sum(axis=0)
                    var h_sum = h_per_interval.sum(axis=0)
                    
                    var left_g_sum: Float32 = 0.0
                    var left_h_sum: Float32 = 0.0
                    var right_g_sum = total_g_sum
                    var right_h_sum = total_h_sum

                    for step in range(len(intervals)-1):
                        left_g_sum += g_sum.data[step]
                        left_h_sum += h_sum.data[step]
                        right_g_sum -= g_sum.data[step]
                        right_h_sum -= h_sum.data[step]

                        var child_loss = leaf_loss_precompute(reg_lambda, reg_alpha, left_g_sum, left_h_sum) + leaf_loss_precompute(reg_lambda, reg_alpha, right_g_sum, right_h_sum)
                        var ig = parent_loss - child_loss
                        if ig > max_gains.data[idx]:
                            max_gains.data[idx] = ig
                            best_thresholds.data[idx] = bins.data[step+1]
        except e:
            print('Error:', e)
    parallelize[p](len(feat_idxs))
    
    var feat_idx = max_gains.argmax()
    return Int(feat_idxs[feat_idx]), best_thresholds.data[feat_idx], max_gains.data[feat_idx]

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
