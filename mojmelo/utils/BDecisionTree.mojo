from mojmelo.DecisionTree import Node
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import findInterval, fill_indices_list
from std.algorithm import parallelize
import std.math as math

struct BDecisionTree(Copyable, ImplicitlyCopyable):
    var min_samples_split: Int
    var max_depth: Int
    var reg_lambda: Float32
    var reg_alpha: Float32
    var gamma: Float32
    var n_bins: Int
    var root: UnsafePointer[Node, MutAnyOrigin]

    def __init__(out self, min_samples_split: Int = 10, max_depth: Int = 3, reg_lambda: Float32 = 1.0, reg_alpha: Float32 = 0.0, gamma: Float32 = 0.0, n_bins: Int = 0):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.n_bins = n_bins if n_bins >= 2 else 0
        self.root = UnsafePointer[Node, MutAnyOrigin]()

    def _moveinit_(mut self, mut take: Self):
        self.min_samples_split = take.min_samples_split
        self.max_depth = take.max_depth
        self.reg_lambda = take.reg_lambda
        self.reg_alpha = take.reg_alpha
        self.gamma = take.gamma
        self.n_bins = take.n_bins
        self.root = take.root
        take.min_samples_split = take.max_depth = 0
        take.reg_lambda = take.reg_alpha = take.gamma = 0.0
        take.root = UnsafePointer[Node, MutAnyOrigin]()

    def __del__(deinit self):
        if self.root:
            delTree(self.root)

    def fit(mut self, X: Matrix, g: Matrix, h: Matrix) raises:
        self.root = self._grow_tree(X, g, h, fill_indices_list(X.height))

    def predict(self, X: Matrix) raises -> Matrix:
        var y_predicted = Matrix(X.height, 1)
        @parameter
        def p(i: Int):
            y_predicted.data[i] = _traverse_tree(X[i, unsafe=True], self.root)
        parallelize[p](X.height)
        return y_predicted^

    def _grow_tree(self, X: Matrix, g: Matrix, h: Matrix, indices: List[Scalar[DType.int]], depth: Int = 0) raises -> UnsafePointer[Node, MutAnyOrigin]:
        var new_node = alloc[Node](1)
        # stopping criteria
        if (
            depth >= self.max_depth
            or len(indices) < self.min_samples_split
        ):
            new_node.init_pointee_move(Node(value = leaf_score(self.reg_lambda, self.reg_alpha, g, h)))
            return new_node

        var feat_idxs = Matrix.rand_choice(X.width, X.width, False)

        # greedily select the best split according to information gain
        var x = X[indices]
        var best_feat: Int
        var best_thresh: Float32
        var best_gain: Float32
        best_feat, best_thresh, best_gain = _best_criteria(self.reg_lambda, self.reg_alpha, x, g, h, feat_idxs, self.n_bins)
        if best_gain <= self.gamma:
            # The best gain is less than gamma
            new_node.init_pointee_move(Node(value = leaf_score(self.reg_lambda, self.reg_alpha, g, h)))
            return new_node
        
        # grow the children that result from the split
        var left_right_idxs = _split(x['', best_feat], best_thresh)
        var left_idxs = left_right_idxs[0].copy()
        var right_idxs = left_right_idxs[1].copy()
        var left_indices = List[Scalar[DType.int]]()
        var right_indices = List[Scalar[DType.int]]()
        for idx in left_idxs:
            left_indices.append(indices[idx])
        for idx in right_idxs:
            right_indices.append(indices[idx])
        var left = self._grow_tree(X, g[left_idxs], h[left_idxs], left_indices, depth + 1)
        var right = self._grow_tree(X, g[right_idxs], h[right_idxs], right_indices, depth + 1)
        new_node.init_pointee_move(Node(best_feat, best_thresh, left, right))
        return new_node

@always_inline
def leaf_score(reg_lambda: Float32, reg_alpha: Float32, g: Matrix, h: Matrix) raises -> Float32:
    var g_sum = g.sum()
    return (-g_sum / (h.sum() + reg_lambda)) - reg_alpha * math.copysign(Float32(1), g_sum)

@always_inline
def leaf_score_precompute(reg_lambda: Float32, reg_alpha: Float32, g_sum: Float32, h_sum: Float32) raises -> Float32:
    return (-g_sum / (h_sum + reg_lambda)) - reg_alpha * math.copysign(Float32(1), g_sum)

@always_inline
def leaf_loss(reg_lambda: Float32, reg_alpha: Float32, g: Matrix, h: Matrix) raises -> Float32:
    var g_sum = g.sum()
    var h_sum = h.sum()
    return (-0.5 * (g_sum ** 2) / (h_sum + reg_lambda)) + reg_alpha * abs(leaf_score_precompute(reg_lambda, reg_alpha, g_sum, h_sum))

@always_inline
def leaf_loss_precompute(reg_lambda: Float32, reg_alpha: Float32, g_sum: Float32, h_sum: Float32) raises -> Float32:
    return (-0.5 * (g_sum ** 2) / (h_sum + reg_lambda)) + reg_alpha * abs(leaf_score_precompute(reg_lambda, reg_alpha, g_sum, h_sum))

def _best_criteria(reg_lambda: Float32, reg_alpha: Float32, X: Matrix, g: Matrix, h: Matrix, feat_idxs: List[Scalar[DType.int]], n_bins: Int) raises -> Tuple[Int, Float32, Float32]:
    var total_g_sum = g.sum()
    var total_h_sum = h.sum()
    var parent_loss = leaf_loss_precompute(reg_lambda, reg_alpha, total_g_sum, total_h_sum)
    var max_gains = Matrix(1, len(feat_idxs))
    max_gains.fill(-math.inf[DType.float32]())
    var best_thresholds = Matrix(1, len(feat_idxs))

    @parameter
    def p(idx: Int):
        try:
            var column = X['', Int(feat_idxs[idx]), unsafe=True]
            if n_bins < 2 or len(column) < n_bins:
                var sorted_indices = column.argsort_inplace()
                var g_sorted = g[sorted_indices]
                var h_sorted = h[sorted_indices]

                var left_g_sum: Float32 = 0.0
                var left_h_sum: Float32 = 0.0

                for step in range(1, X.height):
                    left_g_sum += g_sorted.data[step - 1]
                    left_h_sum += h_sorted.data[step - 1]

                    if column.data[step] == column.data[step - 1]:
                        continue  # skip redundant thresholds

                    var child_loss = leaf_loss_precompute(reg_lambda, reg_alpha, left_g_sum, left_h_sum) + leaf_loss_precompute(reg_lambda, reg_alpha, total_g_sum - left_g_sum, total_h_sum - left_h_sum)
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
                    def find_interval(i: Int):
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

                    for step in range(len(intervals)-1):
                        left_g_sum += g_sum.data[step]
                        left_h_sum += h_sum.data[step]

                        var child_loss = leaf_loss_precompute(reg_lambda, reg_alpha, left_g_sum, left_h_sum) + leaf_loss_precompute(reg_lambda, reg_alpha, total_g_sum - left_g_sum, total_h_sum - left_h_sum)
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
def _split(X_column: Matrix, split_thresh: Float32) -> Tuple[List[Scalar[DType.int]], List[Scalar[DType.int]]]:
    return X_column.argwhere_l(X_column <= split_thresh), X_column.argwhere_l(X_column > split_thresh)

def _traverse_tree(x: Matrix, node: UnsafePointer[Node, MutAnyOrigin]) -> Float32:
    if node[].is_leaf_node():
        return node[].value

    if x.data[node[].feature] <= node[].threshold:
        return _traverse_tree(x, node[].left)
    return _traverse_tree(x, node[].right)

def delTree(node: UnsafePointer[Node, MutAnyOrigin]):
    if node[].left:
        delTree(node[].left)
    if node[].right:
        delTree(node[].right)
    node.free()
