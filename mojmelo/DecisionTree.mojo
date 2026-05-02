from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CV, entropy, entropy_precompute, gini, gini_precompute, mse_loss, mse_loss_precompute, fill_indices_list, MODEL_IDS
from std.algorithm import parallelize
import std.math as math
import std.random as random

struct Node(Copyable):
    var feature: Int
    var threshold: Float32
    var left: UnsafePointer[Node, MutAnyOrigin]
    var right: UnsafePointer[Node, MutAnyOrigin]
    var value: Float32

    def __init__(
        out self, feature: Int = -1, threshold: Float32 = 0.0, left: UnsafePointer[Node, MutAnyOrigin] = UnsafePointer[Node, MutAnyOrigin](), right: UnsafePointer[Node, MutAnyOrigin] = UnsafePointer[Node, MutAnyOrigin](), value: Float32 = math.inf[DType.float32]()
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @always_inline
    def is_leaf_node(self) -> Bool:
        return self.feature == -1

    def __str__(self) -> String:
        if self.is_leaf_node():
            return '{' + String(self.value) + '}'
        return '<' + String(self.feature) + ': ' + String(self.threshold) + '>'

struct DecisionTree(CV, Copyable, ImplicitlyCopyable):
    """A decision tree supporting both classification and regression."""
    var criterion: String
    """The function to measure the quality of a split:
    For classification -> 'entropy', 'gini';
    For regression -> 'mse'.
    """
    var loss_func: fn(Matrix, Matrix, Float32) raises -> Float32
    var c_func: fn(Float32, List[Int]) raises -> Float32
    var r_func: fn(Float32, Float32, Float32) raises -> Float32
    var min_samples_split: Int
    """The minimum number of samples required to split an internal node."""
    var max_depth: Int
    """The maximum depth of the tree."""
    var n_feats: Int
    """The number of features to consider when looking for the best split."""
    var root: UnsafePointer[Node, MutAnyOrigin]
    comptime MODEL_ID = 9
    
    def __init__(out self, criterion: String = 'gini', min_samples_split: Int = 2, max_depth: Int = 100, n_feats: Int = -1, random_state: Int = 42):
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

    def __init__(out self, params: Dict[String, String]) raises:
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

    def _moveinit_(mut self, mut take: Self):
        self.criterion = take.criterion
        self.loss_func = take.loss_func
        self.min_samples_split = take.min_samples_split
        self.max_depth = take.max_depth
        self.n_feats = take.n_feats
        self.root = take.root
        take.criterion = ''
        take.min_samples_split = take.max_depth = take.n_feats = 0
        take.root = UnsafePointer[Node, MutAnyOrigin]()

    def __del__(deinit self):
        if self.root:
            delTree(self.root)

    def fit(mut self, X: Matrix, y: Matrix) raises:
        """Build a decision tree from the training set."""
        self.n_feats = X.width if self.n_feats < 1 else min(self.n_feats, X.width)
        var x = X.asorder('f')
        var sorted_indices = List[List[Scalar[DType.int]]](capacity=X.width)
        sorted_indices.resize(X.width, [])
        @parameter
        def p(i: Int):
            try:
                sorted_indices[i] = x['', i].argsort()
            except e:
                print('Error:', e)
        parallelize[p](X.width)
        if y.width != 1:
            self.root = self._grow_tree(x, y.reshape(y.size, 1), fill_indices_list(X.height), sorted_indices)
        else:
            self.root = self._grow_tree(x, y, fill_indices_list(X.height), sorted_indices)

    def fit_weighted(mut self, X: Matrix, y_with_weights: Matrix) raises:
        """Build a decision tree from a weighted training set."""
        self.n_feats = X.width if self.n_feats < 1 else min(self.n_feats, X.width)
        var x = X.asorder('f')
        var sorted_indices = List[List[Scalar[DType.int]]](capacity=X.width)
        sorted_indices.resize(X.width, [])
        @parameter
        def p(i: Int):
            try:
                sorted_indices[i] = x['', i].argsort()
            except e:
                print('Error:', e)
        parallelize[p](X.width)
        self.root = self._grow_tree(x, y_with_weights, fill_indices_list(X.height), sorted_indices)

    def predict(self, X: Matrix) raises -> Matrix:
        """Predict class or regression value for X.
        
        Returns:
            The predicted values.
        """
        var y_predicted = Matrix(X.height, 1)
        @parameter
        def p(i: Int):
            y_predicted.data[i] = _traverse_tree(X[i, unsafe=True], self.root)
        parallelize[p](X.height)
        return y_predicted^

    def _grow_tree(self, X: Matrix, Y: Matrix, indices: List[Scalar[DType.int]], sorted_indices: List[List[Scalar[DType.int]]], depth: Int = 0) raises -> UnsafePointer[Node, MutAnyOrigin]:
        var y = Y[indices]
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
            or len(indices) < self.min_samples_split
        ):
            new_node.init_pointee_move(Node(value = set_value(_y, weights, freq, self.criterion)))
            return new_node

        var feat_idxs = Matrix.rand_choice(X.width, self.n_feats, False, seed = False)

        # greedily select the best split according to information gain
        var best_feat: Int
        var best_thresh: Float32

        best_feat, best_thresh = _best_criteria(
            X,
            Y,
            _y,
            weights,
            feat_idxs,
            sorted_indices,
            self.loss_func,
            self.c_func,
            self.r_func,
            self.criterion
        )

        # --- partition samples ---
        var goes_left = Dict[Scalar[DType.int], Bool]()
        for idx in indices:
            goes_left[idx] = X[Int(idx), best_feat] <= best_thresh

        var left_indices = List[Scalar[DType.int]]()
        var right_indices = List[Scalar[DType.int]]()

        for idx in indices:
            if goes_left[idx]:
                left_indices.append(idx)
            else:
                right_indices.append(idx)

        # --- partition presorted indices ---
        var left_sorted = List[List[Scalar[DType.int]]](capacity=X.width)
        var right_sorted = List[List[Scalar[DType.int]]](capacity=X.width)
        left_sorted.resize(X.width, [])
        right_sorted.resize(X.width, [])

        @parameter
        def split_feature(f: Int):
            try:
                var left_f = List[Scalar[DType.int]]()
                var right_f = List[Scalar[DType.int]]()

                for idx in sorted_indices[f]:
                    if goes_left[idx]:
                        left_f.append(idx)
                    else:
                        right_f.append(idx)

                # assign the local lists to the global ones
                left_sorted[f] = left_f.copy()
                right_sorted[f] = right_f.copy()
            except e:
                print('Error:', e)
        parallelize[split_feature](X.width)

        # --- recurse ---
        var left_node = self._grow_tree(
            X,
            Y,
            left_indices,
            left_sorted,
            depth + 1
        )

        var right_node = self._grow_tree(
            X,
            Y,
            right_indices,
            right_sorted,
            depth + 1
        )

        new_node.init_pointee_move(
            Node(best_feat, best_thresh, left_node, right_node)
        )

        return new_node

    def save(self, path: String) raises:
        """Save model data necessary for prediction to the specified path."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        with open(_path, "w") as f:
            f.write_bytes(UInt8(Self.MODEL_ID).as_bytes())
            var node_list = List[Node]()
            var children_index_list = List[Tuple[Int, Int]]()
            var stack = [self.root[].copy()]
            while len(stack) > 0:
                var node = stack.pop()
                var children_index = (-1, -1)
                if node.left:
                    stack.insert(0, node.left[].copy())
                    children_index[0] = len(stack) + len(node_list)
                if node.right:
                    stack.insert(0, node.right[].copy())
                    children_index[1] = len(stack) + len(node_list)
                node_list.append(node^)
                children_index_list.append(children_index)
            f.write_bytes(UInt64(len(node_list)).as_bytes())
            for i, node in enumerate(node_list):
                f.write_bytes(UInt64(node.feature).as_bytes())
                f.write_bytes(node.threshold.as_bytes())
                f.write_bytes(UInt64(children_index_list[i][0]).as_bytes())
                f.write_bytes(UInt64(children_index_list[i][1]).as_bytes())
                f.write_bytes(node.value.as_bytes())

    @staticmethod
    def load(path: String) raises -> Self:
        """Load a saved model from the specified path for prediction."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        var model = Self()
        with open(_path, "r") as f:
            var id = f.read_bytes(1)[0]
            if id < 1 or id > UInt8(MODEL_IDS.size-1):
                raise Error('Input file with invalid metadata!')
            elif id != Self.MODEL_ID:
                raise Error('Based on the metadata, ', _path, ' belongs to ', materialize[MODEL_IDS]()[id], ' algorithm!')
            var node_size = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
            var node_list = List[UnsafePointer[Node, MutAnyOrigin]]()
            var children_index_list = List[Tuple[Int, Int]]()
            for i in range(node_size):
                var feature = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
                var threshold = f.read_bytes(4).unsafe_ptr().bitcast[Float32]()[]
                var left = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
                var right = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
                var value = f.read_bytes(4).unsafe_ptr().bitcast[Float32]()[]
                var node = alloc[Node](1)
                node.init_pointee_move(Node(feature=feature, threshold=threshold, value=value))
                node_list.append(node)
                children_index_list.append((left, right))
            model.root = node_list[0]
            for i in range(node_size):
                if children_index_list[i][0] != -1:
                    node_list[i][].left = node_list[children_index_list[i][0]]
                if children_index_list[i][1] != -1:
                    node_list[i][].right = node_list[children_index_list[i][1]]
        return model^

def set_value(y: Matrix, weights: Matrix, freq: List[List[Int]], criterion: String) raises -> Float32:
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

def _best_criteria(X: Matrix, y: Matrix, _y: Matrix, weights: Matrix, feat_idxs: List[Scalar[DType.int]], sorted_indices_per_feature: List[List[Scalar[DType.int]]], loss_func: fn(Matrix, Matrix, Float32) raises -> Float32, c_precompute: fn(Float32, List[Int]) raises -> Float32, r_precompute: fn(Float32, Float32, Float32) raises -> Float32, criterion: String) raises -> Tuple[Int, Float32]:
    var total_samples = Float32(len(_y)) if y.width == 1 else weights.sum()
    var parent_loss = loss_func(_y, weights, total_samples)
    var max_gains = Matrix(1, len(feat_idxs))
    max_gains.fill(-math.inf[DType.float32]())
    var best_thresholds = Matrix(1, len(feat_idxs))
    if criterion != 'mse':
        var num_classes = Int(_y.max() + 1)  # assuming y is 0-indexed
        @parameter
        def p_c(f_idx: Int):
            try:
                var feat = Int(feat_idxs[f_idx])
                var sorted_idx = sorted_indices_per_feature[feat].copy()

                var left_hist = List[Int](capacity=num_classes)
                left_hist.resize(num_classes, 0)

                var right_hist = _y.bincount() if weights.size == 0 else _y.bincount(weights)

                var n_left: Float32 = 0.0

                for i in range(1, len(sorted_idx)):
                    var idx_prev = sorted_idx[i - 1]
                    var c = Int(y.data[idx_prev])

                    if weights.size == 0:
                        n_left += 1
                        left_hist[c] += 1
                        right_hist[c] -= 1
                    else:
                        var w = y[Int(idx_prev), 1]
                        n_left += w
                        left_hist[c] += Int(w)
                        right_hist[c] -= Int(w)

                    var v_prev = X[Int(idx_prev), feat]
                    var v_curr = X[Int(sorted_idx[i]), feat]

                    if v_curr == v_prev:
                        continue

                    var n_right = total_samples - n_left

                    var child_loss =
                        (n_left / total_samples) * c_precompute(n_left, left_hist) +
                        (n_right / total_samples) * c_precompute(n_right, right_hist)

                    var ig = parent_loss - child_loss

                    if ig > max_gains.data[f_idx]:
                        max_gains.data[f_idx] = ig
                        best_thresholds.data[f_idx] = (v_prev + v_curr) / 2.0  # midpoint
            except e:
                print('Error:', e)
        parallelize[p_c](len(feat_idxs))
    else:
        var sum_total = _y.sum() if y.width == 1 else _y.ele_mul(weights).sum()
        var sum_sq_total = _y.ele_mul(_y).sum() if y.width == 1 else (_y.ele_mul(_y).ele_mul(weights)).sum()
        @parameter
        def p_r(f_idx: Int):
            try:
                var feat = Int(feat_idxs[f_idx])
                var sorted_idx = sorted_indices_per_feature[feat].copy()

                var left_sum: Float32 = 0.0
                var left_sum_sq: Float32 = 0.0
                var n_left: Float32 = 0.0

                for i in range(1, len(sorted_idx)):
                    var idx_prev = sorted_idx[i - 1]
                    var yi = y.data[idx_prev]

                    if weights.size == 0:
                        n_left += 1
                        left_sum += yi
                        left_sum_sq += yi * yi
                    else:
                        var w = y[Int(idx_prev), 1]
                        n_left += w
                        left_sum += yi * w
                        left_sum_sq += yi * yi * w

                    var v_prev = X[Int(idx_prev), feat]
                    var v_curr = X[Int(sorted_idx[i]), feat]

                    if v_curr == v_prev:
                        continue

                    var n_right = total_samples - n_left

                    var child_loss =
                        (n_left / total_samples) * r_precompute(n_left, left_sum, left_sum_sq) +
                        (n_right / total_samples) * r_precompute(
                            n_right,
                            sum_total - left_sum,
                            sum_sq_total - left_sum_sq
                        )

                    var ig = parent_loss - child_loss

                    if ig > max_gains.data[f_idx]:
                        max_gains.data[f_idx] = ig
                        best_thresholds.data[f_idx] = (v_prev + v_curr) / 2.0  # midpoint
            except e:
                print('Error:', e)
        parallelize[p_r](len(feat_idxs))
    
    var feat_idx = max_gains.argmax()
    return Int(feat_idxs[feat_idx]), best_thresholds.data[feat_idx]

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
