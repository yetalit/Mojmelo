from mojmelo.DecisionTree import DecisionTree, Node
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CV, MODEL_IDS
from algorithm import parallelize
import math
import random

@always_inline
fn bootstrap_sample(X: Matrix, y: Matrix) raises -> Tuple[Matrix, Matrix]:
    var idxs = Matrix.rand_choice(X.height, X.height, True, seed = False)
    var unique_idxs = List[Scalar[DType.int]]()
    var freqs = Matrix.zeros(X.height, 1)
    for idx in idxs:
        freqs.data[idx] += 1
        if freqs.data[idx] == 1:
            unique_idxs.append(idx)
    var y_with_weights = Matrix(len(unique_idxs), 2, order='f')
    for i in range(len(unique_idxs)):
        y_with_weights[i, 0] = y.data[unique_idxs[i]]
        y_with_weights[i, 1] = freqs.data[unique_idxs[i]]
    return X[unique_idxs], y_with_weights^

fn _predict(y: Matrix, criterion: String) raises -> Float32:
    if criterion == 'mse':
        return y.mean()
    var freq = y.unique()
    var max_val: Int = 0
    var most_common: Int = 0
    for i in range(len(freq)):
        if len(freq[i]) > max_val:
            max_val = len(freq[i])
            most_common = i
    return Float32(most_common)

struct RandomForest(CV, Copyable):
    """A random forest supporting both classification and regression."""
    var n_trees: Int
    """The number of trees in the forest."""
    var min_samples_split: Int
    """The minimum number of samples required to split an internal node."""
    var max_depth: Int
    """The maximum depth of the tree."""
    var n_feats: Int
    """The number of features to consider when looking for the best split."""
    var criterion: String
    """The function to measure the quality of a split:
    For classification -> 'entropy', 'gini';
    For regression -> 'mse'.
    """
    var trees: UnsafePointer[DecisionTree, MutAnyOrigin]
    comptime MODEL_ID = 10
    comptime criterion_ids: List[String] = ['entropy', 'gini', 'mse']

    fn __init__(out self, n_trees: Int = 10, min_samples_split: Int = 2, max_depth: Int = 100, n_feats: Int = -1, criterion: String = 'gini', random_state: Int = 42):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.criterion = criterion.lower()
        random.seed(random_state)
        self.trees = UnsafePointer[DecisionTree, MutAnyOrigin]()

    fn __del__(deinit self):
        if self.trees:
            for i in range(self.n_trees):
                (self.trees + i).destroy_pointee()
            self.trees.free()

    fn fit(mut self, X: Matrix, y: Matrix) raises:
        """Build a forest of trees from the training set."""
        self.trees = alloc[DecisionTree](self.n_trees)
        var _y = y if y.width == 1 else y.reshape(y.size, 1)
        var n_feats = self.n_feats
        if self.n_feats < 1:
            if self.criterion == 'mse':
                n_feats = X.width
            else:
                n_feats = math.sqrt(X.width)
        @parameter
        fn p(i: Int):
            var tree = DecisionTree(
                min_samples_split = self.min_samples_split,
                max_depth = self.max_depth,
                n_feats = n_feats,
                random_state = -1,
                criterion = self.criterion
            )
            try:
                X_samp, y_samp_with_weights = bootstrap_sample(X, _y)
                tree.fit_weighted(X_samp, y_samp_with_weights)
            except e:
                print('Error:', e)
            (self.trees + i).init_pointee_move(tree)
            self.trees[i]._moveinit_(tree)
        parallelize[p](self.n_trees)

    fn predict(self, X: Matrix) raises -> Matrix:
        """Predict class or regression value for X.
        
        Returns:
            The predicted values.
        """
        var tree_preds = Matrix(X.height, self.n_trees)
        @parameter
        fn predict_per_tree(i: Int):
            try:
                tree_preds['', i] = self.trees[i].predict(X)
            except e:
                print('Error:', e)
        parallelize[predict_per_tree](self.n_trees)

        var y_predicted = Matrix(X.height, 1)
        @parameter
        fn predict_per_sample(i: Int):
            try:
                y_predicted.data[i] = _predict(tree_preds[i], self.criterion)
            except e:
                print('Error:', e)
        parallelize[predict_per_sample](tree_preds.height)
        return y_predicted^

    fn save(self, path: String) raises:
        """Save model data necessary for prediction to the specified path."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        with open(_path, "w") as f:
            f.write_bytes(UInt8(Self.MODEL_ID).as_bytes())
            f.write_bytes(UInt8(materialize[self.criterion_ids]().index(self.criterion)).as_bytes())
            f.write_bytes(UInt64(self.n_trees).as_bytes())
            for t_i in range(self.n_trees):
                var node_list = List[Node]()
                var children_index_list = List[Tuple[Int, Int]]()
                var stack = [self.trees[t_i].root[].copy()]
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
    fn load(path: String) raises -> Self:
        """Load a saved model from the specified path for prediction."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        var model = Self()
        with open(_path, "r") as f:
            var id = f.read_bytes(1)[0]
            if id < 1 or id > MODEL_IDS.size-1:
                raise Error('Input file with invalid metadata!')
            elif id != Self.MODEL_ID:
                raise Error('Based on the metadata, ', _path, ' belongs to ', materialize[MODEL_IDS]()[id], ' algorithm!')
            model.criterion = materialize[Self.criterion_ids]()[f.read_bytes(1)[0]]
            model.n_trees = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
            model.trees = alloc[DecisionTree](model.n_trees)
            for t_i in range(model.n_trees):
                var tree = DecisionTree()
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
                tree.root = node_list[0]
                for i in range(node_size):
                    if children_index_list[i][0] != -1:
                        node_list[i][].left = node_list[children_index_list[i][0]]
                    if children_index_list[i][1] != -1:
                        node_list[i][].right = node_list[children_index_list[i][1]]
                (model.trees + t_i).init_pointee_move(tree)
                model.trees[t_i]._moveinit_(tree)
        return model^

    fn __init__(out self, params: Dict[String, String]) raises:
        if 'n_trees' in params:
            self.n_trees = atol(String(params['n_trees']))
        else:
            self.n_trees = 10
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
        if 'criterion' in params:
            self.criterion = params['criterion'].lower()
        else:
            self.criterion = 'gini'
        if 'random_state' in params:
            random.seed(atol(String(params['random_state'])))
        else:
            random.seed(42)
        self.trees = UnsafePointer[DecisionTree, MutAnyOrigin]()
