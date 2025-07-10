from mojmelo.DecisionTree import DecisionTree
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CVM
from memory import UnsafePointer
from algorithm import parallelize
import math
import random

@always_inline
fn bootstrap_sample(X: Matrix, y: Matrix) raises -> Tuple[Matrix, Matrix]:
    var idxs = Matrix.rand_choice(X.height, X.height, True, seed = False)
    return X[idxs], y[idxs]

fn _predict(y: Matrix, criterion: String) raises -> Float32:
    if criterion == 'mse':
        return y.mean()
    var freq = y.unique()
    var max_val: Int = 0
    var most_common: Int = 0
    for k in freq.keys():
        if freq[k] > max_val:
            max_val = freq[k]
            most_common = k
    return Float32(most_common)

struct RandomForest(CVM):
    var n_trees: Int
    var min_samples_split: Int
    var max_depth: Int
    var n_feats: Int
    var criterion: String
    var trees: UnsafePointer[DecisionTree]

    fn __init__(out self, n_trees: Int = 10, min_samples_split: Int = 2, max_depth: Int = 100, n_feats: Int = -1, criterion: String = 'gini', random_state: Int = 42):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.criterion = criterion.lower()
        random.seed(random_state)
        self.trees = UnsafePointer[DecisionTree]()

    fn __del__(owned self):
        if self.trees:
            for i in range(self.n_trees):
                (self.trees + i).destroy_pointee()
            self.trees.free()

    fn fit(mut self, X: Matrix, y: Matrix) raises:
        self.trees = UnsafePointer[DecisionTree].alloc(self.n_trees)
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
                X_samp, y_samp = bootstrap_sample(X, y)
                tree.fit(X_samp, y_samp)
            except:
                print('Error: Tree fitting failed!')
            (self.trees + i).init_pointee_move(tree)
            self.trees[i]._moveinit_(tree)
        parallelize[p](self.n_trees)

    fn predict(self, X: Matrix) raises -> Matrix:
        var tree_preds = Matrix(X.height, self.n_trees)
        @parameter
        fn predict_per_tree(i: Int):
            try:
                tree_preds['', i] = self.trees[i].predict(X)
            except:
                print('Error: Failed to predict!')
        parallelize[predict_per_tree](self.n_trees)

        var y_predicted = Matrix(X.height, 1)
        @parameter
        fn predict_per_sample(i: Int):
            try:
                y_predicted.data[i] = _predict(tree_preds[i], self.criterion)
            except:
                print('Error: Failed to predict!')
        parallelize[predict_per_sample](tree_preds.height)
        return y_predicted^

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
        self.trees = UnsafePointer[DecisionTree]()
