from mojmelo.DecisionTree import DecisionTree
from mojmelo.utils.Matrix import Matrix

fn bootstrap_sample(X: Matrix, y: Matrix) raises -> Tuple[Matrix, Matrix]:
    var idxs = Matrix.rand_choice(X.height, X.height, True)
    return X[idxs], y[idxs]

fn _predict(y: Matrix, criterion: String) raises -> Float32:
    if criterion == 'mse':
        return y.mean()
    var freq = y.unique()
    var max_val: Int = 0
    var most_common: Int = 0
    for k in freq.keys():
        if freq[k[]] > max_val:
            max_val = freq[k[]]
            most_common = k[]
    return Float32(most_common)

struct RandomForest:
    var n_trees: Int
    var min_samples_split: Int
    var max_depth: Int
    var n_feats: Int
    var criterion: String
    var trees: UnsafePointer[DecisionTree]

    fn __init__(inout self, n_trees: Int = 10, min_samples_split: Int = 2, max_depth: Int = 100, n_feats: Int = -1, criterion: String = 'gini'):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.criterion = criterion.lower()
        self.trees = UnsafePointer[DecisionTree]()

    fn __del__(owned self):
        if self.trees:
            for i in range(self.n_trees):
                (self.trees + i).destroy_pointee()
            self.trees.free()

    fn fit(inout self, X: Matrix, y: Matrix) raises:
        self.trees = UnsafePointer[DecisionTree].alloc(self.n_trees)
        for i in range(self.n_trees):
            var tree = DecisionTree(
                min_samples_split = self.min_samples_split,
                max_depth = self.max_depth,
                n_feats = self.n_feats,
                criterion = self.criterion
            )
            var X_samp: Matrix
            var y_samp: Matrix
            X_samp, y_samp = bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            (self.trees + i).init_pointee_move(tree)

    fn predict(self, X: Matrix) raises -> Matrix:
        var tree_preds = Matrix(X.height, self.n_trees)
        for i in range(self.n_trees):
            tree_preds['', i] = self.trees[i].predict(X)
        
        var y_predicted = Matrix(X.height, 1)
        for i in range(tree_preds.height):
            y_predicted.data[i] = _predict(tree_preds[i], self.criterion)
        return y_predicted^
