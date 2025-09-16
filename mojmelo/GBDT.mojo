from mojmelo.utils.BDecisionTree import BDecisionTree
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CVM, sigmoid, log_g, log_h, mse_g, mse_h, softmax_g, softmax_h, softmax_link
from algorithm import parallelize

struct GBDT(CVM):
	"""Gradient Boosting with support for both classification and regression."""
	var criterion: String
    """The method to measure the quality of a split:
    For binary classification -> 'log';
	For multi-class classification -> 'softmax';
    For regression -> 'mse'.
    """
	var loss_g: fn(Matrix, Matrix) raises -> Matrix
	var loss_h: fn(Matrix) raises -> Matrix
	var n_trees: Int
	"""The number of boosting stages to perform."""
	var min_samples_split: Int
	"""The minimum number of samples required to split an internal node."""
	var max_depth: Int
	"""The maximum depth of the tree."""
	var learning_rate: Float32
	"""Learning rate."""
	var reg_lambda: Float32
	"""The L2 regularization parameter."""
	var reg_alpha: Float32
	"""The L1 regularization parameter."""
	var gamma: Float32
	"""Minimum loss reduction required to make a further partition on a leaf node of the tree."""
	var n_bins: Int
	"""Generates histogram boundaries as possible threshold values when n_bins >= 2 instead of all possible values."""
	var trees: UnsafePointer[BDecisionTree]
	var score_start: Float32
	var num_class: Int

	fn __init__(out self,
		criterion: String = 'log',
		n_trees: Int = 10, min_samples_split: Int = 10, max_depth: Int = 3,
		learning_rate: Float32 = 0.1, reg_lambda: Float32 = 1.0, reg_alpha: Float32 = 0.0, gamma: Float32 = 0.0, n_bins: Int = 0
		):
		self.criterion = criterion.lower()
		if self.criterion == 'log':
			self.loss_g = log_g
			self.loss_h = log_h
		elif self.criterion == 'softmax':
			self.loss_g = softmax_g
			self.loss_h = softmax_h
		else:
			self.loss_g = mse_g
			self.loss_h = mse_h
		self.n_trees = n_trees
		self.min_samples_split = min_samples_split
		self.max_depth = max_depth
		self.learning_rate = learning_rate
		self.reg_lambda = reg_lambda
		self.reg_alpha = reg_alpha
		self.gamma = gamma
		self.n_bins = n_bins
		self.trees = UnsafePointer[BDecisionTree]()
		self.score_start = 0.0
		self.num_class = 0

	fn __del__(var self):
		if self.trees:
			for i in range(self.n_trees):
				(self.trees + i).destroy_pointee()
			self.trees.free()

	fn fit(mut self, X: Matrix, y: Matrix) raises:
		"""Fit the gradient boosting model."""
		var X_F = X.asorder('f')
		var score: Matrix
		if self.criterion == 'softmax':
			self.num_class = len(y.unique())
			self.score_start = 0.0
			self.trees = UnsafePointer[BDecisionTree].alloc(self.n_trees * self.num_class)
			score = Matrix.zeros(X.height, self.num_class)
		else:
			self.num_class = 1
			self.trees = UnsafePointer[BDecisionTree].alloc(self.n_trees)
			self.score_start = y.mean()
			score = Matrix.full(X.height, 1, self.score_start)

		for i in range(self.n_trees):
			@parameter
			fn p(k: Int):
				try:
					var g = self.loss_g(y, score)
					var h = self.loss_h(score)
					var tree = BDecisionTree(min_samples_split = self.min_samples_split, max_depth = self.max_depth, reg_lambda = self.reg_lambda, reg_alpha = self.reg_alpha, gamma = self.gamma, n_bins=self.n_bins)
					tree.fit(X_F, g=g['', k], h=h['', k])
					(self.trees + i * self.num_class + k).init_pointee_move(tree)
					self.trees[i * self.num_class + k]._moveinit_(tree)
					score['', k] += self.learning_rate * self.trees[i * self.num_class + k].predict(X)
				except e:
					print('Error:', e)
			parallelize[p](self.num_class)

	fn predict(self, X: Matrix) raises -> Matrix:
		"""Predict class or regression value for X.
        
        Returns:
            The predicted values.
        """
		var scores = Matrix(X.height, self.num_class)
		@parameter
		fn per_class(k: Int):
			var score = Matrix(X.height, self.n_trees)
			@parameter
			fn per_tree(i: Int):
				try:
					score['', i] = self.learning_rate * self.trees[i * self.num_class + k].predict(X)
				except e:
					print('Error:', e)
			parallelize[per_tree](self.n_trees)
			try:
				scores['', k] = score.sum(axis=1) + self.score_start
			except e:
				print('Error:', e)
		parallelize[per_class](self.num_class)
		if self.criterion == 'mse':
			return scores^
		if self.criterion == 'softmax':
			return softmax_link(scores).argmax_f(axis=1) # predicted class
		scores = sigmoid(scores)
		return scores.where(scores > 0.5, 1.0, 0.0)

	fn __init__(out self, params: Dict[String, String]) raises:
		if 'criterion' in params:
			self.criterion = params['criterion'].lower()
		else:
			self.criterion = 'log'
		if self.criterion == 'log':
			self.loss_g = log_g
			self.loss_h = log_h
		else:
			self.loss_g = mse_g
			self.loss_h = mse_h
		if 'n_trees' in params:
			self.n_trees = atol(String(params['n_trees']))
		else:
			self.n_trees = 10
		if 'min_samples_split' in params:
			self.min_samples_split = atol(String(params['min_samples_split']))
		else:
			self.min_samples_split = 10
		if 'max_depth' in params:
			self.max_depth = atol(String(params['max_depth']))
		else:
			self.max_depth = 3
		if 'learning_rate' in params:
			self.learning_rate = atof(String(params['learning_rate'])).cast[DType.float32]()
		else:
			self.learning_rate = 0.1
		if 'reg_lambda' in params:
			self.reg_lambda = atof(String(params['reg_lambda'])).cast[DType.float32]()
		else:
			self.reg_lambda = 1.0
		if 'reg_alpha' in params:
			self.reg_alpha = atof(String(params['reg_alpha'])).cast[DType.float32]()
		else:
			self.reg_alpha = 0.0
		if 'gamma' in params:
			self.gamma = atof(String(params['gamma'])).cast[DType.float32]()
		else:
			self.gamma = 0.0
		if 'n_bins' in params:
			self.n_bins = atol(String(params['n_bins']))
		else:
			self.n_bins = 0
		self.trees = UnsafePointer[BDecisionTree]()
		self.score_start = 0.0
		self.num_class = 0
