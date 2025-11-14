from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import CV, sign
from mojmelo.utils.libsvm.svm_parameter import svm_parameter
from mojmelo.utils.libsvm.svm_problem import svm_problem
from mojmelo.utils.libsvm.svm_node import svm_node
from mojmelo.utils.libsvm.svm_model import svm_model
from mojmelo.utils.libsvm.svm import svm_check_parameter, svm_train, svm_predict, svm_decision_function, svm_free_and_destroy_model
from algorithm import parallelize
import random

struct SVC(CV):
    """Support Vector Classification."""
    var C: Float64
    """Regularization parameter. When C != 0, C-Support Vector Classification model will be used."""
    var nu: Float64
    """An upper bound on the fraction of margin errors and a lower bound of the fraction of support vectors.
    When nu != 0, Nu-Support Vector Classification model will be used.
    """
    var kernel: String
    """Specifies the kernel type to be used in the algorithm: {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}."""
    var degree: Int
    """Degree of the polynomial kernel function ('poly')."""
    var gamma: Float64
    """Kernel coefficient for 'rbf', 'poly' and 'sigmoid':
    if gamma = -1 (default) is passed then it uses 1 / (n_features * X.var());
    if gamma = -0.1, it uses 1 / n_features;
    if custom value, it must be non-negative.
    """
    var coef0: Float64
    """Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'."""

    var cache_size: Float64
    """Specify the size of the kernel cache (in MB)."""
    var tol: Float64
    """Tolerance for stopping criterion."""
    var shrinking: Bool
    """Whether to use the shrinking heuristic."""
    var probability: Bool
    """Whether to enable probability estimates."""
    var _model: UnsafePointer[svm_model, MutOrigin.external]
    var _n_features: Int
    var _x_list: List[List[svm_node]]
    var _x_ptr: List[UnsafePointer[svm_node, MutOrigin.external]]

    fn __init__(out self, C: Float64 = 0.0, nu: Float64 = 0.0, kernel: String = 'rbf',
                degree: Int = 2, gamma: Float64 = -1.0, coef0: Float64 = 0.0, cache_size: Float64 = 200, tol: Float64 = 1e-3, shrinking: Bool = True, probability: Bool = False, random_state: Int = -1):
        self.C = C
        self.nu = nu
        self.kernel = kernel.lower()
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.cache_size = cache_size
        self.tol = tol
        self.shrinking = shrinking
        self.probability = probability
        if random_state != -1:
            random.seed(random_state)
        else:
            random.seed()
        self._model = UnsafePointer[svm_model, MutOrigin.external]()
        self._n_features = 0
        self._x_list = List[List[svm_node]]()
        self._x_ptr = List[UnsafePointer[svm_node, MutOrigin.external]]()

    fn fit(mut self, X: Matrix, y: Matrix) raises:
        """Fit the SVM model according to the given training data."""
        self._n_features = X.width

        var svm_type = 5
        if self.C != 0.0:
            svm_type = svm_parameter.C_SVC
        elif self.nu != 0.0:
            svm_type = svm_parameter.NU_SVC

        if self.gamma == -1.0:
            self.gamma = (1.0 / (X.width * X._var())).cast[DType.float64]()
        elif self.gamma == -0.1:
            self.gamma = 1.0 / X.width

        var svm_kernel = 5
        if self.kernel == 'linear':
            svm_kernel = svm_parameter.LINEAR
        elif self.kernel == 'poly':
            svm_kernel = svm_parameter.POLY
        elif self.kernel == 'rbf':
            svm_kernel = svm_parameter.RBF
        elif self.kernel == 'sigmoid':
            svm_kernel = svm_parameter.SIGMOID
        elif self.kernel == 'precomputed':
            svm_kernel = svm_parameter.PRECOMPUTED

        var param = svm_parameter(
            svm_type = svm_type,
            kernel_type = svm_kernel,
            degree = self.degree,
            gamma = self.gamma,
            coef0 = self.coef0,
            cache_size = self.cache_size,
            eps = self.tol,
            C = self.C,
            nr_weight = 0,
            weight_label = UnsafePointer[Int, MutOrigin.external](),
            weight = UnsafePointer[Float64, MutOrigin.external](),
            nu = self.nu,
            p = 0.0,
            shrinking = Int(self.shrinking),
            probability = Int(self.probability))

        var X_float64 = X.cast_ptr[DType.float64]()

        self._x_list = List[List[svm_node]](capacity=X.height)
        self._x_list.resize(X.height, List[svm_node]())
        self._x_ptr = List[UnsafePointer[svm_node, MutOrigin.external]](capacity=X.height)
        self._x_ptr.resize(X.height, UnsafePointer[svm_node, MutOrigin.external]())

        @parameter
        fn p(i: Int):
            for c in range(X.width):
                var val: Float64
                if X.order == 'c':
                    val = X_float64[(i * X.width) + c]
                else:
                    val = X_float64[(c * X.height) + i]
                if val != 0.0:
                    self._x_list[i].append(svm_node(c+1, val))
            self._x_list[i].append(svm_node(-1, 0))
            self._x_ptr[i] = self._x_list[i]._data
        parallelize[p](X.height)

        X_float64.free()

        var prob = svm_problem()
        prob.l = X.height
        prob.y = y.cast_ptr[DType.float64]()
        prob.x = self._x_ptr._data

        var check = svm_check_parameter(prob, param)
        if check != "":
            prob.y.free()
            raise Error('Error: ' + check)

        self._model = svm_train(prob, param)

        prob.y.free()
    
    fn predict(self, X: Matrix) raises -> Matrix:
        """Perform classification on samples in X.

        Returns:
            The predicted classes.
        """
        var X_float64 = X.cast_ptr[DType.float64]()
        var y_ptr = alloc[Float64](X.height)

        @parameter
        fn p(i: Int):
            var x_list = List[svm_node]()
            for c in range(X.width):
                var val: Float64
                if X.order == 'c':
                    val = X_float64[(i * X.width) + c]
                else:
                    val = X_float64[(c * X.height) + i]
                if val != 0.0:
                    x_list.append(svm_node(c+1, val))
            x_list.append(svm_node(-1, 0))
            y_ptr[i] = svm_predict(self._model[], x_list._data)
            _ = x_list
        parallelize[p](X.height)

        X_float64.free()

        return Matrix(data=y_ptr, height=X.height, width=1)

    fn decision_function(self, X: Matrix) -> List[List[Float64]]:
        """Evaluate the decision function for the samples in X.
        
        Returns:
            The decision values in a 2D List format.
        """
        var X_float64 = X.cast_ptr[DType.float64]()
        var dec_values = List[List[Float64]](capacity=X.height)
        dec_values.resize(X.height, List[Float64]())

        @parameter
        fn p(i: Int):
            var x_list = List[svm_node]()
            for c in range(X.width):
                var val: Float64
                if X.order == 'c':
                    val = X_float64[(i * X.width) + c]
                else:
                    val = X_float64[(c * X.height) + i]
                if val != 0.0:
                    x_list.append(svm_node(c+1, val))
            x_list.append(svm_node(-1, 0))
            var result = svm_decision_function(self._model[], x_list._data)
            dec_values[i] = List[Float64](unsafe_uninit_length=result[1])
            dec_values[i]._data = result[0]
            _ = x_list
        parallelize[p](X.height)

        X_float64.free()

        return dec_values^

    fn __del__(deinit self):
        if self._model:
            svm_free_and_destroy_model(self._model)

    fn support_vectors(self) raises -> Matrix:
        """Get support vectors."""
        var support_vectors_ = Matrix.zeros(self._model[].l, self._n_features)
        for row in range(support_vectors_.height):
            var pointer = 0
            while self._model[].SV[row][pointer].index != -1:
                support_vectors_[row, self._model[].SV[row][pointer].index-1] = self._model[].SV[row][pointer].value.cast[DType.float32]()
                pointer += 1
        return support_vectors_^

    fn __init__(out self, params: Dict[String, String]) raises:
        if 'C' in params:
            self.C = atof(String(params['C']))
        else:
            self.C = 0.0
        if 'nu' in params:
            self.nu = atof(String(params['nu']))
        else:
            self.nu = 0.0
        if 'kernel' in params:
            self.kernel = params['kernel']
        else:
            self.kernel = 'rbf'
        if 'degree' in params:
            self.degree = atol(String(params['degree']))
        else:
            self.degree = 2
        if 'gamma' in params:
            self.gamma = atof(String(params['gamma']))
        else:
            self.gamma = -1.0
        if 'coef0' in params:
            self.coef0 = atof(String(params['coef0']))
        else:
            self.coef0 = 0.0
        if 'cache_size' in params:
            self.cache_size = atof(String(params['cache_size']))
        else:
            self.cache_size = 200
        if 'tol' in params:
            self.tol = atof(String(params['tol']))
        else:
            self.tol = 1e-3
        if 'shrinking' in params:
            if params['shrinking'] == 'True':
                self.shrinking = True
            else:
                self.shrinking = False
        else:
            self.shrinking = True
        if 'probability' in params:
            if params['probability'] == 'True':
                self.probability = True
            else:
                self.probability = False
        else:
            self.probability = False
        if 'random_state' in params and atol(String(params['random_state'])) != -1:
            random.seed(atol(String(params['random_state'])))
        else:
            random.seed()
        self._model = UnsafePointer[svm_model, MutOrigin.external]()
        self._n_features = 0
        self._x_list = List[List[svm_node]]()
        self._x_ptr = List[UnsafePointer[svm_node, MutOrigin.external]]()
