from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import gt, fill_indices_list
from algorithm import parallelize
from python import Python

struct PCA:
    var n_components: Int
    var components: Matrix
    var components_T: Matrix
    var explained_variance: Matrix
    var explained_variance_ratio: Matrix
    var mean: Matrix
    var whiten: Bool
    var whiten_: Matrix
    var lapack: Bool

    fn __init__(out self, n_components: Int, whiten: Bool = False, lapack: Bool = False):
        self.n_components = n_components
        self.components = Matrix(0, 0)
        self.components_T = Matrix(0, 0)
        self.explained_variance = Matrix(0, 0)
        self.explained_variance_ratio = Matrix(0, 0)
        self.mean = Matrix(0, 0)
        self.whiten = whiten
        self.whiten_ = Matrix(0, 0)
        self.lapack = lapack

    fn fit(mut self, X: Matrix) raises:
        # Mean centering
        self.mean = X.mean(0)
        
        var S: Matrix
        if self.lapack:
            numpy_linalg = Python.import_module('numpy.linalg')
            USVt = numpy_linalg.svd((X - self.mean).to_numpy(), full_matrices=False)
            S = Matrix.from_numpy(USVt[1])
            self.components = Matrix.from_numpy(USVt[2]).load_rows(self.n_components)
        else:
            _, S, Vt = (X - self.mean).svd(full_matrices=False)
            var indices = fill_indices_list(S.size)
            mojmelo.utils.sort.partition[gt](Span[Float32, __origin_of(S)](ptr= S.data, length= S.size), indices, self.n_components)
            self.components = Matrix.zeros(self.n_components, Vt.width, order=X.order)
            @parameter
            fn p(i: Int):
                self.components[i, unsafe=True] = Vt[indices[i].value, unsafe=True]
            parallelize[p](self.n_components)
        self.components_T = self.components.T()

        var explained_variance = (S ** 2) / (X.height - 1)
        self.explained_variance = explained_variance.load_columns(self.n_components)
        self.explained_variance_ratio = self.explained_variance / explained_variance.sum()
        if self.whiten:
            self.whiten_ = (self.explained_variance + 1e-8).sqrt() # Avoid division by zero

    fn transform(self, X: Matrix) raises -> Matrix:
        # project data
        if self.whiten:
            return ((X - self.mean) * self.components_T) / self.whiten_
        return (X - self.mean) * self.components_T

    fn inverse_transform(self, X_transformed: Matrix) raises -> Matrix:
        if self.whiten:
            return (X_transformed.ele_mul(self.whiten_) * self.components) + self.mean
        return (X_transformed * self.components) + self.mean
