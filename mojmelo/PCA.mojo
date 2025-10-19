from mojmelo.utils.Matrix import Matrix
from algorithm import parallelize
from python import Python

struct PCA:
    """Principal component analysis (PCA).
    Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
    """
    var n_components: Int
    """Number of components to keep."""
    var components: Matrix
    var components_T: Matrix
    var explained_variance: Matrix
    """The amount of variance explained by each of the selected components."""
    var explained_variance_ratio: Matrix
    """Percentage of variance explained by each of the selected components."""
    var mean: Matrix
    var whiten: Bool
    """To transform data to have zero mean, unit variance, and no correlation between features."""
    var whiten_: Matrix
    var lapack: Bool
    """Use LAPACK to calculate svd."""

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
        """Fit the model."""
        # Mean centering
        self.mean = X.mean(0)
        
        var S: Matrix
        if self.lapack:
            numpy_linalg = Python.import_module('numpy.linalg')
            USVt = numpy_linalg.svd((X - self.mean).to_numpy(), full_matrices=False)
            S = Matrix.from_numpy(USVt[1])
            self.components = Matrix.from_numpy(USVt[2]).load_rows(self.n_components)
        else:
            _, S, Vt = (X - self.mean).svd()
            var indices = S.argsort_inplace[ascending=False]()
            self.components = Matrix.zeros(self.n_components, Vt.width, order=X.order)
            @parameter
            fn p(i: Int):
                self.components[i, unsafe=True] = Vt[Int(indices[i]), unsafe=True]
            parallelize[p](self.n_components)

        S = S.load_columns(self.n_components)
        self.components_T = self.components.T()

        self.explained_variance = (S ** 2) / (X.height - 1)
        self.explained_variance_ratio = self.explained_variance / self.explained_variance.sum()
        if self.whiten:
            self.whiten_ = (self.explained_variance + 1e-8).sqrt() # Avoid division by zero

    fn transform(self, X: Matrix) raises -> Matrix:
        """Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted from a training set.

        Returns:
            Projection of X in the first principal components.
        """
        # project data
        if self.whiten:
            return ((X - self.mean) * self.components_T) / self.whiten_
        return (X - self.mean) * self.components_T

    fn inverse_transform(self, X_transformed: Matrix) raises -> Matrix:
        """Transform data back to its original space.
        
        Returns:
            Original data.
        """
        if self.whiten:
            return (X_transformed.ele_mul(self.whiten_) * self.components) + self.mean
        return (X_transformed * self.components) + self.mean
