from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.svd import svd
from mojmelo.utils.utils import MODEL_IDS
from algorithm import parallelize
from python import Python

struct PCA(Copyable):
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
    comptime MODEL_ID = 12

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
            S = Matrix.from_numpy(USVt[1]).load_columns(self.n_components)
            self.components = Matrix.from_numpy(USVt[2]).load_rows(self.n_components)
        else:
            S, self.components = svd((X - self.mean), self.n_components)

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

    fn save(self, path: String) raises:
        """Save model data necessary for transformation to the specified path."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        with open(_path, "w") as f:
            f.write_bytes(UInt8(Self.MODEL_ID).as_bytes())
            f.write_bytes(UInt64(self.n_components).as_bytes())
            f.write_bytes(UInt64(self.components.width).as_bytes())
            f.write_bytes(Span(ptr=self.components.data.bitcast[UInt8](), length=4*self.components.size))
            f.write_bytes(Span(ptr=self.mean.data.bitcast[UInt8](), length=4*self.mean.size))
            f.write_bytes(UInt8(self.whiten).as_bytes())
            if self.whiten:
                f.write_bytes(Span(ptr=self.whiten_.data.bitcast[UInt8](), length=4*self.whiten_.size))

    @staticmethod
    fn load(path: String) raises -> Self:
        """Load a saved model from the specified path for transformation."""
        var _path = path if path.endswith('.mjml') else path + '.mjml'
        var model = Self(0)
        with open(_path, "r") as f:
            var id = f.read_bytes(1)[0]
            if id < 1 or id > MODEL_IDS.size-1:
                raise Error('Input file with invalid metadata!')
            elif id != Self.MODEL_ID:
                raise Error('Based on the metadata, ', _path, ' belongs to ', materialize[MODEL_IDS]()[id], ' algorithm!')
            var n_components = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
            var components_width = Int(f.read_bytes(8).unsafe_ptr().bitcast[UInt64]()[])
            model.components = Matrix(n_components, components_width, UnsafePointer[Float32, MutAnyOrigin](f.read_bytes(4*n_components*components_width).unsafe_ptr().bitcast[Float32]()))
            model.components_T = model.components.T()
            model.mean = Matrix(1, components_width, UnsafePointer[Float32, MutAnyOrigin](f.read_bytes(4*components_width).unsafe_ptr().bitcast[Float32]()))
            model.whiten = Bool(f.read_bytes(1)[0])
            if model.whiten:
                model.whiten_ = Matrix(1, n_components, UnsafePointer[Float32, MutAnyOrigin](f.read_bytes(4*n_components).unsafe_ptr().bitcast[Float32]()))
        return model^
