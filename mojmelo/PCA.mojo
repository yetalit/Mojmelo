from collections.vector import InlinedFixedVector
from utils import Span
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import gt

struct PCA:
    var n_components: Int
    var components: Matrix
    var mean: Matrix

    fn __init__(inout self, n_components: Int):
        self.n_components = n_components
        self.components = Matrix(0, 0)
        self.mean = Matrix(0, 0)

    fn fit(inout self, X: Matrix) raises:
        # Mean centering
        self.mean = X.mean(0)
        
        # eigenvalues, eigenvectors
        var eigenvalues: Matrix
        var eigenvectors: Matrix
        # covariance, function needs samples as columns
        eigenvalues, eigenvectors = (X - self.mean).T().cov().eigen()
        # transpose for easier calculations
        eigenvectors = eigenvectors.T()

        var indices = InlinedFixedVector[Int](capacity = eigenvalues.size)
        for i in range(eigenvalues.size):
            indices.append(i)
        # sort eigenvectors
        mojmelo.utils.utils.partition[gt](Span[Float32, __lifetime_of(eigenvalues)](unsafe_ptr= eigenvalues.data, len= eigenvalues.size), indices, self.n_components)
        # store first n eigenvectors
        self.components = Matrix.zeros(self.n_components, eigenvectors.width)
        for i in range(self.n_components):
            self.components[i] = eigenvectors[indices[i]]

    fn transform(self, X: Matrix) raises -> Matrix:
        # project data
        return (X - self.mean) * self.components.T()
