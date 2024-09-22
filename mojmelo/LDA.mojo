from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import gt
from collections.vector import InlinedFixedVector
from utils import Span
from python import PythonObject

struct LDA:
    var n_components: Int
    var linear_discriminants: Matrix

    fn __init__(inout self, n_components: Int):
        self.n_components = n_components
        self.linear_discriminants = Matrix(0, 0)

    fn fit(inout self, X: Matrix, y: PythonObject) raises:
        var class_labels: List[String]
        var class_freq: List[Int]
        class_labels, class_freq = Matrix.unique(y)

        # Within class scatter matrix:
        # SW = sum((X_c - mean_X_c)^2 )

        # Between class scatter:
        # SB = sum( n_c * (mean_X_c - mean_overall)^2 )

        var mean_overall = X.mean(0)
        var SW = Matrix.zeros(X.width, X.width)
        var SB = Matrix.zeros(X.width, X.width)
        for i in range(len(class_labels)):
            var X_c = Matrix(class_freq[i], X.width)
            var pointer: Int = 0
            for j in range(X.height):
                if str(y[j]) == class_labels[i]:
                    X_c[pointer] = X[j]
                    pointer += 1
            var mean_c = X_c.mean(0)
            var mean_c_reduce = Matrix(X_c.height, X_c.width)
            for i in range(X_c.height):
                mean_c_reduce[i] = X[i] - mean_c
            # (4, n_c) * (n_c, 4) = (4,4) -> transpose
            SW += (mean_c_reduce).T() * (mean_c_reduce)

            # (4, 1) * (1, 4) = (4,4) -> reshape
            var mean_diff = (mean_c - mean_overall).reshape(X.width, 1)
            SB += X_c.height * (mean_diff * mean_diff.T())

        # Get eigenvalues and eigenvectors of SW^-1 * SB
        var eigenvalues: Matrix
        var eigenvectors: Matrix
        eigenvalues, eigenvectors = (SW.inv() * SB).eigen()
        # transpose for easier calculations
        eigenvectors = eigenvectors.T()
        # sort eigenvalues high to low
        var v_abs = eigenvalues.abs()
        var indices = InlinedFixedVector[Int](capacity = v_abs.size)
        for i in range(v_abs.size):
            indices.append(i)
        # sort eigenvectors
        mojmelo.utils.utils.partition[gt](Span[Float32, __lifetime_of(v_abs)](unsafe_ptr= v_abs.data, len= v_abs.size), indices, self.n_components)
        # store first n eigenvectors
        self.linear_discriminants = Matrix.zeros(self.n_components, eigenvectors.width)
        for i in range(self.n_components):
            self.linear_discriminants[i] = eigenvectors[indices[i]]

    fn transform(self, X: Matrix) raises -> Matrix:
        # project data
        return X * self.linear_discriminants.T()
