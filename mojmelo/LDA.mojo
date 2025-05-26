from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import gt
from algorithm import parallelize
from python import PythonObject

struct LDA:
    var n_components: Int
    var linear_discriminants: Matrix

    fn __init__(out self, n_components: Int):
        self.n_components = n_components
        self.linear_discriminants = Matrix(0, 0)

    fn fit(mut self, X: Matrix, y: PythonObject) raises:
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
                if String(y[j]) == class_labels[i]:
                    X_c[pointer] = X[j]
                    pointer += 1
            var mean_c = X_c.mean(0)
            var X_c_sub_mean_c = X_c - mean_c
            SW += (X_c_sub_mean_c).T() * (X_c_sub_mean_c)

            var mean_diff = (mean_c - mean_overall).reshape(X.width, 1)
            SB += X_c.height * (mean_diff * mean_diff.T())

        # Get eigenvalues and eigenvectors of SW^-1 * SB
        var eigenvalues: Matrix
        var eigenvectors: Matrix
        eigenvalues, eigenvectors = (Matrix.solve(SW, SB)).eigen()
        # transpose for easier calculations
        eigenvectors = eigenvectors.T()
        # sort eigenvalues high to low
        var v_abs = eigenvalues.abs()
        var indices = List[Int](capacity=v_abs.size)
        indices.resize(v_abs.size, 0)
        for i in range(v_abs.size):
            indices[i] = i
        # sort eigenvectors
        mojmelo.utils.utils.partition[gt](Span[Float32, __origin_of(v_abs)](ptr= v_abs.data, length= v_abs.size), indices, self.n_components)
        # store first n eigenvectors
        self.linear_discriminants = Matrix(self.n_components, eigenvectors.width)
        @parameter
        fn p(i: Int):
            self.linear_discriminants[i, unsafe=True] = eigenvectors[indices[i], unsafe=True]
        parallelize[p](self.n_components)

    fn transform(self, X: Matrix) raises -> Matrix:
        # project data
        return X * self.linear_discriminants.T()
