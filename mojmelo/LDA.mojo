from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import gt, fill_indices_list
from algorithm import parallelize
from python import Python, PythonObject

struct LDA:
    var n_components: Int
    var linear_discriminants: Matrix
    var lapack: Bool

    fn __init__(out self, n_components: Int, lapack: Bool = False):
        self.n_components = n_components
        self.linear_discriminants = Matrix(0, 0)
        self.lapack = lapack

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
        if self.lapack:
            numpy_linalg = Python.import_module('numpy.linalg')
            vals_vects = numpy_linalg.eig(numpy_linalg.pinv(SW.to_numpy()).dot(SB.to_numpy()))
            eigenvalues, eigenvectors = Matrix.from_numpy(vals_vects[0]), Matrix.from_numpy(vals_vects[1])
        else:
            eigenvalues, eigenvectors = (Matrix.solve(SW, SB)).eigen()
        eigenvalues = eigenvalues.abs()
        # sort eigenvalues high to low
        var indices = fill_indices_list(eigenvalues.size)
        mojmelo.utils.sort.partition[gt](Span[Float32, __origin_of(eigenvalues)](ptr= eigenvalues.data, length= eigenvalues.size), indices, self.n_components)
        # store first n eigenvectors
        self.linear_discriminants = Matrix(eigenvectors.height, self.n_components)
        @parameter
        fn p(i: Int):
            self.linear_discriminants['', i, unsafe=True] = eigenvectors['', indices[i].value, unsafe=True]
        parallelize[p](self.n_components)

    fn transform(self, X: Matrix) raises -> Matrix:
        # project data
        return X * self.linear_discriminants
