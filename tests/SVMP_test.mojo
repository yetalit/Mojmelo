from mojmelo.SVM import SVM_Primal
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    Python.add_to_path(".")
    svmp_test = Python.import_module("SVMP_test")
    data = svmp_test.get_data() # X_train, X_test, y_train, y_test
    svmp = SVM_Primal()
    svmp.fit(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[2]).T())
    y_pred = svmp.predict(Matrix.from_numpy(data[1]))
    print("SVM_Primal classification accuracy:", accuracy_score(data[3], y_pred))
    svmp_test.test(data[0], data[2], svmp.weights.to_numpy(), svmp.bias)
