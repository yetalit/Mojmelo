from mojmelo.Perceptron import Perceptron
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    Python.add_to_path(".")
    p_test = Python.import_module("Percept_test")
    data = p_test.get_data() # X_train, X_test, y_train, y_test
    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[2]).T())
    y_pred = p.predict(Matrix.from_numpy(data[1]))
    print("Perceptron classification accuracy:", accuracy_score(data[3], y_pred))
    p_test.test(data[0], data[2], p.weights.to_numpy(), p.bias)
