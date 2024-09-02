from mojmelo.Perceptron import Perceptron
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.preprocessing import train_test_split
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    Python.add_to_path(".")
    p_test = Python.import_module("Percept_test")
    data = p_test.get_data() # X, y
    X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[1]).T(), test_size=0.2, random_state=123)
    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    y_pred = p.predict(X_test)
    print("Perceptron classification accuracy:", accuracy_score(y_test, y_pred))
    p_test.test(X_train.to_numpy(), y_train.to_numpy(), p.weights.to_numpy(), p.bias)
