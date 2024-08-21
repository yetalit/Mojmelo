from mojmelo.Adaboost import Adaboost
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    Python.add_to_path(".")
    ab_test = Python.import_module("Adaboost_test")
    data = ab_test.get_data() # X_train, X_test, y_train, y_test
    ab = Adaboost(n_clf=5)
    ab.fit(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[2]).T())
    y_pred = ab.predict(Matrix.from_numpy(data[1]))
    print("Accuracy:", accuracy_score(data[3], y_pred))
