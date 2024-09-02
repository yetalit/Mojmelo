from mojmelo.Adaboost import Adaboost
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.preprocessing import train_test_split
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    Python.add_to_path(".")
    ab_test = Python.import_module("Adaboost_test")
    data = ab_test.get_data() # X, y
    X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[1]).T(), test_size = 0.2, random_state = 5)
    ab = Adaboost(n_clf = 5)
    ab.fit(X_train, y_train)
    y_pred = ab.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
