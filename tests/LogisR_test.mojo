from mojmelo.LogisticRegression import LogisticRegression
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    Python.add_to_path(".")
    lr_test = Python.import_module("LogisR_test")
    data = lr_test.get_data() # X_train, X_test, y_train, y_test
    lr = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    lr.fit(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[2]).T())
    y_pred = lr.predict(Matrix.from_numpy(data[1]))
    print("LR classification accuracy:", accuracy_score(data[3], y_pred))
