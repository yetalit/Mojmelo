from mojmelo.LinearRegression import LinearRegression
from mojmelo.utils.Matrix import Matrix
from python import Python

def main():
    Python.add_to_path(".")
    var test = Python.import_module("LR_test")
    data = test.get_data() # X, X_train, X_test, y_train, y_test
    lr = LinearRegression(0.01, 1000)
    lr.fit(Matrix.from_numpy(data[1]), Matrix.from_numpy(data[3]).T())
    y_pred_line = lr.predict(Matrix.from_numpy(data[0]))
    test.test(data[0], data[1], data[2], data[3], data[4], y_pred_line.to_numpy())
