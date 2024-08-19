from mojmelo.LinearRegression import LinearRegression
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import mse, r2_score
from python import Python

def main():
    Python.add_to_path(".")
    lr_test = Python.import_module("LinR_test")
    data = lr_test.get_data() # X, X_train, X_test, y_train, y_test
    lr = LinearRegression(0.01, 1000)
    lr.fit(Matrix.from_numpy(data[1]), Matrix.from_numpy(data[3]).T())
    y_pred = lr.predict(Matrix.from_numpy(data[2]))
    print("MSE:", mse(Matrix.from_numpy(data[4]).T(), y_pred))
    print("Accuracy:", r2_score(Matrix.from_numpy(data[4]).T(), y_pred))
    y_pred_line = lr.predict(Matrix.from_numpy(data[0]))
    lr_test.test(data[0], data[1], data[2], data[3], data[4], y_pred_line.to_numpy())
