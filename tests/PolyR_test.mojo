from mojmelo.PolynomialRegression import PolyRegression
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import mse, r2_score
from python import Python

def main():
    Python.add_to_path(".")
    pr_test = Python.import_module("PolyR_test")
    data = pr_test.get_data() # X, X_train, X_test, y_train, y_test
    pr = PolyRegression(2, 0.01, 1000)
    pr.fit(Matrix.from_numpy(data[1]).T(), Matrix.from_numpy(data[3]).T())
    y_pred = pr.predict(Matrix.from_numpy(data[2]).T())
    print("MSE:", mse(Matrix.from_numpy(data[4]).T(), y_pred))
    print("Accuracy:", r2_score(Matrix.from_numpy(data[4]).T(), y_pred))
    y_pred_curve = pr.predict(Matrix.from_numpy(data[0]).T())
    pr_test.test(data[0], data[1], data[2], data[3], data[4], y_pred_curve.to_numpy())
