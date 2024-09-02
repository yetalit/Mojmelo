from mojmelo.PolynomialRegression import PolyRegression
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.preprocessing import train_test_split
from mojmelo.utils.utils import mse, r2_score
from python import Python

def main():
    Python.add_to_path(".")
    pr_test = Python.import_module("PolyR_test")
    data = pr_test.get_data() # X, y
    X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]).T(), Matrix.from_numpy(data[1]).T(), test_size=0.2, random_state=1234)
    pr = PolyRegression(2, 0.01, 1000)
    pr.fit(X_train, y_train)
    y_pred = pr.predict(X_test)
    print("MSE:", mse(y_test, y_pred))
    print("Accuracy:", r2_score(y_test, y_pred))
    y_pred_curve = pr.predict(Matrix.from_numpy(data[0]).T())
    pr_test.test(data[0], X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy(), y_pred_curve.to_numpy())
