from mojmelo.LinearRegression import LinearRegression
from mojmelo.utils.Matrix import Matrix
from mojmelo.preprocessing import train_test_split
from mojmelo.utils.utils import mse, r2_score
from python import Python

def main():
    lr_test = Python.import_module("LinR_test")
    data = lr_test.get_data() # X, y
    X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[1]).T(), test_size=0.2, random_state=1234)
    lr = LinearRegression(0.01, 1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("MSE:", mse(y_test, y_pred))
    print("Accuracy:", r2_score(y_test, y_pred))
    y_pred_line = lr.predict(Matrix.from_numpy(data[0]))
    lr_test.test(data[0], X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy(), y_pred_line.to_numpy())
