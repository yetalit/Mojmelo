from mojmelo.RandomForest import RandomForest
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import mse, r2_score
from python import Python

def main():
    Python.add_to_path(".")
    rfr_test = Python.import_module("RandomFR_test")
    data = rfr_test.get_data() # X_train, X_test, y_train, y_test
    rfr = RandomForest(criterion='mse', n_trees = 5)
    rfr.fit(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[2]))
    y_pred = rfr.predict(Matrix.from_numpy(data[1]))
    print("RandomForest regression MSE:", mse(Matrix.from_numpy(data[3]), y_pred))
    print("RandomForest regression Accuracy:", r2_score(Matrix.from_numpy(data[3]), y_pred))
