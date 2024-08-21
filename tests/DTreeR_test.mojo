from mojmelo.DecisionTree import DecisionTree
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import mse, r2_score
from python import Python

def main():
    Python.add_to_path(".")
    dtr_test = Python.import_module("DTreeR_test")
    data = dtr_test.get_data() # X_train, X_test, y_train, y_test
    dtr = DecisionTree(criterion='mse')
    dtr.fit(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[2]))
    y_pred = dtr.predict(Matrix.from_numpy(data[1]))
    print("DecisionTree regression MSE:", mse(Matrix.from_numpy(data[3]), y_pred))
    print("DecisionTree regression Accuracy:", r2_score(Matrix.from_numpy(data[3]), y_pred))
