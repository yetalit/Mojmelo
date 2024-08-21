from mojmelo.DecisionTree import DecisionTree
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    Python.add_to_path(".")
    dtc_test = Python.import_module("DTreeC_test")
    data = dtc_test.get_data() # X_train, X_test, y_train, y_test
    dtc = DecisionTree(criterion='entropy', max_depth=10)
    dtc.fit(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[2]).T())
    y_pred = dtc.predict(Matrix.from_numpy(data[1]))
    print("DecisionTree classification accuracy:", accuracy_score(data[3], y_pred))
