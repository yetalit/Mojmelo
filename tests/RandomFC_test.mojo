from mojmelo.RandomForest import RandomForest
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    Python.add_to_path(".")
    rfc_test = Python.import_module("RandomFC_test")
    data = rfc_test.get_data() # X_train, X_test, y_train, y_test
    rfc = RandomForest(criterion='entropy', n_trees=5, max_depth=10)
    rfc.fit(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[2]).T())
    y_pred = rfc.predict(Matrix.from_numpy(data[1]))
    print("RandomForest classification accuracy:", accuracy_score(data[3], y_pred))
