from mojmelo.RandomForest import RandomForest
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.preprocessing import train_test_split
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    rfc_test = Python.import_module("load_breast_cancer")
    data = rfc_test.get_data() # X, y
    X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[1]).T(), test_size=0.2, random_state=1234)
    rfc = RandomForest(criterion='entropy', n_trees=5, max_depth=10)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    print("RandomForest classification accuracy:", accuracy_score(y_test, y_pred))
