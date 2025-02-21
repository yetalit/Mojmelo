from mojmelo.GBDT import GBDT
from mojmelo.utils.Matrix import Matrix
from mojmelo.preprocessing import train_test_split
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    gbdtc_test = Python.import_module("load_breast_cancer")
    data = gbdtc_test.get_data() # X, y
    X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[1]).T(), test_size=0.2, random_state=1234)
    gbdtc = GBDT(criterion='log', n_trees = 5, max_depth = 6, learning_rate = 0.3, reg_lambda = 0.2, gamma = 0.01)
    gbdtc.fit(X_train, y_train)
    y_pred = gbdtc.predict(X_test)
    print("GBDT classification accuracy:", accuracy_score(y_test, y_pred))
