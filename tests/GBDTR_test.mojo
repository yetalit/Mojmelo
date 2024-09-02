from mojmelo.GBDT import GBDT
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.preprocessing import train_test_split
from mojmelo.utils.utils import mse, r2_score
from python import Python

def main():
    Python.add_to_path(".")
    gbdtr_test = Python.import_module("load_boston")
    data = gbdtr_test.get_data() # X, y
    X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[1]), test_size=0.2, random_state=1234)
    gbdtr = GBDT(criterion='mse', n_trees = 50, max_depth = 6, learning_rate = 0.3, reg_lambda = 0.5, gamma = 0.1)
    gbdtr.fit(X_train, y_train)
    y_pred = gbdtr.predict(X_test)
    print("GBDT regression MSE:", mse(y_test, y_pred))
    print("GBDT regression Accuracy:", r2_score(y_test, y_pred))
