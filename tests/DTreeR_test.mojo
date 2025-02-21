from mojmelo.DecisionTree import DecisionTree
from mojmelo.utils.Matrix import Matrix
from mojmelo.preprocessing import train_test_split
from mojmelo.utils.utils import mse, r2_score
from python import Python

def main():
    dtr_test = Python.import_module("load_boston")
    data = dtr_test.get_data() # X, y
    X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[1]), test_size=0.2, random_state=1234)
    dtr = DecisionTree(criterion='mse')
    dtr.fit(X_train, y_train)
    y_pred = dtr.predict(X_test)
    print("DecisionTree regression MSE:", mse(y_test, y_pred))
    print("DecisionTree regression Accuracy:", r2_score(y_test, y_pred))
