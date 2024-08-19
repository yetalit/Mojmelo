from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

def accuracy(y_true, y_pred):
    y_pred = y_pred.astype(int)
    return np.sum(y_true == y_pred) / len(y_true)

def get_data():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    return [X_train, X_test, y_train, y_test]
