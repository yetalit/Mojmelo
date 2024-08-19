from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

def get_data():
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    return [X_train, X_test, y_train, y_test]

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
