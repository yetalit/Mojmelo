from mojmelo.LogisticRegression import LogisticRegression
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.preprocessing import train_test_split
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    Python.add_to_path(".")
    lr_test = Python.import_module("load_breast_cancer")
    data = lr_test.get_data() # X, y
    X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[1]).T(), test_size=0.2, random_state=1234)
    lr = LogisticRegression(learning_rate=0.001, n_iters=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("LR classification accuracy:", accuracy_score(y_test, y_pred))
