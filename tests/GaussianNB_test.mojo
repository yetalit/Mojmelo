from mojmelo.NaiveBayes import GaussianNB
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    Python.add_to_path(".")
    gnb_test = Python.import_module("GaussianNB_test")
    data = gnb_test.get_data() # X_train, X_test, y_train, y_test
    gnb = GaussianNB()
    gnb.fit(Matrix.from_numpy(data[0]), data[2])
    y_pred = gnb.predict(Matrix.from_numpy(data[1]))
    print("GaussianNB classification accuracy:", accuracy_score(data[3], y_pred))
