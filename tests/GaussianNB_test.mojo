from mojmelo.NaiveBayes import GaussianNB
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.preprocessing import train_test_split
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    gnb_test = Python.import_module("GaussianNB_test")
    data = gnb_test.get_data() # X, y
    X_train, X_test, y_ = train_test_split(Matrix.from_numpy(data[0]), data[1], test_size=0.2, random_state=123)
    gnb = GaussianNB()
    gnb.fit(X_train, y_.train)
    y_pred = gnb.predict(X_test)
    print("GaussianNB classification accuracy:", accuracy_score(y_.test, y_pred))
