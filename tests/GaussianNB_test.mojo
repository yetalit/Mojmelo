from mojmelo.NaiveBayes import GaussianNB
from mojmelo.utils.Matrix import Matrix
from mojmelo.preprocessing import train_test_split, LabelEncoder
from mojmelo.utils.utils import accuracy_score
from std.python import Python
import std.os as os

def main() raises:
    gnb_test = Python.import_module("GaussianNB_test")
    data = gnb_test.get_data() # X, y
    le = LabelEncoder()
    X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]), le.fit_transform(data[1]), test_size=0.2, random_state=123)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb.save('gnb')
    gnb = GaussianNB.load('gnb')
    y_pred = gnb.predict(X_test)
    print("GaussianNB classification accuracy:", accuracy_score(y_test, y_pred))
    os.remove('gnb.mjml')
