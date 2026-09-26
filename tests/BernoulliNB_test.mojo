from mojmelo.NaiveBayes import BernoulliNB
from mojmelo.linalg.Matrix import Matrix
from mojmelo.preprocessing import train_test_split, LabelEncoder
from mojmelo.utils.utils import accuracy_score
from std.python import Python
import std.os as os

def main() raises:
    var mnb_test = Python.import_module("MultinomialNB_test")
    var data = mnb_test.get_data(True) # X, y
    var le = LabelEncoder()
    var X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]), le.fit_transform(data[1]), test_size=0.2, random_state=42)
    var bnb = BernoulliNB(alpha = 1)
    bnb.fit(X_train, y_train)
    bnb.save('bnb')
    bnb = BernoulliNB.load('bnb')
    var y_pred = bnb.predict(X_test)
    print("BernoulliNB classification accuracy:", accuracy_score(y_test, y_pred))
    os.remove('bnb.mjml')
