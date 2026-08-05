from mojmelo.NaiveBayes import MultinomialNB
from mojmelo.utils.Matrix import Matrix
from mojmelo.preprocessing import train_test_split, LabelEncoder
from mojmelo.utils.utils import accuracy_score
from std.python import Python
import std.os as os

def main() raises:
    var mnb_test = Python.import_module("MultinomialNB_test")
    var data = mnb_test.get_data() # X, y
    var le = LabelEncoder()
    var X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]), le.fit_transform(data[1]), test_size=0.2, random_state=42)
    var mnb = MultinomialNB(alpha = 1)
    mnb.fit(X_train, y_train)
    mnb.save('mnb')
    mnb = MultinomialNB.load('mnb')
    var y_pred = mnb.predict(X_test)
    print("MultinomialNB classification accuracy:", accuracy_score(y_test, y_pred))
    os.remove('mnb.mjml')
