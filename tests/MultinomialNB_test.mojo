from mojmelo.NaiveBayes import MultinomialNB
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    Python.add_to_path(".")
    mnb_test = Python.import_module("MultinomialNB_test")
    data = mnb_test.get_data() # X_train, X_test, y_train, y_test
    mnb = MultinomialNB(alpha = 1)
    mnb.fit(Matrix.from_numpy(data[0]), data[2])
    y_pred = mnb.predict(Matrix.from_numpy(data[1]))
    print("MultinomialNB classification accuracy:", accuracy_score(data[3], y_pred))
