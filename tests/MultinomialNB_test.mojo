from mojmelo.NaiveBayes import MultinomialNB
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.preprocessing import train_test_split
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    Python.add_to_path(".")
    mnb_test = Python.import_module("MultinomialNB_test")
    data = mnb_test.get_data() # X, y
    X_train, X_test, y_ = train_test_split(Matrix.from_numpy(data[0]), data[1], test_size=0.2, random_state=42)
    mnb = MultinomialNB(alpha = 1)
    mnb.fit(X_train, y_.train)
    y_pred = mnb.predict(X_test)
    print("MultinomialNB classification accuracy:", accuracy_score(y_.test, y_pred))
