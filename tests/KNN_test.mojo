from mojmelo.KNN import KNN
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.preprocessing import train_test_split
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    Python.add_to_path(".")
    knn_test = Python.import_module("load_iris")
    data = knn_test.get_data() # X, y
    X_train, X_test, y_ = train_test_split(Matrix.from_numpy(data[0]), data[1], test_size=0.2, random_state=1234)
    knn = KNN(k = 3)
    knn.fit(X_train, y_.train)
    y_pred = knn.predict(X_test)
    print("KNN classification accuracy:", accuracy_score(y_.test, y_pred))
