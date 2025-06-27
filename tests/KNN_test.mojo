from mojmelo.KNN import KNN
from mojmelo.utils.Matrix import Matrix
from mojmelo.preprocessing import train_test_split, GridSearchCV
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    knn_test = Python.import_module("load_iris")
    data = knn_test.get_data() # X, y
    X = Matrix.from_numpy(data[0])
    y = data[1]
    params = Dict[String, List[String]]()
    params['k'] = List[String]('3', '5', '7')
    best_params, _ = GridSearchCV[KNN](X, y, params, accuracy_score, cv=4)
    print('tuned parameters: ', best_params.__str__())
    X_train, X_test, y_ = train_test_split(X, y, test_size=0.2, random_state=1234)
    knn = KNN(best_params)
    knn.fit(X_train, y_.train)
    y_pred = knn.predict(X_test)
    print("KNN classification accuracy:", accuracy_score(y_.test, y_pred))
