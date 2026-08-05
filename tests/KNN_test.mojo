from mojmelo.KNN import KNN
from mojmelo.utils.Matrix import Matrix
from mojmelo.preprocessing import train_test_split, GridSearchCV, LabelEncoder
from mojmelo.utils.utils import accuracy_score
from std.python import Python
import std.os as os

def main() raises:
    var knn_test = Python.import_module("load_iris")
    var data = knn_test.get_data() # X, y
    var le = LabelEncoder()
    var X = Matrix.from_numpy(data[0])
    var y = le.fit_transform(data[1])
    var params = Dict[String, List[String]]()
    params['k'] = ['3', '5', '7']
    var best_params = GridSearchCV[KNN](X, y, params, accuracy_score, cv=4, n_jobs=-1)[0].copy()
    print('tuned parameters: ', best_params)
    var X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    var knn = KNN(best_params)
    knn.fit(X_train, y_train)
    knn.save('knn')
    knn = KNN.load('knn')
    var y_pred = knn.predict(X_test)
    print("KNN classification accuracy:", accuracy_score(y_test, y_pred))
    os.remove('knn.mjml')
