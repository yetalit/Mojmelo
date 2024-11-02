from mojmelo.LogisticRegression import LogisticRegression
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.preprocessing import train_test_split, GridSearchCV
from mojmelo.utils.utils import accuracy_score
from collections import Dict
from python import Python

def main():
    lr_test = Python.import_module("load_breast_cancer")
    data = lr_test.get_data() # X, y
    X = Matrix.from_numpy(data[0])
    y = Matrix.from_numpy(data[1]).T()
    params = Dict[String, List[String]]()
    params['learning_rate'] = List[String]('0.001', '0.01', '0.1')
    params['n_iters'] = List[String]('100', '500', '1000')
    params['method'] = List[String]('gradient', 'newton')
    params['tol'] = List[String]('0.001', '0.01', '0.1')
    params['reg_alpha'] = List[String]('0.001', '0.005', '0.01')
    best_params, _ = GridSearchCV[LogisticRegression](X, y, params, accuracy_score, cv=4)
    print('tuned parameters: ', best_params.__str__())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    lr = LogisticRegression(best_params)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("LR classification accuracy:", accuracy_score(y_test, y_pred))
