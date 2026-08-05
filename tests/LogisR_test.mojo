from mojmelo.LogisticRegression import LogisticRegression
from mojmelo.utils.Matrix import Matrix
from mojmelo.preprocessing import train_test_split, GridSearchCV
from mojmelo.utils.utils import accuracy_score
from std.python import Python
import std.os as os

def main() raises:
    var lr_test = Python.import_module("load_breast_cancer")
    var data = lr_test.get_data() # X, y
    var X = Matrix.from_numpy(data[0])
    var y = Matrix.from_numpy(data[1]).T()
    var params = Dict[String, List[String]]()
    params['learning_rate'] = ['0.001', '0.01', '0.1']
    params['n_iters'] = ['100', '500', '1000']
    params['method'] = ['gradient', 'newton']
    params['tol'] = ['0.001', '0.01', '0.1']
    params['reg_alpha'] = ['0.001', '0.005', '0.01']
    var best_params = GridSearchCV[LogisticRegression](X, y, params, accuracy_score, cv=4, n_jobs=-1)[0].copy()
    print('tuned parameters: ', best_params)
    var X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    var lr = LogisticRegression(best_params)
    lr.fit(X_train, y_train)
    lr.save('lgr')
    lr = LogisticRegression.load('lgr')
    var y_pred = lr.predict(X_test)
    print("LR classification accuracy:", accuracy_score(y_test, y_pred))
    os.remove('lgr.mjml')
