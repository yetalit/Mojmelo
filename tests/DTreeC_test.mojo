from mojmelo.DecisionTree import DecisionTree
from mojmelo.utils.Matrix import Matrix
from mojmelo.preprocessing import train_test_split
from mojmelo.utils.utils import accuracy_score
from std.python import Python
import std.os as os

def main() raises:
    var dtc_test = Python.import_module("load_breast_cancer")
    var data = dtc_test.get_data() # X, y
    var X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[1]).T(), test_size=0.2, random_state=1234)
    var dtc = DecisionTree(criterion='entropy', max_depth=10)
    dtc.fit(X_train, y_train)
    dtc.save('dtc')
    dtc = DecisionTree.load('dtc')
    var y_pred = dtc.predict(X_test)
    print("DecisionTree classification accuracy:", accuracy_score(y_test, y_pred))
    os.remove('dtc.mjml')
