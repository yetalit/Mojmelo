from mojmelo.SVM import SVM_Primal
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.preprocessing import train_test_split
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    svmp_test = Python.import_module("SVMP_test")
    data = svmp_test.get_data() # X, y
    X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[1]).T(), test_size=0.2, random_state=1234)
    svmp = SVM_Primal(class_zero = True)
    svmp.fit(X_train, y_train)
    y_pred = svmp.predict(X_test)
    print("SVM_Primal classification accuracy:", accuracy_score(y_test, y_pred))
    svmp_test.test(X_train.to_numpy(), y_train.T().to_numpy(), svmp.weights.to_numpy(), svmp.bias)
