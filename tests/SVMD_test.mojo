from mojmelo.SVM import SVM_Dual
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.preprocessing import train_test_split
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    svmd_test = Python.import_module("SVMD_test")
    data = svmd_test.get_data() # X, y
    X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[1]).T(), test_size=0.2, random_state=1234)
    svmd = SVM_Dual(kernel = 'rbf')
    svmd.fit(X_train, y_train, class_zero = True)
    y_pred = svmd.predict(X_test)
    print("SVM_Primal classification accuracy:", accuracy_score(y_test, y_pred, zero_to_negone = True))
    svmd_test.test(X_train.to_numpy(), svmd.y.T().to_numpy(), svmd.alpha.T().to_numpy(), svmd.sigma, svmd.bias)
