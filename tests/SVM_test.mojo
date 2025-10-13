from mojmelo.SVM import SVC
from mojmelo.utils.Matrix import Matrix
from mojmelo.preprocessing import train_test_split
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    svm_test = Python.import_module("SVM_test")
    data = svm_test.get_data() # X, y, xy
    X_train, X_test, y_train, y_test = train_test_split(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[1]).T(), test_size=0.2, random_state=1234)
    svm = SVC(C = 1.0, random_state=1234)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print("SVC classification accuracy:", accuracy_score(y_test, y_pred))
    xy = Matrix.from_numpy(data[2])
    dec_values_list = svm.decision_function(xy)
    dec_values = Matrix(1, len(dec_values_list))
    for idx in range(len(dec_values_list)):
        dec_values[0, idx] = dec_values_list[idx][0].cast[DType.float32]()
    svm_test.test(data[0], data[1], svm.support_vectors().to_numpy(), dec_values.to_numpy())
