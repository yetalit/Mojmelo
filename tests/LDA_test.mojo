from mojmelo.LDA import LDA   
from mojmelo.utils.Matrix import Matrix 
from python import Python

def main():
    lda_test = Python.import_module("LDA_test")
    data = lda_test.get_data() # X, y
    # Project the data onto the 2 primary linear discriminants
    lda = LDA(2)
    lda.fit(Matrix.from_numpy(data[0]), data[1])
    X_projected = lda.transform(Matrix.from_numpy(data[0]))
    print("Shape of X:", data[0].shape)
    print("Shape of transformed X:", '(' + String(X_projected.height) + ', ' + String(X_projected.width) + ')')
    lda_test.test(X_projected.to_numpy(), data[1])
