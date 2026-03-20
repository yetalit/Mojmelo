from mojmelo.PCA import PCA   
from mojmelo.utils.Matrix import Matrix 
from std.python import Python
import std.os as os

def main() raises:
    pca_test = Python.import_module("PCA_test")
    data = pca_test.get_data() # X, y
    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    pca.fit(Matrix.from_numpy(data[0]))
    pca.save('pca')
    pca = PCA.load('pca')
    X_projected = pca.transform(Matrix.from_numpy(data[0]))
    print("Shape of X:", data[0].shape)
    print("Shape of transformed X:", '(' + String(X_projected.height) + ', ' + String(X_projected.width) + ')')
    pca_test.test(X_projected.to_numpy(), data[1])
    os.remove('pca.mjml')
