from mojmelo.preprocessing import normalize, StandardScaler, MinMaxScaler
from mojmelo.utils.Matrix import Matrix

# Testing preprocessing algorithms with the samples from scikit-learn docs.
def main():
    print('--Normalize--')
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
    X = Matrix([[-2, 1, 2], [-1, 0, 1]])
    print(normalize(X, norm="l1")[0])
    print(normalize(X, norm="l2")[0])
    print('--StandardScaler--')
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    X = Matrix([[0, 0], [0, 0], [1, 1], [1, 1]])
    print(StandardScaler(X)[0])
    print('--MinMaxScaler--')
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    X = Matrix([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
    print(MinMaxScaler(X)[0])
