import matplotlib.pyplot as plt
from sklearn import datasets

def get_data():
    data = datasets.load_iris()

    return[data.data, data.target]

def test(X_projected, y):
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()
