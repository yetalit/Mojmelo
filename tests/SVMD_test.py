import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

def get_data():
    X, y = make_moons(n_samples=200, noise=.05)

    return [X, y]

def test(X, y, alpha, sigma, b):
    def gaussian_kernal(sigma, _X,Z):
        return np.exp(-(1 / sigma ** 2) * np.linalg.norm(_X[:, np.newaxis] - Z[np.newaxis, :], axis=2) ** 2) #e ^-(1/ σ2) ||X-y|| ^2

    def decision_function(_X):
        return (alpha * y).dot(gaussian_kernal(sigma, X, _X)) + b
    
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter', alpha=.5)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, levels=[-1, 0, 1],linestyles=['--', '-', '--'])
    plt.title('SVM Dual (RBF kernel)')
    plt.show()
