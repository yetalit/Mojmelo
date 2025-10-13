import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

def get_data():
    X, y = make_moons(n_samples=200, noise=.05, random_state=0)

    # create grid to evaluate model
    xx = np.linspace(-2.5, 2.5, 30)
    yy = np.linspace(-2.5, 2.5, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    return [X, y, xy]

# defining a function to plot decision boundary according to the svm model
def test(X, y, support_vectors, dec_values):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # Set constant axis limits
    plt.xlim((-2.5, 2.5))
    plt.ylim((-2.5, 2.5))

    xx = np.linspace(-2.5, 2.5, 30)
    yy = np.linspace(-2.5, 2.5, 30)
    YY, XX = np.meshgrid(yy, xx)
    
    Z = dec_values.reshape(XX.shape)

    # plot decision boundary and margins
    ax = plt.gca()
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
            alpha=0.5, linestyles=['--', '-', '--'])

    # plot support vectors
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')
    plt.title('SVM plot')
    plt.show()
