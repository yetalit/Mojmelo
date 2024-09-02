import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

def get_data():
    X, y = make_blobs(n_samples=200, centers=2,random_state=0, cluster_std=0.60)

    return [X, np.where(y <= 0, -1, 1)]

def test(X, y, w, b):

    def get_hyperplane(x, w, b):
        return (-w[0] * x + b) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X[:,0], X[:,1], marker='o',c=y)

    x0_1 = np.amin(X[:,0])
    x0_2 = np.amax(X[:,0])

    x1_1 = get_hyperplane(x0_1, w, b)
    x1_2 = get_hyperplane(x0_2, w, b)
    x1_1_m = x1_1 - 1
    x1_2_m = x1_2 - 1
    x1_1_p = x1_1 + 1
    x1_2_p = x1_2 + 1

    ax.plot([x0_1, x0_2],[x1_1, x1_2], 'y--')
    ax.plot([x0_1, x0_2],[x1_1_m, x1_2_m], 'k')
    ax.plot([x0_1, x0_2],[x1_1_p, x1_2_p], 'k')

    x1_min = np.amin(X[:,1])
    x1_max = np.amax(X[:,1])
    ax.set_ylim([x1_min-3,x1_max+3])

    plt.title('SVM Primal')
    plt.show()
