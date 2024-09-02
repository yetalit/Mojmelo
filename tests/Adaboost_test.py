from sklearn import datasets

def get_data():
    data = datasets.load_breast_cancer()
    y = data.target

    y[y == 0] = -1
    return[data.data, y]
