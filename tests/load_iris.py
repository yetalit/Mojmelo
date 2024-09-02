from sklearn import datasets

def get_data():
    iris = datasets.load_iris()

    return [iris.data, iris.target]
