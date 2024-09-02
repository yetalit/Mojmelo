from sklearn import datasets

def get_data():
    data = datasets.load_breast_cancer()

    return [data.data, data.target]
