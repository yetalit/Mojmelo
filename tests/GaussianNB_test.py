from sklearn import datasets

def get_data():
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )

    return [X, y]
