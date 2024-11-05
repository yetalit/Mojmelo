from mojmelo.DBSCAN import DBSCAN
from mojmelo.utils.Matrix import Matrix
from python import Python

def main():
    db_test = Python.import_module("DBSCAN_test")
    data = db_test.get_data() # X
    db = DBSCAN(eps=0.3, min_samples=10)
    db_y = db.predict(Matrix.from_numpy(data))
    db_test.test(data, db_y.to_numpy())
