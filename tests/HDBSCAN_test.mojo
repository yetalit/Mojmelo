from mojmelo.HDBSCAN import HDBSCAN
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import ids_to_numpy
from python import Python

def main():
    db_test = Python.import_module("DBSCAN_test")
    data = db_test.get_data() # X
    db = HDBSCAN(search_deepness_coef = 3)
    db_y = db.fit_predict(Matrix.from_numpy(data))
    db_test.test(data, ids_to_numpy(db_y))
