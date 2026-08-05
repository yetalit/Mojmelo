from mojmelo.DBSCAN import DBSCAN
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import ids_to_numpy
from std.python import Python

def main() raises:
    var db_test = Python.import_module("DBSCAN_test")
    var data = db_test.get_data() # X
    var db = DBSCAN(eps=0.5, min_samples=10)
    var db_y = db.fit_predict(Matrix.from_numpy(data))
    db_test.test(data, ids_to_numpy(db_y))
