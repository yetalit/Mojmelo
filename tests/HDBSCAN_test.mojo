from mojmelo.HDBSCAN import HDBSCAN
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import ids_to_numpy
from std.python import Python

def main() raises:
    var db_test = Python.import_module("DBSCAN_test")
    var data = db_test.get_data() # X
    var hdb = HDBSCAN(search_depth = 30)
    var hdb_y = hdb.fit_predict(Matrix.from_numpy(data))
    db_test.test(data, ids_to_numpy(hdb_y))
