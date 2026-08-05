from mojmelo.HDBSCAN import HDBSCAN
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import ids_to_numpy
from std.python import Python
import std.random as random
from std.benchmark import keep
import std.time as time

def main() raises:
    random.seed()
    var GLOBAL_SEED = random.random_si64(0, 1_000_000)

    var hdbs_bench = Python.import_module("hdbs_bench")
    var data = Matrix.from_numpy(hdbs_bench.prepare_data(GLOBAL_SEED)[0]) # X

    var WARMUP = 2
    var RUNS = 5
    var labels = List[Scalar[DType.int]]()
    # warm-up
    for _ in range(WARMUP):
        var hdbs = HDBSCAN()
        labels = hdbs.fit_predict(data)

    var times: List[Float64] = []

    # timed runs
    for _ in range(RUNS):
        var hdbs = HDBSCAN()

        var t0 = time.perf_counter()
        keep(hdbs.fit(data))
        var t1 = time.perf_counter()

        times.append(t1 - t0)

    var fit_sum = 0.0

    for i in range(RUNS):
        fit_sum += times[i]

    var fit_mean = fit_sum / Float64(RUNS)

    var fit_var = 0.0

    for i in range(RUNS):
        fit_var += (times[i] - fit_mean) ** 2

    var fit_std = (fit_var / Float64(RUNS)) ** 0.5

    hdbs_bench.run_benchmark(GLOBAL_SEED, fit_mean, fit_std, ids_to_numpy(labels))
