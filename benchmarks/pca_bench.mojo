from mojmelo.PCA import PCA
from mojmelo.utils.Matrix import Matrix
from std.python import Python
import std.random as random
from std.benchmark import keep
import std.time as time

def main() raises:
    random.seed()
    var GLOBAL_SEED = random.random_si64(0, 1_000_000)

    var pca_bench = Python.import_module("pca_bench")
    var data = Matrix.from_numpy(pca_bench.prepare_data(GLOBAL_SEED)) # X

    var WARMUP = 2
    var RUNS = 5
    var explained_var = 0.0
    # warm-up
    for _ in range(WARMUP):
        var pca = PCA(n_components=20)
        pca.fit(data)
        keep(pca.transform(data))
        explained_var = pca.explained_variance_ratio.sum().cast[DType.float64]()

    var fit_times: List[Float64] = []
    var pred_times: List[Float64] = []

    # timed runs
    for _ in range(RUNS):
        var pca = PCA(n_components=20)

        var t0 = time.perf_counter()
        pca.fit(data)
        var t1 = time.perf_counter()

        var t2 = time.perf_counter()
        keep(pca.transform(data))
        var t3 = time.perf_counter()

        fit_times.append(t1 - t0)
        pred_times.append(t3 - t2)

    var fit_sum = 0.0
    var pred_sum = 0.0

    for i in range(RUNS):
        fit_sum += fit_times[i]
        pred_sum += pred_times[i]

    var fit_mean = fit_sum / Float64(RUNS)
    var pred_mean = pred_sum / Float64(RUNS)

    var fit_var = 0.0
    var pred_var = 0.0

    for i in range(RUNS):
        fit_var += (fit_times[i] - fit_mean) ** 2
        pred_var += (pred_times[i] - pred_mean) ** 2

    var fit_std = (fit_var / Float64(RUNS)) ** 0.5
    var pred_std = (pred_var / Float64(RUNS)) ** 0.5

    pca_bench.run_benchmark(GLOBAL_SEED, fit_mean, fit_std, pred_mean, pred_std, explained_var)
