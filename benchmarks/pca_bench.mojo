from mojmelo.PCA import PCA
from mojmelo.utils.Matrix import Matrix
from std.python import Python
import std.random as random
from std.benchmark import keep
import std.time as time

def main() raises:
    random.seed()
    GLOBAL_SEED = random.random_si64(0, 1_000_000)

    pca_bench = Python.import_module("pca_bench")
    data = Matrix.from_numpy(pca_bench.prepare_data(GLOBAL_SEED)) # X

    WARMUP = 2
    RUNS = 5
    explained_var = 0.0
    # warm-up
    for _ in range(WARMUP):
        pca = PCA(n_components=20)
        pca.fit(data)
        keep(pca.transform(data))
        explained_var = pca.explained_variance_ratio.sum().cast[DType.float64]()

    var fit_times: List[Float64] = []
    var pred_times: List[Float64] = []

    # timed runs
    for _ in range(RUNS):
        pca = PCA(n_components=20)

        t0 = time.perf_counter()
        pca.fit(data)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        keep(pca.transform(data))
        t3 = time.perf_counter()

        fit_times.append(t1 - t0)
        pred_times.append(t3 - t2)

    fit_sum = 0.0
    pred_sum = 0.0

    for i in range(RUNS):
        fit_sum += fit_times[i]
        pred_sum += pred_times[i]

    fit_mean = fit_sum / Float64(RUNS)
    pred_mean = pred_sum / Float64(RUNS)

    fit_var = 0.0
    pred_var = 0.0

    for i in range(RUNS):
        fit_var += (fit_times[i] - fit_mean) ** 2
        pred_var += (pred_times[i] - pred_mean) ** 2

    fit_std = (fit_var / Float64(RUNS)) ** 0.5
    pred_std = (pred_var / Float64(RUNS)) ** 0.5

    pca_bench.run_benchmark(GLOBAL_SEED, fit_mean, fit_std, pred_mean, pred_std, explained_var)
