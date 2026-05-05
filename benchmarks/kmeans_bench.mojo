from mojmelo.KMeans import KMeans
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import ids_to_numpy
from std.python import Python
import std.random as random
from std.benchmark import keep
import std.time as time

def main() raises:
    random.seed()
    GLOBAL_SEED = random.random_si64(0, 1_000_000)

    kmeans_bench = Python.import_module("kmeans_bench")
    data = Matrix.from_numpy(kmeans_bench.prepare_data(GLOBAL_SEED)[0]) # X

    WARMUP = 2
    RUNS = 5
    labels = List[Int]()
    # warm-up
    for _ in range(WARMUP):
        kmeans = KMeans(k=5, max_iters=100)
        labels = kmeans.fit_predict(data)

    var times: List[Float64] = []

    # timed runs
    for _ in range(RUNS):
        kmeans = KMeans(k=5, max_iters=100)

        t0 = time.perf_counter()
        keep(kmeans.fit(data))
        t1 = time.perf_counter()

        times.append(t1 - t0)

    fit_sum = 0.0

    for i in range(RUNS):
        fit_sum += times[i]

    fit_mean = fit_sum / Float64(RUNS)

    fit_var = 0.0

    for i in range(RUNS):
        fit_var += (times[i] - fit_mean) ** 2

    fit_std = (fit_var / Float64(RUNS)) ** 0.5

    kmeans_bench.run_benchmark(GLOBAL_SEED, fit_mean, fit_std, ids_to_numpy(labels))
