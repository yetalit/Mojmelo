from mojmelo.DecisionTree import DecisionTree
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import mse as mean_squared_error
from std.python import Python
import std.random as random
from std.benchmark import keep
import std.time as time

def main() raises:
    random.seed()
    GLOBAL_SEED = random.random_si64(0, 1_000_000)

    dtr_bench = Python.import_module("dtr_bench")
    data = dtr_bench.prepare_data(GLOBAL_SEED) # X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = Matrix.from_numpy(data[0]), Matrix.from_numpy(data[1]), Matrix.from_numpy(data[2]).T(), Matrix.from_numpy(data[3]).T()

    WARMUP = 2
    RUNS = 5
    mse = 0.0
    # warm-up
    for _ in range(WARMUP):
        dtr = DecisionTree(criterion='mse', max_depth=10)
        dtr.fit(X_train, y_train)
        y_pred = dtr.predict(X_test)
        mse = mean_squared_error(y_test, y_pred).cast[DType.float64]()

    var fit_times: List[Float64] = []
    var pred_times: List[Float64] = []

    # timed runs
    for _ in range(RUNS):
        dtr = DecisionTree(criterion='mse', max_depth=10)

        t0 = time.perf_counter()
        dtr.fit(X_train, y_train)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        keep(dtr.predict(X_test))
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

    dtr_bench.run_benchmark(GLOBAL_SEED, fit_mean, fit_std, pred_mean, pred_std, mse)
