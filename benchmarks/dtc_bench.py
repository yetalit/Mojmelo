import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def prepare_data(seed, n_samples=30000, n_features=50, test_ratio=0.2):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        random_state=seed
    )

    split = int((1 - test_ratio) * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return (
        X_train.astype(np.float32),
        X_test.astype(np.float32),
        y_train,
        y_test,
    )


# -----------------------
# CORE BENCHMARK
# -----------------------
def benchmark_model(X_train, y_train, X_test, y_test, warmup=2, runs=5):
    accuracy = 0.0
    # warm-up
    for _ in range(warmup):
        m = sklearn_model()
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

    fit_times = []
    pred_times = []

    # timed runs
    for _ in range(runs):
        m = sklearn_model()

        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        _ = m.predict(X_test)
        t3 = time.perf_counter()

        fit_times.append(t1 - t0)
        pred_times.append(t3 - t2)

    fit_sum = 0.0
    pred_sum = 0.0

    for i in range(runs):
        fit_sum += fit_times[i]
        pred_sum += pred_times[i]

    fit_mean = fit_sum / runs
    pred_mean = pred_sum / runs

    fit_var = 0.0
    pred_var = 0.0

    for i in range(runs):
        fit_var += (fit_times[i] - fit_mean) ** 2
        pred_var += (pred_times[i] - pred_mean) ** 2

    fit_std = (fit_var / runs) ** 0.5
    pred_std = (pred_var / runs) ** 0.5

    return {
        "fit_mean": fit_mean,
        "fit_std": fit_std,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "accuracy": accuracy,
    }


def sklearn_model():
    return DecisionTreeClassifier(max_depth=10, random_state=42)


# -----------------------
# HIGH-LEVEL RUNNER
# -----------------------
def run_benchmark(seed, fit_mean, fit_std, pred_mean, pred_std, accuracy):
    X_train, X_test, y_train, y_test = prepare_data(seed)

    sk = benchmark_model(
        X_train, y_train, X_test, y_test
    )

    print("| Model       | Fit Time (s)    | Predict Time (s) | Accuracy |")
    print("|-------------|-----------------|------------------|----------|")

    print(
        f"| sklearn DTC | {sk['fit_mean']:.4f} ± {sk['fit_std']:.4f} "
        f"| {sk['pred_mean']:.4f} ± {sk['pred_std']:.4f}  "
        f"| {sk['accuracy']:.4f}   |"
    )

    print(
        f"| mojmelo DTC | {fit_mean:.4f} ± {fit_std:.4f} "
        f"| {pred_mean:.4f} ± {pred_std:.4f}  "
        f"| {accuracy:.4f}   |"
    )
