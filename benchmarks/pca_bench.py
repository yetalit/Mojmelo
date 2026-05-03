import numpy as np
import time
from sklearn.decomposition import PCA


def prepare_data(seed, n_samples=100000, n_features=100):
    rng = np.random.RandomState(seed)

    # create correlated data
    A = rng.randn(n_features, n_features)
    cov = A @ A.T

    X = rng.multivariate_normal(
        mean=np.zeros(n_features),
        cov=cov,
        size=n_samples
    ).astype(np.float32)

    # center
    X -= X.mean(axis=0, keepdims=True)

    return X


# -----------------------
# CORE BENCHMARK
# -----------------------
def benchmark_model(X, warmup=2, runs=5):
    explained_var = 0.0

    # warm-up
    for _ in range(warmup):
        m = sklearn_model()
        m.fit(X)
        m.transform(X)
        explained_var = np.sum(m.explained_variance_ratio_)

    fit_times = []
    transform_times = []

    for _ in range(runs):
        m = sklearn_model()

        t0 = time.perf_counter()
        m.fit(X)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        _ = m.transform(X)
        t3 = time.perf_counter()

        fit_times.append(t1 - t0)
        transform_times.append(t3 - t2)

    return {
        "fit_mean": np.mean(fit_times),
        "fit_std": np.std(fit_times),
        "trans_mean": np.mean(transform_times),
        "trans_std": np.std(transform_times),
        "explained_variance": explained_var,
    }


def sklearn_model():
    return PCA(n_components=20, svd_solver='full', random_state=42)


# -----------------------
# HIGH-LEVEL RUNNER
# -----------------------
def run_benchmark(seed, fit_mean, fit_std, trans_mean, trans_std, explained_var):
    X = prepare_data(seed)

    sk = benchmark_model(X)

    print("| Model       | Fit Time (s)    | Transform Time (s) | Explained Var |")
    print("|-------------|-----------------|--------------------|---------------|")

    print(
        f"| sklearn PCA | {sk['fit_mean']:.4f} ± {sk['fit_std']:.4f} "
        f"| {sk['trans_mean']:.4f} ± {sk['trans_std']:.4f}    "
        f"| {sk['explained_variance']:.4f}        |"
    )

    print(
        f"| mojmelo PCA | {fit_mean:.4f} ± {fit_std:.4f} "
        f"| {trans_mean:.4f} ± {trans_std:.4f}    "
        f"| {explained_var:.4f}        |"
    )
