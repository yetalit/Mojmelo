import numpy as np
import time
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score


def prepare_data(seed, n_samples=30000, n_features=15):
    rng = np.random.RandomState(seed)

    # --- 1. MIXED DENSITY BLOBS ---
    centers = 5
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        cluster_std=[0.3, 0.8, 1.5, 0.5, 1.2],  # varying density
        random_state=seed
    )

    # --- 2. ADD OVERLAP ---
    X += rng.normal(0, 0.3, size=X.shape)

    # --- 3. NON-LINEAR DISTORTION ---
    X = X + 0.3 * np.sin(X) + 0.2 * np.cos(0.5 * X)

    # --- 4. CORRELATED FEATURES ---
    A = rng.normal(scale=0.8, size=(n_features, n_features))
    X = X @ A

    # --- 5. MORE NOISE DIMENSIONS ---
    noise_dims = int(0.3 * n_features)
    if noise_dims > 0:
        noise = rng.normal(0, 2.0, size=(X.shape[0], noise_dims))
        X[:, :noise_dims] += noise

    # --- 6. SIGNIFICANT OUTLIERS ---
    n_outliers = int(0.05 * n_samples)
    outliers = rng.uniform(low=-20, high=20, size=(n_outliers, n_features))
    X[:n_outliers] = outliers
    y[:n_outliers] = -1

    return X.astype(np.float32), y


def sklearn_model():
    return DBSCAN(eps=10, min_samples=10, algorithm='kd_tree', n_jobs=-1)


# -----------------------
# CORE BENCHMARK
# -----------------------
def benchmark_model(X, warmup=2, runs=5):
    labels = None

    # warm-up
    for _ in range(warmup):
        m = sklearn_model()
        labels = m.fit_predict(X)

    times = []

    # timed runs
    for _ in range(runs):
        m = sklearn_model()

        t0 = time.perf_counter()
        m.fit(X)
        t1 = time.perf_counter()

        times.append(t1 - t0)

    mean = sum(times) / runs
    var = sum((t - mean) ** 2 for t in times) / runs
    std = var ** 0.5

    return {
        "mean": mean,
        "std": std,
        "labels": labels,
    }


# -----------------------
# RUNNER
# -----------------------
def run_benchmark(seed, mean, std, labels):
    X, y_true = prepare_data(seed)

    sk = benchmark_model(X)

    ari_vs_sklearn = adjusted_rand_score(sk["labels"], labels)
    sk_ari_vs_truth = adjusted_rand_score(y_true, sk["labels"])
    ari_vs_truth = adjusted_rand_score(y_true, labels)

    print("| Model       | Fit Time (s)    | ARI vs sklearn | ARI vs truth |")
    print("|-------------|-----------------|----------------|--------------|")

    print(
        f"| sklearn DBS | {sk['mean']:.4f} ± {sk['std']:.4f} "
        f"| {'-':^14} "
        f"| {sk_ari_vs_truth:.4f}       |"
    )

    print(
        f"| mojmelo DBS | {mean:.4f} ± {std:.4f} "
        f"| {ari_vs_sklearn:.4f}         "
        f"| {ari_vs_truth:.4f}       |"
    )
