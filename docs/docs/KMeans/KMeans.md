Mojo struct

# `KMeans`

```mojo
@memory_only
struct KMeans
```

K-Means clustering.

## Aliases

- `__del__is_trivial = False`

## Fields

- **K** (`Int`): The number of clusters to form as well as the number of centroids to generate.
- **init** (`String`): Method for initialization -> 'kmeans++', 'random'.
- **max_iters** (`Int`): Maximum number of iterations of the k-means algorithm for a single run.
- **converge** (`String`): The converge method: Change in centroids <= tol -> 'centroid'; Change in inertia <= tol -> 'inertia'; Exact change in labels -> 'label'.
- **tol** (`Float32`): Relative tolerance value.
- **labels** (`List[Int]`)
- **centroids** (`Matrix`)
- **inertia** (`Float32`): Sum of squared distances of samples to their closest cluster center.
- **X** (`Matrix`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
fn __init__(out self, K: Int = 5, init: String = "kmeans++", max_iters: Int = 100, converge: String = "centroid", tol: Float32 = 1.0E-4, random_state: Int = 42)
```

**Args:**

- **K** (`Int`)
- **init** (`String`)
- **max_iters** (`Int`)
- **converge** (`String`)
- **tol** (`Float32`)
- **random_state** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

### `fit`

```mojo
fn fit(mut self, X: Matrix)
```

Compute cluster centers and cluster index for each sample.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Raises:**

### `fit_predict`

```mojo
fn fit_predict(mut self, X: Matrix) -> List[Int]
```

Compute cluster centers and predict cluster index for each sample.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`List`: List of cluster indices.

**Raises:**


