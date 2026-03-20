Mojo struct

# `KMeans`

```mojo
@memory_only
struct KMeans
```

K-Means clustering.

## Aliases

- `MODEL_ID = 5`

## Fields

- **k** (`Int`): The number of clusters to form as well as the number of centroids to generate.
- **init** (`String`): Method for initialization -> 'kmeans++', 'random'.
- **n_centroid_init** (`Int`): The number of candidate centroids to be initialized.
- **max_iters** (`Int`): Maximum number of iterations of the k-means algorithm for a single run.
- **converge** (`String`): The converge method: Change in centroids <= tol -> 'centroid'; Change in inertia <= tol -> 'inertia'; Exact change in labels -> 'label'.
- **tol** (`Float32`): Relative tolerance value.
- **labels** (`List[Int]`)
- **centroids_** (`Matrix`)
- **inertia** (`Float32`): Sum of squared distances of samples to their closest cluster center.
- **X_mean** (`Matrix`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
def __init__(out self, k: Int = 5, init: String = "kmeans++", n_centroid_init: Int = 1, max_iters: Int = 100, converge: String = "centroid", tol: Float32 = 1.0E-4, random_state: Int = 0)
```

**Args:**

- **k** (`Int`)
- **init** (`String`)
- **n_centroid_init** (`Int`)
- **max_iters** (`Int`)
- **converge** (`String`)
- **tol** (`Float32`)
- **random_state** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

### `fit`

```mojo
def fit(mut self, X: Matrix)
```

Compute cluster centers and cluster index for each sample.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Raises:**

### `predict`

```mojo
def predict(self, X: Matrix) -> List[Int]
```

Predict cluster index for each sample.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`List`: List of cluster indices.

**Raises:**

### `fit_predict`

```mojo
def fit_predict(mut self, X: Matrix) -> List[Int]
```

Compute cluster centers and predict cluster index for each sample.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`List`: List of cluster indices.

**Raises:**

### `save`

```mojo
def save(self, path: String)
```

Save model data necessary for prediction to the specified path.

**Args:**

- **self** (`Self`)
- **path** (`String`)

**Raises:**

### `load`

```mojo
@staticmethod
def load(path: String) -> Self
```

Load a saved model from the specified path for prediction.

**Args:**

- **path** (`String`)

**Returns:**

`Self`

**Raises:**

### `centroids`

```mojo
def centroids(self) -> Matrix
```

**Args:**

- **self** (`Self`)

**Returns:**

`Matrix`

**Raises:**


