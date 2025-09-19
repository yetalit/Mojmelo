Mojo struct

# `KNN`

```mojo
@memory_only
struct KNN
```

Classifier implementing the k-nearest neighbors vote.

## Fields

- **k** (`Int`): Number of neighbors to use.
- **metric** (`String`): Metric to use for distance computation: Euclidean -> 'euc'; Manhattan -> 'man'.
- **n_jobs** (`Int`): The number of parallel jobs to run for neighbors search. `-1` means using all processors.
- **kdtree** (`KDTree`)
- **y_train** (`List[String]`)

## Implemented traits

`AnyType`, `CVP`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, k: Int = 3, metric: String = "euc", n_jobs: Int = 0)
```

**Args:**

- **k** (`Int`)
- **metric** (`String`)
- **n_jobs** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn __init__(out self, params: Dict[String, String])
```

**Args:**

- **params** (`Dict`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `fit`

```mojo
fn fit(mut self, X: Matrix, y: PythonObject)
```

Fit the k-nearest neighbors classifier from the training dataset.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)
- **y** (`PythonObject`)

**Raises:**

### `predict`

```mojo
fn predict(self, X: Matrix) -> List[String]
```

Predict the class labels for the provided data.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`List`: Class labels for each data sample.

**Raises:**


