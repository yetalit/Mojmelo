Mojo struct

# `KNN`

```mojo
@memory_only
struct KNN
```

Classifier implementing the k-nearest neighbors vote.

## Aliases

- `__del__is_trivial = False`

## Fields

- **k** (`Int`): Number of neighbors to use.
- **metric** (`String`): Metric to use for distance computation: Euclidean -> 'euc'; Manhattan -> 'man'.
- **kdtree** (`KDTree`)
- **y_train** (`Matrix`)

## Implemented traits

`AnyType`, `CV`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, k: Int = 3, metric: String = "euc")
```

**Args:**

- **k** (`Int`)
- **metric** (`String`)
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
fn fit(mut self, X: Matrix, y: Matrix)
```

Fit the k-nearest neighbors classifier from the training dataset.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)
- **y** (`Matrix`)

**Raises:**

### `predict`

```mojo
fn predict(self, X: Matrix) -> Matrix
```

Predict the class indices for the provided data.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`Matrix`: Class indices for each data sample.

**Raises:**


