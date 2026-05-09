Mojo struct

# `KNN`

```mojo
@memory_only
struct KNN[metric: String = "euc"]
```

Classifier implementing the k-nearest neighbors vote.

## Aliases

- `MODEL_ID = 4`
- `metric_ids = List(String("euc"), String("man"), __list_literal__=NoneType(None))`

## Parameters

- **metric** (`String`): Metric to use for distance computation:
    Euclidean -> 'euc';
    Manhattan -> 'man'.

## Fields

- **k** (`Int`): Number of neighbors to use.
- **search_depth** (`Int`): Current KDTree implementation applies some approximation to its search results. Increasing search_depth can lead to more accurate results at the cost of performance.
- **kdtree** (`KDTree[True, metric=metric]`)
- **y_train** (`Matrix`)

## Implemented traits

`AnyType`, `CV`, `Copyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
fn __init__(out self, k: Int = 3, search_depth: Int = 1)
```

**Args:**

- **k** (`Int`)
- **search_depth** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn __init__(out self, params: Dict[String, String])
```

**Args:**

- **params** (`Dict[String, String]`)
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
fn predict(mut self, X: Matrix) -> Matrix
```

Predict the class indices for the provided data.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`Matrix`: Class indices for each data sample.

**Raises:**

### `save`

```mojo
fn save(self, path: String)
```

Save model data necessary for prediction to the specified path.

**Args:**

- **self** (`Self`)
- **path** (`String`)

**Raises:**

### `load`

```mojo
@staticmethod
fn load[type: UInt8](path: String) -> KNN[(load_from_mem KNN[metric].metric_ids[type])]
```

Load a saved model from the specified path for prediction.

**Parameters:**

- **type** (`UInt8`)

**Args:**

- **path** (`String`)

**Returns:**

`KNN[(load_from_mem KNN[metric].metric_ids[type])]`

**Raises:**


