Mojo struct

# `KNN`

```mojo
@memory_only
struct KNN[EUC: Bool = False]
```

Classifier implementing the k-nearest neighbors vote.

## Aliases

- `MODEL_ID = 4`
- `metric_ids = List(String("euc"), String("man"), __list_literal__=NoneType(None))`

## Parameters

- **EUC** (`Bool`): Setting EUC=True lets compiler optimize Euclidean distance calculations.

## Fields

- **k** (`Int`): Number of neighbors to use.
- **metric** (`String`): Metric to use for distance computation: Euclidean -> 'euc'; Manhattan -> 'man'.
- **kdtree** (`KDTree[EUC=EUC]`)
- **y_train** (`Matrix`)

## Implemented traits

`AnyType`, `CV`, `Copyable`, `Deinitable`, `Movable`

## Methods

### `__init__`

```mojo
fn def __init__(out self, k: Int = Int(3), metric: String = "euc")
```

**Args:**

- **k** (`Int`)
- **metric** (`String`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def __init__(out self, params: Dict[String, String])
```

**Args:**

- **params** (`Dict[String, String]`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `fit`

```mojo
fn def fit(mut self, X: Matrix, y: Matrix)
```

Fit the k-nearest neighbors classifier from the training dataset.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)
- **y** (`Matrix`)

**Raises:**

### `predict`

```mojo
fn def predict(self, X: Matrix) -> Matrix
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
fn def save(self, path: String)
```

Save model data necessary for prediction to the specified path.

**Args:**

- **self** (`Self`)
- **path** (`String`)

**Raises:**

### `load`

```mojo
@staticmethod
fn def load(path: String) -> Self
```

Load a saved model from the specified path for prediction.

**Args:**

- **path** (`String`)

**Returns:**

`Self`

**Raises:**


