Mojo struct

# `KNN`

```mojo
@memory_only
struct KNN
```

Classifier implementing the k-nearest neighbors vote.

## Aliases

- `MODEL_ID = 4`
- `metric_ids = List(VariadicList("euc", "man"), Tuple())`

## Fields

- **k** (`Int`): Number of neighbors to use.
- **metric** (`String`): Metric to use for distance computation: Euclidean -> 'euc'; Manhattan -> 'man'.
- **kdtree** (`KDTree`)
- **y_train** (`Matrix`)

## Implemented traits

`AnyType`, `CV`, `Copyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
def __init__(out self, k: Int = 3, metric: String = "euc")
```

**Args:**

- **k** (`Int`)
- **metric** (`String`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
def __init__(out self, params: Dict[String, String])
```

**Args:**

- **params** (`Dict`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `fit`

```mojo
def fit(mut self, X: Matrix, y: Matrix)
```

Fit the k-nearest neighbors classifier from the training dataset.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)
- **y** (`Matrix`)

**Raises:**

### `predict`

```mojo
def predict(mut self, X: Matrix) -> Matrix
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


