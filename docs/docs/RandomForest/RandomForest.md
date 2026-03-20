Mojo struct

# `RandomForest`

```mojo
@memory_only
struct RandomForest
```

A random forest supporting both classification and regression.

## Aliases

- `MODEL_ID = 10`
- `criterion_ids = List(VariadicList("entropy", "gini", "mse"), Tuple())`

## Fields

- **n_trees** (`Int`): The number of trees in the forest.
- **min_samples_split** (`Int`): The minimum number of samples required to split an internal node.
- **max_depth** (`Int`): The maximum depth of the tree.
- **n_feats** (`Int`): The number of features to consider when looking for the best split.
- **criterion** (`String`): The function to measure the quality of a split: For classification -> 'entropy', 'gini'; For regression -> 'mse'.
- **trees** (`UnsafePointer[DecisionTree, MutAnyOrigin]`)

## Implemented traits

`AnyType`, `CV`, `Copyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
def __init__(out self, n_trees: Int = 10, min_samples_split: Int = 2, max_depth: Int = 100, n_feats: Int = -1, criterion: String = "gini", random_state: Int = 42)
```

**Args:**

- **n_trees** (`Int`)
- **min_samples_split** (`Int`)
- **max_depth** (`Int`)
- **n_feats** (`Int`)
- **criterion** (`String`)
- **random_state** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
def __init__(out self, params: Dict[String, String])
```

**Args:**

- **params** (`Dict`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `__del__`

```mojo
def __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `fit`

```mojo
def fit(mut self, X: Matrix, y: Matrix)
```

Build a forest of trees from the training set.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)
- **y** (`Matrix`)

**Raises:**

### `predict`

```mojo
def predict(self, X: Matrix) -> Matrix
```

Predict class or regression value for X.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`Matrix`: The predicted values.

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


