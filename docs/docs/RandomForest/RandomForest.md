Mojo struct

# `RandomForest`

```mojo
@memory_only
struct RandomForest[criterion: String = "gini"]
```

A random forest supporting both classification and regression.

## Aliases

- `MODEL_ID = 10`
- `criterion_ids = List(String("mse"), String("entropy"), String("gini"), __list_literal__=NoneType(None))`

## Parameters

- **criterion** (`String`): The function to measure the quality of a split:
    For classification -> 'entropy', 'gini';
    For regression -> 'mse'.

## Fields

- **n_trees** (`Int`): The number of trees in the forest.
- **min_samples_split** (`Int`): The minimum number of samples required to split an internal node.
- **max_depth** (`Int`): The maximum depth of the tree.
- **n_feats** (`Int`): The number of features to consider when looking for the best split.
- **trees** (`UnsafePointer[DecisionTree[criterion], MutAnyOrigin]`)

## Implemented traits

`AnyType`, `CV`, `Copyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
fn __init__(out self, n_trees: Int = 10, min_samples_split: Int = 2, max_depth: Int = 100, n_feats: Int = -1, random_state: Int = 42)
```

**Args:**

- **n_trees** (`Int`)
- **min_samples_split** (`Int`)
- **max_depth** (`Int`)
- **n_feats** (`Int`)
- **random_state** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn __init__(out self, params: Dict[String, String])
```

**Args:**

- **params** (`Dict[String, String]`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `__del__`

```mojo
fn __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `fit`

```mojo
fn fit(mut self, X: Matrix, y: Matrix)
```

Build a forest of trees from the training set.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)
- **y** (`Matrix`)

**Raises:**

### `predict`

```mojo
fn predict(self, X: Matrix) -> Matrix
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
fn load[type: UInt8](path: String) -> RandomForest[(load_from_mem RandomForest[criterion].criterion_ids[type])]
```

Load a saved model from the specified path for prediction.

**Parameters:**

- **type** (`UInt8`)

**Args:**

- **path** (`String`)

**Returns:**

`RandomForest[(load_from_mem RandomForest[criterion].criterion_ids[type])]`

**Raises:**


