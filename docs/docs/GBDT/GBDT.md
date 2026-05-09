Mojo struct

# `GBDT`

```mojo
@memory_only
struct GBDT[criterion: String = "log"]
```

Gradient Boosting with support for both classification and regression.

## Aliases

- `loss_g = log_g if (criterion == String("log")) else softmax_g if (criterion == String("softmax")) else mse_g`
- `loss_h = log_h if (criterion == String("log")) else softmax_h if (criterion == String("softmax")) else mse_h`
- `MODEL_ID = 11`
- `criterion_ids = List(String("mse"), String("log"), String("softmax"), __list_literal__=NoneType(None))`

## Parameters

- **criterion** (`String`): The method to measure the quality of a split:
	For binary classification -> 'log';
	For multi-class classification -> 'softmax';
	For regression -> 'mse'.

## Fields

- **n_trees** (`Int`): The number of boosting stages to perform.
- **min_samples_split** (`Int`): The minimum number of samples required to split an internal node.
- **max_depth** (`Int`): The maximum depth of the tree.
- **learning_rate** (`Float32`): Learning rate.
- **reg_lambda** (`Float32`): The L2 regularization parameter.
- **reg_alpha** (`Float32`): The L1 regularization parameter.
- **gamma** (`Float32`): Minimum loss reduction required to make a further partition on a leaf node of the tree.
- **n_bins** (`Int`): Generates histogram boundaries as possible threshold values when n_bins >= 2 instead of all possible values.
- **trees** (`UnsafePointer[BDecisionTree, MutAnyOrigin]`)
- **score_start** (`Float32`)
- **num_class** (`Int`)

## Implemented traits

`AnyType`, `CV`, `Copyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
fn __init__(out self, n_trees: Int = 10, min_samples_split: Int = 10, max_depth: Int = 3, learning_rate: Float32 = 0.10000000000000001, reg_lambda: Float32 = 1, reg_alpha: Float32 = 0, gamma: Float32 = 0, n_bins: Int = 0)
```

**Args:**

- **n_trees** (`Int`)
- **min_samples_split** (`Int`)
- **max_depth** (`Int`)
- **learning_rate** (`Float32`)
- **reg_lambda** (`Float32`)
- **reg_alpha** (`Float32`)
- **gamma** (`Float32`)
- **n_bins** (`Int`)
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

Fit the gradient boosting model.

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
fn load[type: UInt8](path: String) -> GBDT[(load_from_mem GBDT[criterion].criterion_ids[type])]
```

Load a saved model from the specified path for prediction.

**Parameters:**

- **type** (`UInt8`)

**Args:**

- **path** (`String`)

**Returns:**

`GBDT[(load_from_mem GBDT[criterion].criterion_ids[type])]`

**Raises:**


