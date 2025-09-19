Mojo struct

# `GBDT`

```mojo
@memory_only
struct GBDT
```

Gradient Boosting with support for both classification and regression.

## Fields

- **criterion** (`String`): The method to measure the quality of a split:    For binary classification -> 'log'; For multi-class classification -> 'softmax';    For regression -> 'mse'.
- **loss_g** (`fn(Matrix, Matrix) raises -> Matrix`)
- **loss_h** (`fn(Matrix) raises -> Matrix`)
- **n_trees** (`Int`): The number of boosting stages to perform.
- **min_samples_split** (`Int`): The minimum number of samples required to split an internal node.
- **max_depth** (`Int`): The maximum depth of the tree.
- **learning_rate** (`SIMD[float32, 1]`): Learning rate.
- **reg_lambda** (`SIMD[float32, 1]`): The L2 regularization parameter.
- **reg_alpha** (`SIMD[float32, 1]`): The L1 regularization parameter.
- **gamma** (`SIMD[float32, 1]`): Minimum loss reduction required to make a further partition on a leaf node of the tree.
- **n_bins** (`Int`): Generates histogram boundaries as possible threshold values when n_bins >= 2 instead of all possible values.
- **trees** (`UnsafePointer[BDecisionTree]`)
- **score_start** (`SIMD[float32, 1]`)
- **num_class** (`Int`)

## Implemented traits

`AnyType`, `CVM`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, criterion: String = "log", n_trees: Int = 10, min_samples_split: Int = 10, max_depth: Int = 3, learning_rate: SIMD[float32, 1] = 0.10000000000000001, reg_lambda: SIMD[float32, 1] = 1, reg_alpha: SIMD[float32, 1] = 0, gamma: SIMD[float32, 1] = 0, n_bins: Int = 0)
```

**Args:**

- **criterion** (`String`)
- **n_trees** (`Int`)
- **min_samples_split** (`Int`)
- **max_depth** (`Int`)
- **learning_rate** (`SIMD`)
- **reg_lambda** (`SIMD`)
- **reg_alpha** (`SIMD`)
- **gamma** (`SIMD`)
- **n_bins** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn __init__(out self, params: Dict[String, String])
```

**Args:**

- **params** (`Dict`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `__del__`

```mojo
fn __del__(var self)
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


