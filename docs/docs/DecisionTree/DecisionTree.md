Mojo struct

# `DecisionTree`

```mojo
@memory_only
struct DecisionTree
```

A decision tree supporting both classification and regression.

## Fields

- **criterion** (`String`): The function to measure the quality of a split: For classification -> 'entropy', 'gini'; For regression -> 'mse'.
- **loss_func** (`fn(Matrix) raises -> SIMD[float32, 1]`)
- **c_func** (`fn(SIMD[float32, 1], List[Int]) raises -> SIMD[float32, 1]`)
- **r_func** (`fn(Int, SIMD[float32, 1], SIMD[float32, 1]) raises -> SIMD[float32, 1]`)
- **min_samples_split** (`Int`): The minimum number of samples required to split an internal node.
- **max_depth** (`Int`): The maximum depth of the tree.
- **n_feats** (`Int`): The number of features to consider when looking for the best split.
- **root** (`UnsafePointer[Node]`)

## Implemented traits

`AnyType`, `CVM`, `Copyable`, `Movable`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, criterion: String = "gini", min_samples_split: Int = 2, max_depth: Int = 100, n_feats: Int = -1, random_state: Int = 42)
```

**Args:**

- **criterion** (`String`)
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

Build a decision tree from the training set.

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


