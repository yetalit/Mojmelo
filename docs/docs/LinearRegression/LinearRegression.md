Mojo struct

# `LinearRegression`

```mojo
@memory_only
struct LinearRegression
```

A Gradient Descent based linear regression with mse as the loss function.

## Aliases

- `__del__is_trivial = False`

## Fields

- **lr** (`Float32`): Learning rate.
- **n_iters** (`Int`): The maximum number of iterations.
- **reg_alpha** (`Float32`): Constant that multiplies the regularization term.
- **l1_ratio** (`Float32`): The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
- **tol** (`Float32`): The stopping criterion based on loss.
- **batch_size** (`Int`): Batch size, with batch_size=1 corresponds to SGD, 1 < batch_size < n_samples corresponds to Mini-Batch Gradient Descent.
- **random_state** (`Int`): Used for shuffling the data.
- **weights** (`Matrix`): Weights per feature.
- **bias** (`Float32`): Bias term.

## Implemented traits

`AnyType`, `CV`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, learning_rate: Float32 = 0.001, n_iters: Int = 1000, reg_alpha: Float32 = 0, l1_ratio: Float32 = -1, tol: Float32 = 0, batch_size: Int = 0, random_state: Int = -1)
```

**Args:**

- **learning_rate** (`Float32`)
- **n_iters** (`Int`)
- **reg_alpha** (`Float32`)
- **l1_ratio** (`Float32`)
- **tol** (`Float32`)
- **batch_size** (`Int`)
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

### `fit`

```mojo
fn fit(mut self, X: Matrix, y: Matrix)
```

Fit the model.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)
- **y** (`Matrix`)

**Raises:**

### `predict`

```mojo
fn predict(self, X: Matrix) -> Matrix
```

Predict regression values for X.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`Matrix`: The predicted values.

**Raises:**


