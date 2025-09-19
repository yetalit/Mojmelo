Mojo struct

# `LogisticRegression`

```mojo
@memory_only
struct LogisticRegression
```

A Gradient Descent based logistic regression with binary cross entropy as the loss function.

## Fields

- **lr** (`SIMD[float32, 1]`): Learning rate.
- **n_iters** (`Int`): The maximum number of iterations.
- **method** (`String`): Weight update method -> 'gradient' uses first derivative, 'newton' uses second derivative.
- **reg_alpha** (`SIMD[float32, 1]`): Constant that multiplies the regularization term.
- **l1_ratio** (`SIMD[float32, 1]`): The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
- **tol** (`SIMD[float32, 1]`): The stopping criterion based on loss.
- **batch_size** (`Int`): Batch size, with batch_size=1 corresponds to SGD, 1 < batch_size < n_samples corresponds to Mini-Batch Gradient Descent.
- **random_state** (`Int`): Used for shuffling the data.
- **weights** (`Matrix`): Weights per feature.
- **bias** (`SIMD[float32, 1]`): Bias term.

## Implemented traits

`AnyType`, `CVM`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, learning_rate: SIMD[float32, 1] = 0.001, n_iters: Int = 1000, method: String = "gradient", reg_alpha: SIMD[float32, 1] = 0, l1_ratio: SIMD[float32, 1] = 0, tol: SIMD[float32, 1] = 0, batch_size: Int = 0, random_state: Int = -1)
```

**Args:**

- **learning_rate** (`SIMD`)
- **n_iters** (`Int`)
- **method** (`String`)
- **reg_alpha** (`SIMD`)
- **l1_ratio** (`SIMD`)
- **tol** (`SIMD`)
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

Predict class for X.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`Matrix`: The predicted classes.

**Raises:**


