Mojo struct

# `GaussianNB`

```mojo
@memory_only
struct GaussianNB
```

Gaussian Naive Bayes (GaussianNB).

## Aliases

- `__del__is_trivial = False`

## Fields

- **var_smoothing** (`Float32`): Portion of the largest variance of all features that is added to variances for calculation stability.

## Implemented traits

`AnyType`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, var_smoothing: Float32 = 1.0E-8)
```

**Args:**

- **var_smoothing** (`Float32`)
- **self** (`Self`)

**Returns:**

`Self`

### `fit`

```mojo
fn fit(mut self, X: Matrix, y: Matrix)
```

Fit Gaussian Naive Bayes.

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


