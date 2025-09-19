Mojo struct

# `GaussianNB`

```mojo
@memory_only
struct GaussianNB
```

Gaussian Naive Bayes (GaussianNB).

## Fields

- **var_smoothing** (`SIMD[float32, 1]`): Portion of the largest variance of all features that is added to variances for calculation stability.

## Implemented traits

`AnyType`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, var_smoothing: SIMD[float32, 1] = 1.0E-8)
```

**Args:**

- **var_smoothing** (`SIMD`)
- **self** (`Self`)

**Returns:**

`Self`

### `fit`

```mojo
fn fit(mut self, X: Matrix, y: PythonObject)
```

Fit Gaussian Naive Bayes.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)
- **y** (`PythonObject`)

**Raises:**

### `predict`

```mojo
fn predict(self, X: Matrix) -> List[String]
```

Predict class for X.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`List`: The predicted classes.

**Raises:**


