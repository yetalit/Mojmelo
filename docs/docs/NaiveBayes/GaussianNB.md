Mojo struct

# `GaussianNB`

```mojo
@memory_only
struct GaussianNB
```

Gaussian Naive Bayes (GaussianNB).

## Aliases

- `MODEL_ID = 7`

## Fields

- **var_smoothing** (`Float32`): Portion of the largest variance of all features that is added to variances for calculation stability.

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyDeletable`, `Movable`

## Methods

### `__init__`

```mojo
fn def __init__(out self, var_smoothing: Float32 = 1.0E-8)
```

**Args:**

- **var_smoothing** (`Float32`)
- **self** (`Self`)

**Returns:**

`Self`

### `fit`

```mojo
fn def fit(mut self, X: Matrix, y: Matrix)
```

Fit Gaussian Naive Bayes.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)
- **y** (`Matrix`)

**Raises:**

### `predict`

```mojo
fn def predict(self, X: Matrix) -> Matrix
```

Predict class for X.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`Matrix`: The predicted classes.

**Raises:**

### `save`

```mojo
fn def save(self, path: String)
```

Save model data necessary for prediction to the specified path.

**Args:**

- **self** (`Self`)
- **path** (`String`)

**Raises:**

### `load`

```mojo
@staticmethod
fn def load(path: String) -> Self
```

Load a saved model from the specified path for prediction.

**Args:**

- **path** (`String`)

**Returns:**

`Self`

**Raises:**


