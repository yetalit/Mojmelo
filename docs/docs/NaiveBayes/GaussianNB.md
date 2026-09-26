Mojo struct

# `GaussianNB`

```mojo
@memory_only
struct GaussianNB
```

Gaussian Naive Bayes (GaussianNB).

Assumes the likelihood of each feature, conditioned on the class, follows
a Gaussian distribution. Suited for continuous, real-valued features.

## Aliases

- `MODEL_ID = 7`

## Fields

- **var_smoothing** (`Float32`): Portion of the largest variance of all features that is added to variances for calculation stability.

## Implemented traits

`AnyType`, `CV`, `Copyable`, `Deinitable`, `Movable`

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

```mojo
fn def __init__(out self, params: Dict[String, String])
```

Construct from a hyperparameter dictionary.

**Args:**

- **params** (`Dict[String, String]`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `fit`

```mojo
fn def fit(mut self, X: Matrix, y: Matrix)
```

Fit Gaussian Naive Bayes.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`): Training features of shape (n_samples, n_features).
- **y** (`Matrix`): Training labels of shape (n_samples, 1), encoded as contiguous
    non-negative integers starting at 0.

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


