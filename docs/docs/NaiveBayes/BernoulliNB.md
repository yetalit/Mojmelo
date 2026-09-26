Mojo struct

# `BernoulliNB`

```mojo
@memory_only
struct BernoulliNB
```

Naive Bayes classifier for multivariate Bernoulli models.

Suited for discrete, binary/boolean features. Each feature is binarized
against `binarize` (if it isn't already boolean) before being modelled
with an independent Bernoulli distribution per class.

## Aliases

- `MODEL_ID = 13`

## Fields

- **alpha** (`Float32`): Additive (Laplace/Lidstone) smoothing parameter. Must be non-negative.
- **binarize** (`Float32`): Threshold for binarizing features: values strictly greater than this become 1, others become 0.

## Implemented traits

`AnyType`, `CV`, `Copyable`, `Deinitable`, `Movable`

## Methods

### `__init__`

```mojo
fn def __init__(out self, alpha: Float32 = 0, binarize: Float32 = 0)
```

**Args:**

- **alpha** (`Float32`)
- **binarize** (`Float32`)
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

Fit Bernoulli Naive Bayes classifier.

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


