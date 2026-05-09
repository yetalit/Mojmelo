Mojo struct

# `MultinomialNB`

```mojo
@memory_only
struct MultinomialNB
```

Naive Bayes classifier for multinomial models.

## Aliases

- `MODEL_ID = 8`

## Fields

- **alpha** (`Float32`): Additive smoothing parameter.

## Implemented traits

`AnyType`, `CV`, `Copyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
fn __init__(out self, alpha: Float32 = 0)
```

**Args:**

- **alpha** (`Float32`)
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

### `fit`

```mojo
fn fit(mut self, X: Matrix, y: Matrix)
```

Fit Naive Bayes classifier.

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
fn load(path: String) -> Self
```

Load a saved model from the specified path for prediction.

**Args:**

- **path** (`String`)

**Returns:**

`Self`

**Raises:**


