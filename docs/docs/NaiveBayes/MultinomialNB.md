Mojo struct

# `MultinomialNB`

```mojo
@memory_only
struct MultinomialNB
```

Naive Bayes classifier for multinomial models.

## Aliases

- `__del__is_trivial = False`

## Fields

- **alpha** (`Float32`): Additive smoothing parameter.

## Implemented traits

`AnyType`, `CV`, `ImplicitlyDestructible`

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

- **params** (`Dict`)
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


