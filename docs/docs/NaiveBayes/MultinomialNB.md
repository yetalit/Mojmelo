Mojo struct

# `MultinomialNB`

```mojo
@memory_only
struct MultinomialNB
```

Naive Bayes classifier for multinomial models.

## Fields

- **alpha** (`SIMD[float32, 1]`): Additive smoothing parameter.

## Implemented traits

`AnyType`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, alpha: SIMD[float32, 1] = 0)
```

**Args:**

- **alpha** (`SIMD`)
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
fn fit(mut self, X: Matrix, y: PythonObject)
```

Fit Naive Bayes classifier.

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


