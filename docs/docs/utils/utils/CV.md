Mojo trait

# `CV`

## Aliases

- `__del__is_trivial = `: A flag (often compiler generated) to indicate whether the implementation of `__del__` is trivial. The implementation of `__del__` is considered to be trivial if:
- The struct has a compiler-generated trivial destructor and all its fields
  have a trivial `__del__` method.

In practice, it means that the `__del__` can be considered as no-op.

## Implemented traits

`AnyType`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self: _Self, params: Dict[String, String])
```

**Args:**

- **params** (`Dict`)
- **self** (`_Self`)

**Returns:**

`_Self`

**Raises:**

### `fit`

```mojo
fn fit(mut self: _Self, X: Matrix, y: Matrix)
```

**Args:**

- **self** (`_Self`)
- **X** (`Matrix`)
- **y** (`Matrix`)

**Raises:**

### `predict`

```mojo
fn predict(mut self: _Self, X: Matrix) -> Matrix
```

**Args:**

- **self** (`_Self`)
- **X** (`Matrix`)

**Returns:**

`Matrix`

**Raises:**


