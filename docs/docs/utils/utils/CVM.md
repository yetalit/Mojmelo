Mojo trait

# `CVM`

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
fn predict(self: _Self, X: Matrix) -> Matrix
```

**Args:**

- **self** (`_Self`)
- **X** (`Matrix`)

**Returns:**

`Matrix`

**Raises:**


