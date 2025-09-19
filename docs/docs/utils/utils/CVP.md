Mojo trait

# `CVP`

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
fn fit(mut self: _Self, X: Matrix, y: PythonObject)
```

**Args:**

- **self** (`_Self`)
- **X** (`Matrix`)
- **y** (`PythonObject`)

**Raises:**

### `predict`

```mojo
fn predict(self: _Self, X: Matrix) -> List[String]
```

**Args:**

- **self** (`_Self`)
- **X** (`Matrix`)

**Returns:**

`List`

**Raises:**


