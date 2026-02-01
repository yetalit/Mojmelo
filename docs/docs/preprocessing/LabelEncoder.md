Mojo struct

# `LabelEncoder`

```mojo
@memory_only
struct LabelEncoder
```

Encode target labels with value between 0 and n_classes-1. This transformer can be used to encode target values from numpy, and not the input X.

## Aliases

- `__del__is_trivial = False`

## Fields

- **str_to_index** (`Dict[String, Int]`)
- **index_to_str** (`Dict[Int, String]`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
fn __init__(out self)
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `fit_transform`

```mojo
fn fit_transform(mut self, y: PythonObject) -> Matrix
```

Fit label encoder and return encoded labels.      Args:     y: Targets Python object.

**Args:**

- **self** (`Self`)
- **y** (`PythonObject`)

**Returns:**

`Matrix`: Encoded labels.

**Raises:**

### `transform`

```mojo
fn transform(self, y: PythonObject) -> Matrix
```

Return encoded labels based on fitted encoder.

**Args:**

- **self** (`Self`)
- **y** (`PythonObject`): Targets Python object.

**Returns:**

`Matrix`: Encoded labels.

**Raises:**

### `inverse_transform`

```mojo
fn inverse_transform(self, y: Matrix) -> PythonObject
```

Transform labels back to original encoding.      Args:     y: Encoded targets.

**Args:**

- **self** (`Self`)
- **y** (`Matrix`)

**Returns:**

`PythonObject`: Original targets Python object.

**Raises:**


